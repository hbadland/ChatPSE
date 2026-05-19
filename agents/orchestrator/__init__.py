"""
Orchestrator: end-to-end pipeline controller for the multi-agent flowsheet system.

Pipeline stages
───────────────
1. Basis      — compound name normalisation (alias → DWSIM name)
2. Planner    — natural language → flowsheet JSON (units, streams, connections)
3. Thermo     — assign property packages globally and per-unit
4. Executor   — run DWSIM simulation, collect stream results
5. Critic     — evaluate results: physics checks + LLM diagnosis → routing
6. Refiner    — deterministic + LLM fix application
   Loop 4-6 until PASS or max iterations.

Routing
───────
  PASS    → pipeline complete
  REFINER → RefinerAgent patches the flowsheet, re-run Executor
  THERMO  → ThermoAgent re-assigns packages, re-run Executor
  BASIS   → BasisAgent re-runs LLM stage, PlannerAgent re-plans, restart loop
  HUMAN   → escalate to user, stop
  REPLAN  → PlannerAgent re-invoked with structured failure feedback

OrchestratorResult
──────────────────
  outcome     : "PASS" | "HUMAN" | "MAX_ITER" | "BASIS_FAILED" | "PLAN_FAILED"
  iterations  : full audit trail — IterationRecord per loop
  final_*     : last flowsheet, execution, and critic report
"""
from __future__ import annotations
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Optional

from agents.basis           import BasisAgent,       BasisResult
from agents.planner         import PlannerAgent
from agents.thermo          import ThermoAgent
from agents.executor        import Executor
from agents.critic          import CriticAgent,      CriticReport
from agents.refiner         import RefinerAgent,     RefinementResult
from agents.calibration     import CalibrationAgent, CalibrationResult
from agents.topology_library import match as topology_match, format_hint
from agents.llm             import DEFAULT_MODEL


# ── Iteration record ──────────────────────────────────────────────────────────

@dataclass
class IterationRecord:
    iteration:      int
    routing:        str               # routing decision from the Critic
    flowsheet:      dict              # flowsheet sent to the Executor this turn
    execution:      object            # ExecutionResult (typed loosely to avoid import cycle)
    report:         CriticReport
    refinement:     Optional[RefinementResult] = None
    elapsed_s:      float = 0.0
    thermo_error:   str | None = None
    calibration:    Optional[CalibrationResult] = None


# ── Trial record — one per executed iteration, passed to agents as feedback ───

@dataclass
class TrialRecord:
    iteration:         int
    property_package:  str
    failure_codes:     list[str]   # from CriticReport.failure_codes
    diagnosis:         str          # CriticReport.diagnosis
    execution_summary: str          # "outlet≈feed"|"zero_vapour"|"solver_diverged"|
                                    # "wrong_phase"|"numeric_fail"|"pass"|"unknown"
    refiner_outcome:   str | None   # "success" | "failed" | "not_attempted"
    calibration_tried: bool = False


# ── Final result ──────────────────────────────────────────────────────────────

@dataclass
class OrchestratorResult:
    description:       str
    outcome:           str           # PASS | HUMAN | MAX_ITER | BASIS_FAILED | PLAN_FAILED
    basis_result:      Optional[BasisResult]    = None
    final_flowsheet:   Optional[dict]           = None
    final_execution:   object                   = None   # ExecutionResult
    final_report:      Optional[CriticReport]   = None
    iterations:        list[IterationRecord]    = field(default_factory=list)
    total_time_s:      float                    = 0.0
    basis_reruns:      int                      = 0
    warnings:          list[str]               = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.outcome == "PASS"

    def summary(self) -> str:
        lines = [
            f"Outcome  : {self.outcome}",
            f"Iterations: {len(self.iterations)}   "
            f"Basis reruns: {self.basis_reruns}   "
            f"Time: {self.total_time_s:.1f}s",
        ]
        if self.basis_result:
            lines.append(f"Compounds: {self.basis_result.dwsim_compounds}")
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            lines.extend(f"  ! {w}" for w in self.warnings)
        for rec in self.iterations:
            tag = "✓" if rec.routing == "PASS" else "✗"
            lines.append(f"  [{tag}] iter {rec.iteration}  routing={rec.routing}  "
                         f"solved={getattr(rec.execution, 'solved', '?')}  "
                         f"{rec.elapsed_s:.1f}s")
            if rec.refinement:
                for c in rec.refinement.changes:
                    lines.append(f"      [{c.failure_code}] {c.target}.{c.field}: "
                                 f"{c.old_value!r} → {c.new_value!r}")
        return "\n".join(lines)


# ── Orchestrator ──────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Drives the full multi-agent pipeline from natural language to DWSIM results.

    Args:
        model         : LLM model string (shared across all agents)
        max_iterations: maximum Executor→Critic→Refiner cycles before giving up
        max_basis_reruns: maximum times BASIS routing is allowed before escalating
    """

    def __init__(
            self,
            model:            str = DEFAULT_MODEL,
            max_iterations:   int = 6,
            max_basis_reruns: int = 1,
    ):
        if max_iterations < 2:
            raise ValueError(
                "max_iterations must be at least 2 to allow one refinement attempt "
                "before INFEASIBLE routing.")

        self._model            = model
        self._max_iterations   = max_iterations
        self._max_basis_reruns = max_basis_reruns

        self._basis        = BasisAgent(model=model)
        self._planner      = PlannerAgent(model=model)
        self._thermo       = ThermoAgent(model=model)
        self._executor     = Executor()
        self._critic       = CriticAgent(model=model,
                                         infeasible_threshold=max(1, max_iterations - 1))
        self._refiner      = RefinerAgent(model=model)
        self._calibration  = CalibrationAgent()

    def run(self, description: str) -> OrchestratorResult:
        """
        Run the full pipeline for a natural language process description.
        Returns an OrchestratorResult regardless of outcome — never raises.
        """
        t_start = time.time()

        result = OrchestratorResult(description=description, outcome="MAX_ITER")

        # ── Stage 1: Basis ────────────────────────────────────────────────────
        basis = self._run_basis(description)
        result.basis_result = basis

        if not basis.success:
            result.outcome    = "BASIS_FAILED"
            result.total_time_s = time.time() - t_start
            return result

        # ── Stage 2: Pre-select property package (ThermoAgent) ───────────────
        # ThermoAgent applies its hard rules to the compound set BEFORE the
        # Planner draws topology, so the Planner receives the package as a
        # hard constraint and never needs to reason about thermodynamics.
        original_description = basis.normalised_description
        replan_count = 0
        MAX_REPLANS  = 2

        # ── Topology library: zero-LLM template matching ──────────────────────
        # Match the description against known design patterns. The hint object
        # is passed directly to PlannerAgent; TopologyAgent uses it as Stage 1
        # (free, instant) and only falls back to LLM when no template matches.
        topology_hint = topology_match(original_description, basis.dwsim_compounds)
        if topology_hint:
            result.warnings.append(f"TopologyLibrary: {format_hint(topology_hint)}")

        pre_selected_package = "Raoult's Law"   # safe default
        try:
            pre_selected_package, pkg_reasoning = self._thermo.pre_select(
                basis.dwsim_compounds, original_description)
            result.warnings.append(
                f"ThermoAgent.pre_select: '{pre_selected_package}' — {pkg_reasoning}")
        except Exception as exc:
            result.warnings.append(
                f"ThermoAgent.pre_select failed ({type(exc).__name__}: {exc}); "
                "defaulting to Raoult's Law — ThermoAgent.assign() will correct post-plan.")

        # Build a lightweight condition estimate (bubble-point range) to give
        # the Planner numerical T/P guidance without an extra LLM call.
        condition_estimate = _build_preplan_condition_estimate(
            basis.dwsim_compounds, basis.suggested_compositions or {})

        # ── Stage 3: Planner (topology-only — package already constrained) ────
        try:
            flowsheet = self._planner.plan(
                original_description,
                compounds=basis.dwsim_compounds,
                suggested_compositions=basis.suggested_compositions or None,
                property_package=pre_selected_package,
                condition_estimate=condition_estimate or None,
                topology_hint=topology_hint,
            )
        except Exception as exc:
            result.warnings.append(
                f"PlannerAgent failed: {type(exc).__name__}: {exc}")
            result.outcome      = "PLAN_FAILED"
            result.total_time_s = time.time() - t_start
            return result

        # ── Stage 3b: Thermo assign (unit-level overrides / verification) ─────
        tried_packages: set[str] = set()
        try:
            flowsheet, _ = self._thermo.assign(flowsheet)
            tried_packages.add(flowsheet.get("property_package", ""))
        except Exception as exc:
            result.warnings.append(
                f"ThermoAgent.assign failed at Stage 3b (unit-level verification): "
                f"{type(exc).__name__}: {exc}. Using Planner-assigned package.")

        # ── Stage 3b: Proactive calibration + pre-flight ─────────────────────
        flowsheet, calib_msg = _proactive_calibration(flowsheet, self._calibration)
        if calib_msg:
            result.warnings.append(calib_msg)
        flowsheet, pf_patches = _preflight_patch(flowsheet)
        for msg in pf_patches:
            result.warnings.append(msg)

        # ── Stages 4-6: Execute → Critique → Refine loop ─────────────────────
        basis_reruns        = 0
        run_history:        list[TrialRecord] = []
        consecutive_thermo: int  = 0
        refiner_run_count:  int  = 0
        last_calib_success: bool = False
        recent_codes:       list[str] = []
        seen_flowsheet_hashes: set[str] = set()

        for iteration in range(self._max_iterations):
            t_iter = time.time()

            # ── Pre-flight: zero-LLM fixes before every Executor call ────────
            # A: inject BIPs if corpus has them; no-op if already injected or
            #    package is not NRTL/UNIQUAC.
            flowsheet, calib_msg = _proactive_calibration(flowsheet, self._calibration)
            if calib_msg:
                result.warnings.append(f"[iter {iteration}] {calib_msg}")
            # B: patch sub-bubble-point T_out and obvious Vessel port errors.
            flowsheet, pf_patches = _preflight_patch(flowsheet)
            for msg in pf_patches:
                result.warnings.append(f"[iter {iteration}] {msg}")

            # Cycling detection (after pre-flight so patches can break hash cycles)
            fhash = _flowsheet_hash(flowsheet)
            if fhash in seen_flowsheet_hashes:
                result.outcome = "HUMAN"
                result.warnings.append(
                    f"Cycling detected at iteration {iteration}: "
                    f"flowsheet hash {fhash} was already executed. Escalating to HUMAN."
                )
                break
            seen_flowsheet_hashes.add(fhash)

            execution = self._executor.run(flowsheet)
            report    = self._critic.critique(execution, flowsheet, iteration=iteration)
            routing   = report.routing

            rec = IterationRecord(
                iteration=iteration,
                routing=routing,
                flowsheet=flowsheet,
                execution=execution,
                report=report,
                elapsed_s=time.time() - t_iter,
            )
            result.iterations.append(rec)
            result.final_flowsheet  = flowsheet
            result.final_execution  = execution
            result.final_report     = report

            trial = TrialRecord(
                iteration=iteration,
                property_package=flowsheet.get("property_package", ""),
                failure_codes=list(report.failure_codes),
                diagnosis=report.diagnosis,
                execution_summary=_execution_summary(report),
                refiner_outcome=None,
            )
            run_history.append(trial)

            # ── Trigger 7: CalibrationAgent succeeded last iter but re-solve failed ──
            if last_calib_success and not report.passed and replan_count < MAX_REPLANS:
                replan_count += 1
                last_calib_success = False
                new_fs, err, tier, new_hint = self._do_replan(
                    "post_calibration", report, run_history, flowsheet,
                    original_description, result, replan_count,
                    pre_selected_package=pre_selected_package,
                    topology_hint=topology_hint)
                if tier == "VALUE_PATCH":
                    replan_count -= 1
                if err:
                    result.warnings.append(
                        f"REPLAN (post_calibration) failed: {err}")
                    result.outcome = "HUMAN"
                    break
                flowsheet = new_fs
                topology_hint = new_hint or topology_hint
                if tier != "VALUE_PATCH":
                    tried_packages = {flowsheet.get("property_package", "")}
                    consecutive_thermo = 0
                continue
            last_calib_success = False

            # ── Trigger 6: Stagnation — same failure code 3 consecutive iterations ──
            primary = report.failure_codes[0] if report.failure_codes else "NONE"
            recent_codes.append(primary)
            if len(recent_codes) > 3:
                recent_codes.pop(0)
            if (len(recent_codes) == 3 and len(set(recent_codes)) == 1
                    and primary != "NONE" and replan_count < MAX_REPLANS):
                replan_count += 1
                recent_codes.clear()
                new_fs, err, tier, new_hint = self._do_replan(
                    "stagnation", report, run_history, flowsheet,
                    original_description, result, replan_count,
                    pre_selected_package=pre_selected_package,
                    topology_hint=topology_hint)
                if tier == "VALUE_PATCH":
                    replan_count -= 1
                if err:
                    result.warnings.append(f"REPLAN (stagnation) failed: {err}")
                    result.outcome = "HUMAN"
                    break
                flowsheet = new_fs
                topology_hint = new_hint or topology_hint
                if tier != "VALUE_PATCH":
                    tried_packages = {flowsheet.get("property_package", "")}
                    consecutive_thermo = 0
                continue

            # ── Route ────────────────────────────────────────────────────────

            if routing == "PASS":
                consecutive_thermo = 0
                result.outcome = "PASS"
                break

            # ── Trigger 5: Last-chance REPLAN before HUMAN ────────────────────
            if routing == "HUMAN":
                consecutive_thermo = 0
                if replan_count < MAX_REPLANS:
                    replan_count += 1
                    new_fs, err, tier, new_hint = self._do_replan(
                        "human_last_chance", report, run_history, flowsheet,
                        original_description, result, replan_count,
                    pre_selected_package=pre_selected_package,
                    topology_hint=topology_hint)
                    if tier == "VALUE_PATCH":
                        replan_count -= 1
                    if err:
                        result.warnings.append(
                            f"REPLAN (human_last_chance) failed: {err}")
                        result.outcome = "HUMAN"
                        break
                    flowsheet = new_fs
                    topology_hint = new_hint or topology_hint
                    if tier != "VALUE_PATCH":
                        tried_packages = {flowsheet.get("property_package", "")}
                    continue
                else:
                    result.outcome = "HUMAN"
                    break

            # ── ACTIVITY→CALIBRATION redirect ─────────────────────────────────
            # PARAM_MISSING only fires when DWSIM solved=True (outlet≈feed).
            # When missing BIPs cause a hard crash (SOLVER_FAIL, solved=False),
            # the Critic routes to REFINER — but the Refiner can't inject BIPs.
            # Pre-check the BIP corpus: if it covers this pair, CALIBRATION is
            # the right first step.  If not (topology-caused SOLVER_FAIL), keep
            # REFINER so it can fix the topology without wasting an iteration.
            if (routing == "REFINER"
                    and flowsheet.get("property_package") in ("NRTL", "UNIQUAC")
                    and any(s.code == "SOLVER_FAIL" for s in report.signals)
                    and not trial.calibration_tried
                    and self._calibration.has_coverage(flowsheet)):
                routing = "CALIBRATION"

            if routing == "REFINER":
                consecutive_thermo = 0
                refined = self._refiner.refine(flowsheet, report, run_history=run_history)
                rec.refinement = refined
                if not refined.success:
                    trial.refiner_outcome = "failed"
                    # ── Trigger 1: Refiner structurally failed → REPLAN ───────
                    if replan_count < MAX_REPLANS:
                        replan_count += 1
                        new_fs, err, tier, new_hint = self._do_replan(
                            "refiner_failed", report, run_history, flowsheet,
                            original_description, result, replan_count,
                    pre_selected_package=pre_selected_package,
                    topology_hint=topology_hint)
                        if tier == "VALUE_PATCH":
                            replan_count -= 1
                        if err:
                            result.warnings.append(
                                f"REPLAN (refiner_failed) failed: {err}")
                            result.outcome = "HUMAN"
                            break
                        flowsheet = new_fs
                        topology_hint = new_hint or topology_hint
                        if tier != "VALUE_PATCH":
                            tried_packages = {flowsheet.get("property_package", "")}
                    else:
                        result.outcome = "HUMAN"
                        break
                else:
                    trial.refiner_outcome = "success"
                    flowsheet = refined.updated_flowsheet
                    refiner_run_count += 1
                    # ── Trigger 4: Refiner ran ≥ 2 times without PASS → REPLAN ─
                    if refiner_run_count >= 2 and replan_count < MAX_REPLANS:
                        replan_count += 1
                        new_fs, err, tier, new_hint = self._do_replan(
                            "refiner_ineffective", report, run_history, flowsheet,
                            original_description, result, replan_count,
                    pre_selected_package=pre_selected_package,
                    topology_hint=topology_hint)
                        if tier == "VALUE_PATCH":
                            replan_count -= 1
                        if err:
                            result.warnings.append(
                                f"REPLAN (refiner_ineffective) failed: {err}")
                            result.outcome = "HUMAN"
                            break
                        flowsheet = new_fs
                        topology_hint = new_hint or topology_hint
                        refiner_run_count = 0
                        if tier != "VALUE_PATCH":
                            tried_packages = {flowsheet.get("property_package", "")}
                            consecutive_thermo = 0

            elif routing == "CALIBRATION":
                consecutive_thermo = 0
                calib = self._calibration.run(flowsheet)
                rec.calibration = calib
                trial.calibration_tried = True
                for note in calib.notes:
                    result.warnings.append(f"[CALIBRATION] {note}")
                if calib.success:
                    flowsheet = calib.updated_flowsheet
                    last_calib_success = True
                else:
                    result.warnings.append(
                        "CalibrationAgent: could not supply parameters; "
                        "falling back to ThermoAgent."
                    )
                    try:
                        tried_packages.add(flowsheet.get("property_package", ""))
                        flowsheet.pop("binary_parameters", None)
                        flowsheet, _ = self._thermo.assign(
                            flowsheet,
                            exclude_packages=tried_packages,
                            trial_history=run_history)
                        tried_packages.add(flowsheet.get("property_package", ""))
                    except Exception as exc:
                        err = (f"ThermoAgent failed during CALIBRATION fallback "
                               f"at iteration {iteration}: {type(exc).__name__}: {exc}")
                        result.warnings.append(err)
                        rec.thermo_error = err
                        result.outcome = "HUMAN"
                        break

            elif routing == "THERMO":
                # ── Trigger 3: all packages exhausted — preemptive REPLAN ─────
                from agents.schema import SUPPORTED_PROPERTY_PACKAGES
                if tried_packages.issuperset(SUPPORTED_PROPERTY_PACKAGES):
                    if replan_count < MAX_REPLANS:
                        replan_count += 1
                        new_fs, err, tier, new_hint = self._do_replan(
                            "packages_exhausted", report, run_history, flowsheet,
                            original_description, result, replan_count,
                    pre_selected_package=pre_selected_package,
                    topology_hint=topology_hint)
                        if tier == "VALUE_PATCH":
                            replan_count -= 1
                        if err:
                            result.warnings.append(
                                f"REPLAN (packages_exhausted) failed: {err}")
                            result.outcome = "HUMAN"
                            break
                        flowsheet = new_fs
                        topology_hint = new_hint or topology_hint
                        if tier != "VALUE_PATCH":
                            tried_packages = {flowsheet.get("property_package", "")}
                            consecutive_thermo = 0
                        continue
                    else:
                        result.outcome = "HUMAN"
                        break

                # ── Trigger 2: THERMO ≥ 2 consecutive → REPLAN ───────────────
                consecutive_thermo += 1
                if consecutive_thermo >= 2 and replan_count < MAX_REPLANS:
                    replan_count += 1
                    new_fs, err, tier, new_hint = self._do_replan(
                        "thermo_cycle", report, run_history, flowsheet,
                        original_description, result, replan_count,
                    pre_selected_package=pre_selected_package,
                    topology_hint=topology_hint)
                    if tier == "VALUE_PATCH":
                        replan_count -= 1
                    if err:
                        result.warnings.append(
                            f"REPLAN (thermo_cycle) failed: {err}")
                        result.outcome = "HUMAN"
                        break
                    flowsheet = new_fs
                    topology_hint = new_hint or topology_hint
                    if tier != "VALUE_PATCH":
                        tried_packages = {flowsheet.get("property_package", "")}
                        consecutive_thermo = 0
                    continue

                # Normal THERMO path
                try:
                    tried_packages.add(flowsheet.get("property_package", ""))
                    flowsheet, _ = self._thermo.assign(
                        flowsheet,
                        exclude_packages=tried_packages,
                        trial_history=run_history)
                    tried_packages.add(flowsheet.get("property_package", ""))
                except Exception as exc:
                    err = (f"ThermoAgent failed on THERMO routing at iteration "
                           f"{iteration}: {type(exc).__name__}: {exc}")
                    result.warnings.append(err)
                    rec.thermo_error = err
                    result.outcome = "HUMAN"
                    break

            elif routing == "BASIS":
                consecutive_thermo = 0
                if basis_reruns >= self._max_basis_reruns:
                    result.outcome      = "HUMAN"
                    result.basis_reruns = basis_reruns
                    break
                basis_reruns += 1
                # Combine raw DWSIM errors with the Critic's interpreted diagnosis
                # and suggested fixes. The Critic's natural-language explanation is
                # far more actionable than low-level stack traces, and guarantees
                # Stage 2 always runs (prevents _can_skip_stage2 from re-producing
                # the same compound list and cycling back to the same broken flowsheet).
                execution_feedback: list[str] = (list(rec.execution.errors) +
                                                  list(rec.execution.solver_errors))
                if report.diagnosis:
                    execution_feedback.append(f"Critic diagnosis: {report.diagnosis}")
                for fix in report.suggested_fixes:
                    execution_feedback.append(f"Suggested fix: {fix}")
                new_basis = self._run_basis(description,
                                            feedback=execution_feedback or None)
                if not new_basis.success:
                    result.outcome      = "BASIS_FAILED"
                    result.basis_reruns = basis_reruns
                    break
                result.basis_result = new_basis

                # Re-run pre_select for the new compound set
                try:
                    pre_selected_package, pkg_reasoning = self._thermo.pre_select(
                        new_basis.dwsim_compounds, new_basis.normalised_description)
                    result.warnings.append(
                        f"ThermoAgent.pre_select (BASIS rerun): "
                        f"'{pre_selected_package}' — {pkg_reasoning}")
                except Exception:
                    pass  # keep existing pre_selected_package

                new_ce = _build_preplan_condition_estimate(
                    new_basis.dwsim_compounds,
                    new_basis.suggested_compositions or {})

                # Re-run topology library with the new compound set
                new_topo_hint = topology_match(
                    new_basis.normalised_description, new_basis.dwsim_compounds)
                if new_topo_hint:
                    result.warnings.append(
                        f"TopologyLibrary (BASIS rerun): {format_hint(new_topo_hint)}")

                try:
                    flowsheet = self._planner.plan(
                        new_basis.normalised_description,
                        compounds=new_basis.dwsim_compounds,
                        suggested_compositions=new_basis.suggested_compositions or None,
                        compound_feedback=report.diagnosis or None,
                        property_package=pre_selected_package,
                        condition_estimate=new_ce or None,
                        topology_hint=new_topo_hint,
                    )
                except Exception as exc:
                    result.warnings.append(
                        f"PlannerAgent failed during BASIS re-run: "
                        f"{type(exc).__name__}: {exc}")
                    result.outcome      = "PLAN_FAILED"
                    result.basis_reruns = basis_reruns
                    break

                tried_packages = set()
                run_history    = []   # reset — new compounds, fresh context
                recent_codes   = []
                consecutive_thermo = 0
                refiner_run_count  = 0
                try:
                    flowsheet, _ = self._thermo.assign(flowsheet)
                    tried_packages.add(flowsheet.get("property_package", ""))
                except Exception as exc:
                    err = (f"ThermoAgent failed during BASIS re-run: "
                           f"{type(exc).__name__}: {exc}")
                    result.warnings.append(err)
                    result.outcome      = "HUMAN"
                    result.basis_reruns = basis_reruns
                    break

        result.basis_reruns  = basis_reruns
        result.total_time_s  = time.time() - t_start
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _run_basis(self, description: str,
                   feedback: list[str] | None = None) -> BasisResult:
        return self._basis.identify(description, feedback=feedback)

    def _do_replan(
            self,
            trigger:              str,
            report:               CriticReport,
            run_history:          list[TrialRecord],
            flowsheet:            dict,
            original_description: str,
            result:               "OrchestratorResult",
            replan_count:         int,
            pre_selected_package: str | None = None,
            topology_hint=None,   # TopologyHint | None
    ) -> tuple[dict | None, str | None, str, object]:
        """
        Dispatch one targeted repair.

        Classifies the failure, then executes the minimal fix:
          VALUE_PATCH      — pure Python field write, no LLM
          UNIT_PATCH       — LLM revises exactly one unit
          TOPOLOGY_REBUILD — full LLM flowsheet regeneration

        Returns (new_flowsheet_or_None, error_string_or_None, tier, new_topology_hint).
        new_topology_hint is non-None only after TOPOLOGY_REBUILD — caller should
        update its topology_hint variable so subsequent REPLANs don't re-log it.
        Caller should check error; on error flowsheet is None.
        """
        tier, spec = _classify_repair(trigger, report, run_history, flowsheet)

        # ── VALUE_PATCH: single field write ───────────────────────────────────
        if tier == "VALUE_PATCH":
            result.warnings.append(
                f"REPLAN #{replan_count} [{trigger}]: VALUE_PATCH — "
                f"set {spec['field']} on '{spec['target']}' → {spec['value']}"
            )
            new_fs = _surgical_patch(flowsheet, spec["field"], spec["target"], spec["value"])
            # Re-assign thermo if pressure changed (bubble point shifts → package may change)
            if spec["field"] == "P":
                try:
                    new_fs, _ = self._thermo.assign(new_fs, trial_history=run_history)
                except Exception:
                    pass   # keep existing package
            return new_fs, None, tier, None

        # ── UNIT_PATCH: LLM revises one unit ──────────────────────────────────
        if tier == "UNIT_PATCH":
            result.warnings.append(
                f"REPLAN #{replan_count} [{trigger}]: UNIT_PATCH — "
                f"revising unit '{spec['unit_name']}': {spec['reason'][:80]}"
            )
            try:
                new_fs = self._planner.revise(
                    flowsheet=flowsheet,
                    description=original_description,
                    compounds=result.basis_result.dwsim_compounds,
                    broken_unit_name=spec["unit_name"],
                    reason=spec["reason"],
                    suggested_compositions=(
                        result.basis_result.suggested_compositions or None),
                )
                new_fs, _ = self._thermo.assign(new_fs, trial_history=run_history)
                return new_fs, None, tier, None
            except Exception as exc:
                return None, str(exc), tier, None

        # ── TOPOLOGY_REBUILD: full LLM regeneration ────────────────────────────
        result.warnings.append(
            f"REPLAN #{replan_count} [{trigger}]: TOPOLOGY_REBUILD — full regeneration"
        )
        # Re-run pre_select so the rebuilt flowsheet also gets the right package.
        # Use the stored pre_selected_package as a fallback if pre_select fails.
        rebuild_package = pre_selected_package
        try:
            rebuild_package, _ = self._thermo.pre_select(
                result.basis_result.dwsim_compounds, original_description)
        except Exception:
            pass  # keep pre_selected_package

        rebuild_ce = _build_preplan_condition_estimate(
            result.basis_result.dwsim_compounds,
            result.basis_result.suggested_compositions or {})

        # Re-run topology library in case the previous hint was wrong
        rebuild_hint = topology_match(
            original_description, result.basis_result.dwsim_compounds)
        if rebuild_hint and rebuild_hint != topology_hint:
            result.warnings.append(
                f"TopologyLibrary (REBUILD): {format_hint(rebuild_hint)}")

        topo_feedback = _build_replan_feedback(report)
        cond_feedback = _build_condition_feedback(run_history, flowsheet, report)
        # Distil package feedback into a single CONSTRAINT sentence for ConditionAgent
        # rather than passing the full multi-line block which dilutes focus for smaller LLMs.
        pkg_constraint = _distill_package_feedback(run_history, rebuild_package)
        if pkg_constraint:
            cond_feedback = pkg_constraint + "\n\n" + cond_feedback
            topo_feedback = topo_feedback + "\n\n" + pkg_constraint
        try:
            new_fs = self._planner.plan(
                original_description,
                compounds=result.basis_result.dwsim_compounds,
                suggested_compositions=(
                    result.basis_result.suggested_compositions or None),
                topology_feedback=topo_feedback,
                condition_feedback=cond_feedback,
                property_package=rebuild_package,
                condition_estimate=rebuild_ce or None,
                # Suppress the library hint when we have topology feedback — passing
                # it would route TopologyAgent to Stage 1 (template), which ignores
                # the feedback entirely and regenerates the same wrong topology.
                topology_hint=None if topo_feedback else rebuild_hint,
            )
            try:
                new_fs, _ = self._thermo.assign(new_fs, trial_history=run_history)
            except Exception:
                # All packages in trial_history were exhausted — assign a hard-coded
                # fallback rather than propagating to HUMAN.  Prefer PR (non-polar EOS)
                # as a neutral starting point for the rebuilt topology.
                fallback = rebuild_package or "Peng-Robinson"
                new_fs["property_package"] = fallback
                result.warnings.append(
                    f"ThermoAgent exhausted during TOPOLOGY_REBUILD — "
                    f"falling back to {fallback}"
                )
            return new_fs, None, tier, rebuild_hint
        except Exception as exc:
            return None, str(exc), tier, None


# ── Module-level helpers ──────────────────────────────────────────────────────

def _flowsheet_hash(flowsheet: dict) -> str:
    """Stable hash of a flowsheet dict — used to detect Refiner cycling."""
    return hashlib.md5(
        json.dumps(flowsheet, sort_keys=True).encode()
    ).hexdigest()


def _build_preplan_condition_estimate(
        compounds: list[str],
        suggested_compositions: dict,
) -> str | None:
    """
    Build a lightweight numerical T/P hint block for the Planner.

    Uses the NBP table + Raoult's Law linear interpolation to suggest a
    two-phase operating window without an LLM call.  Returns None when the
    compound set is outside the NBP table (unknown compounds, light gases).

    The hint is advisory — the Planner may deviate if the description says
    otherwise (e.g., cryogenic operation, high-pressure compression).
    """
    # Use the first suggested composition; fall back to equimolar
    comp: dict[str, float] = {}
    for v in suggested_compositions.values():
        if isinstance(v, dict):
            comp = v
            break
    if not comp:
        total = max(len(compounds), 1)
        comp = {c: 1.0 / total for c in compounds}

    t_bub = _estimate_bubble_point(compounds, comp, 101_325.0)
    if t_bub is None:
        return None

    t_lo = round(t_bub + 8,  0)
    t_hi = round(t_bub + 35, 0)

    return (
        f"Mixture bubble point at 1 atm ≈ {t_bub} K "
        f"(Raoult's Law / NBP estimate, ±15 K).\n"
        f"For a Heater or Cooler feeding a flash vessel: "
        f"set T_out in the range {t_lo}–{t_hi} K to produce two phases.\n"
        f"For purely gas-phase or liquid-phase operations, ignore this estimate."
    )


def _execution_summary(report: "CriticReport") -> str:
    """Derive a compact execution outcome label from Critic signals."""
    codes = {s.code for s in report.signals}
    if "PARAM_MISSING"   in codes: return "outlet≈feed"
    if "ZERO_OUTLET"     in codes: return "zero_vapour"
    if "NO_SEPARATION"   in codes: return "no_separation"
    if "SOLVER_FAIL"     in codes: return "solver_diverged"
    if "WRONG_PHASE_DIR" in codes: return "wrong_phase"
    if "NUMERIC_FAIL"    in codes: return "numeric_fail"
    if report.passed:              return "pass"
    return "unknown"


def _build_replan_feedback(report: "CriticReport") -> str:
    """Summarise Critic signals into Planner-digestible topology feedback."""
    lines = []
    for sig in report.signals:
        if sig.code in ("SOLVER_FAIL", "MASS_BALANCE", "NUMERIC_FAIL",
                        "UNPHYSICAL_T", "UNPHYSICAL_P",
                        "WRONG_PHASE_DIR", "ZERO_OUTLET",
                        "NO_SEPARATION", "PARAM_MISSING"):
            lines.append(f"- [{sig.code}] {sig.location}: {sig.evidence}")
    if report.diagnosis:
        lines.append(f"Critic diagnosis: {report.diagnosis}")
    return "\n".join(lines) if lines else "Solver failed — regenerate topology."


# Normal boiling points (K at 1 atm) for common DWSIM compounds.
# Used to estimate bubble/dew points without an LLM call.
_NBP_K: dict[str, float] = {
    "Methanol":           337.85,
    "Ethanol":            351.44,
    "n-Propanol":         370.35,
    "1-Propanol":         370.35,   # DWSIM alias
    "Isopropanol":        355.39,
    "2-Propanol":         355.39,   # DWSIM alias
    "n-Butanol":          390.81,
    "1-Butanol":          390.81,   # DWSIM alias
    "Isobutanol":         381.04,
    "Water":              373.15,
    "Acetone":            329.15,
    "Methyl Ethyl Ketone":352.79,
    "Acetic Acid":        391.15,
    "Ethyl Acetate":      350.26,
    "Diethyl Ether":      307.58,
    "Tetrahydrofuran":    339.12,
    "Acetonitrile":       354.75,
    "Chloroform":         334.35,
    "Dichloromethane":    312.95,
    "Benzene":            353.25,
    "Toluene":            383.78,
    "o-Xylene":           417.58,
    "m-Xylene":           412.27,
    "p-Xylene":           411.51,
    "Cyclohexane":        353.87,
    "Methylcyclohexane":  374.08,
    "n-Pentane":          309.22,
    "n-Hexane":           341.88,
    "n-Heptane":          371.58,
    "n-Octane":           398.82,
    "Methane":            111.66,
    "Ethane":             184.55,
    "Propane":            231.11,
    "n-Butane":           272.65,
    "i-Butane":           261.43,
    "Nitrogen":            77.36,
    "Oxygen":              90.19,
    "Carbon Dioxide":     194.65,
    "Hydrogen Sulfide":   212.84,
    "Ammonia":            239.82,
}

_POLAR_COMPOUNDS = {
    "Water", "Methanol", "Ethanol", "n-Propanol", "Isopropanol",
    "n-Butanol", "Isobutanol", "Acetone", "Methyl Ethyl Ketone",
    "Acetic Acid", "Acetonitrile", "Ammonia",
}

_ACTIVITY_PKGS = {"NRTL", "UNIQUAC"}
_EOS_PKGS      = {"Peng-Robinson", "Soave-Redlich-Kwong", "Lee-Kesler-Plöcker"}


def _estimate_bubble_point(compounds: list[str], composition: dict[str, float],
                            pressure_pa: float = 101325.0) -> float | None:
    """
    Raoult's Law linear interpolation estimate of bubble point.
    Returns None if any compound is missing from _NBP_K.
    Includes a first-order Clausius-Clapeyron pressure correction.
    Only reliable within ~factor-of-5 of 1 atm.
    """
    import math
    if not (20_000 < pressure_pa < 600_000):
        return None
    total = sum(composition.values())
    if total <= 0:
        return None
    for c in compounds:
        if c not in _NBP_K:
            return None
    t_bub = sum((composition.get(c, 0.0) / total) * _NBP_K[c] for c in compounds)
    if abs(pressure_pa - 101_325.0) > 5_000:
        lnP = math.log(pressure_pa / 101_325.0)
        dHvap = 88.0 * t_bub          # Trouton's rule: ΔHvap ≈ 88 R T_b
        t_bub = t_bub / (1.0 - 8.314 * t_bub * lnP / dHvap)
    return round(t_bub, 1)


def _build_condition_feedback(
        history: list[TrialRecord],
        flowsheet: dict,
        report: "CriticReport",
) -> str:
    """
    Generate quantitative, Planner-actionable condition guidance.
    Uses exact values from the flowsheet (not generic ranges) so even
    weaker models receive hard numerical constraints.
    """
    lines: list[str] = []
    summaries  = [r.execution_summary for r in history]
    compounds  = flowsheet.get("compounds", [])

    # Extract first fully-specified feed stream for T, P, composition
    feed = next(
        (s for s in flowsheet.get("streams", []) if s.get("T") is not None),
        None,
    )
    feed_t    = feed.get("T")              if feed else None
    feed_p    = feed.get("P", 101_325.0)  if feed else 101_325.0
    feed_comp = feed.get("composition", {}) if feed else {}

    # ── Zero vapour + Cooler: T_out below bubble point ───────────────────────
    if summaries.count("zero_vapour") >= 1:
        cooler = next(
            (u for u in flowsheet.get("units", []) if u.get("type") == "Cooler"), None
        )
        if cooler is not None:
            t_out = cooler.get("T_out")
            t_bub = _estimate_bubble_point(compounds, feed_comp, feed_p)
            lines.append(
                "CRITICAL — Flash vessel produces zero vapour (sub-bubble-point):")
            if t_out is not None:
                lines.append(f"  Current Cooler ({cooler.get('tag')}) T_out = {t_out} K")
            if t_bub is not None:
                t_lo = round(t_bub + 8,  0)
                t_hi = round(t_bub + 28, 0)
                lines.append(
                    f"  Estimated mixture bubble point at {feed_p:.0f} Pa ≈ {t_bub} K")
                lines.append(
                    f"  T_out MUST be set ABOVE {t_bub} K to produce any vapour.")
                lines.append(
                    f"  ACTION: Set Cooler T_out to {t_lo}–{t_hi} K.")
            else:
                lines.append(
                    "  T_out MUST be raised above the mixture bubble point.")
                lines.append(
                    "  ACTION: Increase Cooler T_out by at least 30 K from current value.")
            lines.append("  DO NOT decrease T_out further.")

    # ── Zero vapour + Heater: Heater T_out below bubble point ────────────────
    if summaries.count("zero_vapour") >= 1:
        heater = next(
            (u for u in flowsheet.get("units", []) if u.get("type") == "Heater"), None
        )
        if heater is not None:
            t_out = heater.get("T_out")
            t_bub = _estimate_bubble_point(compounds, feed_comp, feed_p)
            if t_bub is not None and t_out is not None and t_out < t_bub:
                lines.append(
                    "CRITICAL — Heater T_out is below the estimated bubble point:")
                lines.append(
                    f"  Current Heater ({heater.get('tag')}) T_out = {t_out} K")
                lines.append(
                    f"  Estimated bubble point at {feed_p:.0f} Pa ≈ {t_bub} K")
                t_lo = round(t_bub + 8,  0)
                t_hi = round(t_bub + 28, 0)
                lines.append(
                    f"  ACTION: Set Heater T_out to {t_lo}–{t_hi} K.")

    # ── Outlet ≈ feed (no separation despite solver=True) ────────────────────
    if summaries.count("outlet≈feed") >= 1:
        pkg_tried = [r.property_package for r in history
                     if r.execution_summary == "outlet≈feed"]
        is_polar = bool(_POLAR_COMPOUNDS & set(compounds))
        lines.append(
            "CRITICAL — Outlet composition equals feed; no separation occurred:")
        lines.append(
            f"  Activity models without BIPs: {', '.join(pkg_tried)}")
        if is_polar:
            lines.append(
                "  Polar system — DWSIM lacks binary interaction parameters.")
            lines.append(
                "  OPTION A: Switch to Raoult's Law to confirm topology works,")
            lines.append(
                "    then document that production-quality BIPs must be supplied.")
            lines.append(
                "  OPTION B: If P can be raised above 5 bar, Peng-Robinson or SRK")
            lines.append(
                "    may produce useful VLE without activity-model BIPs.")
        else:
            lines.append(
                "  Non-polar system — use Peng-Robinson or Soave-Redlich-Kwong.")
            lines.append(
                "  If flash shows no separation at 1 atm, raise P to > 5 bar.")

    # ── Wrong phase direction ─────────────────────────────────────────────────
    if summaries.count("wrong_phase") >= 1:
        lines.append(
            "CRITICAL — Phase enrichment direction is inverted "
            "(heavy compound appearing in vapour outlet):")
        lines.append(
            "  FIX 1 (most likely): Swap Vessel outlet port assignments.")
        lines.append(
            "    Vessel vapour outlet MUST use src_port=0.")
        lines.append(
            "    Vessel liquid outlet MUST use src_port=1.")
        lines.append(
            "  FIX 2 (if ports are already correct): feed T/P is outside the")
        lines.append(
            "    two-phase envelope — raise feed temperature by 10–20 K so")
        lines.append(
            "    the mixture enters the vessel as a two-phase stream.")

    # ── Repeated solver divergence ────────────────────────────────────────────
    if summaries.count("solver_diverged") >= 1:
        lines.append(
            "CRITICAL — Solver diverged on two or more consecutive iterations:")
        if feed_t is not None:
            lines.append(
                f"  Feed T = {feed_t} K, P = {feed_p:.0f} Pa.")
        lines.append(
            "  Most common causes: stream tag referenced in connections but not")
        lines.append(
            "  declared in streams list; or a unit output port connected to two")
        lines.append(
            "  streams simultaneously.")
        lines.append(
            "  ACTION: Verify every stream tag appears in both streams[] and")
        lines.append(
            "  connections[]. Each src_port may connect to EXACTLY ONE stream.")

    # ── Unphysical T or P (unit conversion errors) ───────────────────────────
    for sig in report.signals:
        if sig.code == "UNPHYSICAL_T":
            lines.append(
                f"CRITICAL — Temperature out of range at {sig.location}: {sig.evidence}.")
            lines.append(
                "  All temperatures MUST be in Kelvin. "
                "25°C = 298.15 K | 80°C = 353.15 K | 100°C = 373.15 K.")
        if sig.code == "UNPHYSICAL_P":
            lines.append(
                f"CRITICAL — Pressure out of range at {sig.location}: {sig.evidence}.")
            lines.append(
                "  All pressures MUST be in Pascals. "
                "1 atm = 101325 Pa | 1 bar = 100000 Pa | 10 bar = 1000000 Pa.")

    return "\n".join(lines) if lines else (
        "Solver failed across multiple property packages. "
        "Regenerate the flowsheet with different topology or operating conditions.")


def _distill_package_feedback(
        history: list[TrialRecord],
        current_package: str | None,
) -> str:
    """
    Distil multi-trial package failures into a single CONSTRAINT sentence for
    ConditionAgent.  Keeps feedback short so smaller LLMs stay on-task.
    """
    if not history:
        return ""
    failed = [r for r in history if r.execution_summary != "pass"]
    if not failed:
        return ""
    reasons = []
    for r in failed:
        if r.execution_summary == "outlet≈feed":
            reasons.append(
                f"{r.property_package} produced no separation (missing BIPs)")
        elif r.execution_summary == "zero_vapour":
            reasons.append(
                f"{r.property_package} produced zero vapour (T_out below bubble point)")
        elif r.execution_summary == "solver_diverged":
            reasons.append(f"{r.property_package} solver diverged")
    if not reasons:
        return ""
    summary = "; ".join(reasons[:3])
    pkg_note = f" Current package: {current_package}." if current_package else ""
    return f"CONSTRAINT: prior trials failed — {summary}.{pkg_note} Design conditions to avoid these failures."


def _build_package_feedback(history: list[TrialRecord]) -> str:
    """
    Translate package failures into Planner-actionable condition constraints.

    The Planner controls topology and operating conditions, not the property
    package — ThermoAgent does that. So feedback is framed as: given what
    failed, what should the Planner change about conditions or structure?
    """
    if not history:
        return ""

    lines = ["Packages attempted — design implications for the flowsheet:"]

    failed_activity = any(
        r.property_package in _ACTIVITY_PKGS and r.execution_summary == "outlet≈feed"
        for r in history
    )
    failed_eos = any(
        r.property_package in _EOS_PKGS
        and r.execution_summary in {"solver_diverged", "zero_vapour"}
        for r in history
    )

    for r in history:
        pkg  = r.property_package
        summ = r.execution_summary

        if pkg in _ACTIVITY_PKGS and summ == "outlet≈feed":
            lines.append(
                f"  - {pkg}: DWSIM has no binary interaction parameters for this "
                f"compound pair — activity models cannot separate this mixture.")
            lines.append(
                f"    DESIGN IMPLICATION: if compounds are non-polar, design for "
                f"P > 3 bar so Peng-Robinson applies. If polar at ambient P, use "
                f"Raoult's Law as a topology placeholder.")

        elif summ == "zero_vapour":
            lines.append(
                f"  - {pkg}: no vapour produced — mixture is fully liquid at "
                f"the current unit outlet conditions.")
            lines.append(
                f"    DESIGN IMPLICATION: raise Heater/Cooler T_out or feed T "
                f"so the stream enters the flash vessel in the two-phase region.")

        elif pkg in _EOS_PKGS and summ == "solver_diverged":
            lines.append(
                f"  - {pkg}: EOS solver diverged — these models are unreliable "
                f"for polar/hydrogen-bonding compounds at near-ambient pressure.")
            lines.append(
                f"    DESIGN IMPLICATION: this system needs an activity coefficient "
                f"model (NRTL or UNIQUAC). Do NOT design for high-pressure EOS "
                f"conditions unless the compounds are non-polar hydrocarbons.")

        elif summ == "wrong_phase":
            lines.append(
                f"  - {pkg}: vapour/liquid outlets are inverted.")
            lines.append(
                f"    DESIGN IMPLICATION: swap Vessel src_port assignments "
                f"(vapour=0, liquid=1) or raise feed temperature to ensure "
                f"two-phase flow at vessel inlet.")

        elif summ == "numeric_fail":
            lines.append(
                f"  - {pkg}: numerical failure (NaN/Inf in stream results).")
            lines.append(
                f"    DESIGN IMPLICATION: check all T/P/flow values are physical "
                f"(T in K, P in Pa, molar flow > 0 mol/s).")

        elif summ == "solver_diverged":
            lines.append(
                f"  - {pkg}: solver diverged — topology or initial conditions "
                f"likely incorrect.")
            lines.append(
                f"    DESIGN IMPLICATION: simplify unit sequence, ensure every "
                f"intermediate stream appears in both source and destination "
                f"connections.")

        else:
            lines.append(f"  - {pkg}: {summ}")

    if failed_activity and failed_eos:
        lines.append(
            "  WARNING — All major thermodynamic model families have failed.")
        lines.append(
            "  The process as specified may be infeasible at these conditions.")
        lines.append(
            "  Consider a fundamentally different T, P, or composition, or "
            "accept a Raoult's Law approximation to verify the topology only.")

    return "\n".join(lines)


# ── Surgical repair helpers ───────────────────────────────────────────────────

def _classify_repair(
        trigger:   str,
        report:    "CriticReport",
        history:   list[TrialRecord],
        flowsheet: dict,
) -> tuple[str, dict]:
    """
    Classify the minimal repair tier for a given REPLAN trigger.

    Returns (tier, spec) where:
      tier  ∈ {'VALUE_PATCH', 'UNIT_PATCH', 'TOPOLOGY_REBUILD'}
      spec  — for VALUE_PATCH: {'field': str, 'target': str, 'value': Any}
            — for UNIT_PATCH:  {'unit_name': str, 'reason': str}
            — for TOPOLOGY_REBUILD: {}

    Priority: VALUE_PATCH (pure Python) > UNIT_PATCH (single-unit LLM) >
              TOPOLOGY_REBUILD (full LLM regeneration).
    """
    last_summary = history[-1].execution_summary if history else "unknown"
    compounds    = flowsheet.get("compounds", [])

    # ── VALUE_PATCH A: T_out still near or below bubble point ────────────────
    # Pre-flight sets T_out → bubble_point + 15 K; here we try +35 K.
    # Fires when T_out is still < bubble_point + 25 K after pre-flight.
    if last_summary == "zero_vapour":
        feed = next(
            (s for s in flowsheet.get("streams", [])
             if s.get("T") is not None and s.get("composition")),
            None,
        )
        if feed is not None:
            t_bub = _estimate_bubble_point(
                compounds,
                feed.get("composition", {}),
                feed.get("P", 101_325.0),
            )
            if t_bub is not None:
                for unit in flowsheet.get("units", []):
                    if unit.get("type") not in ("Heater", "Cooler"):
                        continue
                    t_out = unit.get("T_out")
                    if t_out is not None and t_out < t_bub + 25.0:
                        return ("VALUE_PATCH", {
                            "field":  "T_out",
                            "target": unit["tag"],
                            "value":  round(t_bub + 35.0, 1),
                        })

    # ── VALUE_PATCH B: outlet≈feed at near-ambient P → raise to 5 bar ────────
    # Only fires when activity models failed with missing BIPs at < 2 bar
    # and an EOS package has not yet been tried.
    if last_summary == "outlet≈feed" and trigger in ("thermo_cycle", "packages_exhausted"):
        tried_activity  = any(
            r.property_package in _ACTIVITY_PKGS and r.execution_summary == "outlet≈feed"
            for r in history
        )
        tried_eos = any(r.property_package in _EOS_PKGS for r in history)
        if tried_activity and not tried_eos:
            feed = next(
                (s for s in flowsheet.get("streams", []) if s.get("P") is not None),
                None,
            )
            if feed is not None and feed.get("P", 101_325.0) < 200_000:
                return ("VALUE_PATCH", {
                    "field":  "P",
                    "target": feed["tag"],
                    "value":  500_000.0,
                })

    # ── UNIT_PATCH: refiner ran ≥ 2× without PASS → one specific unit is wrong
    # Only valid when sig.location is an actual unit tag — the Critic also sets
    # location to "global" (system-wide failure) or "stream:TAG" (stream issue),
    # neither of which exists as a unit; revise() would silently fail on these.
    if trigger == "refiner_ineffective":
        unit_tags = {u.get("tag") for u in flowsheet.get("units", [])}
        for sig in report.signals:
            if (sig.code in ("SOLVER_FAIL", "ZERO_OUTLET", "NO_SEPARATION",
                             "WRONG_PHASE_DIR")
                    and sig.location and sig.location in unit_tags):
                return ("UNIT_PATCH", {
                    "unit_name": sig.location,
                    "reason": (
                        report.diagnosis
                        or f"{sig.code} at {sig.location}: {sig.evidence}"
                    ),
                })

    return ("TOPOLOGY_REBUILD", {})


def _surgical_patch(
        flowsheet: dict,
        field:     str,
        target:    str,
        value,
) -> dict:
    """
    Pure Python: return a deepcopy of flowsheet with exactly one field changed.

    field='T_out'   — set T_out on the unit whose tag==target
    field='P'       — set P on the stream whose tag==target
    field='src_port'— swap Vessel src_port 0↔1 for the vessel whose tag==target
    """
    import copy
    fs = copy.deepcopy(flowsheet)

    if field == "T_out":
        for unit in fs.get("units", []):
            if unit.get("tag") == target:
                unit["T_out"] = value
                break

    elif field == "P":
        for stream in fs.get("streams", []):
            if stream.get("T") is not None and stream.get("composition"):
                stream["P"] = value

    elif field == "src_port":
        conns = [list(c) for c in fs.get("connections", [])]
        p0 = p1 = None
        for i, c in enumerate(conns):
            if len(c) >= 3 and c[0] == target:
                if c[2] == 0:
                    p0 = i
                elif c[2] == 1:
                    p1 = i
        if p0 is not None and p1 is not None:
            conns[p0][2], conns[p1][2] = 1, 0
        fs["connections"] = conns

    return fs


# ── Pre-execution helpers (zero LLM calls) ────────────────────────────────────

def _proactive_calibration(
        flowsheet: dict,
        calibration: "CalibrationAgent",
) -> tuple[dict, str | None]:
    """
    Inject binary interaction parameters BEFORE the Executor runs, if the
    CalibrationAgent corpus covers all compound pairs for the selected package.

    If BIPs are found and not already present: inject and return the updated
    flowsheet with a log message.

    If BIPs are NOT found: return flowsheet unchanged — DWSIM may have its own
    internal parameters, so we do not fall back proactively (that would break
    common pairs like ethanol/water where DWSIM has internal NRTL BIPs).
    """
    pkg = flowsheet.get("property_package", "")
    if pkg not in {"NRTL", "UNIQUAC"}:
        return flowsheet, None
    if flowsheet.get("binary_parameters"):
        return flowsheet, None   # already injected — skip

    from agents.calibration import CalibrationResult
    calib: CalibrationResult = calibration.run(flowsheet)
    if calib.success:
        return calib.updated_flowsheet, (
            f"PRE-FLIGHT: CalibrationAgent injected {pkg} BIPs proactively "
            f"for {flowsheet.get('compounds', [])}."
        )
    return flowsheet, None


_VAP_KEYWORDS = {"VAP", "VAPOR", "VAPOUR", "GAS", "TOP", "OVER", "DIST"}
_LIQ_KEYWORDS = {"LIQ", "LIQUID", "BOT", "BOTTOM", "BOTT", "BASE"}


def _preflight_patch(flowsheet: dict) -> tuple[dict, list[str]]:
    """
    Deterministic physical-feasibility fixes applied before every Executor call.
    No LLM calls. Returns (patched_flowsheet, list_of_patch_messages).

    Fix A — Sub-bubble-point T_out:
      If any Heater or Cooler T_out ≤ estimated bubble point of its feed, raise
      T_out to bubble_point + 15 K.

    Fix B — Vessel vapour/liquid port inversion by stream name:
      Swap src_ports when name keywords contradict the expected assignment.

    Fix C — Bare Vessel with liquid feed:
      If the only unit is a Vessel (no Heater/Cooler) and feed T < bubble point,
      raise the feed stream T to bubble_point + 15 K so the vessel receives a
      two-phase mixture rather than all-liquid feed (which produces ZERO_OUTLET).
    """
    import copy

    fs = copy.deepcopy(flowsheet)
    patches: list[str] = []
    compounds = fs.get("compounds", [])

    # Locate first fully-specified feed stream
    feed = next(
        (s for s in fs.get("streams", [])
         if s.get("T") is not None and s.get("composition")),
        None,
    )

    unit_types = {u.get("type") for u in fs.get("units", [])}
    has_conditioning = bool(unit_types & {"Heater", "Cooler"})

    # Fix A — sub-bubble-point T_out on Heater/Cooler
    if feed is not None:
        feed_comp = feed.get("composition", {})
        feed_p    = feed.get("P", 101_325.0)
        t_bub     = _estimate_bubble_point(compounds, feed_comp, feed_p)
        if t_bub is not None:
            for unit in fs.get("units", []):
                if unit.get("type") not in ("Heater", "Cooler"):
                    continue
                t_out = unit.get("T_out")
                if t_out is not None and t_out <= t_bub:
                    new_t = round(t_bub + 15.0, 1)
                    patches.append(
                        f"PRE-FLIGHT: {unit['type']} {unit['tag']} T_out "
                        f"{t_out} K ≤ bubble point {t_bub} K — patched to {new_t} K."
                    )
                    unit["T_out"] = new_t

    # Fix C — Bare Vessel with sub-bubble-point feed (no conditioning unit)
    if not has_conditioning and "Vessel" in unit_types and feed is not None:
        feed_comp = feed.get("composition", {})
        feed_p    = feed.get("P", 101_325.0)
        t_bub     = _estimate_bubble_point(compounds, feed_comp, feed_p)
        feed_t    = feed.get("T")
        if t_bub is not None and feed_t is not None and feed_t < t_bub:
            new_t = round(t_bub + 15.0, 1)
            for s in fs.get("streams", []):
                if s.get("tag") == feed.get("tag"):
                    s["T"] = new_t
                    break
            patches.append(
                f"PRE-FLIGHT: No Heater/Cooler; feed T={feed_t} K < "
                f"bubble point {t_bub} K — raised feed T to {new_t} K "
                f"for two-phase Vessel entry."
            )

    # Fix B — Vessel port inversion by stream-name heuristic
    conns = [list(c) for c in fs.get("connections", [])]
    for unit in fs.get("units", []):
        if unit.get("type") != "Vessel":
            continue
        utag = unit["tag"]
        port0_idx = port1_idx = None
        port0_tag = port1_tag = None
        for i, c in enumerate(conns):
            if len(c) >= 3 and c[0] == utag:
                if c[2] == 0:
                    port0_idx, port0_tag = i, c[1]
                elif c[2] == 1:
                    port1_idx, port1_tag = i, c[1]

        if port0_idx is None or port1_idx is None:
            continue

        p0_upper = port0_tag.upper()
        p1_upper = port1_tag.upper()
        p0_is_liq  = any(k in p0_upper for k in _LIQ_KEYWORDS)
        p1_is_vap  = any(k in p1_upper for k in _VAP_KEYWORDS)
        should_swap = p0_is_liq or p1_is_vap

        if should_swap:
            conns[port0_idx] = list(conns[port0_idx])
            conns[port1_idx] = list(conns[port1_idx])
            conns[port0_idx][2] = 1
            conns[port1_idx][2] = 0
            patches.append(
                f"PRE-FLIGHT: Vessel {utag} port swap — "
                f"{port0_tag} moved to src_port=1 (liquid), "
                f"{port1_tag} moved to src_port=0 (vapour)."
            )
        fs["connections"] = conns

    return fs, patches
