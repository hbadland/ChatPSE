"""
v2 Orchestrator — lean pipeline controller.

Pipeline:
  Stage 1  UnitExtractor + StreamExtractor  (2 LLM calls)
  Stage 2  GraphBuilder                      (0 LLM calls — assembly)
  Normalise + Validate                       (0 LLM calls)
  Stage 3  ThermoMapper + ParamMapper        (1-2 LLM calls)
  Normalise + Validate                       (0 LLM calls)
  IR → DWSIM JSON                            (0 LLM calls)
  Stage 4  loop:
    Executor → ErrorClassifier → RepairAgent (0-1 LLM calls per iteration)

All error routing lives in ErrorClassifier + the error taxonomy.
The orchestrator only manages the loop and aggregates results.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Optional

from agents.basis import BasisAgent, BasisResult
from agents.executor import Executor
from agents.llm import DEFAULT_MODEL

from agents.stage1 import UnitExtractor, StreamExtractor
from agents.stage2 import GraphBuilder
from agents.stage3 import ThermoMapper, ParamMapper
from agents.stage4 import ErrorClassifier, ClassifiedError, RepairAgent
from agents.stage4.repair_agent import RepairMemory
from agents.stage4.sim_hints import SimulationHints, EMPTY_HINTS
from agents.rule_store import FailureRuleStore

from ir import FlowsheetGraph, normalise, validate, to_dwsim
from ir.margin_model import get_global_margin_model
from ir.consistency import GlobalConsistencyPass
from ir.validate import ValidationReport
from rag.retriever import Retriever


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class IterationRecord:
    iteration:    int
    errors:       list[ClassifiedError]
    changes:      list[str]
    flowsheet:    dict
    execution:    object
    elapsed_s:    float


@dataclass
class PipelineResult:
    description:        str
    outcome:            str              # PASS | HUMAN | MAX_ITER | BASIS_FAILED
                                         # INVALID_IR | INVALID_JSON | PLAN_FAILED
    basis_result:       Optional[BasisResult]      = None
    final_graph:        Optional[FlowsheetGraph]   = None
    final_flowsheet:    Optional[dict]             = None
    final_execution:    object                     = None
    ir_report:          Optional[ValidationReport] = None
    iterations:         list[IterationRecord]      = field(default_factory=list)
    warnings:           list[str]                  = field(default_factory=list)
    total_time_s:       float                      = 0.0
    margin_snapshot:    dict                       = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.outcome == "PASS"

    def summary(self) -> str:
        lines = [
            f"Outcome    : {self.outcome}",
            f"Iterations : {len(self.iterations)}",
            f"Time       : {self.total_time_s:.1f}s",
        ]
        if self.basis_result:
            lines.append(f"Compounds  : {self.basis_result.dwsim_compounds}")
        for rec in self.iterations:
            tag = "✓" if not rec.errors else "✗"
            lines.append(f"  [{tag}] iter {rec.iteration}  {rec.elapsed_s:.1f}s")
            for c in rec.changes:
                lines.append(f"      {c}")
        if self.warnings:
            for w in self.warnings[:5]:
                lines.append(f"  ! {w}")
        return "\n".join(lines)


# ── Orchestrator ───────────────────────────────────────────────────────────────

class OrchestratorV2:
    """
    End-to-end v2 pipeline.

    Args:
        model          : LLM model for all agents
        max_iterations : Stage 4 execution loop limit
    """

    def __init__(
        self,
        model:          str = DEFAULT_MODEL,
        max_iterations: int = 6,
        rule_store:     Optional[FailureRuleStore] = None,
    ):
        self._model      = model
        self._retriever  = Retriever()
        self._max_iter   = max_iterations
        self._rule_store = rule_store or FailureRuleStore()

        self._basis       = BasisAgent(model=model)
        self._unit_ext    = UnitExtractor(model=model)
        self._stream_ext  = StreamExtractor(model=model)
        self._builder     = GraphBuilder()
        self._thermo      = ThermoMapper(model=model, retriever=self._retriever)
        self._params      = ParamMapper(model=model, retriever=self._retriever)
        self._consistency = GlobalConsistencyPass()
        self._executor    = Executor()
        self._classifier  = ErrorClassifier(model=model)
        self._repair      = RepairAgent(model=model, retriever=self._retriever)

    def run(self, description: str) -> PipelineResult:
        t_start = time.time()
        result  = PipelineResult(description=description, outcome="MAX_ITER")

        # ── Stage 0: Basis ─────────────────────────────────────────────────────
        print("[ORCH] step: basis.identify START", flush=True)
        basis = self._basis.identify(description)
        print(f"[ORCH] step: basis.identify END  success={basis.success}", flush=True)
        result.basis_result = basis
        if not basis.success:
            result.outcome    = "BASIS_FAILED"
            result.total_time_s = time.time() - t_start
            return result

        desc      = basis.normalised_description
        compounds = basis.dwsim_compounds
        print(f"[ORCH] compounds={compounds}", flush=True)

        # ── Stage 1: Semantic parsing ──────────────────────────────────────────
        # Pass concentration_hints and suggested_compositions from BasisAgent
        # so StreamExtractor does not need to re-derive feed composition from prose.
        try:
            print("[ORCH] step: unit_ext.extract START", flush=True)
            sem_units = self._unit_ext.extract(desc, compounds)
            print(f"[ORCH] step: unit_ext.extract END  units={[u.tag for u in sem_units.units]}", flush=True)

            print("[ORCH] step: stream_ext.extract START", flush=True)
            sem_topo  = self._stream_ext.extract(
                desc, compounds,
                unit_tags               = [u.tag  for u in sem_units.units],
                unit_roles              = {u.tag: u.role for u in sem_units.units},
                concentration_hints     = basis.concentration_hints or [],
                suggested_compositions  = basis.suggested_compositions or {},
            )
            print("[ORCH] step: stream_ext.extract END", flush=True)
        except RuntimeError as exc:
            print(f"[ORCH] Stage 1 EXCEPTION: {exc}", flush=True)
            result.warnings.append(f"Stage 1 failed: {exc}")
            result.outcome      = "PLAN_FAILED"
            result.total_time_s = time.time() - t_start
            return result

        # ── Stage 2: IR construction ───────────────────────────────────────────
        print("[ORCH] step: builder.build START", flush=True)
        graph = self._builder.build(sem_units, sem_topo, compounds)
        print("[ORCH] step: builder.build END", flush=True)

        # Reconcile: augment graph.compounds with any compounds found in stream
        # compositions that BasisAgent missed (e.g. second compound in a
        # hyphenated pair like "ethanol-water" where Stage 1 only found "Ethanol").
        _known_lower = {c.lower() for c in graph.compounds}
        for _stream in graph.streams():
            for _name in _stream.composition:
                if _name.lower() not in _known_lower:
                    graph.compounds.append(_name)
                    _known_lower.add(_name.lower())
                    result.warnings.append(
                        f"[compounds] '{_name}' found in feed composition "
                        f"but missing from basis list — added automatically"
                    )
        if graph.compounds != list(compounds):
            print(f"[ORCH] compounds reconciled: {graph.compounds}", flush=True)

        # Back-fill: any stream that already carries composition data must have
        # every compound in graph.compounds (use 0.0 for absent ones).
        # pre_execution_check rejects feed streams missing compounds from the
        # global list — this can happen when reconciliation adds a compound
        # that only appears in one of several feed streams.
        for _stream in graph.streams():
            if _stream.composition:
                for _name in graph.compounds:
                    if _name not in _stream.composition:
                        _stream.composition[_name] = 0.0

        print("[ORCH] step: normalise(graph) #1 START", flush=True)
        graph = normalise(graph)
        print("[ORCH] step: normalise(graph) #1 END", flush=True)

        print(f"[ORCH] step: validate(graph) #1 START  graph.compounds={graph.compounds}", flush=True)
        ir_report = validate(graph)
        print(f"[ORCH] step: validate(graph) #1 END  valid={ir_report.valid}", flush=True)
        result.ir_report = ir_report
        if not ir_report.valid:
            result.warnings    += [str(i) for i in ir_report.errors()]
            result.outcome      = "INVALID_IR"
            for e in ir_report.errors():
                print(f"[ORCH] INVALID_IR: {e}", flush=True, file=sys.stderr)
            result.total_time_s = time.time() - t_start
            return result
        if ir_report.warnings():
            result.warnings += [str(w) for w in ir_report.warnings()]

        # ── Stage 3: Simulation mapping ────────────────────────────────────────
        # ParamMapper: deterministic-first, LLM fallback only for unknown params.
        # GlobalConsistencyPass: enforce cross-unit T/P constraints + backward pass.
        # FailureRuleStore: apply any synthesized rules from prior benchmark cases.
        try:
            print("[ORCH] step: thermo.assign START", flush=True)
            graph = self._thermo.assign(graph, description=desc)
            print(f"[ORCH] step: thermo.assign END  pkg={getattr(graph, 'property_package', '?')}", flush=True)

            print("[ORCH] step: params.assign START", flush=True)
            graph = self._params.assign(graph, description=desc)
            print("[ORCH] step: params.assign END", flush=True)

            print("[ORCH] step: consistency.apply START", flush=True)
            graph, consistency_changes = self._consistency.apply(graph)
            print(f"[ORCH] step: consistency.apply END  changes={len(consistency_changes)}", flush=True)
            if consistency_changes:
                result.warnings += [f"[consistency] {c}" for c in consistency_changes]

            # Apply rules synthesized from previous failures (cross-run learning)
            print("[ORCH] step: rule_store.apply_to_graph START", flush=True)
            graph, rule_changes = self._rule_store.apply_to_graph(
                graph, basis.dwsim_compounds)
            print(f"[ORCH] step: rule_store.apply_to_graph END  changes={len(rule_changes)}", flush=True)
            if rule_changes:
                result.warnings += [f"[rule] {c}" for c in rule_changes]
        except Exception as exc:
            print(f"[ORCH] Stage 3 EXCEPTION: {type(exc).__name__}: {exc}", flush=True)
            result.warnings.append(f"Stage 3 failed: {exc}")
            result.outcome      = "PLAN_FAILED"
            result.total_time_s = time.time() - t_start
            return result

        print("[ORCH] step: normalise(graph) #2 START", flush=True)
        graph = normalise(graph)
        print("[ORCH] step: normalise(graph) #2 END", flush=True)

        print(f"[ORCH] step: validate(graph) #2 START  graph.compounds={graph.compounds}", flush=True)
        post_report = validate(graph)
        print(f"[ORCH] step: validate(graph) #2 END  valid={post_report.valid}", flush=True)
        if not post_report.valid:
            result.warnings    += [str(i) for i in post_report.errors()]
            result.outcome      = "INVALID_JSON"
            result.total_time_s = time.time() - t_start
            return result

        result.final_graph = graph

        # ── Stage 3→4 bridge: IR → DWSIM JSON ─────────────────────────────────
        print("[ORCH] step: to_dwsim(graph) START", flush=True)
        dwsim_json = to_dwsim(graph)
        print("[ORCH] step: to_dwsim(graph) END", flush=True)

        # ── Stage 4: Execution loop ────────────────────────────────────────────
        # RepairMemory persists across iterations so the agent never repeats a
        # failed strategy and can detect stagnation.
        # SimulationHints carries signals from the last DWSIM execution into
        # the repair agent so it can prioritise actually-failed units.
        tried_packages: set[str] = {graph.property_package}
        repair_memory             = RepairMemory()
        sim_hints                 = EMPTY_HINTS

        for iteration in range(self._max_iter):
            t_iter    = time.time()
            repair_memory.tick()
            print(f"[ORCH] step: executor.run iteration={iteration} START", flush=True)
            execution = self._executor.run(dwsim_json)
            print(f"[ORCH] step: executor.run iteration={iteration} END  solved={getattr(execution, 'solved', '?')}", flush=True)

            # Build simulation hints from the execution result for this iteration
            sim_hints = SimulationHints.from_execution(execution, iteration=iteration)

            if getattr(execution, "solved", False) and _no_critic_failures(execution):
                result.outcome         = "PASS"
                result.final_flowsheet = dwsim_json
                result.final_execution = execution
                result.iterations.append(IterationRecord(
                    iteration=iteration, errors=[], changes=["PASS"],
                    flowsheet=dwsim_json, execution=execution,
                    elapsed_s=time.time() - t_iter))
                break

            errors = self._classifier.classify(execution, graph)

            if any(e.is_terminal for e in errors):
                result.outcome         = "HUMAN"
                result.final_flowsheet = dwsim_json
                result.final_execution = execution
                result.iterations.append(IterationRecord(
                    iteration=iteration, errors=errors, changes=[],
                    flowsheet=dwsim_json, execution=execution,
                    elapsed_s=time.time() - t_iter))
                break

            try:
                graph, changes = self._repair.repair(
                    graph, errors, tried_packages,
                    description=desc, memory=repair_memory,
                    sim_hints=sim_hints)
                _record_repairs_in_store(
                    errors, changes, graph, self._rule_store,
                    compounds=basis.dwsim_compounds)
                graph  = normalise(graph)
                post   = validate(graph)
                if not post.valid:
                    result.warnings += [str(i) for i in post.errors()]

                tried_packages.add(graph.property_package)
                dwsim_json = to_dwsim(graph)
            except Exception as _repair_exc:
                print(f"[ORCH] repair/normalise error iter={iteration}: {_repair_exc}",
                      flush=True, file=sys.stderr)
                result.warnings.append(f"repair error iter={iteration}: {_repair_exc}")
                result.iterations.append(IterationRecord(
                    iteration=iteration, errors=errors, changes=[],
                    flowsheet=dwsim_json, execution=execution,
                    elapsed_s=time.time() - t_iter))
                break

            result.final_graph     = graph
            result.final_flowsheet = dwsim_json
            result.final_execution = execution
            result.iterations.append(IterationRecord(
                iteration=iteration, errors=errors, changes=changes,
                flowsheet=dwsim_json, execution=execution,
                elapsed_s=time.time() - t_iter))

        result.total_time_s  = time.time() - t_start
        result.margin_snapshot = get_global_margin_model().snapshot()
        return result


def _no_critic_failures(execution) -> bool:
    """True when the execution has no error signals from the v1 critic (if present)."""
    report = getattr(execution, "_critic_report", None)
    if report:
        return getattr(report, "passed", False)
    return not getattr(execution, "errors", [])


def _record_repairs_in_store(
    errors:     list,
    changes:    list[str],
    graph,
    store:      "FailureRuleStore",
    compounds:  Optional[list[str]] = None,
) -> None:
    """
    Parse applied CONDITION_FIX repairs from the change log and record
    each pattern in the FailureRuleStore for future rule synthesis.
    """
    import re
    from agents.rule_store import _outlet_unit_types

    for change in changes:
        m = re.match(
            r"CONDITION_FIX\[.*?\]: (\S+)\.(\w+) .*?→([\d.]+)", change)
        if m is None:
            continue
        unit_tag, param, val_str = m.group(1), m.group(2), m.group(3)
        try:
            applied_val = float(val_str)
        except ValueError:
            continue

        node = graph.unit(unit_tag)
        if node is None:
            continue
        downstream = _outlet_unit_types(graph, unit_tag)
        # Use the first downstream type (most relevant constraint)
        dst_type = next(iter(downstream), None)

        store.record_fix(
            unit_type       = node.unit_type,
            error_code      = "CONDITION_FIX",
            downstream_type = dst_type,
            param           = param,
            applied_value   = applied_val,
            compounds       = compounds,
        )
