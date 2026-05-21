"""
benchmark/diagnostics.py

Research-grade diagnostic engine for the CCS benchmark.

Produces structured diagnostics across ten categories (A–J):
  A  Case summary
  B  Physical validity report
  C  Constraint system diagnostics
  D  Search behaviour trace
  E  Repair dynamics analysis
  F  Trajectory credit analysis
  G  Margin model evaluation  (proxy — pipeline internals not yet instrumented)
  H  Coordinate descent effectiveness  (proxy)
  I  Coupling system effectiveness
  J  Generalisation info

Cross-model ablation comparison and three global summary tables are
produced by DiagnosticReport.  Interpretation / bottleneck ranking is
driven purely by ablation deltas, so no ground-truth labels are needed.

Usage:
    from benchmark.diagnostics import DiagnosticEngine
    engine = DiagnosticEngine()
    report = engine.analyse(ablation_results)   # dict[mode → BenchmarkRunSet]
    print(report.format())
    report.save("results/diagnostics")
"""
from __future__ import annotations

import json
import os
import re
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional

from benchmark.runner import BenchmarkRunSet, CaseRunResult
from benchmark.metrics import RunMetrics
from benchmark.logger import IterationLog


# ══════════════════════════════════════════════════════════════════════════════
# Compound / domain knowledge
# ══════════════════════════════════════════════════════════════════════════════

_POLAR = frozenset({
    "ethanol", "water", "methanol", "propanol", "acetone",
    "mek", "methyl ethyl ketone", "acetic acid",
})
_GAS_PHASE = frozenset({
    "methane", "nitrogen", "co2", "carbon dioxide", "propylene",
    "ethylene", "hydrogen", "oxygen",
})
_AZEOTROPIC_PAIRS: set = {
    frozenset({"ethanol", "water"}),
    frozenset({"hexane", "cyclohexane"}),
    frozenset({"acetone", "methanol"}),
    frozenset({"mek", "water"}),
    frozenset({"hexane", "heptane"}),
}
_KNOWN_IN_TRAINING = frozenset({
    "ethanol", "water", "benzene", "toluene", "propane", "butane", "pentane",
    "methane", "acetone", "methanol", "propanol", "propylene", "hexane",
    "cyclohexane", "co2", "carbon dioxide", "nitrogen", "xylene",
})

_TEMP_TYPES     = frozenset({"temp_increases_across", "temp_decreases_across",
                              "outlet_t_range", "temp_consistency_inlet_outlet"})
_PRESSURE_TYPES = frozenset({"pressure_increases_across"})
_PHASE_TYPES    = frozenset({"two_phase_outlet", "single_phase_vapor_ok"})
_COUPLING_TYPES = frozenset({"unit_type_present", "n_units_of_type",
                              "property_package_class", "bip_injected"})

_FAILURE_OUTCOMES = [
    "EXCEPTION", "INVALID_IR", "INVALID_JSON", "PHYSICS_VIOLATION",
    "MISSING_PARAMS", "REPAIR_EXHAUSTED", "ESCALATED",
    "NO_CONVERGENCE", "MAX_ITER",
]


def _classify_compounds(compounds: list) -> list:
    norm = {c.lower().strip() for c in compounds}
    classes: list = []
    pairs = {frozenset({a, b}) for a in norm for b in norm if a < b}
    if pairs & _AZEOTROPIC_PAIRS:
        classes.append("azeotropic")
    if norm & _POLAR:
        classes.append("polar")
    if norm & _GAS_PHASE:
        classes.append("gas")
    if norm - _POLAR - _GAS_PHASE:
        classes.append("hydrocarbon")
    return classes or ["unknown"]


def _compound_familiarity(compounds: list) -> str:
    norm = {c.lower().strip() for c in compounds}
    return "known" if norm <= _KNOWN_IN_TRAINING else "unseen"


# ══════════════════════════════════════════════════════════════════════════════
# Section dataclasses  A – J
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhysicalValidityReport:                       # B
    temperature_pass:   bool = True
    temperature_detail: str  = ""
    pressure_pass:      bool = True
    pressure_detail:    str  = ""
    phase_pass:         bool = True
    phase_detail:       str  = ""
    coupling_pass:      bool = True
    coupling_detail:    str  = ""
    violated_unit_tags: list = field(default_factory=list)

    @property
    def overall(self) -> str:
        n_fail = sum([
            not self.temperature_pass, not self.pressure_pass,
            not self.phase_pass,       not self.coupling_pass,
        ])
        return "PASS" if n_fail == 0 else ("FAIL" if n_fail >= 2 else "PARTIAL")


@dataclass
class ConstraintDiagnostic:                         # C
    violations_before:          int  = 0
    violations_after_settling:  int  = 0
    violations_after_repair:    int  = 0
    violations_after_local_opt: int  = 0
    violations_after_beam:      int  = 0
    dropped_constraints:   list = field(default_factory=list)
    resolved_constraints:  list = field(default_factory=list)
    physical_conflicts:    list = field(default_factory=list)
    heuristic_conflicts:   list = field(default_factory=list)
    constraint_resolution_by_stage: dict = field(default_factory=dict)
    dominant_resolution_stage:      str  = "unknown"
    instrumented:                   bool = False


@dataclass
class SearchTrace:                                   # D
    beam_width_used:           int   = 0
    total_states_generated:    int   = 0
    total_states_validated:    int   = 0
    state_cache_hits:          int   = 0
    diversity_acceptance_rate: float = 0.0
    coupling_boost_count:      int   = 0
    explore_phase_steps:       int   = 0
    exploit_phase_steps:       int   = 0
    exploration_ratio:         float = 0.0
    stagnation_events:         int   = 0
    oscillation_events:        int   = 0


@dataclass
class RepairDynamics:                                # E
    repairs_per_unit:            dict  = field(default_factory=dict)
    success_rate_deterministic:  float = 0.0
    success_rate_physics:        float = 0.0
    success_rate_llm:            float = 0.0
    success_rate_unknown:        float = 0.0
    avg_fix_magnitude:           float = 0.0
    rejected_fix_rate:           float = 0.0
    oscillation_detection_triggered: bool = False
    oscillation_escape_used:         bool = False
    coordinate_descent_improved:     bool = False


@dataclass
class TrajectoryCreditAnalysis:                      # F
    total_credit:          float = 0.0
    top_credited_moves:    list  = field(default_factory=list)
    credit_label:          str   = "INSUFFICIENT_DATA"
    credit_collapse:       bool  = False
    credit_diffusion:      bool  = False
    credit_alignment_good: bool  = False


@dataclass
class MarginModelReport:                             # G
    n_updates:            int   = 0
    margin_variance:      float = 0.0
    wildcard_fallbacks:   int   = 0
    stability:            float = 0.0
    drift_detected:       bool  = False
    hard_bounds_hit_freq: float = 0.0
    cold_start_dominant:  bool  = False
    instrumented:         bool  = False


@dataclass
class CoordDescentReport:                            # H
    improved:                  bool  = False
    improvement_magnitude:     float = 0.0
    level_0_triggered:         int   = 0
    level_1_triggered:         int   = 0
    level_2_triggered:         int   = 0
    constraint_rejection_rate: float = 0.0
    instrumented:              bool  = False


@dataclass
class CouplingReport:                                # I
    corrections_triggered:    int = 0
    ping_pong_events:         int = 0
    coupled_settler_resolved: int = 0
    propagation_lag_steps:    int = 0


# ══════════════════════════════════════════════════════════════════════════════
# CaseDiagnostic: aggregates all sections for one case × mode run
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CaseDiagnostic:
    # A — case summary
    case_id:               str
    tier:                  str
    ablation_mode:         str
    n_units:               int
    n_constraints:         int
    compound_classes:      list
    convergence_status:    str
    final_ir_error_count:  int
    iterations_to_converge: int

    # B – I
    physical_validity: PhysicalValidityReport
    constraint_diag:   ConstraintDiagnostic
    search_trace:      SearchTrace
    repair_dynamics:   RepairDynamics
    trajectory_credit: TrajectoryCreditAnalysis
    margin_model:      MarginModelReport
    coord_descent:     CoordDescentReport
    coupling:          CouplingReport

    # J / cross-cutting
    compound_familiarity: str
    compounds:            list

    # raw metrics pass-through
    metrics: RunMetrics


# ══════════════════════════════════════════════════════════════════════════════
# Extraction: one helper per section
# ══════════════════════════════════════════════════════════════════════════════

def _phys(result: CaseRunResult) -> PhysicalValidityReport:
    checks = result.metrics.physics_check_details or []

    def _assess(items):
        if not items:
            return True, "no checks defined"
        failures = [c for c in items if not c.get("passed", True)]
        if not failures:
            return True, "all pass"
        return False, "; ".join(c.get("detail", "") for c in failures[:3])

    temp_pass,  temp_d  = _assess([c for c in checks if c.get("check") in _TEMP_TYPES])
    pres_pass,  pres_d  = _assess([c for c in checks if c.get("check") in _PRESSURE_TYPES])
    phase_pass, phase_d = _assess([c for c in checks if c.get("check") in _PHASE_TYPES])
    coup_pass,  coup_d  = _assess([c for c in checks if c.get("check") in _COUPLING_TYPES])

    violated: list = []
    for c in checks:
        if not c.get("passed", True):
            tags = re.findall(r'\b([A-Z][A-Z0-9\-]{1,})\b', c.get("detail", ""))
            violated.extend(tags[:3])

    return PhysicalValidityReport(
        temperature_pass=temp_pass, temperature_detail=temp_d,
        pressure_pass=pres_pass,    pressure_detail=pres_d,
        phase_pass=phase_pass,      phase_detail=phase_d,
        coupling_pass=coup_pass,    coupling_detail=coup_d,
        violated_unit_tags=list(dict.fromkeys(violated)),
    )


def _constraint(result: CaseRunResult) -> ConstraintDiagnostic:
    iters = result.run_log.iterations
    m     = result.metrics
    traj  = m.score_trajectory

    if not iters:
        before = int(traj[0]) if traj else 0
        after  = int(traj[-1]) if traj else m.n_constraint_violations_final
        return ConstraintDiagnostic(
            violations_before=before,
            violations_after_settling=before,
            violations_after_repair=after,
            violations_after_local_opt=after,
            violations_after_beam=after,
        )

    v_before   = iters[0].n_errors_before
    v_settling = iters[0].n_errors_after
    v_repair   = iters[-1].n_errors_after

    # Use instrumented sub-stage counts when available (Fix 3)
    beam_vals  = [it.stage_after_beam_errors
                  for it in iters if it.stage_after_beam_errors is not None]
    local_vals = [it.stage_after_local_opt_errors
                  for it in iters if it.stage_after_local_opt_errors is not None]
    instrumented = bool(beam_vals)

    v_after_beam      = int(statistics.mean(beam_vals))  if beam_vals  else v_repair
    v_after_local_opt = int(statistics.mean(local_vals)) if local_vals else v_repair

    # Per-stage resolution attribution
    delta_settling  = max(0, v_before    - v_settling)
    delta_beam      = max(0, v_settling  - v_after_beam)
    delta_local_opt = max(0, v_after_beam - v_after_local_opt)
    resolution_by_stage = {
        "settling":    delta_settling,
        "beam_search": delta_beam,
        "local_opt":   delta_local_opt,
    }
    total_resolved = delta_settling + delta_beam + delta_local_opt
    dominant = (max(resolution_by_stage, key=resolution_by_stage.get)
                if total_resolved > 0 else "unknown")

    init_set  = set(iters[0].constraint_violations)
    final_set = set(iters[-1].constraint_violations)
    resolved  = sorted(init_set - final_set)[:10]

    phys_kws  = ("TEMP", "PRESSURE", "PHASE", "BUBBLE", "DEW", "FLASH")
    physical  = [v for v in init_set if any(kw in v.upper() for kw in phys_kws)][:5]
    heuristic = [v for v in init_set if v not in physical][:5]

    return ConstraintDiagnostic(
        violations_before=v_before,
        violations_after_settling=v_settling,
        violations_after_repair=v_repair,
        violations_after_local_opt=v_after_local_opt,
        violations_after_beam=v_after_beam,
        dropped_constraints=[v for v in resolved if v not in {
            c for it in iters[1:] for c in it.constraint_violations}][:5],
        resolved_constraints=resolved,
        physical_conflicts=physical,
        heuristic_conflicts=heuristic,
        constraint_resolution_by_stage=resolution_by_stage,
        dominant_resolution_stage=dominant,
        instrumented=instrumented,
    )


def _search(result: CaseRunResult) -> SearchTrace:
    m    = result.metrics
    traj = m.score_trajectory

    bw         = max(m.beam_width_used, 1)
    total_val  = m.n_candidates_total
    total_gen  = total_val * bw + m.n_state_cache_hits
    diversity  = ((total_val - m.n_state_cache_hits) / total_gen
                  if total_gen > 0 else 0.0)

    stagnation = sum(1 for i in range(1, len(traj)) if traj[i] == traj[i - 1])
    oscillation = sum(
        1 for i in range(1, len(traj) - 1)
        if traj[i] < traj[i - 1] and traj[i + 1] > traj[i]
    )

    exp    = m.explore_steps
    exploit = m.exploit_steps
    total  = exp + exploit
    ratio  = exp / total if total > 0 else 0.0

    return SearchTrace(
        beam_width_used=bw,
        total_states_generated=total_gen,
        total_states_validated=total_val,
        state_cache_hits=m.n_state_cache_hits,
        diversity_acceptance_rate=round(diversity, 3),
        coupling_boost_count=m.n_coupling_boosts,
        explore_phase_steps=exp,
        exploit_phase_steps=exploit,
        exploration_ratio=round(ratio, 3),
        stagnation_events=stagnation,
        oscillation_events=oscillation,
    )


def _repair(result: CaseRunResult) -> RepairDynamics:
    iters = result.run_log.iterations
    m     = result.metrics

    unit_counts: dict = {}
    src_tried:   dict = {k: 0 for k in ("deterministic", "physics", "llm", "unknown")}
    src_accepted: dict = {k: 0 for k in ("deterministic", "physics", "llm", "unknown")}
    magnitudes: list = []
    total_tried = total_accepted = 0

    for it in iters:
        # Unit tags from accepted changes
        for chg in it.changes:
            if isinstance(chg, str):
                for tag in re.findall(r'\b([A-Z][A-Z0-9\-]{1,8})\b', chg):
                    unit_counts[tag] = unit_counts.get(tag, 0) + 1
                nums = re.findall(r'[-+]?\d+\.?\d*', chg)
                if len(nums) >= 2:
                    try:
                        mag = abs(float(nums[-1]) - float(nums[-2]))
                        if 0 < mag < 1e6:
                            magnitudes.append(mag)
                    except ValueError:
                        pass

        for cand in it.candidates_tried:
            raw = cand.get("source", "").lower()
            src = ("deterministic" if any(k in raw for k in ("det", "rule", "deterministic"))
                   else "physics"  if any(k in raw for k in ("physics", "bubble", "thermo"))
                   else "llm"      if "llm" in raw
                   else "unknown")
            src_tried[src]  += 1
            total_tried     += 1
            score = cand.get("score")
            if score is not None and float(score) > 0:
                src_accepted[src] += 1
                total_accepted   += 1

    def _rate(src: str) -> float:
        t = src_tried[src]
        return round(src_accepted[src] / t, 3) if t > 0 else 0.0

    rejected = ((total_tried - total_accepted) / total_tried
                if total_tried > 0 else 0.0)

    traj = m.score_trajectory
    osc  = any(traj[i] < traj[i - 1] and traj[i + 1] > traj[i]
               for i in range(1, len(traj) - 1)) if len(traj) >= 3 else False
    all_changes = [c for it in iters for c in it.changes if isinstance(c, str)]
    osc_escape  = any("ESCAPE" in c.upper() for c in all_changes)
    cd_improved = len(traj) >= 2 and traj[-1] < traj[-2]

    # keep top-10 most repaired units
    top_units = dict(sorted(unit_counts.items(), key=lambda x: x[1], reverse=True)[:10])

    return RepairDynamics(
        repairs_per_unit=top_units,
        success_rate_deterministic=_rate("deterministic"),
        success_rate_physics=_rate("physics"),
        success_rate_llm=_rate("llm"),
        success_rate_unknown=_rate("unknown"),
        avg_fix_magnitude=round(statistics.mean(magnitudes), 3) if magnitudes else 0.0,
        rejected_fix_rate=round(rejected, 3),
        oscillation_detection_triggered=osc,
        oscillation_escape_used=osc_escape,
        coordinate_descent_improved=cd_improved,
    )


def _credit(result: CaseRunResult) -> TrajectoryCreditAnalysis:
    traj  = result.metrics.score_trajectory
    iters = result.run_log.iterations

    if len(traj) < 2:
        return TrajectoryCreditAnalysis(credit_label="INSUFFICIENT_DATA")

    improvements = [max(0.0, traj[i] - traj[i + 1]) for i in range(len(traj) - 1)]
    total = sum(improvements)

    if total == 0:
        return TrajectoryCreditAnalysis(total_credit=0.0, credit_label="NO_IMPROVEMENT")

    credits = [imp / total for imp in improvements]
    indexed = sorted(enumerate(credits), key=lambda x: x[1], reverse=True)
    top3: list = []
    for idx, cred in indexed[:3]:
        changes = iters[idx].changes[:3] if idx < len(iters) else []
        top3.append({
            "step":         idx,
            "credit":       round(cred, 3),
            "changes":      [str(c) for c in changes],
            "errors_before": int(traj[idx]),
            "errors_after":  int(traj[idx + 1]),
        })

    last_credit  = credits[-1]
    max_credit   = max(credits)
    n_steps      = len(credits)
    collapse   = last_credit > 0.70
    diffusion  = max_credit < 0.20 and n_steps > 3
    aligned    = not collapse and not diffusion and max_credit >= 0.25

    if collapse:
        label = "CREDIT_COLLAPSE"
    elif diffusion:
        label = "CREDIT_DIFFUSION"
    elif aligned:
        label = "CREDIT_ALIGNMENT_GOOD"
    else:
        label = "MIXED"

    return TrajectoryCreditAnalysis(
        total_credit=total,
        top_credited_moves=top3,
        credit_label=label,
        credit_collapse=collapse,
        credit_diffusion=diffusion,
        credit_alignment_good=aligned,
    )


def _margin(result: CaseRunResult) -> MarginModelReport:
    snapshot = result.metrics.margin_snapshot

    if snapshot:
        # Real data from instrumented pipeline (Fix 1)
        n_entries  = len(snapshot)
        all_means  = [v["mean"] for v in snapshot.values()]
        all_stds   = [v["std"]  for v in snapshot.values()]
        all_n_obs  = [v["n_obs"] for v in snapshot.values()]
        wildcards  = sum(1 for n in all_n_obs if n < 3)
        hard_hits  = sum(1 for v in snapshot.values()
                         if v["mean"] >= 65.0 or v["mean"] <= 4.0)

        variance  = statistics.variance(all_means) if len(all_means) >= 2 else 0.0
        mean_val  = statistics.mean(all_means) if all_means else 1.0
        mean_std  = statistics.mean(all_stds)  if all_stds  else 0.0
        stability = round(mean_std / max(mean_val, 1e-9), 4)
        drift     = variance > 100.0 or any(v["std"] > 25.0 for v in snapshot.values())

        return MarginModelReport(
            n_updates=n_entries,
            margin_variance=round(variance, 4),
            wildcard_fallbacks=wildcards,
            stability=stability,
            drift_detected=drift,
            hard_bounds_hit_freq=round(hard_hits / max(n_entries, 1), 3),
            cold_start_dominant=(wildcards / max(n_entries, 1) > 0.5),
            instrumented=True,
        )

    # Proxy fallback for runs without snapshot
    iters       = result.run_log.iterations
    all_changes = [c for it in iters for c in it.changes if isinstance(c, str)]
    n_updates  = sum(1 for c in all_changes if any(s in c for s in ("→", ":=", "->", "=")))
    wildcards  = sum(1 for c in all_changes
                     if any(kw in c.upper() for kw in ("WILDCARD", "DEFAULT", "FALLBACK")))
    hard_hits  = sum(1 for c in all_changes
                     if any(kw in c.upper() for kw in ("CLIP", "BOUND", "LIMIT")))

    traj = result.metrics.score_trajectory
    if len(traj) >= 2:
        mags      = [abs(traj[i] - traj[i - 1]) for i in range(1, len(traj))]
        variance  = statistics.variance(mags) if len(mags) >= 2 else 0.0
        last_n    = mags[-10:]
        stability = statistics.variance(last_n) if len(last_n) >= 2 else 0.0
        mid   = len(mags) // 2
        drift = (statistics.mean(mags[mid:]) > statistics.mean(mags[:mid]) * 1.2
                 if mid > 0 and mags[mid:] else False)
    else:
        variance = stability = 0.0
        drift = False

    return MarginModelReport(
        n_updates=n_updates,
        margin_variance=round(variance, 4),
        wildcard_fallbacks=wildcards,
        stability=round(stability, 4),
        drift_detected=drift,
        hard_bounds_hit_freq=round(hard_hits / max(n_updates, 1), 3),
        cold_start_dominant=(wildcards / max(n_updates, 1) > 0.5),
        instrumented=False,
    )


def _coord(result: CaseRunResult) -> CoordDescentReport:
    iters = result.run_log.iterations
    traj  = result.metrics.score_trajectory

    cd_iters = [it for it in iters if it.coord_descent_triggered]

    if cd_iters:
        # Real data from instrumented pipeline (Fix 2)
        total_improvement = sum(it.coord_descent_improvement for it in cd_iters)
        l0 = sum(1 for it in cd_iters if it.coord_descent_level >= 0)
        l1 = sum(1 for it in cd_iters if it.coord_descent_level >= 1)
        l2 = sum(1 for it in cd_iters if it.coord_descent_level >= 2)

        all_changes = [c for it in iters for c in it.changes if isinstance(c, str)]
        rejected    = sum(1 for c in all_changes if "REJECT" in c.upper())
        rej_rate    = rejected / max(len(all_changes), 1)

        return CoordDescentReport(
            improved=total_improvement > 0,
            improvement_magnitude=round(total_improvement, 2),
            level_0_triggered=l0,
            level_1_triggered=l1,
            level_2_triggered=l2,
            constraint_rejection_rate=round(rej_rate, 3),
            instrumented=True,
        )

    # Proxy fallback
    all_changes  = [c for it in iters for c in it.changes if isinstance(c, str)]
    improved     = len(traj) >= 2 and traj[-1] < traj[-2]
    magnitude    = max(0.0, traj[-2] - traj[-1]) if len(traj) >= 2 else 0.0

    def _count_level(n: int) -> int:
        patterns = (f"LEVEL_{n}", f"LEVEL {n}", f"L{n}")
        return sum(1 for c in all_changes if any(p in c.upper() for p in patterns))

    last_changes = iters[-1].changes if iters else []
    rejected     = sum(1 for c in last_changes if isinstance(c, str) and "REJECT" in c.upper())
    rej_rate     = rejected / max(len(last_changes), 1)

    return CoordDescentReport(
        improved=improved,
        improvement_magnitude=magnitude,
        level_0_triggered=_count_level(0),
        level_1_triggered=_count_level(1),
        level_2_triggered=_count_level(2),
        constraint_rejection_rate=round(rej_rate, 3),
        instrumented=False,
    )


def _coupling(result: CaseRunResult) -> CouplingReport:
    m     = result.metrics
    iters = result.run_log.iterations

    all_boost_tags: list = [t for it in iters for t in it.coupling_boosts]
    tag_freq: dict = {}
    for t in all_boost_tags:
        tag_freq[t] = tag_freq.get(t, 0) + 1
    ping_pong = sum(1 for freq in tag_freq.values() if freq >= 3)

    settler_resolved = sum(
        1 for it in iters
        if it.coupling_boosts and it.n_errors_after < it.n_errors_before
    )

    return CouplingReport(
        corrections_triggered=m.n_coupling_boosts,
        ping_pong_events=ping_pong,
        coupled_settler_resolved=settler_resolved,
        propagation_lag_steps=m.steps_to_full_consistency,
    )


# ── Main extractor ─────────────────────────────────────────────────────────────

def extract_case_diagnostic(result: CaseRunResult) -> CaseDiagnostic:
    case = result.case
    m    = result.metrics
    ref  = case.reference_structure

    n_units       = ref.n_units if ref else 0
    n_constraints = (len(case.expected.physics_checks) +
                     (len(ref.connections) if ref else 0))

    traj         = m.score_trajectory
    final_errors = int(traj[-1]) if traj else m.n_constraint_violations_final

    if m.success:
        status = "converged"
    elif m.valid_ir and traj and final_errors < (traj[0] / 2):
        status = "partially converged"
    else:
        status = "failed"

    return CaseDiagnostic(
        case_id=case.id, tier=case.tier, ablation_mode=m.ablation_mode,
        n_units=n_units, n_constraints=n_constraints,
        compound_classes=_classify_compounds(case.compounds),
        convergence_status=status,
        final_ir_error_count=final_errors,
        iterations_to_converge=m.n_iterations,
        physical_validity = _phys(result),
        constraint_diag   = _constraint(result),
        search_trace      = _search(result),
        repair_dynamics   = _repair(result),
        trajectory_credit = _credit(result),
        margin_model      = _margin(result),
        coord_descent     = _coord(result),
        coupling          = _coupling(result),
        compound_familiarity=_compound_familiarity(case.compounds),
        compounds=case.compounds,
        metrics=m,
    )


# ══════════════════════════════════════════════════════════════════════════════
# DiagnosticReport
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DiagnosticReport:
    """Full diagnostic report across one or more ablation modes."""
    timestamp:   str
    modes:       list
    diagnostics: dict   # {mode: [CaseDiagnostic, ...]}
    run_sets:    dict   # {mode: BenchmarkRunSet}

    def format(self, verbose: bool = False) -> str:
        return format_full_report(self, verbose=verbose)

    def global_tables(self) -> str:
        return _format_global_tables(self)

    def interpretation(self) -> str:
        return interpret(self)

    def save(self, out_dir: str = "results/diagnostics") -> str:
        os.makedirs(out_dir, exist_ok=True)
        fname = f"diagnostic_{self.timestamp}.json"
        path  = os.path.join(out_dir, fname)
        payload = {
            "timestamp": self.timestamp,
            "modes":     self.modes,
            "cases": {
                mode: [_diag_to_dict(d) for d in diags]
                for mode, diags in self.diagnostics.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        return path


# ══════════════════════════════════════════════════════════════════════════════
# DiagnosticEngine
# ══════════════════════════════════════════════════════════════════════════════

class DiagnosticEngine:
    """
    Entry point.  Pass the output of BenchmarkRunner.run_ablation() directly:

        engine = DiagnosticEngine()
        report = engine.analyse(runner.run_ablation())
    """

    def analyse(
        self,
        results,  # dict[str, BenchmarkRunSet] | BenchmarkRunSet
    ) -> DiagnosticReport:
        if isinstance(results, BenchmarkRunSet):
            results = {results.ablation_mode: results}

        diags: dict = {}
        for mode, run_set in results.items():
            diags[mode] = [extract_case_diagnostic(r) for r in run_set.case_results]

        return DiagnosticReport(
            timestamp=time.strftime("%Y%m%d_%H%M%S"),
            modes=list(results.keys()),
            diagnostics=diags,
            run_sets=results,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Formatting helpers
# ══════════════════════════════════════════════════════════════════════════════

_W = 68  # report column width


def _bar(char: str = "═") -> str:
    return char * _W


def _row(label: str, value, width: int = 22) -> str:
    return f"  {label:<{width}}: {value}"


def format_case_diagnostic(d: CaseDiagnostic) -> str:
    """Full per-case formatted block covering sections A–I."""
    lines: list = []
    lines += [
        _bar(),
        f"  {d.case_id}  |  {d.tier}  |  {d.ablation_mode}",
        _bar(),
        "",
        "[A] CASE SUMMARY",
        _row("case_id",            d.case_id),
        _row("tier",               d.tier),
        _row("ablation_mode",      d.ablation_mode),
        _row("n_units",            d.n_units),
        _row("n_constraints",      d.n_constraints),
        _row("compound_classes",   ", ".join(d.compound_classes)),
        _row("compounds",          ", ".join(d.compounds)),
        _row("familiarity",        d.compound_familiarity),
        _row("convergence_status", d.convergence_status.upper()),
        _row("final_ir_errors",    d.final_ir_error_count),
        _row("iterations_used",    f"{d.iterations_to_converge}/{d.metrics.n_iterations or '?'}"),
        "",
    ]

    # B — physical validity
    pv = d.physical_validity
    lines += [
        "[B] PHYSICAL VALIDITY",
        _row("Temperature", ("PASS" if pv.temperature_pass else f"FAIL  {pv.temperature_detail[:50]}")),
        _row("Pressure",    ("PASS" if pv.pressure_pass    else f"FAIL  {pv.pressure_detail[:50]}")),
        _row("Phase",       ("PASS" if pv.phase_pass       else f"FAIL  {pv.phase_detail[:50]}")),
        _row("Coupling",    ("PASS" if pv.coupling_pass    else f"FAIL  {pv.coupling_detail[:50]}")),
        _row("Violated units", (", ".join(pv.violated_unit_tags) or "none")),
        f"  ── Overall: {pv.overall}",
        "",
    ]

    # C — constraint diagnostics
    cd       = d.constraint_diag
    proxy_c  = "" if cd.instrumented else " (proxy)"
    lines += [
        "[C] CONSTRAINT SYSTEM",
        _row("Violations before",                  cd.violations_before),
        _row("After settling",                     cd.violations_after_settling),
        _row(f"After beam search{proxy_c}",        cd.violations_after_beam),
        _row(f"After local opt{proxy_c}",          cd.violations_after_local_opt),
        _row("After repair",                       cd.violations_after_repair),
        _row("Dropped",   (", ".join(cd.dropped_constraints[:3]) or "none")),
        _row("Resolved",  (f"{len(cd.resolved_constraints)} constraints")),
        _row("Physical",  (f"{len(cd.physical_conflicts)} conflicts")),
        _row("Heuristic", (f"{len(cd.heuristic_conflicts)} conflicts")),
    ]
    if cd.constraint_resolution_by_stage:
        rs = cd.constraint_resolution_by_stage
        lines.append(
            _row("Stage attribution",
                 f"settling={rs.get('settling',0)}  "
                 f"beam={rs.get('beam_search',0)}  "
                 f"local_opt={rs.get('local_opt',0)}")
        )
        lines.append(_row("Dominant stage", cd.dominant_resolution_stage))
    lines.append("")

    # D — search trace
    st  = d.search_trace
    m_d = d.metrics
    rec_bd = m_d.error_recurrence_breakdown
    lines += [
        "[D] SEARCH BEHAVIOUR",
        _row("Beam width",           st.beam_width_used),
        _row("States generated",     st.total_states_generated),
        _row("States validated",     st.total_states_validated),
        _row("Cache hits",           st.state_cache_hits),
        _row("Diversity rate",       f"{st.diversity_acceptance_rate:.3f}"),
        _row("Coupling boosts",      st.coupling_boost_count),
        _row("Explore / exploit",    f"{st.explore_phase_steps} / {st.exploit_phase_steps}"),
        _row("Exploration ratio",    f"{st.exploration_ratio:.3f}"),
        _row("Stagnation events",    st.stagnation_events),
        _row("Oscillation events",   st.oscillation_events),
        _row("Error recurrences",
             f"{m_d.n_error_recurrences}  (rate={m_d.error_recurrence_rate:.3f})"),
    ]
    if rec_bd:
        lines.append(
            _row("  Recurrence types",
                 f"osc={rec_bd.get('oscillation',0)}  "
                 f"coup={rec_bd.get('coupling',0)}  "
                 f"prop={rec_bd.get('propagation',0)}")
        )
    lines.append("")

    # E — repair dynamics
    rd = d.repair_dynamics
    top_units = sorted(rd.repairs_per_unit.items(), key=lambda x: x[1], reverse=True)[:4]
    lines += [
        "[E] REPAIR DYNAMICS",
        _row("Most-repaired units",
             (", ".join(f"{t}({n}×)" for t, n in top_units) or "none")),
        _row("Source rates",
             (f"det={rd.success_rate_deterministic:.2f}  "
              f"phys={rd.success_rate_physics:.2f}  "
              f"llm={rd.success_rate_llm:.2f}")),
        _row("Avg fix magnitude",        f"{rd.avg_fix_magnitude:.2f}"),
        _row("Rejected fix rate",        f"{rd.rejected_fix_rate:.3f}"),
        _row("Oscillation detected",     "YES" if rd.oscillation_detection_triggered else "NO"),
        _row("Oscillation escape used",  "YES" if rd.oscillation_escape_used else "NO"),
        _row("Coord descent improved",   "YES" if rd.coordinate_descent_improved else "NO"),
        "",
    ]

    # F — trajectory credit
    tc = d.trajectory_credit
    lines += ["[F] TRAJECTORY CREDIT", _row("Total credit", tc.total_credit),
              _row("Credit label", tc.credit_label)]
    if tc.top_credited_moves:
        lines.append("  Top credited moves:")
        for mv in tc.top_credited_moves:
            chg_str = " | ".join(str(c) for c in mv.get("changes", []))[:50]
            lines.append(
                f"    step {mv['step']:2d}  credit={mv['credit']:.3f}"
                f"  err:{mv['errors_before']}→{mv['errors_after']}"
                f"  [{chg_str}]"
            )
    lines += [
        _row("Credit collapse",  "YES" if tc.credit_collapse else "NO"),
        _row("Credit diffusion", "YES" if tc.credit_diffusion else "NO"),
        "",
    ]

    # G — margin model
    mm       = d.margin_model
    proxy_g  = "" if mm.instrumented else "  (proxy)"
    n_label  = "n_entries" if mm.instrumented else "n_updates"
    lines += [
        f"[G] MARGIN MODEL{proxy_g}",
        _row(n_label,          mm.n_updates),
        _row("Margin variance", f"{mm.margin_variance:.4f}"),
        _row("Wildcards",       mm.wildcard_fallbacks),
        _row("Stability",       f"{mm.stability:.4f}"),
        _row("Drift detected",  "YES" if mm.drift_detected else "NO"),
        _row("Hard bounds freq",f"{mm.hard_bounds_hit_freq:.3f}"),
        _row("Cold-start dom.", "YES" if mm.cold_start_dominant else "NO"),
        "",
    ]

    # H — coordinate descent
    hd      = d.coord_descent
    proxy_h = "" if hd.instrumented else "  (proxy)"
    lines += [
        f"[H] COORDINATE DESCENT{proxy_h}",
        _row("Improved after beam",    "YES" if hd.improved else "NO"),
        _row("Improvement magnitude",  f"{hd.improvement_magnitude:.2f}"),
        _row("Level-0 triggered",      hd.level_0_triggered),
        _row("Level-1 triggered",      hd.level_1_triggered),
        _row("Level-2 triggered",      hd.level_2_triggered),
        _row("Constraint rejection",   f"{hd.constraint_rejection_rate:.3f}"),
        "",
    ]

    # I — coupling
    cp        = d.coupling
    ping_flag = "  *** PING-PONG DETECTED ***" if cp.ping_pong_events > 0 else ""
    rec_bd_i  = d.metrics.error_recurrence_breakdown
    lines += [
        "[I] COUPLING SYSTEM",
        _row("Corrections triggered",  cp.corrections_triggered),
        _row("Ping-pong events",       f"{cp.ping_pong_events}{ping_flag}"),
        _row("Settler resolved",       cp.coupled_settler_resolved),
        _row("Propagation lag",        f"{cp.propagation_lag_steps} step(s)"),
    ]
    if rec_bd_i:
        osc  = rec_bd_i.get("oscillation", 0)
        coup = rec_bd_i.get("coupling", 0)
        if osc + coup > 0:
            lines.append(
                _row("Recurrence (osc/coup)",
                     f"{osc} oscillation  {coup} coupling-induced")
            )
    lines.append("")

    return "\n".join(lines)


def _format_global_tables(report: DiagnosticReport) -> str:
    lines: list = []

    # ── Table 1: Tier Performance (full_ccs or first mode) ────────────────────
    ref_mode = "full_ccs" if "full_ccs" in report.run_sets else report.modes[0]
    ref_set  = report.run_sets.get(ref_mode)

    lines += [
        _bar("─"),
        "TABLE 1 — TIER PERFORMANCE",
        _bar("─"),
        f"  Mode: {ref_mode}",
        f"  {'Tier':<16} {'Success':>8} {'Avg IR↓':>9} {'Avg iters':>10} {'Oscillation':>12} {'Cases':>6}",
        f"  {'─'*16} {'─'*8} {'─'*9} {'─'*10} {'─'*12} {'─'*6}",
    ]

    if ref_set and ref_set.tier_aggregates:
        for tier, agg in sorted(ref_set.tier_aggregates.items()):
            ref_diags = [d for d in report.diagnostics.get(ref_mode, []) if d.tier == tier]
            osc_rate  = (sum(d.search_trace.oscillation_events > 0 for d in ref_diags)
                         / max(len(ref_diags), 1))
            traj_vals = [d.metrics.score_trajectory for d in ref_diags]
            avg_reduction = statistics.mean(
                [(t[0] - t[-1]) for t in traj_vals if len(t) >= 2]
            ) if any(len(t) >= 2 for t in traj_vals) else 0.0
            lines.append(
                f"  {tier:<16} {agg.success_rate:>7.1%} {avg_reduction:>9.2f}"
                f" {agg.mean_iterations:>10.2f} {osc_rate:>11.1%} {len(ref_diags):>6}"
            )
    else:
        lines.append("  (no tier data available)")
    lines.append("")

    # ── Table 2: Ablation Impact ──────────────────────────────────────────────
    baseline_agg = (ref_set.aggregate if ref_set else None)
    lines += [
        _bar("─"),
        "TABLE 2 — ABLATION IMPACT  (Δ vs full_ccs)",
        _bar("─"),
        f"  {'Mode':<16} {'Δ Success':>9} {'Δ Conv spd':>11} {'Δ Physics':>10} "
        f"{'Δ Oscilln':>10} {'Δ GenGap':>9}",
        f"  {'─'*16} {'─'*9} {'─'*11} {'─'*10} {'─'*10} {'─'*9}",
    ]

    for mode in report.modes:
        rs  = report.run_sets.get(mode)
        agg = rs.aggregate if rs else None
        if not agg or not baseline_agg:
            lines.append(f"  {mode:<16}  (no aggregate)")
            continue
        delta_succ  = agg.success_rate      - baseline_agg.success_rate
        delta_iters = agg.mean_iterations   - baseline_agg.mean_iterations
        delta_phys  = agg.physics_pass_rate - baseline_agg.physics_pass_rate
        delta_osc   = agg.pct_score_oscillated - baseline_agg.pct_score_oscillated

        # generalisation gap: generalisation tier success vs overall
        gen_diags = [d for d in report.diagnostics.get(mode, []) if d.tier == "generalisation"]
        gen_succ  = (sum(d.metrics.success for d in gen_diags) / len(gen_diags)
                     if gen_diags else None)
        gap_str = (f"{(gen_succ - agg.success_rate):+.1%}" if gen_succ is not None else "  n/a")

        marker = "◀ baseline" if mode == ref_mode else ""
        lines.append(
            f"  {mode:<16} {delta_succ:>+8.1%} {delta_iters:>+10.2f}"
            f" {delta_phys:>+9.1%} {delta_osc:>+9.1%} {gap_str:>9}  {marker}"
        )
    lines.append("")

    # ── Table 3: Failure Mode Distribution ────────────────────────────────────
    lines += [
        _bar("─"),
        "TABLE 3 — FAILURE MODE DISTRIBUTION",
        _bar("─"),
        f"  {'Outcome':<22} {'Count':>7} {'% of all':>9}  by tier",
        f"  {'─'*22} {'─'*7} {'─'*9}",
    ]

    all_diags = [d for diags in report.diagnostics.values() for d in diags]
    total_cases = len(all_diags)
    outcome_by_tier: dict = {}
    outcome_counts: dict  = {}
    for d in all_diags:
        oc = d.metrics.outcome or ("PASS" if d.metrics.success else "UNKNOWN")
        outcome_counts[oc] = outcome_counts.get(oc, 0) + 1
        if d.tier not in outcome_by_tier:
            outcome_by_tier[d.tier] = {}
        outcome_by_tier[d.tier][oc] = outcome_by_tier[d.tier].get(oc, 0) + 1

    for oc in sorted(outcome_counts, key=lambda x: outcome_counts[x], reverse=True):
        count = outcome_counts[oc]
        pct   = count / total_cases if total_cases else 0.0
        tier_parts = ", ".join(
            f"{tier}:{cnt}" for tier, oc_map in sorted(outcome_by_tier.items())
            if (cnt := oc_map.get(oc, 0)) > 0
        )
        lines.append(f"  {oc:<22} {count:>7} {pct:>8.1%}  [{tier_parts}]")
    lines.append("")

    return "\n".join(lines)


def _diag_to_dict(d: CaseDiagnostic) -> dict:
    pv = d.physical_validity
    cd = d.constraint_diag
    st = d.search_trace
    rd = d.repair_dynamics
    tc = d.trajectory_credit
    mm = d.margin_model
    hd = d.coord_descent
    cp = d.coupling
    return {
        "case_id": d.case_id, "tier": d.tier, "ablation_mode": d.ablation_mode,
        "n_units": d.n_units, "n_constraints": d.n_constraints,
        "compound_classes": d.compound_classes, "compounds": d.compounds,
        "convergence_status": d.convergence_status,
        "final_ir_error_count": d.final_ir_error_count,
        "iterations": d.iterations_to_converge,
        "compound_familiarity": d.compound_familiarity,
        "B_physical_validity": {
            "temperature": pv.temperature_pass, "temperature_detail": pv.temperature_detail,
            "pressure":    pv.pressure_pass,    "pressure_detail":    pv.pressure_detail,
            "phase":       pv.phase_pass,        "phase_detail":       pv.phase_detail,
            "coupling":    pv.coupling_pass,     "coupling_detail":    pv.coupling_detail,
            "overall":     pv.overall,           "violated_units":     pv.violated_unit_tags,
        },
        "C_constraint_diag": {
            "violations_before":         cd.violations_before,
            "violations_after_settling": cd.violations_after_settling,
            "violations_after_beam":     cd.violations_after_beam,
            "violations_after_local_opt": cd.violations_after_local_opt,
            "violations_after_repair":   cd.violations_after_repair,
            "dropped":  cd.dropped_constraints,
            "resolved": cd.resolved_constraints,
            "physical_conflicts":  cd.physical_conflicts,
            "heuristic_conflicts": cd.heuristic_conflicts,
            "resolution_by_stage": cd.constraint_resolution_by_stage,
            "dominant_stage":      cd.dominant_resolution_stage,
            "instrumented":        cd.instrumented,
        },
        "D_search_trace": {
            "beam_width":        st.beam_width_used,
            "states_generated":  st.total_states_generated,
            "states_validated":  st.total_states_validated,
            "cache_hits":        st.state_cache_hits,
            "diversity_rate":    st.diversity_acceptance_rate,
            "coupling_boosts":   st.coupling_boost_count,
            "explore_steps":     st.explore_phase_steps,
            "exploit_steps":     st.exploit_phase_steps,
            "exploration_ratio": st.exploration_ratio,
            "stagnation_events": st.stagnation_events,
            "oscillation_events": st.oscillation_events,
            "n_error_recurrences": d.metrics.n_error_recurrences,
            "error_recurrence_rate": d.metrics.error_recurrence_rate,
            "recurrence_breakdown": d.metrics.error_recurrence_breakdown,
        },
        "E_repair_dynamics": {
            "repairs_per_unit":           rd.repairs_per_unit,
            "sr_deterministic":           rd.success_rate_deterministic,
            "sr_physics":                 rd.success_rate_physics,
            "sr_llm":                     rd.success_rate_llm,
            "avg_fix_magnitude":          rd.avg_fix_magnitude,
            "rejected_fix_rate":          rd.rejected_fix_rate,
            "oscillation_escape":         rd.oscillation_escape_used,
            "coord_descent_improved":     rd.coordinate_descent_improved,
        },
        "F_trajectory_credit": {
            "total_credit": tc.total_credit, "label": tc.credit_label,
            "collapse": tc.credit_collapse,  "diffusion": tc.credit_diffusion,
            "alignment_good": tc.credit_alignment_good,
            "top_moves": tc.top_credited_moves,
        },
        "G_margin_model": {
            "n_entries": mm.n_updates, "variance": mm.margin_variance,
            "wildcards": mm.wildcard_fallbacks, "stability": mm.stability,
            "drift": mm.drift_detected, "cold_start": mm.cold_start_dominant,
            "hard_bounds_freq": mm.hard_bounds_hit_freq, "instrumented": mm.instrumented,
        },
        "H_coord_descent": {
            "improved": hd.improved, "magnitude": hd.improvement_magnitude,
            "level_0": hd.level_0_triggered, "level_1": hd.level_1_triggered,
            "level_2": hd.level_2_triggered,
            "rejection_rate": hd.constraint_rejection_rate, "instrumented": hd.instrumented,
        },
        "I_coupling": {
            "corrections": cp.corrections_triggered, "ping_pong": cp.ping_pong_events,
            "settler_resolved": cp.coupled_settler_resolved,
            "propagation_lag": cp.propagation_lag_steps,
        },
        "score_trajectory": d.metrics.score_trajectory,
        "outcome": d.metrics.outcome,
    }


def format_full_report(report: DiagnosticReport, verbose: bool = False) -> str:
    lines: list = [
        _bar(),
        f"  CCS DIAGNOSTIC REPORT  |  {report.timestamp}",
        f"  Modes: {', '.join(report.modes)}",
        _bar(),
        "",
    ]

    # Global tables always present
    lines.append(report.global_tables())

    # Per-mode case diagnostics
    for mode in report.modes:
        diags = report.diagnostics.get(mode, [])
        lines += [_bar("─"), f"  MODE: {mode}  ({len(diags)} cases)", _bar("─"), ""]

        # Pick cases to show in detail: failures + flagged
        def _is_flagged(d: CaseDiagnostic) -> bool:
            return (
                d.convergence_status != "converged"
                or d.search_trace.oscillation_events > 0
                or d.coupling.ping_pong_events > 0
                or d.trajectory_credit.credit_collapse
                or d.physical_validity.overall != "PASS"
            )

        show = diags if verbose else [d for d in diags if _is_flagged(d)]
        if not show and not verbose:
            lines.append("  All cases converged cleanly — no flagged diagnostics.\n")
        else:
            for d in show:
                lines.append(format_case_diagnostic(d))

        # Mode summary line
        if diags:
            n_conv   = sum(1 for d in diags if d.convergence_status == "converged")
            n_osc    = sum(1 for d in diags if d.search_trace.oscillation_events > 0)
            n_ping   = sum(1 for d in diags if d.coupling.ping_pong_events > 0)
            n_phys   = sum(1 for d in diags if d.physical_validity.overall != "PASS")
            lines += [
                f"  SUMMARY [{mode}]: converged={n_conv}/{len(diags)}"
                f"  oscillation={n_osc}  ping-pong={n_ping}  phys-fail={n_phys}",
                "",
            ]

    # J — Generalisation breakdown across all modes
    lines += [_bar("─"), "SECTION J — GENERALISATION BREAKDOWN", _bar("─"), ""]
    for mode in report.modes:
        diags = report.diagnostics.get(mode, [])
        known  = [d for d in diags if d.compound_familiarity == "known"]
        unseen = [d for d in diags if d.compound_familiarity == "unseen"]

        def _stats(subset, label):
            if not subset:
                return f"  {label:<10}: n/a"
            succ = sum(d.metrics.success for d in subset) / len(subset)
            traj_reds = [(t[0] - t[-1]) for d in subset
                         if (t := d.metrics.score_trajectory) and len(t) >= 2]
            ir_red = statistics.mean(traj_reds) if traj_reds else 0.0
            vals   = [d.search_trace.total_states_validated for d in subset]
            err_red = [(t[0] - t[-1]) for d in subset
                       if (t := d.metrics.score_trajectory) and len(t) >= 2]
            cands   = [d.search_trace.total_states_validated for d in subset]
            beam_eff = statistics.mean(
                [er / max(c, 1) for er, c in zip(err_red, cands) if c > 0]
            ) if err_red else 0.0
            fail_modes = {}
            for d in subset:
                oc = d.metrics.outcome or ("PASS" if d.metrics.success else "?")
                if oc != "PASS":
                    fail_modes[oc] = fail_modes.get(oc, 0) + 1
            fm_str = ", ".join(f"{k}:{v}" for k, v in sorted(fail_modes.items()))
            return (f"  {label:<10}: n={len(subset)}"
                    f"  succ={succ:.1%}  IR↓={ir_red:.2f}"
                    f"  beam_eff={beam_eff:.3f}  failures=[{fm_str or 'none'}]")

        lines.append(f"  {mode}:")
        lines.append(_stats(known, "known"))
        lines.append(_stats(unseen, "unseen"))
        lines.append("")

    # Interpretation
    lines += ["", report.interpretation()]

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Interpretation / bottleneck ranking
# ══════════════════════════════════════════════════════════════════════════════

def _agg_success(report: DiagnosticReport, mode: str) -> Optional[float]:
    rs = report.run_sets.get(mode)
    return rs.aggregate.success_rate if rs and rs.aggregate else None


def interpret(report: DiagnosticReport) -> str:
    lines: list = [_bar("═"), "  SYSTEM INTERPRETATION", _bar("═"), ""]

    # ── A. Bottleneck ranking from ablation deltas ────────────────────────────
    base = _agg_success(report, "full_ccs")
    contributions: list = []

    _pairs = [
        ("physics",    "no_physics",    "Thermo estimation (bubble_point_K, pkg selection)"),
        ("rule_store", "no_rule_store", "Rule store / RAG (BIP lookup, unit context)"),
        ("search",     "greedy",        "Beam search (multi-candidate parallel exploration)"),
        ("coupling",   "no_coupling",   "Coupling propagation (ParameterCouplingMap)"),
    ]
    for key, ablated_mode, label in _pairs:
        ablated = _agg_success(report, ablated_mode)
        if base is not None and ablated is not None:
            delta = base - ablated   # positive = ablation hurts → component contributes
            contributions.append((delta, key, label, ablated_mode))
        else:
            contributions.append((0.0, key, label, ablated_mode))

    contributions.sort(reverse=True)

    lines += ["[A] SYSTEM DIAGNOSIS", ""]

    if base is not None:
        lines.append(f"  Baseline success rate (full_ccs): {base:.1%}")
    else:
        lines.append("  Baseline (full_ccs) not in results — using first mode.")

    lines.append("")
    lines.append("  Contribution of each subsystem (Δ success rate when ablated):")
    lines.append("")

    for rank, (delta, key, label, mode) in enumerate(contributions, 1):
        flag = " *** PRIMARY BOTTLENECK ***" if rank == 1 else ""
        lines.append(f"    #{rank}  {label}")
        lines.append(f"        Key: {key}  |  Ablated mode: {mode}  |  Δ={delta:+.1%}{flag}")
        lines.append("")

    # ── B. Dominant failure mode ──────────────────────────────────────────────
    all_diags = [d for diags in report.diagnostics.values() for d in diags]
    outcome_counts: dict = {}
    for d in all_diags:
        oc = d.metrics.outcome or ("PASS" if d.metrics.success else "UNKNOWN")
        outcome_counts[oc] = outcome_counts.get(oc, 0) + 1

    non_pass = {k: v for k, v in outcome_counts.items() if k != "PASS"}
    total_failures = sum(non_pass.values())
    dominant = max(non_pass, key=lambda k: non_pass[k]) if non_pass else "NONE"

    lines += [
        "[B] DOMINANT FAILURE MODE",
        f"  Most common: {dominant}  ({non_pass.get(dominant, 0)} / {total_failures} failures)",
        "",
    ]

    # Categorise failure source
    search_dominated   = non_pass.get("REPAIR_EXHAUSTED", 0) + non_pass.get("MAX_ITER", 0)
    physics_dominated  = non_pass.get("PHYSICS_VIOLATION", 0) + non_pass.get("MISSING_PARAMS", 0)
    ir_dominated       = non_pass.get("INVALID_IR", 0) + non_pass.get("INVALID_JSON", 0)

    if total_failures > 0:
        lines.append(f"  Search failures  : {search_dominated:>4} ({search_dominated/total_failures:.1%})")
        lines.append(f"  Physics failures : {physics_dominated:>4} ({physics_dominated/total_failures:.1%})")
        lines.append(f"  IR/format failures:{ir_dominated:>3} ({ir_dominated/total_failures:.1%})")
    lines.append("")

    # ── C. Generalisation diagnosis ───────────────────────────────────────────
    known_diags  = [d for d in all_diags if d.compound_familiarity == "known"]
    unseen_diags = [d for d in all_diags if d.compound_familiarity == "unseen"]

    known_succ  = (sum(d.metrics.success for d in known_diags) / len(known_diags)
                   if known_diags else None)
    unseen_succ = (sum(d.metrics.success for d in unseen_diags) / len(unseen_diags)
                   if unseen_diags else None)

    lines += ["[C] GENERALISATION DIAGNOSIS"]
    if known_succ is not None and unseen_succ is not None:
        gen_gap = known_succ - unseen_succ
        if gen_gap > 0.15:
            verdict = "LIMITED — large generalisation gap; likely thermodynamic overfitting"
        elif gen_gap > 0.05:
            verdict = "MODERATE — some generalisation gap; review BIP coverage for unseen compounds"
        else:
            verdict = "GOOD — generalisation gap within noise"
        lines += [
            f"  Known-compound success : {known_succ:.1%}",
            f"  Unseen-compound success: {unseen_succ:.1%}",
            f"  Gap                    : {gen_gap:+.1%}",
            f"  Verdict                : {verdict}",
            "",
        ]
    else:
        lines.append("  Insufficient data (run generalisation tier to assess).\n")

    # ── D. Bottleneck ranking summary ─────────────────────────────────────────
    lines += ["[D] BOTTLENECK RANKING (in order of impact)", ""]
    for rank, (delta, key, label, mode) in enumerate(contributions, 1):
        lines.append(f"  {rank}. {label}")
        lines.append(f"     Ablation drop: {delta:+.1%}  (mode: {mode})")
        lines.append("")

    # ── E. Tuning recommendations ─────────────────────────────────────────────
    lines += ["[E] TUNING RECOMMENDATIONS", ""]

    # Generate data-driven recommendations
    recs: list = []

    # Margin drift
    any_drift = any(d.margin_model.drift_detected for d in all_diags)
    if any_drift:
        recs.append(("margins",
                     "Margin drift detected across cases — "
                     "add upper bound clipping in MarginModel.update()"))

    # Oscillation rate
    osc_rate = (sum(d.search_trace.oscillation_events > 0 for d in all_diags)
                / max(len(all_diags), 1))
    if osc_rate > 0.25:
        recs.append(("beam_width",
                     f"Oscillation rate {osc_rate:.1%} — "
                     "increase beam_width or tighten oscillation escape threshold"))

    # Ping-pong
    n_ping = sum(d.coupling.ping_pong_events > 0 for d in all_diags)
    if n_ping > 0:
        recs.append(("coupling_thresholds",
                     f"{n_ping} case(s) with ping-pong coupling — "
                     "raise ParameterCouplingMap priority threshold or add dampening"))

    # Level-2 coord descent triggering frequently
    n_l2 = sum(d.coord_descent.level_2_triggered > 0 for d in all_diags)
    if n_l2 > len(all_diags) * 0.3:
        recs.append(("coordinate_descent_step_sizes",
                     f"Level-2 expansion triggered in {n_l2} cases — "
                     "reduce step sizes or tighten level-1 convergence criterion"))

    # Rule store contribution top 2
    if contributions and contributions[0][1] == "rule_store":
        recs.append(("rule_store_sensitivity",
                     "Rule store is the primary bottleneck — "
                     "expand BIP corpus for underrepresented compound pairs"))

    if not recs:
        recs.append(("general",
                     "No single dominant failure — consider increasing max_iterations "
                     "for hard-tier cases"))

    for param, text in recs:
        lines.append(f"  [{param}]")
        lines.append(f"    {text}")
        lines.append("")

    return "\n".join(lines)
