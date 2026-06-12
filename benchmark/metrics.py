"""
Metrics collection for the CCS benchmark.

RunMetrics captures every observable from one pipeline execution.
MetricCollector extracts RunMetrics from a PipelineResult.
AggregateMetrics summarises a set of RunMetrics for reporting.

Metric taxonomy (NeurIPS paper sections):
  Core          — convergence, iterations, simulation calls
  Search        — beam width, candidates evaluated, explore/exploit ratio
  Quality       — parameter change magnitude, score trajectory stability
  Robustness    — recovery rate (perturbation tier)
  Generalisation — domain-split performance gap
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Optional


# ── Per-run metrics ────────────────────────────────────────────────────────────

@dataclass
class RunMetrics:
    # Identity
    case_id:             str
    tier:                str
    difficulty:          str
    domain:              str
    coupling_level:      str
    perturbation:        str
    ablation_mode:       str = "full"

    # Core
    success:             bool  = False   # pipeline outcome == PASS
    n_iterations:        int   = 0       # Stage 4 loop iterations
    n_sim_calls:         int   = 0       # executor.run() calls
    n_llm_calls:         int   = 0       # LLM chat calls
    elapsed_s:           float = 0.0
    outcome:             str   = ""      # PASS|HUMAN|MAX_ITER|INVALID_IR|...

    # IR quality
    valid_ir:            bool  = False
    valid_json:          bool  = False
    n_schema_errors:     int   = 0
    n_physics_errors:    int   = 0
    n_constraint_violations_final: int = 0

    # Search behaviour (beam search)
    beam_width_used:     int   = 0       # actual beam width in final repair
    n_candidates_total:  int   = 0       # total candidates evaluated across all iterations
    explore_steps:       int   = 0       # iterations in EXPLORE phase
    exploit_steps:       int   = 0       # iterations in EXPLOIT phase
    n_coupling_boosts:   int   = 0       # times ParameterCouplingMap boosted a target
    n_state_cache_hits:  int   = 0       # StateCache hits (avoided re-validations)

    # Score trajectory
    score_trajectory:    list[float] = field(default_factory=list)  # IR error count per iteration
    score_improved:      bool  = False   # monotonically decreased?
    score_oscillated:    bool  = False   # non-monotone path?
    score_delta_total:   float = 0.0     # initial_errors - final_errors

    # Quality
    param_changes:       list[dict] = field(default_factory=list)   # {tag, param, old, new}
    param_change_magnitude: float = 0.0  # L2 norm of parameter changes
    n_thermo_switches:   int  = 0        # property package switches
    bip_injected:        bool = False

    # Robustness (perturbation tier)
    recovery_success:    bool  = False   # perturbed case successfully repaired
    n_repair_rounds:     int   = 0       # repair iterations before valid

    # CCS-specific: coupling failure recovery (A)
    n_error_recurrences:     int   = 0   # errors that were fixed then reappeared
    error_recurrence_rate:   float = 0.0 # n_recurrences / total_fixes

    # CCS-specific: propagation lag (B)
    steps_to_full_consistency: int = 0   # iters from last change to first zero-error iter

    # CCS-specific: search efficiency normalisation (C)
    error_reduction_per_candidate: float = 0.0  # Δerrors / n_candidates_total

    # CCS-specific: recurrence breakdown (Fix 4)
    error_recurrence_breakdown: dict = field(default_factory=dict)  # oscillation/coupling/propagation counts
    recurrence_type_ratios:     dict = field(default_factory=dict)  # same keys as fractions

    # Margin model snapshot (Fix 1)
    margin_snapshot: dict = field(default_factory=dict)

    # Physics checks passed (all severities)
    physics_checks_run:  int  = 0
    physics_checks_passed: int = 0
    physics_check_details: list[dict] = field(default_factory=list)

    # Ground-truth comparison (validation tier only)
    match_score:      float = 0.0   # 0.0–1.0 from compare_flowsheets()
    validation_pass:  bool  = False  # overall_pass from compare_flowsheets()
    failure_modes:    dict  = field(default_factory=dict)  # structured failure categories
    mape_T_pct:       float = 0.0   # mean |ΔT/T_ref|×100 across matched streams

    # Reference-match scoring (stricter ±5 K / ±5% / ±0.05 thresholds)
    has_reference:         bool  = False  # case had a reference_file and comparison ran
    reference_mape_T:      float = 0.0   # mean |ΔT/T_ref|×100 at ±5 K threshold
    reference_mape_P:      float = 0.0   # mean |ΔP/P_ref|×100 at ±5% threshold
    reference_mape_vf:     float = 0.0   # mean |ΔVF| at ±0.05 threshold
    reference_match_pass:  bool  = False  # all streams within T/P/VF thresholds

    # Physics checks — CRITICAL severity only
    # Only CRITICAL failures count against the critical_physics_pass_rate.
    # WARNING/INFO failures are surfaced in physics_check_details but do not
    # affect this rate, so a pass/fail decision on thermodynamic correctness
    # is not diluted by structural/presence checks.
    critical_physics_checks_run:    int = 0
    critical_physics_checks_passed: int = 0

    @property
    def explore_exploit_ratio(self) -> Optional[float]:
        total = self.explore_steps + self.exploit_steps
        if total == 0:
            return None
        return self.explore_steps / total

    @property
    def physics_check_pass_rate(self) -> float:
        if self.physics_checks_run == 0:
            return 1.0
        return self.physics_checks_passed / self.physics_checks_run

    @property
    def critical_physics_pass_rate(self) -> float:
        if self.critical_physics_checks_run == 0:
            return 1.0
        return self.critical_physics_checks_passed / self.critical_physics_checks_run

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["explore_exploit_ratio"]        = self.explore_exploit_ratio
        d["physics_check_pass_rate"]      = self.physics_check_pass_rate
        d["critical_physics_pass_rate"]   = self.critical_physics_pass_rate
        return d


# ── CCS metric helpers ────────────────────────────────────────────────────────

def _error_sig(e) -> str:
    """Stable string key for an error record (string, dict, or object)."""
    if isinstance(e, str):
        return e
    if isinstance(e, dict):
        return f"{e.get('type', e.get('code', '?'))}:{e.get('tag', e.get('target', ''))}"
    for attr in ("error_type", "type", "code"):
        v = getattr(e, attr, None)
        if v:
            tag = getattr(e, "tag", getattr(e, "target", ""))
            return f"{v}:{tag}"
    return str(e)


def _error_triple(e) -> tuple:
    """Return (error_type, unit_tag, param) for recurrence classification."""
    if isinstance(e, str):
        parts = e.split(":")
        return (parts[0], parts[1] if len(parts) > 1 else "", "")
    if isinstance(e, dict):
        return (
            e.get("type", e.get("code", "?")),
            e.get("tag",   e.get("target", "")),
            e.get("param", e.get("parameter", "")),
        )
    etype = ""
    for attr in ("error_type", "type", "code"):
        v = getattr(e, attr, None)
        if v:
            etype = str(v)
            break
    tag   = str(getattr(e, "tag", getattr(e, "target", getattr(e, "unit_tag", ""))))
    param = str(getattr(e, "param", getattr(e, "parameter", "")))
    return (etype or "?", tag, param)


def _compute_recurrence(iterations) -> tuple:
    """
    Count and classify error recurrences.

    Returns (n_recurrences, recurrence_rate, breakdown, ratios).
      breakdown = {"oscillation": int, "coupling": int, "propagation": int}
      ratios    = same keys as fractions of n_recurrences

    oscillation — exact (type, tag, param) triple fixed and reappears
    coupling    — same unit_tag has new errors after the unit was fully fixed
    """
    triple_sets: list[set] = []
    for rec in iterations:
        triple_sets.append({_error_triple(e) for e in getattr(rec, "errors", [])})

    total_fixes   = 0
    n_recurrences = 0
    breakdown = {"oscillation": 0, "coupling": 0, "propagation": 0}

    for i in range(1, len(triple_sets)):
        prev = triple_sets[i - 1]
        curr = triple_sets[i]
        fixed = prev - curr
        if not fixed:
            continue
        total_fixes += len(fixed)

        fixed_tags  = {t[1] for t in fixed}
        clean_tags  = fixed_tags - {t[1] for t in curr}   # units now fully error-free
        remaining   = set(fixed)

        for j in range(i + 1, len(triple_sets)):
            later = triple_sets[j]

            # Oscillation: exact triple reappears
            osc = remaining & later
            n_recurrences += len(osc)
            breakdown["oscillation"] += len(osc)
            remaining -= osc

            # Coupling: same unit_tag, different error, after unit was fully clean
            if clean_tags:
                coup = {t for t in later if t[1] in clean_tags and t not in prev}
                if coup:
                    n_recurrences += len(coup)
                    breakdown["coupling"] += len(coup)
                    clean_tags -= {t[1] for t in coup}

            if not remaining and not clean_tags:
                break

    rate = n_recurrences / total_fixes if total_fixes > 0 else 0.0
    ratios = (
        {k: round(v / n_recurrences, 3) for k, v in breakdown.items()}
        if n_recurrences > 0
        else {"oscillation": 0.0, "coupling": 0.0, "propagation": 0.0}
    )
    return n_recurrences, rate, breakdown, ratios


def _compute_propagation_lag(iterations, trajectory: list[float]) -> int:
    """
    Iterations from the last iteration that made any parameter change to the
    first subsequent iteration with zero errors.  Returns 0 when no changes
    were made, or when zero errors were never reached after the last change.
    """
    last_change = -1
    for i, rec in enumerate(iterations):
        if getattr(rec, "changes", []):
            last_change = i
    if last_change < 0:
        return 0
    for i in range(last_change + 1, len(trajectory)):
        if trajectory[i] == 0.0:
            return i - last_change
    return 0


def _compute_search_efficiency(delta: float, n_candidates: int) -> float:
    """Error reduction per candidate evaluated.  0 when no candidates recorded."""
    return delta / n_candidates if n_candidates > 0 else 0.0


# ── Extraction from PipelineResult ────────────────────────────────────────────

def extract_metrics(
    pipeline_result,
    case,
    ablation_mode:  str = "full",
    llm_calls_delta: int = 0,
) -> RunMetrics:
    """
    Extract RunMetrics from an OrchestratorV2 PipelineResult and a BenchmarkCaseSpec.

    Reads iteration records, repair memory (if exposed), and validation report.
    Falls back gracefully when fields are absent (v1 pipeline compatibility).
    """
    from benchmark.case_schema import BenchmarkCaseSpec
    pr = pipeline_result

    n_iter   = len(getattr(pr, "iterations", []))
    n_sim    = n_iter  # one executor call per iteration in v2
    outcome  = getattr(pr, "outcome", "")
    success  = outcome == "PASS"

    # IR validation counts
    ir_report = getattr(pr, "ir_report", None)
    n_schema  = 0
    n_physics = 0
    if ir_report is not None:
        issues = getattr(ir_report, "issues", [])
        n_schema  = sum(1 for i in issues
                        if getattr(i, "level", "") in ("SCHEMA", "ERROR"))
        n_physics = sum(1 for i in issues
                        if getattr(i, "level", "") == "PHYSICS")

    # Score trajectory from iteration records
    trajectory: list[float] = []
    all_changes: list[dict] = []
    n_candidates = 0
    explore_steps = 0
    exploit_steps = 0
    n_coupling    = 0
    n_cache_hits  = 0
    n_thermo      = 0

    for rec in getattr(pr, "iterations", []):
        n_errors = len(getattr(rec, "errors", []))
        trajectory.append(float(n_errors))

        changes = getattr(rec, "changes", [])
        for chg in changes:
            if isinstance(chg, str):
                if "THERMO_SWITCH" in chg or "PACKAGE" in chg.upper():
                    n_thermo += 1
                if "EXPLORE" in chg.upper():
                    explore_steps += 1
                elif "EXPLOIT" in chg.upper():
                    exploit_steps += 1
                if "COUPLING_BOOST" in chg.upper():
                    n_coupling += 1
                if "CACHE_HIT" in chg.upper():
                    n_cache_hits += 1
                if ":=" in chg or "→" in chg or "->" in chg:
                    all_changes.append({"change": chg})
                if "CANDIDATE" in chg.upper():
                    try:
                        n_candidates += int(chg.split("CANDIDATE")[0].strip().split()[-1])
                    except Exception:
                        n_candidates += 1
            elif isinstance(chg, dict):
                all_changes.append(chg)

    # Score properties
    improved   = all(trajectory[i] >= trajectory[i+1] for i in range(len(trajectory)-1))
    oscillated = not improved and len(trajectory) > 1
    delta      = (trajectory[0] - trajectory[-1]) if trajectory else 0.0

    # CCS-specific metrics
    iters = getattr(pr, "iterations", [])
    n_recurrences, recurrence_rate, recurrence_breakdown, recurrence_ratios = _compute_recurrence(iters)
    steps_consistency = _compute_propagation_lag(iters, trajectory)
    err_per_candidate = _compute_search_efficiency(delta, n_candidates)

    # Margin snapshot from instrumented pipeline
    margin_snapshot = getattr(pr, "margin_snapshot", {})

    # Beam width from repair configuration
    beam_width = 0
    repair_agent = getattr(pr, "_repair_agent", None)
    if repair_agent:
        beam_width = getattr(repair_agent, "_beam_width", 0)

    # BIP injection flag
    bip_injected = False
    graph = getattr(pr, "final_graph", None)
    if graph is not None:
        bips = getattr(graph, "binary_parameters", [])
        bip_injected = len(bips) > 0

    # Parameter change magnitude (sum of absolute changes if parseable)
    change_mag = float(len(all_changes))  # proxy when exact values not parsed

    # Final constraint violations
    n_violations = 0
    if ir_report is not None:
        errors = getattr(ir_report, "errors", None)
        if callable(errors):
            n_violations = len(errors())
        elif errors:
            n_violations = len(errors)

    return RunMetrics(
        case_id             = case.id,
        tier                = case.tier,
        difficulty          = case.difficulty,
        domain              = case.domain,
        coupling_level      = case.coupling_level,
        perturbation        = case.perturbation,
        ablation_mode       = ablation_mode,
        success             = success,
        n_iterations        = n_iter,
        n_sim_calls         = n_sim,
        n_llm_calls         = llm_calls_delta,
        elapsed_s           = getattr(pr, "total_time_s", 0.0),
        outcome             = outcome,
        valid_ir            = getattr(pr, "ir_valid",   False),
        valid_json          = getattr(pr, "json_valid", False),
        n_schema_errors     = n_schema,
        n_physics_errors    = n_physics,
        n_constraint_violations_final = n_violations,
        beam_width_used     = beam_width,
        n_candidates_total  = n_candidates,
        explore_steps       = explore_steps,
        exploit_steps       = exploit_steps,
        n_coupling_boosts   = n_coupling,
        n_state_cache_hits  = n_cache_hits,
        score_trajectory    = trajectory,
        score_improved      = improved,
        score_oscillated    = oscillated,
        score_delta_total   = delta,
        param_changes       = all_changes[:20],
        param_change_magnitude = change_mag,
        n_thermo_switches   = n_thermo,
        bip_injected        = bip_injected,
        recovery_success    = success and case.perturbation not in ("none",),
        n_repair_rounds     = n_iter,
        n_error_recurrences              = n_recurrences,
        error_recurrence_rate            = recurrence_rate,
        steps_to_full_consistency        = steps_consistency,
        error_reduction_per_candidate    = err_per_candidate,
        error_recurrence_breakdown       = recurrence_breakdown,
        recurrence_type_ratios           = recurrence_ratios,
        margin_snapshot                  = margin_snapshot,
    )


# ── Aggregate metrics ──────────────────────────────────────────────────────────

@dataclass
class AggregateMetrics:
    """Summary statistics over a set of RunMetrics."""
    n_cases:              int
    ablation_mode:        str

    # Core rates
    success_rate:                float = 0.0
    valid_ir_rate:               float = 0.0
    valid_json_rate:             float = 0.0
    physics_pass_rate:           float = 0.0
    critical_physics_pass_rate:  float = 0.0

    # Core averages
    mean_iterations:      float = 0.0
    mean_sim_calls:       float = 0.0
    mean_llm_calls:       float = 0.0
    mean_elapsed_s:       float = 0.0

    # Search
    mean_candidates:      float = 0.0
    mean_explore_ratio:   float = 0.0
    mean_cache_hits:      float = 0.0
    pct_score_improved:   float = 0.0
    pct_score_oscillated: float = 0.0
    mean_score_delta:     float = 0.0

    # Quality
    mean_param_changes:   float = 0.0
    mean_thermo_switches: float = 0.0
    pct_bip_injected:     float = 0.0

    # Robustness (perturbation tier only)
    recovery_rate:        Optional[float] = None

    # Validation tier — ground-truth comparison (archive tolerances)
    mean_match_score:     float = 0.0   # mean over cases that had a reference
    pct_validation_pass:  float = 0.0   # fraction of validation cases that passed
    mean_mape_T_pct:      float = 0.0   # mean temperature MAPE across compared cases

    # Reference-match scoring (strict ±5 K / ±5% / ±0.05 thresholds)
    ref_match_rate:   float = 0.0   # fraction of reference cases passing all T+P checks
    mean_ref_mape_T:  float = 0.0   # mean temperature MAPE at strict threshold
    mean_ref_mape_P:  float = 0.0   # mean pressure MAPE at strict threshold
    mean_ref_mape_vf: float = 0.0   # mean vapour-fraction MAE at strict threshold

    # CCS-specific
    mean_recurrence_rate:               float = 0.0
    mean_steps_to_consistency:          float = 0.0
    mean_error_reduction_per_candidate: float = 0.0
    mean_oscillation_recurrence_rate:   float = 0.0
    mean_coupling_recurrence_rate:      float = 0.0

    # Breakdown by sub-group
    by_difficulty:        dict = field(default_factory=dict)
    by_domain:            dict = field(default_factory=dict)
    by_tier:              dict = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"=== AggregateMetrics [{self.ablation_mode}]  n={self.n_cases} ===",
            f"  success_rate    : {self.success_rate:.1%}",
            f"  valid_ir_rate   : {self.valid_ir_rate:.1%}",
            f"  valid_json_rate : {self.valid_json_rate:.1%}",
            f"  physics_pass    : {self.physics_pass_rate:.1%}",
            f"  crit_phys_pass  : {self.critical_physics_pass_rate:.1%}",
            f"  mean_iterations : {self.mean_iterations:.2f}",
            f"  mean_sim_calls  : {self.mean_sim_calls:.2f}",
            f"  mean_candidates : {self.mean_candidates:.1f}",
            f"  explore_ratio   : {self.mean_explore_ratio:.2f}",
            f"  cache_hits      : {self.mean_cache_hits:.1f}",
            f"  score_improved  : {self.pct_score_improved:.1%}",
            f"  score_oscillated: {self.pct_score_oscillated:.1%}",
            f"  bip_injected    : {self.pct_bip_injected:.1%}",
        ]
        if self.recovery_rate is not None:
            lines.append(f"  recovery_rate   : {self.recovery_rate:.1%}")
        if self.mean_match_score > 0.0 or self.pct_validation_pass > 0.0:
            lines += [
                f"  mean_match_score: {self.mean_match_score:.1%}",
                f"  validation_pass : {self.pct_validation_pass:.1%}",
                f"  mean_mape_T_pct : {self.mean_mape_T_pct:.2f}%",
            ]
        if self.ref_match_rate > 0.0 or self.mean_ref_mape_T > 0.0:
            lines += [
                f"  ref_match_rate  : {self.ref_match_rate:.1%}",
                f"  ref_mape_T      : {self.mean_ref_mape_T:.2f}%",
                f"  ref_mape_P      : {self.mean_ref_mape_P:.2f}%",
                f"  ref_mape_vf     : {self.mean_ref_mape_vf:.4f}",
            ]
        lines += [
            f"  recurrence_rate : {self.mean_recurrence_rate:.3f}",
            f"  osc_recur_rate  : {self.mean_oscillation_recurrence_rate:.3f}",
            f"  coup_recur_rate : {self.mean_coupling_recurrence_rate:.3f}",
            f"  steps_to_consist: {self.mean_steps_to_consistency:.2f}",
            f"  err_per_cand    : {self.mean_error_reduction_per_candidate:.3f}",
        ]
        return "\n".join(lines)


def _ref_match_aggregate(metrics: list[RunMetrics]) -> dict:
    """
    Compute reference-match aggregate fields from a list of RunMetrics.

    Only cases where has_reference=True are included in the averages, so
    mixed batches (sanity + validation) don't dilute the rates with zeros.
    """
    ref_cases = [m for m in metrics if m.has_reference]
    if not ref_cases:
        return {
            "ref_match_rate":   0.0,
            "mean_ref_mape_T":  0.0,
            "mean_ref_mape_P":  0.0,
            "mean_ref_mape_vf": 0.0,
        }

    def _mean(vals):
        return statistics.mean(vals) if vals else 0.0

    t_vals  = [m.reference_mape_T  for m in ref_cases if m.reference_mape_T  > 0.0]
    p_vals  = [m.reference_mape_P  for m in ref_cases if m.reference_mape_P  > 0.0]
    vf_vals = [m.reference_mape_vf for m in ref_cases]

    return {
        "ref_match_rate":   sum(m.reference_match_pass for m in ref_cases) / len(ref_cases),
        "mean_ref_mape_T":  round(_mean(t_vals),  2),
        "mean_ref_mape_P":  round(_mean(p_vals),  2),
        "mean_ref_mape_vf": round(_mean(vf_vals), 4),
    }


def aggregate(
    metrics: list[RunMetrics],
    ablation_mode: str = "full",
) -> AggregateMetrics:
    """Aggregate a list of RunMetrics into summary statistics."""
    n = len(metrics)
    if n == 0:
        return AggregateMetrics(n_cases=0, ablation_mode=ablation_mode)

    def mean(vals):
        return statistics.mean(vals) if vals else 0.0

    def rate(vals):
        return sum(vals) / len(vals) if vals else 0.0

    pert_metrics = [m for m in metrics if m.perturbation != "none"]
    explore_ratios = [r for m in metrics
                      if (r := m.explore_exploit_ratio) is not None]

    # Per-group breakdown
    def _group_rate(key):
        groups: dict[str, list[float]] = {}
        for m in metrics:
            g = getattr(m, key, "unknown")
            groups.setdefault(g, []).append(float(m.success))
        return {k: mean(v) for k, v in groups.items()}

    return AggregateMetrics(
        n_cases             = n,
        ablation_mode       = ablation_mode,
        success_rate        = rate([m.success for m in metrics]),
        valid_ir_rate       = rate([m.valid_ir for m in metrics]),
        valid_json_rate     = rate([m.valid_json for m in metrics]),
        physics_pass_rate          = mean([m.physics_check_pass_rate for m in metrics]),
        critical_physics_pass_rate = mean([m.critical_physics_pass_rate for m in metrics]),
        mean_iterations     = mean([m.n_iterations for m in metrics]),
        mean_sim_calls      = mean([m.n_sim_calls for m in metrics]),
        mean_llm_calls      = mean([m.n_llm_calls for m in metrics]),
        mean_elapsed_s      = mean([m.elapsed_s for m in metrics]),
        mean_candidates     = mean([m.n_candidates_total for m in metrics]),
        mean_explore_ratio  = mean(explore_ratios) if explore_ratios else 0.0,
        mean_cache_hits     = mean([m.n_state_cache_hits for m in metrics]),
        pct_score_improved  = rate([m.score_improved for m in metrics]),
        pct_score_oscillated = rate([m.score_oscillated for m in metrics]),
        mean_score_delta    = mean([m.score_delta_total for m in metrics]),
        mean_param_changes  = mean([m.param_change_magnitude for m in metrics]),
        mean_thermo_switches = mean([m.n_thermo_switches for m in metrics]),
        pct_bip_injected    = rate([m.bip_injected for m in metrics]),
        recovery_rate       = rate([m.recovery_success for m in pert_metrics])
                              if pert_metrics else None,
        mean_match_score    = mean([m.match_score for m in metrics
                                    if m.match_score > 0.0]) if any(
                                    m.match_score > 0.0 for m in metrics) else 0.0,
        pct_validation_pass = rate([m.validation_pass for m in metrics
                                    if m.match_score > 0.0]) if any(
                                    m.match_score > 0.0 for m in metrics) else 0.0,
        mean_mape_T_pct     = mean([m.mape_T_pct for m in metrics
                                    if m.mape_T_pct > 0.0]) if any(
                                    m.mape_T_pct > 0.0 for m in metrics) else 0.0,
        **_ref_match_aggregate(metrics),
        mean_recurrence_rate               = mean([m.error_recurrence_rate for m in metrics]),
        mean_steps_to_consistency          = mean([m.steps_to_full_consistency for m in metrics]),
        mean_error_reduction_per_candidate = mean([m.error_reduction_per_candidate for m in metrics]),
        mean_oscillation_recurrence_rate   = mean([m.recurrence_type_ratios.get("oscillation", 0.0) for m in metrics]),
        mean_coupling_recurrence_rate      = mean([m.recurrence_type_ratios.get("coupling", 0.0) for m in metrics]),
        by_difficulty       = _group_rate("difficulty"),
        by_domain           = _group_rate("domain"),
        by_tier             = _group_rate("tier"),
    )
