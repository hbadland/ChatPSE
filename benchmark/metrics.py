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

    # Physics checks passed
    physics_checks_run:  int  = 0
    physics_checks_passed: int = 0
    physics_check_details: list[dict] = field(default_factory=list)

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

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["explore_exploit_ratio"] = self.explore_exploit_ratio
        d["physics_check_pass_rate"] = self.physics_check_pass_rate
        return d


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
    )


# ── Aggregate metrics ──────────────────────────────────────────────────────────

@dataclass
class AggregateMetrics:
    """Summary statistics over a set of RunMetrics."""
    n_cases:              int
    ablation_mode:        str

    # Core rates
    success_rate:         float = 0.0
    valid_ir_rate:        float = 0.0
    valid_json_rate:      float = 0.0
    physics_pass_rate:    float = 0.0

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
        return "\n".join(lines)


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
        physics_pass_rate   = mean([m.physics_check_pass_rate for m in metrics]),
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
        by_difficulty       = _group_rate("difficulty"),
        by_domain           = _group_rate("domain"),
        by_tier             = _group_rate("tier"),
    )
