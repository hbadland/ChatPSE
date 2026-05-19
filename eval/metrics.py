"""
Evaluation metrics for v2 pipeline.

Primary metrics:
  pct_valid_ir        — % of cases where Stage 2 exits with a valid FlowsheetGraph
  pct_valid_json      — % of cases where IR → DWSIM JSON passes schema.validate()
  pct_converged       — % of cases where DWSIM execution solved=True
  avg_repair_iters    — mean number of Stage 4 loop iterations across all cases

Outcome distribution:
  PASS | HUMAN | MAX_ITER | BASIS_FAILED | INVALID_IR | INVALID_JSON | PLAN_FAILED
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from collections import Counter


@dataclass
class CaseResult:
    """One benchmark case result."""
    case_id:          str
    outcome:          str       # PipelineResult.outcome
    valid_ir:         bool      # ir_report.valid (True even if outcome != PASS)
    valid_json:       bool      # schema.validate() passed
    converged:        bool      # execution.solved
    repair_iterations: int      # len(iterations)
    llm_calls:        int       # agents/llm.get_call_count() delta
    elapsed_s:        float
    warnings:         list[str] = field(default_factory=list)


@dataclass
class BenchmarkMetrics:
    n_total:             int
    pct_valid_ir:        float
    pct_valid_json:      float
    pct_converged:       float
    avg_repair_iters:    float
    avg_llm_calls:       float
    outcome_counts:      dict  = field(default_factory=dict)
    # Ablation delta vs full system (filled by ablation runner)
    ablation_mode:       str   = "full"
    delta_converged:     Optional[float] = None

    def __str__(self) -> str:
        lines = [
            f"=== Benchmark Metrics [{self.ablation_mode}] ===",
            f"  n_total         : {self.n_total}",
            f"  valid IR        : {self.pct_valid_ir:.1%}",
            f"  valid JSON      : {self.pct_valid_json:.1%}",
            f"  converged       : {self.pct_converged:.1%}",
            f"  avg repair iter : {self.avg_repair_iters:.2f}",
            f"  avg LLM calls   : {self.avg_llm_calls:.1f}",
        ]
        if self.outcome_counts:
            lines.append("  outcomes:")
            for k, v in sorted(self.outcome_counts.items()):
                lines.append(f"    {k:15s}: {v:3d}  ({v/self.n_total:.0%})")
        if self.delta_converged is not None:
            lines.append(
                f"  Δconverged vs full: {self.delta_converged:+.1%}")
        return "\n".join(lines)


def compute_metrics(
    results: list[CaseResult],
    ablation_mode: str = "full",
) -> BenchmarkMetrics:
    n = len(results)
    if n == 0:
        return BenchmarkMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                ablation_mode=ablation_mode)
    return BenchmarkMetrics(
        n_total          = n,
        pct_valid_ir     = sum(r.valid_ir    for r in results) / n,
        pct_valid_json   = sum(r.valid_json  for r in results) / n,
        pct_converged    = sum(r.converged   for r in results) / n,
        avg_repair_iters = sum(r.repair_iterations for r in results) / n,
        avg_llm_calls    = sum(r.llm_calls   for r in results) / n,
        outcome_counts   = dict(Counter(r.outcome for r in results)),
        ablation_mode    = ablation_mode,
    )
