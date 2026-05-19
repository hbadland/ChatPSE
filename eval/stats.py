"""
Statistical evaluation and experimental claims (Items 8, 9).

Provides:
  bootstrap_ci()       — bootstrap 95% CI for any scalar metric
  run_trials()         — repeat benchmark N times, aggregate statistics
  TrialSummary         — per-metric mean ± CI across trials
  ExperimentalClaims   — structured quantitative claims for the paper
  compute_claims()     — derive ExperimentalClaims from empirical results
  format_claims()      — format claims as LaTeX or Markdown table

Statistical methodology:
  - All metrics are means over the benchmark case set
  - Trial variance captures model stochasticity (temperature > 0)
  - Bootstrap CI uses B=1000 resamples of (cases × trials) for stability
  - Significance: Welch's t-test for system vs baseline (unpaired, unequal var)
  - Reproducibility: set PYTHONHASHSEED=0 and LLM temperature=0 for
    deterministic runs; report ± 0 CI in that case

Experimental claims format:
  Each claim is: metric, system_value ± CI, baseline_value ± CI,
                 absolute_delta, relative_delta (%), p_value
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Optional

from eval.metrics import CaseResult, BenchmarkMetrics, compute_metrics
from eval.benchmark_cases import BenchmarkCase


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(
    values:     list[float],
    n_bootstrap: int = 1000,
    confidence:  float = 0.95,
    seed:        int = 42,
) -> tuple[float, float, float]:
    """
    Non-parametric bootstrap confidence interval.

    Returns (mean, ci_lower, ci_upper).
    ci_lower and ci_upper are absolute bounds (not half-widths).
    """
    if not values:
        return 0.0, 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0], values[0]

    rng    = random.Random(seed)
    n      = len(values)
    means  = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(values) for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()

    alpha    = 1.0 - confidence
    lo_idx   = int(alpha / 2 * n_bootstrap)
    hi_idx   = int((1 - alpha / 2) * n_bootstrap)
    mean_val = statistics.mean(values)
    return mean_val, means[lo_idx], means[hi_idx]


def welch_t_test(
    a: list[float],
    b: list[float],
) -> float:
    """
    Welch's t-test p-value (two-tailed, unequal variances).
    Returns 1.0 if either sample has < 2 elements.
    """
    if len(a) < 2 or len(b) < 2:
        return 1.0
    mean_a = statistics.mean(a)
    mean_b = statistics.mean(b)
    var_a  = statistics.variance(a)
    var_b  = statistics.variance(b)
    na, nb = len(a), len(b)
    se     = math.sqrt(var_a / na + var_b / nb)
    if se == 0:
        return 0.0 if mean_a != mean_b else 1.0
    t_stat = (mean_a - mean_b) / se
    # Welch-Satterthwaite degrees of freedom
    df_num = (var_a / na + var_b / nb) ** 2
    df_den = (var_a / na) ** 2 / (na - 1) + (var_b / nb) ** 2 / (nb - 1)
    df     = df_num / df_den if df_den > 0 else 1.0
    # Approximate p-value using normal distribution for large df
    p = 2 * (1 - _normal_cdf(abs(t_stat)))
    return p


def _normal_cdf(z: float) -> float:
    """Approximation of Φ(z) — standard normal CDF."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# ── Trial structures ───────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    """Results from one full benchmark run (all cases, one LLM temperature sample)."""
    run_id:     int
    results:    list[CaseResult]
    metrics:    BenchmarkMetrics


@dataclass
class TrialSummary:
    """Aggregated statistics over multiple trial runs for one metric."""
    metric_name: str
    values:      list[float]    # one per trial
    mean:        float = 0.0
    std:         float = 0.0
    ci_lower:    float = 0.0
    ci_upper:    float = 0.0

    def __post_init__(self) -> None:
        if self.values:
            self.mean, self.ci_lower, self.ci_upper = bootstrap_ci(self.values)
            self.std = statistics.stdev(self.values) if len(self.values) > 1 else 0.0

    def __str__(self) -> str:
        return (f"{self.metric_name}: {self.mean:.3f} "
                f"[{self.ci_lower:.3f}, {self.ci_upper:.3f}] "
                f"(std={self.std:.3f}, n={len(self.values)})")


# ── Multi-trial runner ─────────────────────────────────────────────────────────

def run_trials(
    orchestrator,
    cases:      list[BenchmarkCase] | None = None,
    n_trials:   int = 5,
    ablation:   str = "full",
    verbose:    bool = False,
) -> tuple[list[TrialResult], dict[str, TrialSummary]]:
    """
    Run the benchmark n_trials times and compute per-metric statistics.

    Returns (trial_results, summaries) where summaries is
    {metric_name: TrialSummary}.
    """
    from eval.benchmark import run_benchmark, BENCHMARK_CASES
    cases = cases or BENCHMARK_CASES
    trials: list[TrialResult] = []

    for i in range(n_trials):
        if verbose:
            print(f"\n=== Trial {i+1}/{n_trials} ===")
        results, metrics = run_benchmark(
            orchestrator, cases=cases, ablation=ablation, verbose=verbose)
        trials.append(TrialResult(run_id=i, results=results, metrics=metrics))

    summaries = _summarise_trials(trials)
    return trials, summaries


def _summarise_trials(trials: list[TrialResult]) -> dict[str, TrialSummary]:
    keys = ["pct_valid_ir", "pct_valid_json", "pct_converged",
            "avg_repair_iters", "avg_llm_calls"]
    out: dict[str, TrialSummary] = {}
    for key in keys:
        values = [getattr(t.metrics, key) for t in trials]
        out[key] = TrialSummary(metric_name=key, values=values)
    return out


# ── Experimental claims ────────────────────────────────────────────────────────

@dataclass
class Claim:
    """One quantitative claim for the paper."""
    metric:          str
    system_mean:     float
    system_ci:       tuple[float, float]  # (lower, upper)
    baseline_mean:   float
    baseline_ci:     tuple[float, float]
    delta_abs:       float                # system - baseline
    delta_rel:       float                # (system - baseline) / baseline × 100
    p_value:         float
    significant:     bool = False         # p < 0.05

    def __post_init__(self) -> None:
        self.significant = self.p_value < 0.05

    def __str__(self) -> str:
        sig = "✓" if self.significant else "~"
        return (
            f"{self.metric:30s}: system={self.system_mean:.3f} "
            f"[{self.system_ci[0]:.3f},{self.system_ci[1]:.3f}] "
            f"vs baseline={self.baseline_mean:.3f} "
            f"[{self.baseline_ci[0]:.3f},{self.baseline_ci[1]:.3f}] "
            f"Δ={self.delta_abs:+.3f} ({self.delta_rel:+.1f}%) "
            f"p={self.p_value:.3f} {sig}"
        )


@dataclass
class ExperimentalClaims:
    """All quantitative claims extracted from a set of trials."""
    claims:          list[Claim] = field(default_factory=list)
    system_label:    str = "v2-system"
    baseline_label:  str = "baseline"
    n_system_trials: int = 0
    n_baseline_trials: int = 0

    def __str__(self) -> str:
        lines = [
            f"=== Experimental Claims ({self.system_label} vs {self.baseline_label}) ===",
            f"  System trials:   {self.n_system_trials}",
            f"  Baseline trials: {self.n_baseline_trials}",
            "",
        ]
        for c in self.claims:
            lines.append(f"  {c}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        header = ("| Metric | System | Baseline | Δ (abs) | Δ (%) | p | Sig |\n"
                  "|--------|--------|----------|---------|-------|---|-----|\n")
        rows = []
        for c in self.claims:
            sig = "✓" if c.significant else ""
            rows.append(
                f"| {c.metric} "
                f"| {c.system_mean:.3f} [{c.system_ci[0]:.3f},{c.system_ci[1]:.3f}] "
                f"| {c.baseline_mean:.3f} [{c.baseline_ci[0]:.3f},{c.baseline_ci[1]:.3f}] "
                f"| {c.delta_abs:+.3f} "
                f"| {c.delta_rel:+.1f}% "
                f"| {c.p_value:.3f} "
                f"| {sig} |"
            )
        return header + "\n".join(rows)

    def to_latex(self) -> str:
        """Minimal LaTeX tabular for direct inclusion in the paper."""
        lines = [
            r"\begin{tabular}{lcccccc}",
            r"\hline",
            r"Metric & System & Baseline & $\Delta$ & $\Delta\%$ & $p$ & Sig \\ \hline",
        ]
        for c in self.claims:
            sig = r"$\checkmark$" if c.significant else ""
            lines.append(
                f"{c.metric} & "
                f"{c.system_mean:.3f} & "
                f"{c.baseline_mean:.3f} & "
                f"{c.delta_abs:+.3f} & "
                f"{c.delta_rel:+.1f}\\% & "
                f"{c.p_value:.3f} & "
                f"{sig} \\\\"
            )
        lines += [r"\hline", r"\end{tabular}"]
        return "\n".join(lines)


def compute_claims(
    system_trials:   list[TrialResult],
    baseline_trials: list[TrialResult],
    system_label:    str = "v2-system",
    baseline_label:  str = "baseline",
) -> ExperimentalClaims:
    """
    Derive ExperimentalClaims from multiple trial results.

    Each claim compares the distribution of system metric values across
    trials vs baseline metric values across trials.
    """
    ec = ExperimentalClaims(
        system_label    = system_label,
        baseline_label  = baseline_label,
        n_system_trials  = len(system_trials),
        n_baseline_trials = len(baseline_trials),
    )

    metrics_to_compare = [
        ("pct_valid_json",   "Valid JSON (%)"),
        ("pct_converged",    "Convergence rate (%)"),
        ("pct_valid_ir",     "Valid IR (%)"),
        ("avg_repair_iters", "Avg repair iterations"),
        ("avg_llm_calls",    "Avg LLM calls"),
    ]

    for attr, label in metrics_to_compare:
        sys_vals  = [getattr(t.metrics, attr) for t in system_trials]
        base_vals = [getattr(t.metrics, attr) for t in baseline_trials]

        sys_mean,  sys_lo,  sys_hi  = bootstrap_ci(sys_vals)
        base_mean, base_lo, base_hi = bootstrap_ci(base_vals)
        p_val  = welch_t_test(sys_vals, base_vals)
        d_abs  = sys_mean - base_mean
        d_rel  = (d_abs / base_mean * 100) if base_mean != 0 else float("inf")

        ec.claims.append(Claim(
            metric        = label,
            system_mean   = sys_mean,
            system_ci     = (sys_lo, sys_hi),
            baseline_mean = base_mean,
            baseline_ci   = (base_lo, base_hi),
            delta_abs     = d_abs,
            delta_rel     = d_rel,
            p_value       = p_val,
        ))

    return ec


# ── Ablation comparison ────────────────────────────────────────────────────────

def ablation_table(
    ablation_results: dict[str, list[TrialResult]],
    reference_key:    str = "full",
) -> str:
    """
    Format an ablation table as Markdown.
    ablation_results: {ablation_mode: trial_results}
    """
    metrics = ["pct_valid_ir", "pct_valid_json", "pct_converged",
               "avg_repair_iters"]
    header = "| Mode | " + " | ".join(metrics) + " |\n"
    sep    = "|------|" + "|".join(["------"] * len(metrics)) + "|\n"
    rows   = []

    for mode, trials in ablation_results.items():
        summaries = _summarise_trials(trials)
        vals = []
        for m in metrics:
            s = summaries.get(m)
            if s:
                vals.append(f"{s.mean:.3f} ± {s.std:.3f}")
            else:
                vals.append("—")
        rows.append(f"| {mode} | " + " | ".join(vals) + " |")

    return header + sep + "\n".join(rows)
