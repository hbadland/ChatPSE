"""
Interpretable benchmark reporting.

All output is ASCII-only so it renders correctly in terminals, log files,
and CI artefacts.

Entry points:
  print_report(run_result)        — full formatted report to stdout
  format_report(run_result) -> str — same as string

BenchmarkRunResult carries everything needed:
  - per-split BenchmarkMetrics
  - FailureLog
  - optional RobustnessReport
  - optional per-case CaseResult list
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Optional

from eval.metrics import BenchmarkMetrics, CaseResult
from eval.failure_log import FailureLog, FAILURE_MODES
from eval.dataset import split_summary


# ── BenchmarkRunResult (carrier) ───────────────────────────────────────────────

@dataclass
class BenchmarkRunResult:
    """
    Output of a full benchmark run.  Produced by the updated run_benchmark().
    """
    split:            str                    # "dev" | "holdout" | "stress" | "all"
    ablation:         str                    # e.g. "full"
    case_results:     list[CaseResult]       = field(default_factory=list)

    # Per-split metrics (populated for split="all"; single entry otherwise)
    metrics_all:      Optional[BenchmarkMetrics]  = None
    metrics_dev:      Optional[BenchmarkMetrics]  = None
    metrics_holdout:  Optional[BenchmarkMetrics]  = None
    metrics_stress:   Optional[BenchmarkMetrics]  = None

    failure_log:      Optional[FailureLog]        = None
    robustness:       Optional[object]            = None  # RobustnessReport

    # Generalisation gap (holdout - dev), filled when split="all"
    gap_valid_ir:     Optional[float] = None
    gap_converged:    Optional[float] = None


# ── Width constant ─────────────────────────────────────────────────────────────
_W = 72


def _hr(char: str = "=") -> str:
    return char * _W


def _centre(text: str) -> str:
    return text.center(_W)


def _pct(v: Optional[float], missing: str = "  n/a  ") -> str:
    if v is None:
        return missing
    return f"{v:6.1%}"


def _f2(v: Optional[float], missing: str = "  n/a") -> str:
    if v is None:
        return missing
    return f"{v:5.2f}"


# ── Section renderers ──────────────────────────────────────────────────────────

def _header(run: BenchmarkRunResult) -> str:
    lines = [
        _hr("="),
        _centre(f"BENCHMARK REPORT  |  split={run.split}  ablation={run.ablation}"),
        _hr("="),
        "",
    ]
    ds = split_summary()
    lines.append(
        f"  Dataset  : {ds['total']} cases total  "
        f"(dev={ds['dev']['n']}  holdout={ds['holdout']['n']}  "
        f"stress={ds['stress']['n']}  frozen={ds['frozen']['n']})"
    )
    lines.append(f"  Cases run: {len(run.case_results)}")
    lines.append("")
    return "\n".join(lines)


def _metrics_table(run: BenchmarkRunResult) -> str:
    col_w = 10
    hdr = (f"  {'Metric':<28}"
           f"{'All':>{col_w}}"
           f"{'Dev':>{col_w}}"
           f"{'Holdout':>{col_w}}"
           f"{'Stress':>{col_w}}")

    rows = [
        _hr("-"),
        "  METRICS BY SPLIT",
        _hr("-"),
        hdr,
        "  " + "-" * (_W - 2),
    ]

    def row(label: str, getter) -> str:
        vals = [
            _pct(getter(run.metrics_all)),
            _pct(getter(run.metrics_dev)),
            _pct(getter(run.metrics_holdout)),
            _pct(getter(run.metrics_stress)),
        ]
        return f"  {label:<28}" + "".join(f"{v:>{col_w}}" for v in vals)

    def row_f(label: str, getter) -> str:
        vals = [
            _f2(getter(run.metrics_all)),
            _f2(getter(run.metrics_dev)),
            _f2(getter(run.metrics_holdout)),
            _f2(getter(run.metrics_stress)),
        ]
        return f"  {label:<28}" + "".join(f"{v:>{col_w}}" for v in vals)

    rows.append(row("Valid IR (%)",
                    lambda m: m.pct_valid_ir if m else None))
    rows.append(row("Valid JSON (%)",
                    lambda m: m.pct_valid_json if m else None))
    rows.append(row("Converged (%)",
                    lambda m: m.pct_converged if m else None))
    rows.append(row_f("Avg repair iters",
                      lambda m: m.avg_repair_iters if m else None))
    rows.append(row_f("Avg LLM calls",
                      lambda m: m.avg_llm_calls if m else None))
    rows.append("")

    # Generalisation gap
    if run.gap_valid_ir is not None or run.gap_converged is not None:
        rows.append("  Generalisation gap (holdout - dev):")
        if run.gap_valid_ir is not None:
            rows.append(f"    valid_ir  : {run.gap_valid_ir:+.1%}")
        if run.gap_converged is not None:
            rows.append(f"    converged : {run.gap_converged:+.1%}")
        rows.append("")

    return "\n".join(rows)


def _failure_table(run: BenchmarkRunResult) -> str:
    if run.failure_log is None:
        return ""

    log = run.failure_log
    by_split = log.by_split()
    rows = [
        _hr("-"),
        "  FAILURE MODE DISTRIBUTION  (primary mode per case)",
        _hr("-"),
    ]

    col_w = 10
    hdr = (f"  {'Mode':<22}"
           f"{'Total':>{col_w}}"
           f"{'Dev':>{col_w}}"
           f"{'Holdout':>{col_w}}"
           f"{'Stress':>{col_w}}")
    rows.append(hdr)
    rows.append("  " + "-" * (_W - 2))

    total_counts = log.mode_counts()
    dev_counts     = by_split.get("dev",     {})
    holdout_counts = by_split.get("holdout", {})
    stress_counts  = by_split.get("stress",  {})

    for mode in FAILURE_MODES:
        t = total_counts.get(mode, 0)
        d = dev_counts.get(mode, 0)
        h = holdout_counts.get(mode, 0)
        s = stress_counts.get(mode, 0)
        if t == 0:
            continue
        rows.append(
            f"  {mode:<22}"
            f"{t:>{col_w}d}"
            f"{d:>{col_w}d}"
            f"{h:>{col_w}d}"
            f"{s:>{col_w}d}"
        )

    rows.append("")

    # Pass rates
    rows.append("  Pass rates:")
    for label, split in [("All", None), ("Dev", "dev"),
                          ("Holdout", "holdout"), ("Stress", "stress")]:
        pr = log.pass_rate(split)
        rows.append(f"    {label:<10}: {pr:.1%}")
    rows.append("")

    # Frozen failures
    frozen_fails = log.frozen_failures()
    if frozen_fails:
        rows.append(f"  FROZEN FAILURES [{len(frozen_fails)}]  <-- investigate immediately")
        for r in frozen_fails:
            rows.append(f"    {r.case_id}  [{r.primary_mode}]  outcome={r.outcome}")
    else:
        rows.append("  Frozen failures: none")
    rows.append("")

    # Worst cases
    worst = log.worst_cases(n=5)
    if worst:
        rows.append("  Worst cases (most repair iterations):")
        for r in worst:
            rows.append(f"    {r.case_id:<10} iters={r.repair_iters}  "
                        f"mode={r.primary_mode}  split={r.split}")
    rows.append("")

    return "\n".join(rows)


def _tier_table(run: BenchmarkRunResult) -> str:
    if run.failure_log is None:
        return ""

    by_tier = run.failure_log.by_tier()
    if not by_tier:
        return ""

    rows = [
        _hr("-"),
        "  FAILURE MODE DISTRIBUTION  (by tier)",
        _hr("-"),
    ]

    all_modes = [m for m in FAILURE_MODES if m != "PASS"]
    tier_order = ["easy", "medium", "hard", "underspec",
                  "ambiguous", "adversarial", "edge", "real"]

    col_w = 9
    hdr = f"  {'Tier':<14}" + "".join(f"{m[:col_w-1]:>{col_w}}" for m in all_modes) + f"{'PASS':>{col_w}}"
    rows.append(hdr)
    rows.append("  " + "-" * (_W - 2))

    for tier in tier_order:
        counts = by_tier.get(tier, {})
        if not counts:
            continue
        cols = [f"{counts.get(m, 0):>{col_w}d}" for m in all_modes]
        cols.append(f"{counts.get('PASS', 0):>{col_w}d}")
        rows.append(f"  {tier:<14}" + "".join(cols))

    rows.append("")
    return "\n".join(rows)


def _robustness_section(run: BenchmarkRunResult) -> str:
    if run.robustness is None:
        return ""

    rob = run.robustness
    rows = [
        _hr("-"),
        "  ROBUSTNESS",
        _hr("-"),
    ]

    by_pert = rob.by_perturbation()
    for pert, summaries in by_pert.items():
        rows.append(f"  Perturbation: {pert}")
        for s in summaries:
            flag = ""
            if s.cv >= 0.15:
                flag = "  <-- SENSITIVE"
            elif s.cv >= 0.05:
                flag = "  <-- moderate"
            rows.append(
                f"    {s.metric_name:<28} "
                f"mean={s.mean:.3f}  std={s.std:.3f}  cv={s.cv:.3f}"
                f"  {s.stability_label()}{flag}"
            )
        rows.append("")

    if rob.all_stable():
        rows.append("  Overall: ALL METRICS STABLE (cv < 0.05 for all)")
    else:
        sensitive = [s.metric_name for s in rob.sensitive_metrics()]
        rows.append(f"  Overall: SENSITIVE metrics: {', '.join(sensitive)}")
    rows.append("")
    return "\n".join(rows)


def _overfitting_check(run: BenchmarkRunResult) -> str:
    rows = []
    if run.metrics_dev is None or run.metrics_holdout is None:
        return ""

    rows = [
        _hr("-"),
        "  OVERFITTING CHECK",
        _hr("-"),
    ]

    gap_ir   = run.metrics_holdout.pct_valid_ir   - run.metrics_dev.pct_valid_ir
    gap_json = run.metrics_holdout.pct_valid_json  - run.metrics_dev.pct_valid_json
    gap_conv = run.metrics_holdout.pct_converged   - run.metrics_dev.pct_converged

    def verdict(delta: float) -> str:
        if delta >= -0.05:
            return "OK"
        if delta >= -0.15:
            return "WATCH"
        return "OVERFIT?"

    rows.append(f"  {'Metric':<22} {'Dev':>8} {'Holdout':>8} {'Delta':>8}  Verdict")
    rows.append("  " + "-" * (_W - 2))
    for label, dev_v, hold_v, gap in [
        ("valid_ir",   run.metrics_dev.pct_valid_ir,   run.metrics_holdout.pct_valid_ir,   gap_ir),
        ("valid_json", run.metrics_dev.pct_valid_json,  run.metrics_holdout.pct_valid_json,  gap_json),
        ("converged",  run.metrics_dev.pct_converged,   run.metrics_holdout.pct_converged,   gap_conv),
    ]:
        rows.append(
            f"  {label:<22} {dev_v:>7.1%} {hold_v:>8.1%} {gap:>+8.1%}  {verdict(gap)}"
        )
    rows.append("")
    return "\n".join(rows)


def _outcome_distribution(run: BenchmarkRunResult) -> str:
    if not run.case_results:
        return ""

    from collections import Counter
    counts = Counter(r.outcome for r in run.case_results)
    n      = len(run.case_results)

    rows = [
        _hr("-"),
        "  OUTCOME DISTRIBUTION",
        _hr("-"),
    ]
    for outcome, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "#" * int(cnt / n * 40)
        rows.append(f"  {outcome:<18} {cnt:>4}  ({cnt/n:5.1%})  {bar}")
    rows.append("")
    return "\n".join(rows)


def _footer(run: BenchmarkRunResult) -> str:
    lines = [_hr("="), ""]
    return "\n".join(lines)


# ── Public API ─────────────────────────────────────────────────────────────────

def format_report(run: BenchmarkRunResult) -> str:
    """Return the full benchmark report as a formatted string."""
    sections = [
        _header(run),
        _metrics_table(run),
        _overfitting_check(run),
        _failure_table(run),
        _tier_table(run),
        _robustness_section(run),
        _outcome_distribution(run),
        _footer(run),
    ]
    return "\n".join(s for s in sections if s)


def print_report(run: BenchmarkRunResult) -> None:
    """Print the full report to stdout."""
    print(format_report(run))
