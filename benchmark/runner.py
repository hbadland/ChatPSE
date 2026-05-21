"""
BenchmarkRunner — main entry point for the CCS benchmark suite.

Usage:
    from benchmark.runner import BenchmarkRunner
    from benchmark.ablation import CONFIGS

    runner = BenchmarkRunner(model="qwen3:14b")

    # Full suite
    results = runner.run_all()

    # Single tier
    results = runner.run_tier("hard")

    # Ablation study
    all_results = runner.run_ablation(tiers=["easy", "medium", "hard"])

    # Single case
    result = runner.run_case("HARD_01")

BenchmarkRunner returns BenchmarkRunSet, which serialises to JSON
and produces publication-ready tables via .to_markdown() and .to_latex().
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from benchmark.case_schema import (
    BenchmarkCaseSpec, load_all, load_tier, load_by_id, TIERS
)
from benchmark.metrics import (
    RunMetrics, extract_metrics, aggregate, AggregateMetrics
)
from benchmark.logger import RunLog, extract_run_log
from benchmark.physics_eval import run_physics_checks
from benchmark.ablation import AblationConfig, CONFIGS, apply_ablation, make_orchestrator

_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "results"
)


# ── Result containers ──────────────────────────────────────────────────────────

@dataclass
class CaseRunResult:
    case:      BenchmarkCaseSpec
    metrics:   RunMetrics
    run_log:   RunLog
    log_path:  str = ""


@dataclass
class BenchmarkRunSet:
    """Full results from one benchmark run (one ablation mode, one or more tiers)."""
    ablation_mode:  str
    model:          str
    tiers:          list[str]
    timestamp:      str
    case_results:   list[CaseRunResult] = field(default_factory=list)
    aggregate:      Optional[AggregateMetrics] = None

    # Per-tier aggregates
    tier_aggregates: dict[str, AggregateMetrics] = field(default_factory=dict)

    def metrics_list(self) -> list[RunMetrics]:
        return [r.metrics for r in self.case_results]

    def to_dict(self) -> dict:
        return {
            "ablation_mode":   self.ablation_mode,
            "model":           self.model,
            "tiers":           self.tiers,
            "timestamp":       self.timestamp,
            "n_cases":         len(self.case_results),
            "aggregate":       self.aggregate.__dict__ if self.aggregate else {},
            "tier_aggregates": {k: v.__dict__ for k, v in self.tier_aggregates.items()},
            "case_results": [
                {
                    "case_id":   r.case.id,
                    "tier":      r.case.tier,
                    "difficulty": r.case.difficulty,
                    "domain":    r.case.domain,
                    **r.metrics.to_dict(),
                }
                for r in self.case_results
            ],
        }

    def save(self, results_dir: str | None = None) -> str:
        d = results_dir or os.path.join(_RESULTS_DIR, "summaries")
        os.makedirs(d, exist_ok=True)
        tiers_str = "_".join(self.tiers[:3])
        fname = f"{self.ablation_mode}_{tiers_str}_{self.timestamp}.json"
        path  = os.path.join(d, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path

    def to_markdown(self) -> str:
        lines = [
            f"## Benchmark Results — {self.ablation_mode}  ({self.timestamp})\n",
            f"Model: `{self.model}`  |  Tiers: {', '.join(self.tiers)}\n",
        ]
        agg = self.aggregate
        if agg:
            lines += [
                "### Aggregate\n",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Success rate | {agg.success_rate:.1%} |",
                f"| Valid IR | {agg.valid_ir_rate:.1%} |",
                f"| Valid JSON | {agg.valid_json_rate:.1%} |",
                f"| Physics checks pass | {agg.physics_pass_rate:.1%} |",
                f"| Mean iterations | {agg.mean_iterations:.2f} |",
                f"| Mean sim calls | {agg.mean_sim_calls:.2f} |",
                f"| Mean candidates | {agg.mean_candidates:.1f} |",
                f"| Explore/exploit ratio | {agg.mean_explore_ratio:.2f} |",
                f"| Score improved (%) | {agg.pct_score_improved:.1%} |",
                f"| Score oscillated (%) | {agg.pct_score_oscillated:.1%} |",
                f"| BIP injected (%) | {agg.pct_bip_injected:.1%} |",
            ]
            if agg.recovery_rate is not None:
                lines.append(f"| Recovery rate | {agg.recovery_rate:.1%} |")
            lines.append("")

        if self.tier_aggregates:
            lines += ["### By Tier\n",
                      "| Tier | Success | Valid IR | Physics | Mean iters |",
                      "|------|---------|----------|---------|-----------|"]
            for tier, ta in sorted(self.tier_aggregates.items()):
                lines.append(
                    f"| {tier} | {ta.success_rate:.1%} | {ta.valid_ir_rate:.1%} "
                    f"| {ta.physics_pass_rate:.1%} | {ta.mean_iterations:.1f} |"
                )
            lines.append("")

        lines += ["### Per-case Results\n",
                  "| ID | Tier | Difficulty | Domain | Success | Iter | Candidates | Phys ✓ |",
                  "|----|------|------------|--------|---------|------|------------|--------|"]
        for r in self.case_results:
            m = r.metrics
            lines.append(
                f"| {r.case.id} | {r.case.tier} | {r.case.difficulty} "
                f"| {r.case.domain} | {'✓' if m.success else '✗'} "
                f"| {m.n_iterations} | {m.n_candidates_total} "
                f"| {m.physics_checks_passed}/{m.physics_checks_run} |"
            )
        return "\n".join(lines)

    def to_latex(self) -> str:
        agg = self.aggregate
        if not agg:
            return ""
        rows = []
        data = [
            ("Success rate",       f"{agg.success_rate:.1%}"),
            ("Valid IR",           f"{agg.valid_ir_rate:.1%}"),
            ("Physics check pass", f"{agg.physics_pass_rate:.1%}"),
            ("Mean iterations",    f"{agg.mean_iterations:.2f}"),
            ("Mean sim calls",     f"{agg.mean_sim_calls:.2f}"),
            ("Mean candidates",    f"{agg.mean_candidates:.1f}"),
            ("Explore ratio",      f"{agg.mean_explore_ratio:.2f}"),
            ("Score improved",     f"{agg.pct_score_improved:.1%}"),
            ("Score oscillated",   f"{agg.pct_score_oscillated:.1%}"),
            ("BIP injected",       f"{agg.pct_bip_injected:.1%}"),
        ]
        if agg.recovery_rate is not None:
            data.append(("Recovery rate", f"{agg.recovery_rate:.1%}"))
        for label, val in data:
            rows.append(f"  {label} & {val} \\\\")
        return (
            "\\begin{tabular}{lc}\n"
            "\\hline\n"
            "Metric & Value \\\\ \\hline\n"
            + "\n".join(rows)
            + "\n\\hline\n\\end{tabular}"
        )


# ── BenchmarkRunner ────────────────────────────────────────────────────────────

class BenchmarkRunner:
    """
    Runs benchmark cases through OrchestratorV2 with full metrics collection.

    Parameters
    ----------
    model           : LLM model name (default: qwen3:14b via Ollama)
    max_iterations  : Stage 4 repair loop limit per case
    save_logs       : write per-run trajectory JSON to results/per_run/
    verbose         : print per-case progress
    """

    def __init__(
        self,
        model:          str  = "qwen3:14b",
        max_iterations: int  = 6,
        save_logs:      bool = True,
        verbose:        bool = True,
    ) -> None:
        self._model     = model
        self._max_iter  = max_iterations
        self._save_logs = save_logs
        self._verbose   = verbose

    # ── Public API ─────────────────────────────────────────────────────────────

    def run_case(
        self,
        case_id:       str,
        ablation_mode: str = "full_ccs",
    ) -> CaseRunResult:
        """Run a single case by ID."""
        case   = load_by_id(case_id)
        config = CONFIGS.get(ablation_mode, CONFIGS["full_ccs"])
        return self._run_one(case, config)

    def run_tier(
        self,
        tier:          str,
        ablation_mode: str = "full_ccs",
    ) -> BenchmarkRunSet:
        """Run all cases in one tier."""
        cases  = load_tier(tier)
        config = CONFIGS.get(ablation_mode, CONFIGS["full_ccs"])
        return self._run_set(cases, config, tiers=[tier])

    def run_all(
        self,
        tiers:         list[str] | None = None,
        ablation_mode: str = "full_ccs",
    ) -> BenchmarkRunSet:
        """Run all cases across specified tiers (default: all)."""
        selected = tiers or TIERS
        cases    = load_all(tiers=selected)
        config   = CONFIGS.get(ablation_mode, CONFIGS["full_ccs"])
        return self._run_set(cases, config, tiers=selected)

    def run_ablation(
        self,
        tiers:   list[str] | None = None,
        modes:   list[str] | None = None,
        verbose: bool = True,
    ) -> dict[str, BenchmarkRunSet]:
        """
        Run the full ablation study across all modes.

        Returns {ablation_mode: BenchmarkRunSet}.
        """
        selected_modes = modes or list(CONFIGS.keys())
        selected_tiers = tiers or ["easy", "medium", "hard"]
        results: dict[str, BenchmarkRunSet] = {}

        for mode in selected_modes:
            if verbose:
                print(f"\n{'='*60}")
                print(f"  Ablation: {mode}")
                print(f"{'='*60}")
            run_set = self.run_all(tiers=selected_tiers, ablation_mode=mode)
            results[mode] = run_set

            if verbose:
                print(run_set.aggregate)

        return results

    # ── Internal ───────────────────────────────────────────────────────────────

    def _run_one(
        self,
        case:   BenchmarkCaseSpec,
        config: AblationConfig,
    ) -> CaseRunResult:
        from agents.llm import get_call_count, reset_call_count

        orch, _ = make_orchestrator(config, self._model, self._max_iter)
        reset_call_count()

        if self._verbose:
            print(f"  [{case.id}/{case.tier}] {case.name[:50]} …", end=" ", flush=True)

        t0 = time.time()
        try:
            with apply_ablation(config):
                pr = orch.run(case.description)
        except Exception as exc:
            import traceback as _tb
            if self._verbose:
                print(f"\n  [EXCEPTION] {exc}")
                _tb.print_exc()
            pr = _make_failed_result(str(exc))

        llm_calls = get_call_count()

        # Physics checks
        checks = run_physics_checks(case, pr)
        n_checks_run    = len(checks)
        n_checks_passed = sum(1 for c in checks if c.get("passed", False))

        # Attach physics check results to pr so metrics extractor can read them
        pr._physics_checks = checks   # type: ignore[attr-defined]

        # Derive compatible attributes for metrics extractor
        pr.ir_valid   = getattr(pr, "ir_valid",   False) or (
            getattr(pr, "ir_report", None) is not None and
            getattr(getattr(pr, "ir_report", None), "valid", False))
        pr.json_valid = getattr(pr, "json_valid", False) or (
            getattr(pr, "final_flowsheet", None) is not None)
        pr.converged  = getattr(pr, "converged", False) or (
            getattr(pr, "outcome", "") == "PASS")

        # Extract metrics
        m = extract_metrics(pr, case, config.mode, llm_calls)
        m.physics_checks_run    = n_checks_run
        m.physics_checks_passed = n_checks_passed
        m.physics_check_details = checks

        # Extract trajectory log
        run_log  = extract_run_log(pr, case.id, config.mode, self._model)
        log_path = ""
        if self._save_logs:
            log_path = run_log.save(os.path.join(_RESULTS_DIR, "per_run"))

        elapsed = time.time() - t0
        if self._verbose:
            outcome = getattr(pr, "outcome", "?")
            print(f"{outcome}  iter={m.n_iterations}  "
                  f"phys={n_checks_passed}/{n_checks_run}  {elapsed:.1f}s")

        return CaseRunResult(case=case, metrics=m, run_log=run_log, log_path=log_path)

    def _run_set(
        self,
        cases:  list[BenchmarkCaseSpec],
        config: AblationConfig,
        tiers:  list[str],
    ) -> BenchmarkRunSet:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_set = BenchmarkRunSet(
            ablation_mode = config.mode,
            model         = self._model,
            tiers         = tiers,
            timestamp     = timestamp,
        )

        for case in cases:
            result = self._run_one(case, config)
            run_set.case_results.append(result)

        # Aggregates
        all_metrics = run_set.metrics_list()
        run_set.aggregate = aggregate(all_metrics, config.mode)

        for tier in tiers:
            tier_m = [m for m in all_metrics if m.tier == tier]
            if tier_m:
                run_set.tier_aggregates[tier] = aggregate(tier_m, config.mode)

        return run_set


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_failed_result(error_msg: str):
    """Minimal PipelineResult-like object for exception cases."""
    class _FailedResult:
        outcome       = "EXCEPTION"
        ir_valid      = False
        json_valid    = False
        converged     = False
        ir_report     = None
        final_graph   = None
        final_flowsheet = None
        final_execution = None
        iterations    = []
        warnings      = []
        basis_result  = None
        total_time_s  = 0.0

    r = _FailedResult()
    r.warnings = [f"EXCEPTION: {error_msg}"]
    return r


def ablation_table(results: dict[str, BenchmarkRunSet]) -> str:
    """
    Format ablation comparison table as Markdown.
    results: {ablation_mode: BenchmarkRunSet}
    """
    modes   = list(results.keys())
    headers = ["Mode", "Success", "Valid IR", "Physics", "Mean iter", "Candidates",
               "Explore%", "Oscillated%"]
    rows    = [headers, ["---"] * len(headers)]

    for mode, rs in results.items():
        agg = rs.aggregate
        if agg is None:
            continue
        rows.append([
            mode,
            f"{agg.success_rate:.1%}",
            f"{agg.valid_ir_rate:.1%}",
            f"{agg.physics_pass_rate:.1%}",
            f"{agg.mean_iterations:.2f}",
            f"{agg.mean_candidates:.1f}",
            f"{agg.mean_explore_ratio:.1%}",
            f"{agg.pct_score_oscillated:.1%}",
        ])

    col_w = [max(len(r[i]) for r in rows) for i in range(len(headers))]
    lines = []
    for row in rows:
        cells = [cell.ljust(col_w[i]) for i, cell in enumerate(row)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
