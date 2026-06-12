"""
Top-level benchmark runner script.

Run inside the VSCode Dev Container:

    # Full suite, all tiers, full CCS
    PYTHONPATH=. OLLAMA_BASE_URL=http://host.docker.internal:11434/v1 \\
        python3.9 benchmark_runner.py

    # Single tier
    PYTHONPATH=. ... python3.9 benchmark_runner.py --tiers sanity easy

    # Multi-run (N=5): runs the full suite 5 times and reports mean ± std
    PYTHONPATH=. ... python3.9 benchmark_runner.py --runs 5 --tiers easy medium hard

    # Ablation study (all 5 modes)
    PYTHONPATH=. ... python3.9 benchmark_runner.py --ablation

    # Single case
    PYTHONPATH=. ... python3.9 benchmark_runner.py --case HARD_01

    # Specific ablation mode
    PYTHONPATH=. ... python3.9 benchmark_runner.py --mode greedy --tiers easy medium hard

Results are saved to results/summaries/ (single run) or results/runs/ (--runs N).
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONNET_RUNTIME", "coreclr")  # must precede any pythonnet/clr import

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CCS Benchmark Runner")
    parser.add_argument("--model", default="qwen3:30b-a3b",
                        help="LLM model (default: qwen3:30b-a3b)")
    parser.add_argument("--tiers", nargs="+",
                        choices=["sanity", "easy", "medium", "hard",
                                 "perturbation", "generalisation",
                                 "multi_unit", "missing_bip",
                                 "ambiguous", "adversarial",
                                 "validation"],
                        help="Tiers to run (default: all). The four extended hard-benchmark "
                             "tiers (multi_unit, missing_bip, ambiguous, adversarial) map "
                             "to hard_benchmark_{short}.json files automatically.")
    parser.add_argument("--case-files", nargs="+", metavar="FILE",
                        help="Load cases from one or more JSON files directly, bypassing "
                             "the tier system. Paths may be absolute or relative to the "
                             "repo root. Incompatible with --ablation; takes precedence "
                             "over --tiers when both are given.")
    parser.add_argument("--mode", default="full_ccs",
                        choices=["full_ccs", "no_physics", "no_rule_store",
                                 "greedy", "no_coupling"],
                        help="Ablation mode (default: full_ccs)")
    parser.add_argument("--ablation", action="store_true",
                        help="Run full ablation study (all 5 modes)")
    parser.add_argument("--case", type=str,
                        help="Run a single case by ID")
    parser.add_argument("--no-save-logs", action="store_true",
                        help="Skip per-run trajectory JSON logs")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-case progress output")
    parser.add_argument("--max-iter", type=int, default=6,
                        help="Stage 4 repair loop limit per case")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Run full diagnostic analysis on results and print report")
    parser.add_argument("--diag-dir", default="results/diagnostics",
                        help="Output directory for diagnostic JSON + plot data")
    parser.add_argument("--runs", type=int, default=1, metavar="N",
                        help="Repeat the full benchmark N times and report mean ± std "
                             "(addresses LLM variance; cannot be combined with --ablation). "
                             "Each run is saved to results/runs/; "
                             "a multi-run summary is written to "
                             "results/runs/multi_run_summary_<timestamp>.json.")
    parser.add_argument("--accumulate-rules", action="store_true",
                        help="Do NOT delete results/rule_store.json between runs in "
                             "--runs N mode. Default behaviour (without this flag) is "
                             "to clear the rule store before each run so that runs are "
                             "independent. Use this flag only to measure the effect of "
                             "accumulated cross-run learning on performance.")
    args = parser.parse_args()

    if args.runs > 1 and args.ablation:
        print("ERROR: --runs and --ablation cannot be combined. "
              "Run ablation separately.", file=sys.stderr)
        sys.exit(1)

    if args.case_files and args.ablation:
        print("ERROR: --case-files and --ablation cannot be combined. "
              "Run ablation by tier instead.", file=sys.stderr)
        sys.exit(1)

    if args.case_files and args.tiers:
        print("WARNING: --case-files and --tiers are both set; "
              "--case-files takes precedence.", file=sys.stderr)

    from benchmark.runner import BenchmarkRunner, ablation_table
    from benchmark.case_schema import summary
    from benchmark.diagnostics import DiagnosticEngine
    from benchmark.visualisation import save_plot_data

    # Print dataset summary
    ds = summary()
    print(f"\nDataset: {ds}")

    runner = BenchmarkRunner(
        model          = args.model,
        max_iterations = args.max_iter,
        save_logs      = not args.no_save_logs,
        verbose        = not args.quiet,
    )

    def _maybe_diagnose(results, label: str) -> None:
        if not args.diagnostics:
            return
        engine = DiagnosticEngine()
        report = engine.analyse(results)
        print(report.format())
        diag_path = report.save(args.diag_dir)
        plot_path = save_plot_data(report, args.diag_dir)
        print(f"\nDiagnostics → {diag_path}")
        print(f"Plot data   → {plot_path}")

    if args.case:
        # Single case by ID — searches all tiers including the new extended ones.
        result  = runner.run_case(args.case, ablation_mode=args.mode)
        print(f"\n{result.metrics}")
        if args.diagnostics:
            from benchmark.runner import BenchmarkRunSet
            import time as _t
            rs_single = BenchmarkRunSet(
                ablation_mode=args.mode, model=args.model,
                tiers=[result.case.tier], timestamp=_t.strftime("%Y%m%d_%H%M%S"),
                case_results=[result],
            )
            _maybe_diagnose(rs_single, args.case)
        return

    if args.case_files:
        # Load cases directly from JSON files, bypassing the tier system.
        run_set = runner.run_case_files(args.case_files, ablation_mode=args.mode)
        md = run_set.to_markdown()
        print(f"\n{md}")
        path = run_set.save()
        print(f"\nSaved → {path}")
        _maybe_diagnose(run_set, args.mode)
        return

    if args.ablation:
        # Full ablation study
        tiers = args.tiers or ["easy", "medium", "hard"]
        print(f"\nRunning ablation study on tiers: {tiers}")
        ablation_results = runner.run_ablation(tiers=tiers)
        table = ablation_table(ablation_results)
        print(f"\n{table}")
        for mode, rs in ablation_results.items():
            path = rs.save()
            print(f"  [{mode}] saved → {path}")
        _maybe_diagnose(ablation_results, "ablation")
        return

    # Multi-run (--runs N > 1)
    if args.runs > 1:
        _run_multi(runner, args, clear_rules_between_runs=not args.accumulate_rules)
        return

    # Standard single run
    run_set = runner.run_all(tiers=args.tiers, ablation_mode=args.mode)
    md = run_set.to_markdown()
    print(f"\n{md}")
    path = run_set.save()
    print(f"\nSaved → {path}")
    _maybe_diagnose(run_set, args.mode)


def _run_multi(runner, args, clear_rules_between_runs: bool = True) -> None:
    """
    Run the full benchmark N times and report mean ± std for key metrics.

    Each run's BenchmarkRunSet is saved to results/runs/.
    A combined summary JSON is written to results/runs/multi_run_summary_<ts>.json.

    Parameters
    ----------
    clear_rules_between_runs : bool (default True)
        Delete results/rule_store.json before each run so that every run starts
        from a cold rule store and results are statistically independent.
        Set False (via --accumulate-rules CLI flag) only to measure the effect of
        cross-run learning on performance — results will NOT be independent.
    """
    runs_dir = ROOT / "results" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Path mirrors agents/rule_store.py: RULES_PATH
    rule_store_path = ROOT / "results" / "rule_store.json"

    run_sets  = []
    run_paths = []
    N = args.runs

    print(f"\nMulti-run mode: N={N}  model={args.model}  "
          f"tiers={args.tiers or 'all'}  mode={args.mode}  "
          f"clear_rules={clear_rules_between_runs}")
    if not clear_rules_between_runs:
        print("WARNING: --accumulate-rules set — runs share rule store state and "
              "are NOT statistically independent. Results are not publishable as "
              "variance estimates.\n")

    for n in range(1, N + 1):
        print(f"{'='*60}")
        print(f"  Run {n}/{N}")
        print(f"{'='*60}")

        # Delete rule store file so each run starts from a clean state.
        if clear_rules_between_runs:
            if rule_store_path.exists():
                rule_store_path.unlink()
                print(f"  [rule_store] deleted {rule_store_path}")
            else:
                print(f"  [rule_store] not present — nothing to clear")

        # Clear thermodynamic estimate caches for run independence.
        try:
            from ir.thermo_estimation import clear_cache
            clear_cache()
        except Exception:
            pass

        if getattr(args, "case_files", None):
            run_set = runner.run_case_files(args.case_files, ablation_mode=args.mode)
        else:
            run_set = runner.run_all(tiers=args.tiers, ablation_mode=args.mode)
        path    = run_set.save(results_dir=str(runs_dir))
        run_sets.append(run_set)
        run_paths.append(path)
        print(f"  → {path}")

        if run_set.aggregate:
            a = run_set.aggregate
            print(f"  success={a.success_rate:.1%}  "
                  f"crit_phys={a.critical_physics_pass_rate:.1%}  "
                  f"phys_all={a.physics_pass_rate:.1%}  "
                  f"iter={a.mean_iterations:.2f}")

    stats        = _multi_run_stats(run_sets, cleared=clear_rules_between_runs)
    summary_path = _save_multi_run_summary(stats, run_paths, args, str(runs_dir))
    _print_multi_run_stats(stats, N)
    print(f"\nMulti-run summary → {summary_path}")


def _multi_run_stats(run_sets, cleared: bool = True) -> dict:
    """
    Compute mean ± std for each tracked metric across all runs.

    Parameters
    ----------
    cleared : bool
        Whether results/rule_store.json was deleted before each run.
        Stored in the returned dict as ``rule_store_cleared_between_runs``
        so readers can judge whether variance estimates are independent.
    """

    def _vals(attr: str) -> list[float]:
        return [getattr(rs.aggregate, attr)
                for rs in run_sets
                if rs.aggregate is not None]

    def _stat(vals: list[float]) -> dict:
        if not vals:
            return {"mean": None, "std": None, "n": 0, "values": []}
        mean_ = statistics.mean(vals)
        std_  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return {
            "mean":   round(mean_, 4),
            "std":    round(std_,  4),
            "n":      len(vals),
            "values": [round(v, 4) for v in vals],
        }

    return {
        "n_runs":                           len(run_sets),
        "rule_store_cleared_between_runs":  cleared,
        "success_rate":                     _stat(_vals("success_rate")),
        "critical_physics_pass_rate":       _stat(_vals("critical_physics_pass_rate")),
        "physics_pass_rate":                _stat(_vals("physics_pass_rate")),
        "mean_iterations":                  _stat(_vals("mean_iterations")),
    }


def _print_multi_run_stats(stats: dict, N: int) -> None:
    sep     = "=" * 60
    cleared = stats.get("rule_store_cleared_between_runs", True)
    indep   = "YES (runs are independent)" if cleared else "NO  (runs share rule state — not independent)"
    print(f"\n{sep}")
    print(f"  Multi-run Statistics  (N={N})")
    print(f"  rule_store_cleared_between_runs: {indep}")
    print(sep)
    rows = [
        ("success_rate",                   stats["success_rate"]),
        ("critical_physics_pass_rate",     stats["critical_physics_pass_rate"]),
        ("physics_pass_rate (all checks)", stats["physics_pass_rate"]),
        ("mean_iterations",                stats["mean_iterations"]),
    ]
    for label, s in rows:
        if s["mean"] is None:
            print(f"  {label:<35s}: no data")
            continue
        std_str  = f"± {s['std']:.4f}"
        vals_str = f"  values={s['values']}" if N <= 10 else ""
        print(f"  {label:<35s}: {s['mean']:.4f} {std_str}{vals_str}")
    print(sep)


def _save_multi_run_summary(stats: dict, run_paths: list, args, runs_dir: str) -> str:
    ts   = time.strftime("%Y%m%d_%H%M%S")
    data = {
        "n_runs":        stats["n_runs"],
        "model":         args.model,
        "tiers":         args.tiers or [],
        "ablation_mode": args.mode,
        "timestamp":     ts,
        "run_files":     run_paths,
        "statistics":    stats,
        "caveats": [
            "FailureRuleStore accumulates knowledge across runs in multi-run mode. "
            "Later runs may benefit from rules learned in earlier runs, reducing "
            "variance estimates vs a cold-start scenario. "
            "Use --mode no_rule_store to disable cross-run rule learning.",
            "bubble_point_K estimates in vle_bubble_point_spot_check assume "
            "equimolar feed via Raoult's Law — approximate for non-ideal systems.",
        ],
    }
    path = os.path.join(runs_dir, f"multi_run_summary_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return path


if __name__ == "__main__":
    main()
