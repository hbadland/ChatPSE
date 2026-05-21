"""
Top-level benchmark runner script.

Run inside the VSCode Dev Container:

    # Full suite, all tiers, full CCS
    PYTHONPATH=. OLLAMA_BASE_URL=http://host.docker.internal:11434/v1 \\
        python3.9 benchmark_runner.py

    # Single tier
    PYTHONPATH=. ... python3.9 benchmark_runner.py --tiers sanity easy

    # Ablation study (all 5 modes)
    PYTHONPATH=. ... python3.9 benchmark_runner.py --ablation

    # Single case
    PYTHONPATH=. ... python3.9 benchmark_runner.py --case HARD_01

    # Specific ablation mode
    PYTHONPATH=. ... python3.9 benchmark_runner.py --mode greedy --tiers easy medium hard

Results are saved to results/summaries/ (JSON) and printed as Markdown.
"""
from __future__ import annotations

import argparse
import os
import sys
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
                                 "perturbation", "generalisation"],
                        help="Tiers to run (default: all)")
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
    args = parser.parse_args()

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
        # Single case — wrap into a minimal RunSet so DiagnosticEngine can consume it
        result  = runner.run_case(args.case, ablation_mode=args.mode)
        print(f"\n{result.metrics}")
        if args.diagnostics:
            # Build a one-case RunSet
            from benchmark.runner import BenchmarkRunSet
            import time as _t
            rs_single = BenchmarkRunSet(
                ablation_mode=args.mode, model=args.model,
                tiers=[result.case.tier], timestamp=_t.strftime("%Y%m%d_%H%M%S"),
                case_results=[result],
            )
            _maybe_diagnose(rs_single, args.case)
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

    # Standard run
    run_set = runner.run_all(tiers=args.tiers, ablation_mode=args.mode)
    md = run_set.to_markdown()
    print(f"\n{md}")
    path = run_set.save()
    print(f"\nSaved → {path}")
    _maybe_diagnose(run_set, args.mode)


if __name__ == "__main__":
    main()
