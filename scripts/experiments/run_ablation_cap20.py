"""
Component ablation study — 20-case capability set.

Population (20 cases, all with expert-specified reference flowsheets):
  capability tier (13): P1-P3, F1-F4, S1-S2, C1-C3, M1
  easy tier (3):        EASY_01, EASY_02, EASY_04
  sanity tier (2):      SAN_03, SAN_04
  generalisation (2):   GEN_01, GEN_03

Configurations (4 total):
  full_ccs      — complete system (baseline)
  no_rule_store — NullRetriever + empty FailureRuleStore
  no_physics    — bubble_point_K returns None (thermo estimation disabled)
  no_coupling   — ParameterCouplingMap.get_coupled_boosts returns {}

One execution per case per configuration (no best-of-N, no repeated sampling).
Execution path: OrchestratorV2 (USE_LANGGRAPH unset).
All ablation patches verified on the v2 path before this run.

Usage (HPC):
  PYTHONPATH=. OLLAMA_BASE_URL=http://localhost:11434/v1 \\
      python3.9 scripts/experiments/run_ablation_cap20.py \
          --model qwen3:30b-a3b [--modes full_ccs no_rule_store ...]

Outputs per-run JSON to results/per_run/ and prints a summary table at the end.
"""
import argparse, json, os, sys, time
from datetime import datetime

CASE_IDS = [
    "P1", "P2", "P3",
    "F1", "F2", "F3", "F4",
    "S1", "S2",
    "C1", "C2", "C3",
    "M1",
    "EASY_01", "EASY_02", "EASY_04",
    "SAN_03", "SAN_04",
    "GEN_01", "GEN_03",
]

ALL_MODES = ["full_ccs", "no_rule_store", "no_physics", "no_coupling"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:30b-a3b")
    ap.add_argument("--max-iter", type=int, default=6)
    ap.add_argument("--modes", nargs="+", default=ALL_MODES, choices=ALL_MODES)
    args = ap.parse_args()

    commit = os.popen("git rev-parse HEAD 2>/dev/null").read().strip()[:12] or "unknown"
    run_date = datetime.utcnow().strftime("%Y-%m-%d")

    print(f"[ablation] commit={commit}  date={run_date}  model={args.model}")
    print(f"[ablation] cases={len(CASE_IDS)}  modes={args.modes}")
    print(f"[ablation] 80 total runs (20 cases × 4 modes), single execution each\n")

    from benchmark.runner import BenchmarkRunner

    runner = BenchmarkRunner(model=args.model, max_iterations=args.max_iter)
    all_results = runner.run_targeted_ablation(
        case_ids=CASE_IDS,
        modes=args.modes,
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"ABLATION SUMMARY  commit={commit}  date={run_date}  model={args.model}")
    print("=" * 72)
    print(f"{'Mode':<18} {'N':>4} {'PASS':>6} {'MAX_ITER':>10} {'OTHER':>8} {'Rate':>8}")
    print("-" * 72)

    baseline_rate = None
    for mode in args.modes:
        rs = all_results.get(mode)
        if rs is None:
            print(f"  {mode}: NO RESULTS")
            continue
        n = len(rs.case_metrics)
        passes    = sum(1 for m in rs.case_metrics if m.outcome == "PASS")
        max_iters = sum(1 for m in rs.case_metrics if m.outcome == "MAX_ITER")
        others    = n - passes - max_iters
        rate = passes / n if n else 0.0
        delta = f"({rate - baseline_rate:+.1%})" if baseline_rate is not None else "(baseline)"
        if baseline_rate is None:
            baseline_rate = rate
        print(f"  {mode:<16} {n:>4} {passes:>6} {max_iters:>10} {others:>8} "
              f"  {rate:.1%} {delta}")

    print()
    print("Per-case breakdown:")
    print(f"{'Case':<12} {'tier':<16}", end="")
    for mode in args.modes:
        print(f" {mode[:12]:>12}", end="")
    print()
    print("-" * (28 + 13 * len(args.modes)))

    for cid in CASE_IDS:
        from benchmark.case_schema import load_by_id
        c = load_by_id(cid)
        print(f"  {cid:<10} {c.tier:<16}", end="")
        for mode in args.modes:
            rs = all_results.get(mode)
            if rs is None:
                print(f" {'?':>12}", end="")
                continue
            m = next((m for m in rs.case_metrics if m.case_id == cid), None)
            outcome = m.outcome if m else "MISSING"
            print(f" {outcome:>12}", end="")
        print()

    print()
    print(f"[ablation] DONE  commit={commit}  date={run_date}")


if __name__ == "__main__":
    main()
