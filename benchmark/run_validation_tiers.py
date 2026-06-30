"""
Run the validation-family complexity tiers end-to-end (full pipeline).

Tiers — real-sourced DWSIM/FOSSEE reference cases, binned by CORRECTED material
unit count (replaces the old flat "validation" tier):
    val_3_5      FOS_01 (FOSSEE ORC)
    val_6_9      VAL_09, VAL_01, VAL_07
    val_10_14    VAL_02, VAL_03, VAL_06, VAL_08, VAL_10
    val_15plus   VAL_05, VAL_04

Usage (HPC):
  PYTHONPATH=. USE_LANGGRAPH=1 OLLAMA_BASE_URL=http://localhost:11434/v1 \
      python3.9 benchmark/run_validation_tiers.py \
          --model qwen3:30b-a3b [--tiers val_3_5 val_6_9 ...] [--repeats N]

Writes results/per_run/<case>_full_ccs_<ts>.json per run and prints a per-tier
pass summary at the end (most-recent <repeats> runs per case).
"""
import argparse, glob, json, os
from collections import Counter
from benchmark.runner import BenchmarkRunner
from benchmark.case_schema import load_tier

VAL_TIERS = ["val_3_5", "val_6_9", "val_10_14", "val_15plus"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:30b-a3b")
    ap.add_argument("--tiers", nargs="+", default=VAL_TIERS, choices=VAL_TIERS)
    ap.add_argument("--repeats", type=int, default=1,
                    help="runs per case (use >1 to measure pass-rate / flakiness)")
    args = ap.parse_args()

    runner = BenchmarkRunner(model=args.model)
    order = [(t, c.id) for t in args.tiers for c in load_tier(t)]
    print(f"[run] {len(order)} cases x {args.repeats} repeat(s) over {args.tiers} "
          f"| model={args.model}", flush=True)

    for t, cid in order:
        for r in range(args.repeats):
            print(f"\n##### {t} / {cid}  (run {r + 1}/{args.repeats}) #####", flush=True)
            runner.run_case(cid)

    # ── Per-tier summary from the most-recent <repeats> per-case logs ────────────
    print("\n" + "=" * 76)
    print("VALIDATION-TIER RUN SUMMARY")
    print("=" * 76)
    grand_pass = grand_n = 0
    for t in args.tiers:
        rows, tpass, tn = [], 0, 0
        for c in load_tier(t):
            fs = sorted(glob.glob(f"results/per_run/{c.id}_full_ccs_*.json"),
                        key=os.path.getmtime)[-args.repeats:]
            ds = [json.load(open(f)) for f in fs]
            npass = sum(1 for d in ds if (d.get("outcome") or "") == "PASS")
            rows.append((c.id, npass, len(ds),
                         dict(Counter((d.get("outcome") or "?") for d in ds))))
            tpass += npass
            tn += len(ds)
        grand_pass += tpass
        grand_n += tn
        print(f"\n[{t}]  pass {tpass}/{tn}")
        for cid, npass, n, outs in rows:
            print(f"   {cid:10} {npass}/{n}  {outs}")
    print("\n" + "-" * 76)
    print(f"OVERALL  pass {grand_pass}/{grand_n}")


if __name__ == "__main__":
    main()
