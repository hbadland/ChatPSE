"""
Aggregate results/per_run/ (HPC) — inventory + two diagnostic audits.

Runs ON HPC, reads results/per_run/ directly (no syncing). Pure stdlib + the
benchmark case specs; NO model calls.

It is deliberately staged so you VET the dataset before any rate is computed:

  1. INVENTORY (always printed): per-case run counts, date ranges, per-model
     breakdown, the full observed `outcome` vocabulary, ablation modes, and any
     unreadable/old-schema files.  This tells you the shape of the data.

  2. COHERENT SET (only when --since is given): the most-recent run per case on
     the target model (default qwen3:30b-a3b) with timestamp >= --since (the
     code-era cutoff).  Reports coverage: cases with a valid recent-target run
     vs cases that only have old / wrong-model data.  NEVER mixes models or eras.

  3. AUDITS over the coherent set only:
       (a) failure-stage census  — terminal stage per case
       (b) complexity curve      — rates binned by CORRECTED unit count

Usage:
  # Step 1 — see the dataset first (pick a cutoff from the printed date ranges):
  PYTHONPATH=. python3.9 benchmark/aggregate_per_run.py

  # Step 2 — compute the coherent-set audits with your chosen cutoff:
  PYTHONPATH=. python3.9 benchmark/aggregate_per_run.py \
      --since 20260615 --model qwen3:30b-a3b --out results/aggregate_summary.json

Notes / honesty:
  * RunLog persists NO explicit `converged` or `reference_match` flag.  So:
      - "convergence" is PROXIED (reached-DWSIM proxy below); labelled as such.
      - reference-matching failures are NOT separable from this schema -> flagged.
  * Code "era" can only be proxied by run DATE (no git sha in the JSON).
  * CORRECTED unit counts exist only for cases we manually established; every
    other case falls back to its reference/spec count and is flagged RAW.
"""
from __future__ import annotations
import argparse, glob, json, os
from collections import Counter, defaultdict

# ── Manually-established CORRECTED unit counts (override raw reference counts) ──
# Only add a case here once its true count is verified against the source flowsheet.
# VAL_01: reference lists 16 but 8 are phantom energy/duty artifacts -> real = 8.
CORRECTED_UNIT_COUNTS = {
    "VAL_01": 8,
}

# Complexity bins (lo, hi inclusive, label).  1-2 added to cover easy/sanity cases.
BINS = [(1, 2, "1-2"), (3, 5, "3-5"), (6, 9, "6-9"), (10, 14, "10-14"), (15, 10**9, "15+")]


# ── Loading ────────────────────────────────────────────────────────────────────

def load_runs(per_run_dir):
    """Read every per-run JSON defensively. Returns (runs, n_unreadable)."""
    runs, bad = [], 0
    for path in glob.glob(os.path.join(per_run_dir, "*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            d["_path"] = os.path.basename(path)
            runs.append(d)
        except Exception:
            bad += 1
    return runs, bad


def norm_model(rec):
    """Normalise the model tag. Falls back to filename, then 'unknown'."""
    m = (rec.get("model") or "").lower()
    if not m:
        m = rec.get("_path", "").lower()
    if "30b" in m:
        return "30b"
    if "14b" in m:
        return "14b"
    return "unknown"


def ts_of(rec):
    """Timestamp 'YYYYMMDD_HHMMSS' -> comparable string 'YYYYMMDDHHMMSS' (or '')."""
    t = str(rec.get("timestamp") or "")
    digits = t.replace("_", "")
    return digits if digits.isdigit() else ""


def datestr(rec):
    t = str(rec.get("timestamp") or "")
    return t.split("_")[0] if t else "????????"


# ── Unit-count resolution (corrected -> reference/spec -> None) ──────────────────

def build_unit_count_map():
    """case_id -> (count, source) where source in {corrected, reference_file, spec, unknown}."""
    out = {}
    try:
        import benchmark.case_schema as C
        for tier in C.TIERS:
            try:
                cases = C.load_tier(tier)
            except Exception:
                continue
            for c in cases:
                cid = c.id
                cnt, src = None, "unknown"
                rf = getattr(c, "reference_file", None)
                if rf and os.path.exists(rf):
                    try:
                        cnt = len(json.load(open(rf)).get("units", []))
                        src = "reference_file"
                    except Exception:
                        pass
                if cnt is None:
                    rs = getattr(c, "reference_structure", None)
                    if rs is not None and getattr(rs, "n_units", None):
                        cnt, src = rs.n_units, "spec"
                if cid in CORRECTED_UNIT_COUNTS:
                    cnt, src = CORRECTED_UNIT_COUNTS[cid], "corrected"
                out[cid] = (cnt, src)
    except Exception as e:
        print(f"[WARN] could not import case specs ({e}); unit counts limited to "
              "validation reference files + corrections.")
    # Ensure manual corrections are present even if specs failed to load.
    for cid, n in CORRECTED_UNIT_COUNTS.items():
        out[cid] = (n, "corrected")
    return out


def bin_label(n):
    if n is None:
        return "unknown"
    for lo, hi, lab in BINS:
        if lo <= n <= hi:
            return lab
    return "unknown"


# ── Failure-stage classification (from persisted fields only) ────────────────────
# Maps to the requested taxonomy: extraction / ir_build / thermo / convergence /
# repair_exhaustion / reference_match, plus pass and exception (ambiguous).

def underlying_error(rec):
    """Scan iteration constraint_violations + ir issue summaries for an error class."""
    blob = []
    for it in (rec.get("iterations") or []):
        blob += [str(x) for x in (it.get("constraint_violations") or [])]
    irr = rec.get("ir_report_json") or {}
    blob += [str(x) for x in (irr.get("issue_summaries") or [])]
    text = " ".join(blob).lower()
    if "dwsim convergence" in text or "did not converge" in text:
        return "dwsim_convergence"
    if "mass_balance" in text or "mass balance" in text:
        return "mass_balance"
    if "param" in text and "missing" in text:
        return "param_missing"
    if "phase" in text:
        return "phase"
    return None


def classify(rec, corrected_count):
    """Return (terminal_stage, detail, flags[])."""
    outcome = (rec.get("outcome") or "").upper()
    fg      = rec.get("final_graph_summary")
    irr     = rec.get("ir_report_json") or {}
    iters   = rec.get("iterations") or []
    warns   = " ".join(rec.get("warnings") or [])
    sc      = rec.get("score_curve") or []
    flags   = []

    # Under-capture is a SECONDARY signal (not a terminal stage on its own).
    n_units = (fg or {}).get("n_units")
    if n_units is not None and corrected_count:
        if n_units < corrected_count:
            flags.append(f"under_capture({n_units}/{corrected_count})")
        elif n_units > corrected_count:
            flags.append(f"over_capture({n_units}/{corrected_count})")

    if outcome == "PASS":
        return ("pass", None, flags)

    if outcome == "EXCEPTION":
        w = warns.lower()
        if "langgraph is not installed" in w:
            sub = "env_error(langgraph_missing)"
        elif "recursion limit" in w or "graph_recursion_limit" in w:
            sub = "recursion_limit"
        elif "timeout" in w or "wall-clock" in w:
            sub = "timeout"
        else:
            sub = "other"
        flags.append("ambiguous_exception")
        return ("exception", sub, flags)

    if outcome in ("BASIS_FAILED",):
        return ("extraction", "basis_failed", flags)
    if outcome in ("PLAN_FAILED", "EMPTY_RESPONSE"):
        return ("extraction", "plan_failed/no_topology", flags)

    if outcome in ("INVALID_IR", "INVALID_TOPOLOGY"):
        return ("ir_build", outcome.lower(), flags)
    if fg is None:
        return ("ir_build", "no_graph_built", flags)
    if irr.get("valid") is False:
        return ("ir_build", "ir_invalid", flags)

    if "THERMO" in outcome or "PACKAGE" in outcome:
        return ("thermo", outcome.lower(), flags)

    if outcome in ("MAX_ITER", "HUMAN", "INFEASIBLE"):
        ue = underlying_error(rec)
        if ue:
            flags.append(f"underlying={ue}")
        # If the underlying error is convergence, the run reached DWSIM and the
        # repair loop could not clear it -> repair-exhaustion is the terminal
        # stage, convergence is the cause (multi-stage).
        if ue == "dwsim_convergence":
            flags.append("multi_stage")
        detail = f"{outcome.lower()};persistent_score={sc[-1] if sc else '?'}"
        return ("repair_exhaustion", detail, flags)

    # Unknown / unmapped outcome string -> surface verbatim, do not guess.
    flags.append("unmapped_outcome")
    return ("unclassified", outcome.lower() or "blank", flags)


def reached_dwsim_proxy(rec):
    """Proxy: a graph was built AND the run entered the execute/repair loop."""
    fg = rec.get("final_graph_summary")
    if fg is None:
        return False
    return bool(rec.get("iterations")) or bool(rec.get("score_curve")) \
        or (rec.get("outcome", "").upper() == "PASS")


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-run-dir", default="results/per_run")
    ap.add_argument("--model", default="qwen3:30b-a3b",
                    help="target model for the coherent set (matched by 30b/14b tag)")
    ap.add_argument("--since", default=None,
                    help="code-era cutoff YYYYMMDD; required to compute the audits. "
                         "Pick it from the per-model date ranges in the inventory.")
    ap.add_argument("--out", default=None, help="optional JSON dump path")
    args = ap.parse_args()

    runs, n_bad = load_runs(args.per_run_dir)
    unit_map = build_unit_count_map()
    target_tag = "30b" if "30b" in args.model.lower() else \
                 "14b" if "14b" in args.model.lower() else "unknown"

    # ── 1. INVENTORY ────────────────────────────────────────────────────────────
    print("=" * 92)
    print(f"INVENTORY — {args.per_run_dir}")
    print("=" * 92)
    print(f"total readable runs : {len(runs)}")
    print(f"unreadable/corrupt  : {n_bad}")
    print(f"distinct outcomes   : {dict(Counter((r.get('outcome') or '?') for r in runs))}")
    print(f"distinct models     : {dict(Counter(norm_model(r) for r in runs))}")
    print(f"ablation modes      : {dict(Counter((r.get('ablation_mode') or '?') for r in runs))}")
    all_ts = sorted(datestr(r) for r in runs if datestr(r) != "????????")
    print(f"overall date range  : {all_ts[0] if all_ts else '?'} .. {all_ts[-1] if all_ts else '?'}")

    by_case = defaultdict(list)
    for r in runs:
        by_case[r.get("case_id") or "?"].append(r)

    print("\nper-case  (runs | date range | model breakdown)")
    print("-" * 92)
    for cid in sorted(by_case):
        rs = by_case[cid]
        ds = sorted(datestr(r) for r in rs if datestr(r) != "????????")
        mb = dict(Counter(norm_model(r) for r in rs))
        cnt, src = unit_map.get(cid, (None, "unknown"))
        print(f"  {cid:14} {len(rs):4d} | {ds[0] if ds else '?'}..{ds[-1] if ds else '?'} "
              f"| {mb} | units={cnt}({src})")

    # ── 2. COHERENT SET ──────────────────────────────────────────────────────────
    if not args.since:
        print("\n" + "=" * 92)
        print("NO --since GIVEN -> stopping after inventory (audits not computed).")
        print("Pick a cutoff from the per-model date ranges above (the date the "
              "reactor/robustness fixes landed), then re-run with:")
        print(f"  --since YYYYMMDD --model {args.model}")
        print("=" * 92)
        if args.out:
            json.dump({"inventory_only": True,
                       "per_case_runs": {c: len(v) for c, v in by_case.items()}},
                      open(args.out, "w"), indent=2, default=str)
        return

    coherent, only_old = {}, []
    for cid, rs in by_case.items():
        elig = [r for r in rs
                if norm_model(r) == target_tag and ts_of(r) >= args.since.replace("_", "")]
        if elig:
            coherent[cid] = max(elig, key=ts_of)   # most recent
        else:
            only_old.append(cid)

    print("\n" + "=" * 92)
    print(f"COHERENT SET — most recent run per case on model~{target_tag}, "
          f"timestamp >= {args.since}")
    print("=" * 92)
    print(f"cases with valid recent-{target_tag} run : {len(coherent)}")
    print(f"cases WITHOUT (only old/14b/other)       : {len(only_old)}"
          + (f" -> {sorted(only_old)}" if only_old else ""))

    # ── 3a. FAILURE-STAGE CENSUS ─────────────────────────────────────────────────
    print("\n" + "-" * 92)
    print("AUDIT (a) — FAILURE-STAGE CENSUS  (coherent set only)")
    print(f"{'case':14}{'outcome':12}{'terminal_stage':20}{'detail':28}flags")
    print("-" * 92)
    census, stage_counter = [], Counter()
    for cid in sorted(coherent):
        r = coherent[cid]
        cnt, _src = unit_map.get(cid, (None, "unknown"))
        stage, detail, flags = classify(r, cnt)
        stage_counter[stage] += 1
        census.append({"case": cid, "outcome": r.get("outcome"),
                       "terminal_stage": stage, "detail": detail, "flags": flags,
                       "model": r.get("model"), "timestamp": r.get("timestamp")})
        print(f"  {cid:12}{str(r.get('outcome')):12}{stage:20}"
              f"{str(detail)[:26]:28}{','.join(flags)}")
    print("-" * 92)
    print(f"terminal-stage distribution: {dict(stage_counter)}")
    print("NOTE: 'reference_match' cannot be detected from the persisted schema; "
          "'convergence' surfaces only as underlying=dwsim_convergence within "
          "repair_exhaustion. 'exception' rows are ambiguous (flagged).")

    # ── 3b. COMPLEXITY-SCALING CURVE ─────────────────────────────────────────────
    print("\n" + "-" * 92)
    print("AUDIT (b) — COMPLEXITY SCALING  (coherent set; binned by CORRECTED unit count)")
    print(f"{'bin':8}{'n_cases':9}{'valid_IR':10}{'reached_DWSIM*':16}{'success(PASS)':14} cases")
    print("-" * 92)
    bins = defaultdict(list)
    for cid in coherent:
        cnt, _src = unit_map.get(cid, (None, "unknown"))
        bins[bin_label(cnt)].append(cid)
    curve = []
    for _lo, _hi, lab in BINS + [(0, 0, "unknown")]:
        cids = bins.get(lab, [])
        if not cids:
            continue
        n = len(cids)
        valid_ir = sum(1 for c in cids
                       if (coherent[c].get("ir_report_json") or {}).get("valid") is True
                       or (coherent[c].get("final_graph_summary") and
                           (coherent[c].get("ir_report_json") or {}).get("valid") is None
                           and coherent[c].get("outcome", "").upper() == "PASS"))
        reached = sum(1 for c in cids if reached_dwsim_proxy(coherent[c]))
        passed  = sum(1 for c in cids if (coherent[c].get("outcome") or "").upper() == "PASS")
        curve.append({"bin": lab, "n": n, "valid_ir": valid_ir,
                      "reached_dwsim_proxy": reached, "pass": passed, "cases": sorted(cids)})
        print(f"  {lab:6}{n:7d}  {valid_ir}/{n:<7} {reached}/{n:<13} {passed}/{n:<11} "
              f"{sorted(cids)}")
    print("-" * 92)
    print("* reached_DWSIM is a PROXY (graph built AND entered execute/repair loop); "
          "RunLog stores no explicit converged flag. success = outcome==PASS.")
    print("Rates are raw counts (n/N), NOT percentages — N per bin is small; do not "
          "read as smooth percentages.")

    if args.out:
        json.dump({"target_model": target_tag, "since": args.since,
                   "coherent_cases": sorted(coherent), "only_old_or_wrong_model": sorted(only_old),
                   "census": census, "complexity_curve": curve},
                  open(args.out, "w"), indent=2, default=str)
        print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
