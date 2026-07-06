"""
Retroactively apply the insufficient_match gate to stored per-run JSONs.

Old per-run JSONs (before the gate landed in the runner) store n_matched only
inside the reference_stream_matching check, and their TOP-LEVEL reference_comparison
still reads e.g. mape_status=None / reference_mape_T=0.0 / match_pass=True even when
only 1 stream matched.  This rewrites the top-level fields to the GATED values so
the stored JSON matches what a fresh run now writes.

Gate (identical to the runner / physics_eval):
  sufficient          = n_matched >= _MIN_MATCH_FOR_MAPE
  mape_status         = "computed" if sufficient else "insufficient_match"
  reference_mape_*    = the stored number if sufficient, else "insufficient_match"
  reference_match_pass= stored_match_pass AND sufficient
Also writes top-level n_matched + min_match_threshold.  Idempotent.

Usage:
  PYTHONPATH=. python3.9 benchmark/regate_reference_mape.py [glob ...] [--dry-run]
  (default glob: results/per_run/*.json)
"""
import argparse, glob, json, os, sys
from benchmark.physics_eval import _MIN_MATCH_FOR_MAPE

_INS = "insufficient_match"


def _n_matched(rc: dict):
    if rc.get("n_matched") is not None:
        return rc["n_matched"]
    for c in (rc.get("checks") or []):
        if c.get("check") == "reference_stream_matching":
            return c.get("n_matched")
    return None


def regate(rc: dict) -> tuple[dict, bool]:
    """Return (rewritten reference_comparison, changed?)."""
    nm  = _n_matched(rc)
    nmv = nm if nm is not None else 0
    sufficient = nmv >= _MIN_MATCH_FOR_MAPE

    new = dict(rc)
    new["n_matched"]           = nmv
    new["min_match_threshold"] = _MIN_MATCH_FOR_MAPE
    new["mape_status"]         = "computed" if sufficient else _INS
    new["reference_match_pass"] = bool(rc.get("reference_match_pass")) and sufficient
    for k in ("reference_mape_T", "reference_mape_P", "reference_mape_vf"):
        if not sufficient:
            new[k] = _INS
        # sufficient: keep the stored numeric MAPE unchanged
    return new, (new != rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="*", default=["results/per_run/*.json"])
    ap.add_argument("--dry-run", action="store_true", help="preview, do not write")
    args = ap.parse_args()

    paths = sorted({p for g in args.globs for p in glob.glob(g)})
    n_seen = n_changed = n_no_ref = 0
    for path in paths:
        try:
            d = json.load(open(path))
        except Exception:
            continue
        rc = d.get("reference_comparison")
        if not isinstance(rc, dict):
            n_no_ref += 1
            continue
        n_seen += 1
        new_rc, changed = regate(rc)
        if not changed:
            continue
        n_changed += 1
        print(f"  {os.path.basename(path)}: n_matched={new_rc['n_matched']} "
              f"status={new_rc['mape_status']} "
              f"mape_T {rc.get('reference_mape_T')!r}->{new_rc['reference_mape_T']!r} "
              f"pass {rc.get('reference_match_pass')}->{new_rc['reference_match_pass']}")
        if not args.dry_run:
            d["reference_comparison"] = new_rc
            json.dump(d, open(path, "w"), indent=2, default=str)

    mode = "DRY-RUN (no writes)" if args.dry_run else "written"
    print(f"\n{mode}: {n_changed} changed / {n_seen} with reference_comparison "
          f"({n_no_ref} without) across {len(paths)} files")


if __name__ == "__main__":
    main()
