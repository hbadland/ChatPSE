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
from benchmark.solve_status import (
    compute_solve_status, gate_mape_status, specified_outlet_temps)

_INS = "insufficient_match"


def _n_matched(rc: dict):
    if rc.get("n_matched") is not None:
        return rc["n_matched"]
    for c in (rc.get("checks") or []):
        if c.get("check") == "reference_stream_matching":
            return c.get("n_matched")
    return None


def regate(d: dict) -> tuple[dict, dict, bool]:
    """Return (rewritten reference_comparison, solve_status, changed?).

    Applies BOTH gates with precedence partial_solve > insufficient_match >
    computed: a flowsheet that didn't fully solve can't yield a valid correctness
    MAPE regardless of match count. fully_solved is recomputed from the stored
    system_streams (exact); unit counts are best-effort.
    """
    rc = d.get("reference_comparison") or {}
    nm  = _n_matched(rc)
    nmv = nm if nm is not None else 0
    sufficient = nmv >= _MIN_MATCH_FOR_MAPE

    _fgs    = d.get("final_graph_summary") or {}
    n_units = _fgs.get("n_units")
    _spec_T = specified_outlet_temps(_fgs.get("unit_conditions"))
    solve   = compute_solve_status(d.get("system_streams"), n_units, _spec_T)
    status  = gate_mape_status(solve["fully_solved"], sufficient)
    valid   = (status == "computed")

    new = dict(rc)
    new["n_matched"]            = nmv
    new["min_match_threshold"]  = _MIN_MATCH_FOR_MAPE
    new["fully_solved"]         = solve["fully_solved"]
    new["n_streams_at_default"] = solve["n_streams_at_default"]
    new["mape_status"]          = status
    new["reference_match_pass"] = (bool(rc.get("reference_match_pass"))
                                   and solve["fully_solved"] and sufficient)
    for k in ("reference_mape_T", "reference_mape_P", "reference_mape_vf"):
        if not valid:
            new[k] = status
        # valid: keep the stored numeric MAPE unchanged
    return new, solve, (new != rc)


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
        new_rc, solve, rc_changed = regate(d)
        top_changed = (d.get("fully_solved")   != solve["fully_solved"] or
                       d.get("n_units_solved") != solve["n_units_solved"] or
                       d.get("n_units_total")  != solve["n_units_total"])
        if not (rc_changed or top_changed):
            continue
        n_changed += 1
        print(f"  {os.path.basename(path)}: fully_solved={solve['fully_solved']} "
              f"(default_streams={solve['n_streams_at_default']}) "
              f"status {rc.get('mape_status')!r}->{new_rc['mape_status']!r} "
              f"mape_T {rc.get('reference_mape_T')!r}->{new_rc['reference_mape_T']!r} "
              f"pass {rc.get('reference_match_pass')}->{new_rc['reference_match_pass']}")
        if not args.dry_run:
            d["reference_comparison"] = new_rc
            d["fully_solved"]   = solve["fully_solved"]
            d["n_units_solved"] = solve["n_units_solved"]
            d["n_units_total"]  = solve["n_units_total"]
            json.dump(d, open(path, "w"), indent=2, default=str)

    mode = "DRY-RUN (no writes)" if args.dry_run else "written"
    print(f"\n{mode}: {n_changed} changed / {n_seen} with reference_comparison "
          f"({n_no_ref} without) across {len(paths)} files")


if __name__ == "__main__":
    main()
