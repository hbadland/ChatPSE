"""
Retroactive vapor_fraction migration for stored per-run JSONs.

The wrapper read the LIQUID phase fraction as vapor_fraction (fixed in
dwsim_wrapper b173e29). Stored SYSTEM streams therefore carry an inverted vf.
This applies vf -> 1 - vf to system_streams ONLY (making them physically correct)
and recomputes the numeric reference_mape_vf against the reference (which is NOT
touched — VAL references come from .dwxmz with correct vf). Idempotent via a
'vf_migrated' flag.
"""
import argparse, glob, json, os
from benchmark.stream_matcher import match_streams

_REFDIR = "benchmark/reference_flowsheets"
_TOL_VF = 0.05


def _load_ref_streams(case_id: str):
    p = os.path.join(_REFDIR, f"{case_id}_reference.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return {t: {"T_K": s.get("T_K"), "P_Pa": s.get("P_Pa"),
                "vapor_fraction": s.get("vapor_fraction"),
                "composition": s.get("composition"), "is_feed": s.get("is_feed")}
            for t, s in (d.get("streams") or {}).items()}


def _sys_streams(d: dict, invert: bool):
    out = {}
    for t, s in (d.get("system_streams") or {}).items():
        if not isinstance(s, dict):
            continue
        vf = s.get("vapor_fraction")
        vf2 = (1.0 - vf) if (invert and vf is not None) else vf
        out[t] = {"T_K": s.get("T_K"), "P_Pa": s.get("P_Pa"), "vapor_fraction": vf2,
                  "composition": s.get("composition"), "is_feed": s.get("is_feed")}
    return out


def _vf_metrics(sysd, refd):
    if not sysd or not refd:
        return None
    m = match_streams(sysd, refd)
    dvfs = [abs(p["dvf"]) for p in m["pairs"] if p.get("dvf") is not None]
    if not dvfs:
        return {"n_matched": m["n_matched"], "mape_vf": None, "within": 0, "n_vf": 0}
    return {"n_matched": m["n_matched"], "mape_vf": round(sum(dvfs) / len(dvfs), 4),
            "within": sum(1 for x in dvfs if x <= _TOL_VF), "n_vf": len(dvfs)}


def migrate(path: str, write: bool):
    d = json.load(open(path))
    if d.get("vf_migrated"):
        return None
    case_id = d.get("case_id", "")
    refd = _load_ref_streams(case_id)
    before = _vf_metrics(_sys_streams(d, invert=False), refd)
    after  = _vf_metrics(_sys_streams(d, invert=True),  refd)
    # invert system_streams vf in place (physically correct)
    for s in (d.get("system_streams") or {}).values():
        if isinstance(s, dict) and s.get("vapor_fraction") is not None:
            s["vapor_fraction"] = round(1.0 - s["vapor_fraction"], 6)
    d["vf_migrated"] = True
    # update the numeric reference_mape_vf only when it was a computed number
    rc = d.get("reference_comparison")
    if isinstance(rc, dict) and isinstance(rc.get("reference_mape_vf"), (int, float)) \
            and after and after["mape_vf"] is not None:
        rc["reference_mape_vf"] = after["mape_vf"]
    if write:
        json.dump(d, open(path, "w"), indent=2, default=str)
    return {"case_id": case_id, "before": before, "after": after}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="*", default=["results/per_run/VAL_*.json"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    paths = sorted({p for g in args.globs for p in glob.glob(g)})
    latest = {}   # case_id -> (mtime, metrics) for the report
    n = 0
    for p in paths:
        r = migrate(p, write=not args.dry_run)
        if r is None:
            continue
        n += 1
        cid = r["case_id"]
        mt = os.path.getmtime(p)
        if cid not in latest or mt > latest[cid][0]:
            latest[cid] = (mt, r)
    print(f"{'DRY-RUN' if args.dry_run else 'migrated'}: {n} runs\n")
    print(f"{'case':8} {'ref?':5} {'mape_vf before->after':26} {'within±0.05 before->after'}")
    for cid in sorted(latest):
        r = latest[cid][1]
        b, a = r["before"], r["after"]
        if not b:
            print(f"  {cid:8} {'no':5} (no reference / no streams)")
            continue
        print(f"  {cid:8} {'yes':5} "
              f"{str(b['mape_vf'])+' -> '+str(a['mape_vf']):26} "
              f"{b['within']}/{b['n_vf']} -> {a['within']}/{a['n_vf']}")


if __name__ == "__main__":
    main()
