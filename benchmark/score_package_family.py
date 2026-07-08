"""
Score property-package family selection over the labeled benchmark cases.

Runs the ACTUAL selection logic used by the active pipeline
(Retriever.select_package -> ThermoRetriever.select, which is what
ThermoMapper/PackageSelector call) over every case that carries an expected
family label (BenchmarkCaseSpec.expected.property_package_class), maps the pick
to a family, and scores it.

This is the deterministic (rule-based) selection — the LLM tie-break
(ThermoLLMFallback) is NOT exercised here (it requires the model and can only
reorder WITHIN the rule-produced shortlist). candidates[0] is exactly what the
pipeline assigns when the LLM abstains/fails, so this is the reproducible,
GPU-free baseline. BIP-corpus coverage IS exercised (select_package passes the
BIP retriever), so activity picks reflect real corpus coverage.

Usage:
  PYTHONPATH=. python3.9 benchmark/score_package_family.py [--out PATH]
"""
from __future__ import annotations
import argparse, json, os

from benchmark.case_schema import load_tier
from benchmark.package_family import package_to_family, normalize_expected, family_correct
from rag.retriever import Retriever

_TIERS = ["val_3_5", "val_6_9", "val_10_14", "val_15plus"]
_DEFAULT_T = 300.0
_DEFAULT_P = 101_325.0


def _bip_coverage(bip, compounds: list[str]) -> dict:
    """Best-effort corpus coverage for NRTL/UNIQUAC (for BIP-contribution report)."""
    out = {}
    for model in ("NRTL", "UNIQUAC"):
        try:
            out[model] = bool(bip.has_full_coverage(compounds, model))
        except Exception:
            out[model] = None
    return out


def score() -> dict:
    r = Retriever()
    bip = getattr(r, "bip", None)
    rows = []
    for tier in _TIERS:
        try:
            cases = load_tier(tier)
        except Exception:
            continue
        for c in cases:
            exp = c.expected.property_package_class
            if not exp:
                continue
            cands = r.select_package(c.compounds, c.description, _DEFAULT_P, _DEFAULT_T)
            pick = cands[0] if cands else ""
            fam = package_to_family(pick)
            rows.append({
                "case_id":        c.id,
                "compounds":      c.compounds,
                "selected":       pick,
                "candidates":     cands,
                "selected_family": fam,
                "expected_label": exp,
                "expected_family": normalize_expected(exp),
                "correct":        family_correct(pick, exp),
                "bip_coverage":   _bip_coverage(bip, c.compounds) if bip else {},
            })
    n = len(rows)
    n_ok = sum(1 for x in rows if x["correct"])
    return {"n_labeled": n, "n_correct": n_ok,
            "accuracy": (n_ok / n) if n else None, "cases": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/package_family_baseline.json")
    args = ap.parse_args()
    res = score()

    print(f"{'case':9} {'selected':22} {'fam':9} {'exp':9} {'':4} {'candidates'}")
    for x in res["cases"]:
        mark = "OK" if x["correct"] else "MISS"
        print(f"  {x['case_id']:7} {x['selected']:22} {x['selected_family']:9} "
              f"{str(x['expected_family']):9} {mark:4} {x['candidates']}")
    print(f"\nFAMILY-SELECTION ACCURACY: {res['n_correct']}/{res['n_labeled']} "
          f"= {100*res['accuracy']:.0f}%" if res['accuracy'] is not None else "no labeled cases")

    print("\nBIP-corpus cases (activity picks) — corpus coverage:")
    any_act = False
    for x in res["cases"]:
        if x["selected_family"] == "activity":
            any_act = True
            print(f"  {x['case_id']:7} {x['selected']:8} coverage={x['bip_coverage']} {x['compounds']}")
    if not any_act:
        print("  (none of the labeled cases selected an activity model)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"\npersisted -> {args.out}")


if __name__ == "__main__":
    main()
