"""
Variant B driver — reference-assisted topology injection diagnostic.

Runs each validation case through the LangGraph pipeline with VARIANT_B=1, so the
LLM basis/topology nodes are bypassed and topology is injected from the reference
flowsheet.  Everything from build_node onward (GraphBuilder → validate →
topology_repair → thermo → execute/DWSIM → repair) runs as the real system.

Per-case it prints the diagnostic ladder and aggregates a summary, answering:
given (reference) topology, how many cases build valid IR / reach DWSIM /
converge / match the reference within tolerance.

Requires DWSIM (Docker/Singularity) + an LLM for the thermo half — i.e. HPC.
VAL_03 is reported first.

Run (HPC):
    VARIANT_B=1 PYTHONPATH=. python3.9 agents/variant_b_diag.py
    VARIANT_B=1 PYTHONPATH=. python3.9 agents/variant_b_diag.py --case VAL_03
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from agents.graph_pipeline import GraphPipeline, variant_b_summary

_CASES_FILE = "benchmark/cases/validation.json"


def _load_cases() -> list[dict]:
    d = json.load(open(_CASES_FILE))
    return d if isinstance(d, list) else d.get("cases", [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("VARIANT_B_MODEL", "qwen3:30b-a3b"))
    ap.add_argument("--max-iter", type=int, default=6)
    ap.add_argument("--case", default=None, help="run a single case id (e.g. VAL_03)")
    args = ap.parse_args()

    if not os.environ.get("VARIANT_B"):
        print("WARNING: VARIANT_B is not set — forcing VARIANT_B=1 for this run.",
              file=sys.stderr)
        os.environ["VARIANT_B"] = "1"

    cases = [c for c in _load_cases() if c.get("reference_file")]
    # VAL_03 first (priority case we understand best), then the rest in order.
    cases.sort(key=lambda c: (c.get("id") != "VAL_03", c.get("id", "")))
    if args.case:
        cases = [c for c in cases if c.get("id") == args.case]

    pipe = GraphPipeline(model=args.model, max_iterations=args.max_iter)

    diags: list[dict] = []
    for c in cases:
        cid = c.get("id")
        print(f"\n{'#'*80}\n# VARIANT B — {cid}: {c.get('name','')}\n{'#'*80}", flush=True)
        try:
            result = pipe.run(
                description=c["description"],
                reference_file=c["reference_file"],
                tier="validation",
            )
            diag = getattr(result, "variant_b_diag", None)
            if diag is None:
                diag = {"case": cid, "topology_source": None, "built_valid_ir": False,
                        "reached_dwsim": False, "converged": False,
                        "n_repair_iterations": 0, "reference_mape_T": None,
                        "reference_mape_P": None, "reference_mape_vf": None,
                        "failure_stage": "other", "outcome": result.outcome}
        except Exception as exc:  # noqa: BLE001
            import traceback; traceback.print_exc()
            diag = {"case": cid, "topology_source": None, "built_valid_ir": False,
                    "reached_dwsim": False, "converged": False,
                    "n_repair_iterations": 0, "reference_mape_T": None,
                    "reference_mape_P": None, "reference_mape_vf": None,
                    "failure_stage": f"exception:{type(exc).__name__}", "outcome": "ERROR"}
        print(f"[LADDER] {cid}: {json.dumps(diag)}", flush=True)
        diags.append(diag)

    print(variant_b_summary(diags))


if __name__ == "__main__":
    main()
