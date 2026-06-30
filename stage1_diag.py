"""
Stage-1-only topology-extraction diagnostic for the 10 validation cases.

Runs ONLY the extraction + IR-build path (basis -> topology -> build -> validate,
plus the deterministic, no-LLM topology_repair when applicable) and STOPS before
thermo/execute/repair.  This isolates extraction/IR-build quality from the slow
convergence layer.  It is NOT a convergence or MAPE result.

Per case: valid_ir, units extracted vs reference unit count, extraction-stage
failure reason (if any).

Run:  PYTHONPATH=. OLLAMA_BASE_URL=http://localhost:11434/v1 \
          python3.9 stage1_diag.py --model qwen3:30b-a3b
Writes <diag-dir>/stage1_diag_results.json and prints a summary table.
"""
import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter

from benchmark.case_schema import load_tier
from agents.graph_pipeline import GraphPipeline
from agents.stage1.stream_extractor import _EMPTY_ERRORS
from ir import validate

# Default model matches benchmark_runner.py (the canonical entry point).
# STAGE1_MODEL env var overrides the default; an explicit --model wins over both.
DEFAULT_MODEL = os.environ.get("STAGE1_MODEL", "qwen3:30b-a3b")


def initial_state(description: str) -> dict:
    return {
        "description": description, "tier": "validation", "reference_file": None,
        "max_iterations": 10, "t_start": time.time(),
        "variant_b_active": False, "topology_source": None,
        "reference_unit_params": {}, "variant_b_inferred_feed": False,
        "basis_result": None, "norm_desc": "", "compounds": [],
        "sem_units": None, "sem_topo": None, "recycle_origin": {},
        "ir_graph": None, "ir_report": None, "missing_units": [],
        "dwsim_json": None, "reference_data": None, "tried_packages": [],
        "iteration": 0, "eff_max_iter": 10, "beam_extended": False,
        "repair_memory": None, "sim_hints": None, "execution": None,
        "errors": [], "prev_hash": None, "warnings": [], "iterations_log": [],
        "outcome": "MAX_ITER",
    }


def ref_unit_count(reference_file: str) -> int:
    try:
        d = json.load(open(reference_file))
        u = d.get("units")
        return len(u) if u else 0
    except Exception:
        return -1


def run_stage1(gp: GraphPipeline, case) -> dict:
    desc = getattr(case, "prompt", None) or case.description
    st = initial_state(desc)
    rec = {
        "case": case.id, "valid_ir": False,
        "units_extracted": None, "ir_units": None, "unit_types": None,
        # Raw UnitExtractor output (LLM call #2) BEFORE normalise auto-inserts
        # mixers/splitters — tags + multiplicity preserved, so "three stages"
        # producing one vs three units is visible.  extracted_type_counts is the
        # per-type multiplicity; graph_type_counts is the post-normalise count, so
        # (graph_type_counts - extracted_type_counts) = deterministic padding.
        "extracted_units": None, "extracted_type_counts": None,
        "graph_type_counts": None,
        "ref_units": ref_unit_count(getattr(case, "reference_file", "") or ""),
        "fail_stage": None, "fail_reason": None,
    }
    try:
        # 1. basis
        st.update(gp._basis_node(st))
        if st["outcome"] == "BASIS_FAILED":
            rec["fail_stage"] = "basis"
            br = st.get("basis_result")
            rec["fail_reason"] = getattr(br, "reason", "BASIS_FAILED") or "BASIS_FAILED"
            return rec

        # 2. topology (unit + stream extraction)
        st.update(gp._topology_node(st))
        if st["outcome"] == "PLAN_FAILED":
            # _topology_node degrades an exhausted/empty Stage-1 LLM extraction to
            # PLAN_FAILED, stashing the real exception text in warnings.  An empty
            # stream-extractor response (ValueError "empty response" / "only
            # markdown" / a "line 1 column 1" JSON-decode of "") is an EXTRACTION
            # failure, NOT a topology failure — relabel it so the diagnostic does
            # not mask empty-response failures inside a generic "topology" bucket.
            detail = next((w for w in (st.get("warnings") or [])
                           if "Stage 1 failed" in w), "")
            if any(m in detail for m in _EMPTY_ERRORS):
                rec["fail_stage"] = "empty_response"
                rec["fail_reason"] = (
                    detail or "Stage 1 stream extractor returned an empty response")
            else:
                rec["fail_stage"] = "topology"
                rec["fail_reason"] = (
                    detail or "PLAN_FAILED (extraction yielded no usable topology)")
            return rec
        sem_units = st.get("sem_units")
        rec["units_extracted"] = len(sem_units.units) if sem_units else 0
        # Capture the raw extraction VERBATIM (call #2 output, pre-normalise):
        # tags + types + reaction, in extraction order, with multiplicity intact.
        if sem_units:
            rec["extracted_units"] = [
                {"tag": u.tag, "type": u.type, "reaction": u.reaction or None}
                for u in sem_units.units
            ]
            rec["extracted_type_counts"] = dict(
                Counter(u.type for u in sem_units.units))

        # 3. build IR
        st.update(gp._build_node(st))

        # 4. validate
        st.update(gp._validate_node(st))

        # 4b. deterministic topology_repair (NO LLM) if repairable
        if st["outcome"] == "INVALID_TOPOLOGY":
            try:
                st.update(gp._topology_repair_node(st))
            except Exception as e:  # noqa: BLE001
                rec["fail_reason"] = f"topology_repair raised: {e!r}"

        graph = st.get("ir_graph")
        if graph is not None:
            units = graph.units()
            rec["ir_units"] = len(units)
            # NodeIR exposes .unit_type (the canonical type string); .type is the
            # SemanticUnit attribute and does NOT exist on the built graph nodes —
            # using it crashed every successfully-built case with an AttributeError.
            rec["unit_types"] = sorted({u.unit_type for u in units})
            # Post-normalise per-type multiplicity; diff against
            # extracted_type_counts to isolate deterministic mixer/splitter padding.
            rec["graph_type_counts"] = dict(Counter(u.unit_type for u in units))
            rep = validate(graph)
            rec["valid_ir"] = bool(rep.valid)
            if not rep.valid:
                rec["fail_stage"] = "ir_build"
                errs = [str(e) for e in rep.errors()]
                rec["fail_reason"] = f"{st['outcome']}: " + " | ".join(errs[:3])
        else:
            rec["fail_stage"] = "ir_build"
            rec["fail_reason"] = "no ir_graph produced"
    except Exception as e:  # noqa: BLE001
        rec["fail_stage"] = rec["fail_stage"] or "exception"
        rec["fail_reason"] = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()
        rec["_traceback"] = tb
        # Surface the full traceback immediately — a swallowed exception is
        # undiagnosable.  Print to stderr so it lands in the run log.
        print(f"[diag] {rec['case']} FAILED ({rec['fail_stage']}): {rec['fail_reason']}\n{tb}",
              flush=True, file=sys.stderr)
    return rec


def main():
    parser = argparse.ArgumentParser(
        description="Stage-1 topology-extraction diagnostic (extraction/IR-build only)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"LLM model (default: {DEFAULT_MODEL})")
    parser.add_argument("--diag-dir", default="results/diagnostics",
                        help="output directory (default: results/diagnostics)")
    args = parser.parse_args()

    out_file = os.path.join(args.diag_dir, "stage1_diag_results.json")
    os.makedirs(args.diag_dir, exist_ok=True)
    cases = [c for t in ("val_3_5", "val_6_9", "val_10_14", "val_15plus")
             for c in load_tier(t)]
    print(f"[diag] {len(cases)} validation cases; model={args.model}", flush=True)
    gp = GraphPipeline(model=args.model, max_iterations=10)

    results = []
    for case in cases:
        t0 = time.time()
        print(f"\n===== {case.id} START =====", flush=True)
        rec = run_stage1(gp, case)
        rec["elapsed_s"] = round(time.time() - t0, 1)
        results.append(rec)
        json.dump(results, open(out_file, "w"), indent=2, default=str)  # checkpoint each case
        print(f"===== {case.id} DONE valid_ir={rec['valid_ir']} "
              f"ir_units={rec['ir_units']} ref={rec['ref_units']} "
              f"fail={rec['fail_stage']} ({rec['elapsed_s']}s) =====", flush=True)

    print(f"\n[diag] wrote {out_file}", flush=True)

    # Summary table
    print("\n" + "=" * 92)
    print("STAGE-1 TOPOLOGY-EXTRACTION DIAGNOSTIC (extraction/IR-build only; NOT convergence/MAPE)")
    print("=" * 92)
    print(f"{'case':9}{'valid_ir':10}{'extracted':11}{'ir_units':10}{'ref':6}{'fail_stage':13}reason")
    print("-" * 92)
    for r in results:
        reason = (r["fail_reason"] or "")[:30]
        print(f"{r['case']:9}{str(r['valid_ir']):10}{str(r['units_extracted']):11}"
              f"{str(r['ir_units']):10}{str(r['ref_units']):6}{str(r['fail_stage'] or '—'):13}{reason}")
    n = len(results)
    print("-" * 92)
    print(f"valid IR built: {sum(1 for r in results if r['valid_ir'])}/{n}")


if __name__ == "__main__":
    main()
