"""
Stage-1-only topology-extraction diagnostic for the 10 validation cases.

Runs ONLY the extraction + IR-build path (basis -> topology -> build -> validate,
plus the deterministic, no-LLM topology_repair when applicable) and STOPS before
thermo/execute/repair.  This isolates extraction/IR-build quality from the slow
convergence layer.  It is NOT a convergence or MAPE result.

Per case: valid_ir, units extracted vs reference unit count, extraction-stage
failure reason (if any).

Run:  PYTHONPATH=. OLLAMA_BASE_URL=http://localhost:11434/v1 python3.9 stage1_diag.py
Writes results/diagnostics/stage1_diag_results.json and prints a summary table.
"""
import json
import os
import time
import traceback

from benchmark.case_schema import load_tier
from agents.graph_pipeline import GraphPipeline
from ir import validate

MODEL = os.environ.get("STAGE1_MODEL", "qwen3:14b")
OUT_DIR = "results/diagnostics"
OUT_FILE = os.path.join(OUT_DIR, "stage1_diag_results.json")


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
            rec["fail_stage"] = "topology"
            rec["fail_reason"] = "PLAN_FAILED (extraction yielded no usable topology)"
            return rec
        sem_units = st.get("sem_units")
        rec["units_extracted"] = len(sem_units.units) if sem_units else 0

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
            rec["unit_types"] = sorted({u.type for u in units})
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
        rec["_traceback"] = traceback.format_exc()
    return rec


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cases = load_tier("validation")
    print(f"[diag] {len(cases)} validation cases; model={MODEL}", flush=True)
    gp = GraphPipeline(model=MODEL, max_iterations=10)

    results = []
    for case in cases:
        t0 = time.time()
        print(f"\n===== {case.id} START =====", flush=True)
        rec = run_stage1(gp, case)
        rec["elapsed_s"] = round(time.time() - t0, 1)
        results.append(rec)
        json.dump(results, open(OUT_FILE, "w"), indent=2, default=str)  # checkpoint each case
        print(f"===== {case.id} DONE valid_ir={rec['valid_ir']} "
              f"ir_units={rec['ir_units']} ref={rec['ref_units']} "
              f"fail={rec['fail_stage']} ({rec['elapsed_s']}s) =====", flush=True)

    print(f"\n[diag] wrote {OUT_FILE}", flush=True)

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
