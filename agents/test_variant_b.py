"""
Host-side tests for Variant B topology sourcing in agents/graph_pipeline.py.

Confirms the reference_topology node uses a POPULATED connections array VERBATIM
(no inference) and only INFERS connectivity when the array is empty.

Run: PYTHONPATH=. python3.9 agents/test_variant_b.py
"""
from __future__ import annotations

import json
import os
import tempfile

from agents.graph_pipeline import (
    _topology_from_connections,
    _infer_sequential_topology,
    _reference_trust,
    variant_b_summary,
)
from agents.stage1.unit_extractor import SemanticUnits, SemanticUnit
from agents.stage2 import GraphBuilder
from ir import normalise, validate

_passed = 0
_failed = 0


def check(cond: bool, msg: str) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS: {msg}")
    else:
        _failed += 1
        print(f"  FAIL: {msg}")


def _units(*pairs) -> SemanticUnits:
    return SemanticUnits(units=[SemanticUnit(tag=t, type=ty, role="reference")
                                for t, ty in pairs])


# A populated reference in the canonical DWSIM/schema format: connections are
# [src, dst, src_port, dst_port] pairing a unit with a stream.
_REF_POPULATED = {
    "case_id": "SYNTH_EXACT",
    "compounds": ["Water", "Ethanol"],
    "units": [{"tag": "HT-01", "type": "Heater", "params": {"T_out": 350.0}},
              {"tag": "V-01",  "type": "Vessel",  "params": {}}],
    "connections": [
        ["FEED", "HT-01", 0, 0],   # stream FEED → unit HT-01  (FEED is a feed)
        ["HT-01", "HOT",  0, 0],   # unit HT-01 → stream HOT
        ["HOT",  "V-01",  0, 0],   # stream HOT → unit V-01
        ["V-01", "VAP",   0, 0],   # unit V-01 → stream VAP
        ["V-01", "LIQ",   1, 0],   # unit V-01 → stream LIQ
    ],
    "streams": {
        "FEED": {"T_K": 350.0, "P_Pa": 101325.0, "flow_mol_s": 1.0,
                 "vapor_fraction": 0.0,
                 "composition": {"Water": 0.5, "Ethanol": 0.5}},
        "HOT":  {"T_K": 360.0, "P_Pa": 101325.0, "flow_mol_s": 1.0},
        "VAP":  {"T_K": 360.0, "P_Pa": 101325.0, "flow_mol_s": 0.4},
        "LIQ":  {"T_K": 360.0, "P_Pa": 101325.0, "flow_mol_s": 0.6},
    },
}


def test_connections_used_verbatim():
    print("\n[Variant B] populated connections (DWSIM format) used VERBATIM")
    sem_units = _units(("HT-01", "Heater"), ("V-01", "Vessel"))
    topo = _topology_from_connections(
        _REF_POPULATED["connections"], sem_units, _REF_POPULATED)

    by_tag = {s.tag: s for s in topo.streams}
    check(set(by_tag) == {"FEED", "HOT", "VAP", "LIQ"},
          f"streams keep their real reference tags (got {sorted(by_tag)})")
    check(not any(s.tag.startswith("VB-") for s in topo.streams),
          "NO inferred (VB-*) streams — connections taken verbatim")

    check(by_tag["FEED"].src is None and by_tag["FEED"].dst == "HT-01"
          and by_tag["FEED"].is_feed,
          "FEED: src=None, dst=HT-01, is_feed=True")
    check(by_tag["HOT"].src == "HT-01" and by_tag["HOT"].dst == "V-01",
          "HOT: HT-01 → V-01 (unit→unit edge recovered from unit↔stream pairs)")
    check(by_tag["VAP"].src == "V-01" and by_tag["VAP"].dst is None,
          "VAP: V-01 → product")
    check(by_tag["LIQ"].src == "V-01" and by_tag["LIQ"].dst is None,
          "LIQ: V-01 → product")
    check(by_tag["FEED"].T == 350.0 and by_tag["FEED"].composition.get("Water") == 0.5,
          "FEED conditions (T/composition) carried over from reference streams")

    g = normalise(GraphBuilder().build(sem_units,
                                       topo, _REF_POPULATED["compounds"]))
    check(validate(g).valid, "verbatim topology builds a VALID IR")


def test_empty_connections_infers():
    print("\n[Variant B] empty connections → inference (VB-* streams)")
    sem_units = _units(("HT-01", "Heater"), ("V-01", "Vessel"))
    topo, inferred_feed = _infer_sequential_topology(sem_units, ["Water", "Ethanol"])
    check(inferred_feed, "inferred_feed=True for the inference path")
    check(all(s.tag.startswith("VB-") for s in topo.streams),
          "all streams are inferred (VB-*) — distinct from the verbatim path")


def test_node_branch_selects_verbatim_vs_infer():
    print("\n[Variant B] node uses verbatim when present, infers when empty")
    try:
        from agents.graph_pipeline import GraphPipeline, LANGGRAPH_AVAILABLE
        if not LANGGRAPH_AVAILABLE:
            print("  SKIP: langgraph not installed")
            return
    except Exception as e:  # noqa: BLE001
        print(f"  SKIP: cannot import GraphPipeline ({e})")
        return

    pipe = GraphPipeline(model="qwen3:14b")
    tmp = tempfile.mkdtemp()

    # (a) populated connections → reference-exact, verbatim tags
    p_pop = os.path.join(tmp, "pop.json")
    json.dump(_REF_POPULATED, open(p_pop, "w"))
    out = pipe._reference_topology_node(
        {"reference_file": p_pop, "description": "x", "variant_b_active": True})
    check(out["topology_source"] == "reference-exact",
          "populated connections → topology_source='reference-exact'")
    tags = {s.tag for s in out["sem_topo"].streams}
    check(tags == {"FEED", "HOT", "VAP", "LIQ"} and not any(t.startswith("VB-") for t in tags),
          f"node used connections verbatim (tags={sorted(tags)})")
    check(out["variant_b_inferred_feed"] is False, "inferred_feed=False on verbatim path")

    # (b) empty connections → reference-inferred-connections
    ref_empty = dict(_REF_POPULATED, case_id="SYNTH_EMPTY", connections=[])
    p_emp = os.path.join(tmp, "emp.json")
    json.dump(ref_empty, open(p_emp, "w"))
    out2 = pipe._reference_topology_node(
        {"reference_file": p_emp, "description": "x", "variant_b_active": True})
    check(out2["topology_source"] == "reference-inferred-connections",
          "empty connections → topology_source='reference-inferred-connections'")
    check(all(s.tag.startswith("VB-") for s in out2["sem_topo"].streams),
          "node inferred (VB-*) only when connections were empty")


def test_untrusted_reference_gating():
    print("\n[Variant B] reactive case missing reactor → untrusted (auto-clears)")
    # Reactive case (VAL_03), no reactor unit → untrusted.
    r_missing = {"case_id": "VAL_03", "units": [{"tag": "V-01", "type": "Vessel"}]}
    check(_reference_trust(r_missing) == "missing reactor",
          "VAL_03 with no reactor → 'missing reactor'")
    # Same reactive case but reactor present → trusted (gate auto-clears).
    r_fixed = {"case_id": "VAL_03",
               "units": [{"tag": "RX-01", "type": "ConversionReactor"},
                         {"tag": "V-01", "type": "Vessel"}]}
    check(_reference_trust(r_fixed) is None,
          "VAL_03 WITH a reactor → trusted (gate auto-clears once ref is fixed)")
    # Non-reactive case with no reactor → trusted (no reactor expected).
    r_nonreactive = {"case_id": "VAL_07", "units": [{"tag": "V-01", "type": "Vessel"}]}
    check(_reference_trust(r_nonreactive) is None,
          "VAL_07 (non-reactive) with no reactor → trusted")


def test_summary_excludes_untrusted_convergence():
    print("\n[Variant B] summary flags untrusted + excludes them from convergence")
    diags = [
        {"case": "VAL_03", "topology_source": "reference-inferred-connections",
         "untrusted_reference": "missing reactor", "built_valid_ir": True,
         "reached_dwsim": True, "converged": True, "n_repair_iterations": 1,
         "reference_mape_T": 3.2, "failure_stage": None},
        {"case": "VAL_07", "topology_source": "reference-inferred-connections",
         "untrusted_reference": None, "built_valid_ir": True,
         "reached_dwsim": True, "converged": True, "n_repair_iterations": 0,
         "reference_mape_T": 1.1, "failure_stage": None},
    ]
    out = variant_b_summary(diags)
    check("UNTRUSTED*" in out, "untrusted case flagged with UNTRUSTED* in the table")
    check("converged (TRUSTED only): 1/1" in out,
          "headline converged counts trusted cases only (untrusted excluded)")
    check("UNTRUSTED: 1/2" in out and "VAL_03" in out,
          "aggregate names the untrusted case(s)")


def main():
    test_connections_used_verbatim()
    test_empty_connections_infers()
    test_node_branch_selects_verbatim_vs_infer()
    test_untrusted_reference_gating()
    test_summary_excludes_untrusted_convergence()
    print(f"\n{'='*60}\nRESULT: {_passed} passed, {_failed} failed\n{'='*60}")
    raise SystemExit(1 if _failed else 0)


if __name__ == "__main__":
    main()
