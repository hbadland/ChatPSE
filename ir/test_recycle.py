"""
Unit tests for recycle stream support in the IR layer.
Run with: PYTHONPATH=. python3.9 -m ir.test_recycle
"""
from __future__ import annotations

from ir.graph import FlowsheetGraph, EdgeIR, MixerNode, HeaterNode
from ir.validate import validate


def _make_recycle_graph() -> FlowsheetGraph:
    """
    Minimal A → S1 → B → S2 → A graph where S2 is a recycle stream.
    A and B are Mixer and Heater respectively (each requires ≥1 inlet).
    """
    g = FlowsheetGraph()
    g.compounds = ["Methane", "Ethane"]
    g.property_package = "Peng-Robinson"

    g.add_unit(MixerNode(tag="A"), strict=False)
    g.add_unit(HeaterNode(tag="B", params={"T_out": 400.0}), strict=False)

    forward = EdgeIR(
        tag="S1",
        T=300.0, P=200000.0,
        composition={"Methane": 0.8, "Ethane": 0.2},
    )
    g.add_stream(forward, src_tag="A", dst_tag="B", enforce_phase=False)

    recycle = EdgeIR(
        tag="S2",
        T=400.0, P=200000.0,
        composition={"Methane": 0.8, "Ethane": 0.2},
        is_recycle=True,
        recycle_target="A",
    )
    g.add_stream(recycle, src_tag="B", dst_tag="A", enforce_phase=False)

    return g


def test_has_recycles() -> None:
    g = _make_recycle_graph()
    assert g.has_recycles, "has_recycles should be True when a recycle stream exists"
    assert len(g.recycle_edges()) == 1
    assert g.recycle_edges()[0].tag == "S2"
    print("PASS test_has_recycles")


def test_validate_dag_does_not_raise_on_recycle_cycle() -> None:
    g = _make_recycle_graph()
    # The graph has a real cycle (A→S1→B→S2→A), but validate_dag() should
    # exclude the recycle stream and return True.
    assert not g.is_acyclic(), "underlying graph should contain a cycle"
    assert g.validate_dag(), "validate_dag() should return True (cycle is from recycle)"
    print("PASS test_validate_dag_does_not_raise_on_recycle_cycle")


def test_validate_passes_with_valid_recycle_target() -> None:
    g = _make_recycle_graph()
    report = validate(g)
    cycle_errors = [
        i for i in report.issues
        if "cycle" in i.error.evidence.lower()
    ]
    recycle_errors = [
        i for i in report.issues
        if "recycle" in i.error.evidence.lower()
    ]
    assert not cycle_errors, f"Unexpected cycle errors: {cycle_errors}"
    assert not recycle_errors, f"Unexpected recycle errors: {recycle_errors}"
    print("PASS test_validate_passes_with_valid_recycle_target")


def test_invalid_recycle_target_fails_validation() -> None:
    g = _make_recycle_graph()
    # Corrupt the recycle_target to a non-existent unit tag.
    bad_stream = g.stream("S2")
    object.__setattr__(bad_stream, "recycle_target", "DOES_NOT_EXIST") \
        if hasattr(bad_stream, "__dataclass_fields__") else None
    # EdgeIR is a mutable dataclass, so direct assignment works.
    bad_stream.recycle_target = "DOES_NOT_EXIST"

    report = validate(g)
    recycle_errors = [
        i for i in report.issues
        if "recycle_target" in i.error.evidence.lower()
    ]
    assert recycle_errors, "Should have a CRITICAL error for invalid recycle_target"
    assert all(i.severity == "CRITICAL" for i in recycle_errors)
    print("PASS test_invalid_recycle_target_fails_validation")


def test_non_recycle_edges_unaffected() -> None:
    g = FlowsheetGraph()
    g.compounds = ["Water"]
    g.property_package = "Peng-Robinson"
    g.add_unit(MixerNode(tag="M"), strict=False)
    g.add_unit(HeaterNode(tag="H", params={"T_out": 350.0}), strict=False)
    feed = EdgeIR(tag="FEED", T=298.15, P=101325.0, composition={"Water": 1.0})
    out  = EdgeIR(tag="OUT",  T=350.0,  P=101325.0, composition={"Water": 1.0})
    g.add_stream(feed, src_tag=None, dst_tag="M", enforce_phase=False)
    g.add_stream(out,  src_tag="M",  dst_tag="H",  enforce_phase=False)
    assert not g.has_recycles
    assert g.recycle_edges() == []
    assert g.validate_dag()
    assert g.is_acyclic()
    print("PASS test_non_recycle_edges_unaffected")


def test_sanity_cases_have_no_recycles() -> None:
    """
    Regression guard: sanity benchmark cases must not contain recycle trigger
    phrases.  Any hit means the phrase guard in orchestrator_v2 could fire on
    a case that is not a recycle flowsheet.
    """
    import json
    import os
    from agents.orchestrator_v2 import _RECYCLE_PHRASES

    cases_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "benchmark", "cases", "sanity.json",
    )
    with open(cases_path) as f:
        cases = json.load(f)

    for case in cases[:5]:
        desc_lower = case["description"].lower()
        hit = next((p for p in _RECYCLE_PHRASES if p in desc_lower), None)
        assert hit is None, (
            f"Sanity case '{case['id']}' description contains recycle phrase "
            f"'{hit}' — false-positive risk: {case['description'][:80]!r}"
        )

    g = FlowsheetGraph()
    assert not g.has_recycles, "Empty graph must have has_recycles=False"

    print("PASS test_sanity_cases_have_no_recycles")


if __name__ == "__main__":
    test_has_recycles()
    test_validate_dag_does_not_raise_on_recycle_cycle()
    test_validate_passes_with_valid_recycle_target()
    test_invalid_recycle_target_fails_validation()
    test_non_recycle_edges_unaffected()
    test_sanity_cases_have_no_recycles()
    print("\nAll recycle tests passed.")
