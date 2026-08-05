"""Non-circularity invariant: reference-injected values are labelled Source.REFERENCE
and MUST NOT survive into a scored run.

Reference-guided refinement (the VARIANT_B ablation arm) is the only writer of
REFERENCE-tagged values; it applies them via node.correct_param(..., Source.REFERENCE)
— the same call exercised here. _assert_scored_run_reference_free enforces that a
scored (VARIANT_B-off) graph carries none, turning "the arm is disabled" into a
provable, tested guarantee.

Run: PYTHONPATH=. python3.9 agents/test_reference_invariant.py
"""
from ir import (
    FlowsheetGraph, make_node, Source,
    reference_provenance_tags, ReferenceProvenanceLeak,
)
from agents.orchestrator_v2 import _assert_scored_run_reference_free


def _graph(inject_reference: bool):
    g = FlowsheetGraph()
    g.compounds = ["Propane"]
    n = make_node("Cooler", "CL-01", params={"T_out": 300.0,
                                             "_temperature_source": "specified",
                                             "_desc_T_out": True})
    if inject_reference:
        # mirrors _reference_guided_refinement's write (orchestrator_v2:241)
        n.correct_param("T_out", 317.0, Source.REFERENCE)
    g.add_unit(n, strict=False)
    return g


def test_reference_in_vocabulary():
    from ir.graph import _SOURCE_TO_STR
    assert Source.REFERENCE.name == "REFERENCE"
    assert int(Source.REFERENCE) == 7                       # highest authority
    assert _SOURCE_TO_STR[Source.REFERENCE] == "reference"
    print("OK: Source.REFERENCE in vocabulary (=7, 'reference')")


def test_correct_param_tags_reference_and_clears_sentinel():
    n = _graph(inject_reference=True).unit("CL-01")
    assert n.params["T_out"] == 317.0
    assert n.params["_temperature_source"] == "reference"   # honest reference tag
    assert "_desc_T_out" not in n.params                    # description sentinel cleared
    print("OK: correct_param(REFERENCE) tags 'reference' + clears sentinel")


def test_audit_detects_reference_tags_only():
    assert reference_provenance_tags(_graph(True)) == ["CL-01._temperature_source"]
    assert reference_provenance_tags(_graph(False)) == []   # 'specified' is not flagged
    print("OK: reference_provenance_tags flags reference tags, ignores others")


def test_invariant_raises_only_on_scored_leak():
    g_ref, g_clean = _graph(True), _graph(False)
    raised = False
    try:
        _assert_scored_run_reference_free(g_ref, scored=True)      # scored + leak → fail
    except ReferenceProvenanceLeak:
        raised = True
    assert raised, "scored run with a reference tag must raise ReferenceProvenanceLeak"
    _assert_scored_run_reference_free(g_ref, scored=False)         # VARIANT_B → allowed
    _assert_scored_run_reference_free(g_clean, scored=True)        # clean scored → ok
    print("OK: raises on scored leak; silent for VARIANT_B and clean scored runs")


def _run_all():
    test_reference_in_vocabulary()
    test_correct_param_tags_reference_and_clears_sentinel()
    test_audit_detects_reference_tags_only()
    test_invariant_raises_only_on_scored_leak()
    print("\nALL REFERENCE-INVARIANT TESTS PASSED")


if __name__ == "__main__":
    _run_all()
