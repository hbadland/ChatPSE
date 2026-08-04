"""
Phase 5: the provenance invariant end-to-end (the artifact 3.2 cites).

One origin (param_mapper, not exercised here) stamps provenance; the two guarded
mutators — GlobalConsistencyPass (correct_param) and FailureRuleStore (set_param) — are
the ONLY things that change a stamped value, and every change is logged. Demonstrates:
  (1) a specified, physically-plausible value survives BOTH mutators untouched;
  (2) a specified but IMPLAUSIBLE value is corrected by consistency (-> computed, logged),
      and then legitimately refined by a learned rule (-> rule, logged) — the guard
      reaching a case it previously could not, exactly as intended.

Run: PYTHONPATH=. python3.9 agents/test_provenance_end_to_end.py
"""
from ir.graph import FlowsheetGraph, EdgeIR, HeaterNode
from ir.consistency import GlobalConsistencyPass
from agents.rule_store import FailureRuleStore

_FEED = dict(T=350.0, P=101325.0, composition={"Ethanol": 0.5, "Water": 0.5})


def _graph(node):
    g = FlowsheetGraph(); g.compounds = ["Ethanol", "Water"]
    g.add_unit(node, strict=False)
    g.add_stream(EdgeIR(tag="FEED", **_FEED), None, node.tag)
    g.add_stream(EdgeIR(tag="OUT"), node.tag, None)
    return g


def _rule_store():
    rs = FailureRuleStore()
    for _ in range(3):
        rs.record_fix("Heater", "ENERGY_UNPHYSICAL", None, "T_out", 400.0,
                      ["Ethanol", "Water"])
    return rs


def test_specified_plausible_survives_both():
    g = _graph(HeaterNode(tag="H1", params={
        "T_out": 375.0, "_desc_T_out": True, "_temperature_source": "specified"}))
    g2, c1 = GlobalConsistencyPass().apply(g)
    n2 = g2.units()[0]
    assert n2.params["T_out"] == 375.0, ("consistency changed it", n2.params["T_out"])
    assert n2.params["_temperature_source"] == "specified"
    g3, c2 = _rule_store().apply_to_graph(g2, ["Ethanol", "Water"])
    n3 = g3.units()[0]
    assert n3.params["T_out"] == 375.0, ("rule changed it", n3.params["T_out"])
    assert n3.params["_temperature_source"] == "specified"
    assert any("SUPPRESSED" in c for c in c2), c2
    print("OK: specified+plausible untouched by consistency AND rule (protected end-to-end)")


def test_implausible_specified_corrected_then_rule_refined():
    g = _graph(HeaterNode(tag="H1", params={
        "T_out": 200.0, "_desc_T_out": True, "_temperature_source": "specified"}))
    g2, c1 = GlobalConsistencyPass().apply(g)
    n2 = g2.units()[0]
    # channel 1: physical correction, recorded
    assert n2.params["T_out"] == 366.3 and n2.params["_temperature_source"] == "computed"
    assert any("CONSISTENCY" in c for c in c1), c1
    g3, c2 = _rule_store().apply_to_graph(g2, ["Ethanol", "Water"])
    n3 = g3.units()[0]
    # channel 2: learned rule now permitted to refine the estimate, recorded
    assert n3.params["T_out"] == 400.0 and n3.params["_temperature_source"] == "rule"
    assert any("RULE[" in c and "SUPPRESSED" not in c for c in c2), c2
    print("OK: implausible specified -> corrected (computed, logged) -> rule-refined (rule, logged)")


if __name__ == "__main__":
    test_specified_plausible_survives_both()
    test_implausible_specified_corrected_then_rule_refined()
    print("\nALL END-TO-END PROVENANCE TESTS PASSED")
