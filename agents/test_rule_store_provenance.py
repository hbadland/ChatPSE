"""
Phase 2 behaviour test: FailureRuleStore.apply_to_graph must, after routing through
NodeIR.set_param(Source.RULE), reproduce the legacy provenance guard end-to-end —
suppress on description-derived/extracted/inherited/prior-rule values, apply on
estimates/untagged, and re-tag applied values to 'rule'.

Run: PYTHONPATH=. python3.9 agents/test_rule_store_provenance.py
"""
from agents.rule_store import FailureRuleStore
from ir.graph import FlowsheetGraph, HeaterNode


def _store_with_rule():
    rs = FailureRuleStore()
    for _ in range(3):  # RULE_THRESHOLD
        rs.record_fix("Heater", "ENERGY_UNPHYSICAL", None, "T_out", 400.0,
                      ["Ethanol", "Water"])
    assert rs.active_rules(), "no active rule synthesized"
    return rs


def _graph_with_heater(tag_source, sentinel=False):
    g = FlowsheetGraph(); g.compounds = ["Ethanol", "Water"]
    params = {"T_out": 300.0}
    if tag_source:
        params["_temperature_source"] = tag_source
    if sentinel:
        params["_desc_T_out"] = True
    g.add_unit(HeaterNode(tag="H-01", params=params), strict=False)
    return g


def test_suppressed_on_protected():
    rs = _store_with_rule()
    for tag in ("specified", "extracted", "inherited", "rule"):
        g2, changes = rs.apply_to_graph(_graph_with_heater(tag), ["Ethanol", "Water"])
        node = g2.units()[0]
        assert node.params["T_out"] == 300.0, (tag, node.params["T_out"])
        assert any("SUPPRESSED" in c for c in changes), (tag, changes)
    print("OK: suppressed on specified/extracted/inherited/rule")


def test_applied_on_estimate():
    rs = _store_with_rule()
    for tag in ("computed", "default_fallback", "fallback", None):
        g2, changes = rs.apply_to_graph(_graph_with_heater(tag), ["Ethanol", "Water"])
        node = g2.units()[0]
        assert node.params["T_out"] == 400.0, (tag, node.params["T_out"])
        assert node.params["_temperature_source"] == "rule", (tag, node.params.get("_temperature_source"))
        assert any("RULE[" in c and "SUPPRESSED" not in c for c in changes), (tag, changes)
    print("OK: applied + re-tagged 'rule' on computed/default/fallback/untagged")


def test_untagged_with_sentinel_protected():
    rs = _store_with_rule()
    g2, changes = rs.apply_to_graph(_graph_with_heater(None, sentinel=True),
                                    ["Ethanol", "Water"])
    assert g2.units()[0].params["T_out"] == 300.0
    assert any("SUPPRESSED" in c for c in changes)
    print("OK: untagged + description sentinel protected")


if __name__ == "__main__":
    test_suppressed_on_protected()
    test_applied_on_estimate()
    test_untagged_with_sentinel_protected()
    print("\nALL RULE-STORE PROVENANCE TESTS PASSED")
