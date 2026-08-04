"""
Phase 4 regression test: GlobalConsistencyPass now writes through NodeIR.correct_param,
so its physical corrections/derivations are value-identical to the old direct writes but
carry honest provenance (fixing the self-certifying-tag bug in the twin of rule_store).

Run: PYTHONPATH=. python3.9 -m ir.test_consistency_provenance
"""
from ir.graph import FlowsheetGraph, EdgeIR, HeaterNode, CoolerNode, CompressorNode
from ir.consistency import GlobalConsistencyPass

_FEED = dict(T=350.0, P=101325.0, composition={"Ethanol": 0.5, "Water": 0.5})


def _run(node):
    g = FlowsheetGraph(); g.compounds = ["Ethanol", "Water"]
    g.add_unit(node, strict=False)
    g.add_stream(EdgeIR(tag="FEED", **_FEED), None, node.tag)
    g.add_stream(EdgeIR(tag="OUT"), node.tag, None)
    g2, changes = GlobalConsistencyPass().apply(g)
    return g2.units()[0], changes


def test_implausible_specified_corrected_and_retagged():
    # value: physical (bubble-point) correction, unchanged from the legacy direct write
    n, _ = _run(HeaterNode(tag="H1", params={
        "T_out": 200.0, "_desc_T_out": True, "_temperature_source": "specified"}))
    assert n.params["T_out"] == 366.3, n.params["T_out"]
    # honesty: the corrected value is no longer the user's spec
    assert n.params["_temperature_source"] == "computed"
    assert "_desc_T_out" not in n.params          # sentinel cleared (self-cert bug fixed)
    print("OK: implausible specified T_out corrected -> computed, sentinel cleared")


def test_fill_missing_now_tagged():
    n, _ = _run(HeaterNode(tag="H3", params={}))
    assert n.params["T_out"] == 371.3
    assert n.params["_temperature_source"] == "computed"   # was untagged before Phase 4
    print("OK: filled T_out carries computed provenance")


def test_already_computed_value_identical():
    for node, exp in [
        (HeaterNode(tag="H2", params={"T_out": 300.0, "_temperature_source": "computed"}), 366.3),
        (CoolerNode(tag="C1", params={"T_out": 400.0, "_temperature_source": "computed"}), 325.0),
        (CompressorNode(tag="K1", params={"P_out": 50000.0, "_pressure_source": "computed"}), 506625.0),
    ]:
        n, _ = _run(node)
        key = "T_out" if "T_out" in n.params else "P_out"
        assert n.params[key] == exp, (node.tag, n.params[key])
    print("OK: already-computed corrections value-identical")


if __name__ == "__main__":
    test_implausible_specified_corrected_and_retagged()
    test_fill_missing_now_tagged()
    test_already_computed_value_identical()
    print("\nALL CONSISTENCY-PROVENANCE TESTS PASSED")
