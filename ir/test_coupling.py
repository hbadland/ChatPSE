"""
Tests for CoupledSettler joint-settling directionality.

Run: PYTHONPATH=. python3.9 -m ir.test_coupling
     (run as a module — ir/types.py shadows stdlib `types` if run as a script)

Covers the coupling-inert fix: a Compressor/Expander P_out change must settle the
DOWNSTREAM Cooler/Heater (compress->cool->flash trains), not only the upstream
Cooler (the pre-existing Pump.P_out->upstream-Cooler path).  Without this the beam
adjusts P_out and the coupled cooler T_out independently and never converges
(VAL_01 / HB_ADV_07 / HB_AMB_03 signature).
"""
from ir.graph import FlowsheetGraph, EdgeIR, make_node
from ir.coupling import CoupledSettler


def _chain(units):
    g = FlowsheetGraph()
    g.compounds = ["Propane"]   # sub-critical BP defined at the test pressures
    for tag, ty, params in units:
        g.add_unit(make_node(ty, tag, params=params))
    return g


def test_downstream_cooler_settles_after_compressor_pout():
    """Compressor.P_out change -> DOWNSTREAM Cooler.T_out settled jointly (the fix)."""
    g = _chain([("CP-01", "Compressor", {"P_out": 2_000_000.0}),
                ("CL-01", "Cooler",     {"T_out": 350.0}),   # deliberately too hot
                ("V-01",  "Vessel",     {})])
    g.add_stream(EdgeIR(tag="FEED"), None, "CP-01")
    g.add_stream(EdgeIR(tag="S1"), "CP-01", "CL-01")
    g.add_stream(EdgeIR(tag="S2"), "CL-01", "V-01")
    g.add_stream(EdgeIR(tag="VAP", phase="vapour", src_port=0), "V-01", None)
    g.add_stream(EdgeIR(tag="LIQ", phase="liquid", src_port=1), "V-01", None)

    g2, changes = CoupledSettler().settle(g, "CP-01", "P_out")
    after = g2.unit("CL-01").params["T_out"]
    assert any("CL-01" in c for c in changes), f"settler did not engage: {changes}"
    assert after < 350.0, f"cooler not retargeted toward BP: {after}"
    assert g.unit("CL-01").params["T_out"] == 350.0, "input graph was mutated"
    print("OK downstream-cooler settles after Compressor.P_out:", after, changes)


def test_upstream_cooler_still_settles_after_pump_pout():
    """Regression: Pump.P_out change still settles the UPSTREAM Cooler."""
    g = _chain([("CL-01", "Cooler", {"T_out": 350.0}),
                ("PM-01", "Pump",   {"P_out": 2_000_000.0})])
    g.add_stream(EdgeIR(tag="FEED"), None, "CL-01")
    g.add_stream(EdgeIR(tag="S1"), "CL-01", "PM-01")
    g.add_stream(EdgeIR(tag="OUT"), "PM-01", None)

    g2, changes = CoupledSettler().settle(g, "PM-01", "P_out")
    after = g2.unit("CL-01").params["T_out"]
    assert any("CL-01" in c for c in changes), f"upstream path regressed: {changes}"
    assert after < 350.0
    print("OK upstream-cooler still settles after Pump.P_out:", after, changes)


def test_settle_retags_provenance_honestly():
    """The settler applies via correct_param: a settled value that was description-
    specified is retagged 'computed' and its sentinel cleared — no silent overwrite
    (the universal-provenance property extended to the coupling settler)."""
    g = _chain([("CP-01", "Compressor", {"P_out": 2_000_000.0}),
                ("CL-01", "Cooler",     {"T_out": 350.0,
                                         "_temperature_source": "specified",
                                         "_desc_T_out": True}),
                ("V-01",  "Vessel",     {})])
    g.add_stream(EdgeIR(tag="FEED"), None, "CP-01")
    g.add_stream(EdgeIR(tag="S1"), "CP-01", "CL-01")
    g.add_stream(EdgeIR(tag="S2"), "CL-01", "V-01")
    g.add_stream(EdgeIR(tag="VAP", phase="vapour", src_port=0), "V-01", None)
    g.add_stream(EdgeIR(tag="LIQ", phase="liquid", src_port=1), "V-01", None)

    g2, _ = CoupledSettler().settle(g, "CP-01", "P_out")
    n = g2.unit("CL-01")
    assert n.params["T_out"] < 350.0, "settler did not change the value"
    assert n.params["_temperature_source"] == "computed", n.params   # honest retag
    assert "_desc_T_out" not in n.params, "description sentinel must be cleared"
    # input graph untouched (correct_param acts on the settler's copy)
    assert g.unit("CL-01").params["_temperature_source"] == "specified"
    print("OK settler retags specified→computed + clears sentinel:", n.params["T_out"])


if __name__ == "__main__":
    test_downstream_cooler_settles_after_compressor_pout()
    test_upstream_cooler_still_settles_after_pump_pout()
    test_settle_retags_provenance_honestly()
    print("\nALL COUPLING TESTS PASSED")
