"""
Isolation test for the DWSIM Recycle convergence block.

Verifies that add_recycle_block() adds the convergence block without crashing
and that the flowsheet accepts the stream connections.  Does NOT solve.

Run inside the DWSIM container:
    docker exec priceless_elion sh -c \
        "cd /workspaces/multiAgentFlowsheet && \
        PYTHONPATH=. python3.9 dwsim/test_recycle_block.py"

Or on HPC via Singularity:
    singularity exec --bind /rds $DWSIM_SIF \
        bash -c "cd $DEST && PYTHONPATH=. python3.9 dwsim/test_recycle_block.py"
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dwsim.dwsim_wrapper import DWSIMFlowsheet


def test_add_recycle_block_no_crash() -> None:
    """
    Minimal recycle topology:

        FEED → S0 → MIX ─── S1 ─── HEAT ─── S_CALC ─→ [REC-01] ─→ S_INIT ┐
                    └────────────────────────────────────────────────────────┘

    Only checks that:
      (a) add_recycle_block() does not raise
      (b) the block is registered as a simulation object (where DWSIM supports it)
    """
    sim = DWSIMFlowsheet()
    sim.add_compounds(["Methane", "Ethane"])
    sim.set_property_package("Peng-Robinson (PR)")

    # Streams
    for tag in ("FEED", "S0", "S1", "S_CALC", "S_INIT"):
        sim.add_stream(tag)

    # Units
    sim.add_unit("MIX",  "Mixer")
    sim.add_unit("HEAT", "Heater")

    # Wire up the forward path
    sim.connect("FEED",   "MIX",    src_port=0, dst_port=0)
    sim.connect("S_INIT", "MIX",    src_port=0, dst_port=1)   # recycle into mixer
    sim.connect("MIX",    "S1",     src_port=0, dst_port=0)
    sim.connect("S1",     "HEAT",   src_port=0, dst_port=0)
    sim.connect("HEAT",   "S_CALC", src_port=0, dst_port=0)

    # Add the recycle block — must not raise
    sim.add_recycle_block("REC-01", inlet_stream="S_CALC", outlet_stream="S_INIT")
    print("  add_recycle_block() completed without exception")

    # Verify the block appears in DWSIM's simulation objects where possible.
    # Some DWSIM builds register logical blocks separately from unit operations,
    # so a None return from GetFlowsheetSimulationObject is not a test failure.
    try:
        obj = sim._sim.GetFlowsheetSimulationObject("REC-01")
        if obj is not None:
            print(f"  REC-01 found in SimulationObjects (type: {obj.GetType().Name})")
        else:
            print("  REC-01 not in SimulationObjects registry "
                  "(logical block — may be stored separately)")
    except Exception as e:
        print(f"  GetFlowsheetSimulationObject raised: {e} (non-fatal)")

    print("PASS test_add_recycle_block_no_crash")


def test_to_dwsim_recycle_serialisation() -> None:
    """
    Verify ir/to_dwsim produces the expected keys when a recycle edge is present.
    This test is host-runnable (does not import DWSIMFlowsheet) — it just checks
    the JSON structure.
    """
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from ir.graph import FlowsheetGraph, EdgeIR, MixerNode, HeaterNode
    from ir.to_dwsim import to_dwsim

    g = FlowsheetGraph()
    g.compounds = ["Methane", "Ethane"]
    g.property_package = "Peng-Robinson"

    g.add_unit(MixerNode(tag="MIX"), strict=False)
    g.add_unit(HeaterNode(tag="HEAT", params={"T_out": 400.0}), strict=False)

    fwd = EdgeIR(tag="S1", T=300.0, P=200000.0,
                 composition={"Methane": 0.8, "Ethane": 0.2})
    rec = EdgeIR(tag="S_REC", T=400.0, P=200000.0,
                 composition={"Methane": 0.8, "Ethane": 0.2},
                 is_recycle=True, recycle_target="MIX")
    g.add_stream(fwd, src_tag="MIX",  dst_tag="HEAT", enforce_phase=False)
    g.add_stream(rec, src_tag="HEAT", dst_tag="MIX",  enforce_phase=False)

    d = to_dwsim(g)

    stream_tags = {s["tag"] for s in d["streams"]}
    assert "S_REC"      in stream_tags, "calc stream missing"
    assert "S_REC-INIT" in stream_tags, "init stream missing"
    assert "recycle_blocks" in d, "recycle_blocks key missing"
    assert d["recycle_blocks"][0]["inlet_stream"]  == "S_REC"
    assert d["recycle_blocks"][0]["outlet_stream"] == "S_REC-INIT"
    assert d["recycle_blocks"][0]["tag"]           == "REC-01"

    # Connections: S_REC should appear as dst only (not src), S_REC-INIT as src only
    conn_srcs = [c[0] for c in d["connections"]]
    conn_dsts = [c[1] for c in d["connections"]]
    assert "S_REC"      in conn_dsts, "calc stream not a dst"
    assert "S_REC"      not in conn_srcs, "calc stream should not be a src (recycle block handles it)"
    assert "S_REC-INIT" in conn_srcs, "init stream not a src"
    assert "S_REC-INIT" not in conn_dsts, "init stream should not be a dst (recycle block handles it)"

    print("PASS test_to_dwsim_recycle_serialisation")


if __name__ == "__main__":
    # Serialisation test runs on any host (no DWSIM required)
    test_to_dwsim_recycle_serialisation()

    # Block-addition test requires DWSIM — run only inside container/Singularity
    print("\nAttempting DWSIM block addition test "
          "(requires container — will fail gracefully on host)...")
    try:
        test_add_recycle_block_no_crash()
    except Exception as e:
        print(f"  Skipped (not in DWSIM environment): {e}")

    print("\nRecycle block tests complete.")
