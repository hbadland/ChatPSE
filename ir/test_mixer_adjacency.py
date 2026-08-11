"""
Tests for the Mixer-adjacency recycle-INIT composition heuristic.

Run with: PYTHONPATH=. python3.9 -m ir.test_mixer_adjacency
"""
from __future__ import annotations

from ir.graph import (
    FlowsheetGraph, EdgeIR,
    MixerNode, CompressorNode, ColumnNode,
)
from ir.to_dwsim import _mixer_secondary_feed_composition, _feed_stream_composition


def _val09_graph() -> tuple[FlowsheetGraph, EdgeIR]:
    """
    Minimal VAL_09-style graph: two feed streams into an explicit Mixer, then a
    Column; the recycle from the Column targets the Mixer but has no graph edge
    (unlinked, recycle_target set — the topology seen before normalise() wires it).

      FEED (Benzene=0.5/Ethanol=0.5) ──┐
                                        ├── MIX-COL-01 ── COL-01 ──> DIST
      SOLV (p-Xylene=1.0) ─────────────┘                        └──> BOT
                                                   ^
                                                   │  REC (is_recycle=True,
                                                   └─ recycle_target=MIX-COL-01,
                                                         dst=None)
    """
    g = FlowsheetGraph()
    g.compounds = ["Benzene", "Ethanol", "p-Xylene"]
    g.property_package = "NRTL"

    g.add_unit(MixerNode(tag="MIX-COL-01"), strict=False)
    g.add_unit(ColumnNode(tag="COL-01",
                          params={"Nstages": 20, "feed_stage": 10,
                                  "reflux_ratio": 2.0, "reboil_ratio": 1.0,
                                  "condenser_pressure": 101325.0,
                                  "key_comp_light": "Ethanol",
                                  "key_comp_heavy": "p-Xylene",
                                  "light_recovery": 0.99,
                                  "heavy_recovery": 0.99}), strict=False)

    feed = EdgeIR(tag="FEED", composition={"Benzene": 0.5, "Ethanol": 0.5, "p-Xylene": 0.0})
    g.add_stream(feed, src_tag=None, dst_tag="MIX-COL-01", enforce_phase=False)

    solv = EdgeIR(tag="SOLV", composition={"p-Xylene": 1.0})
    g.add_stream(solv, src_tag=None, dst_tag="MIX-COL-01", enforce_phase=False)

    mixed = EdgeIR(tag="S-MIX-OUT")
    g.add_stream(mixed, src_tag="MIX-COL-01", dst_tag="COL-01", enforce_phase=False)

    dist = EdgeIR(tag="DIST", src_port=0)
    g.add_stream(dist, src_tag="COL-01", dst_tag=None, enforce_phase=False)

    bot = EdgeIR(tag="BOT", src_port=1)
    g.add_stream(bot, src_tag="COL-01", dst_tag=None, enforce_phase=False)

    rec = EdgeIR(tag="REC", is_recycle=True, recycle_target="MIX-COL-01")
    g.add_stream(rec, src_tag="COL-01", dst_tag=None, enforce_phase=False)

    return g, rec


def _vals01_graph() -> tuple[FlowsheetGraph, EdgeIR]:
    """
    VALS_01-style graph after _insert_mixers: auto-inserted MIX-CP-01 receives
    a single feed (FEED, Methane=1.0) and the recycle stream VAP wired to it.

      FEED (Methane=1.0) ── MIX-CP-01 ── CP-01
                               ^
                               │  VAP (is_recycle, graph edge to MIX-CP-01)
    """
    g = FlowsheetGraph()
    g.compounds = ["Methane"]
    g.property_package = "Peng-Robinson"

    g.add_unit(MixerNode(tag="MIX-CP-01", metadata={"auto_inserted": True}), strict=False)
    g.add_unit(CompressorNode(tag="CP-01",
                              params={"P_out": 5000000.0, "efficiency": 0.75}), strict=False)

    feed = EdgeIR(tag="FEED", composition={"Methane": 1.0})
    g.add_stream(feed, src_tag=None, dst_tag="MIX-CP-01", enforce_phase=False)

    link = EdgeIR(tag="S-MIX-CP-01-OUT")
    g.add_stream(link, src_tag="MIX-CP-01", dst_tag="CP-01", enforce_phase=False)

    vap = EdgeIR(tag="VAP", is_recycle=True, recycle_target="CP-01")
    g.add_stream(vap, src_tag="CP-01", dst_tag="MIX-CP-01", enforce_phase=False)

    return g, vap


def test_val09_selects_solvent_feed() -> None:
    """Heuristic returns the p-Xylene=1.0 composition, not the Benzene/Ethanol feed."""
    g, rec = _val09_graph()
    comp = _mixer_secondary_feed_composition(rec, g)
    assert comp == {"p-Xylene": 1.0}, (
        f"Expected p-Xylene=1.0, got {comp!r}")
    print("PASS test_val09_selects_solvent_feed")


def test_vals01_single_feed_returns_empty() -> None:
    """Auto-inserted Mixer with only one feed: heuristic yields {}, fallback takes over."""
    g, vap = _vals01_graph()
    comp = _mixer_secondary_feed_composition(vap, g)
    assert comp == {}, f"Expected empty dict, got {comp!r}"

    # Full fallback chain should still return Methane=1.0 unchanged
    fallback = _feed_stream_composition(g)
    assert fallback == {"Methane": 1.0}, f"Feed fallback wrong: {fallback!r}"
    print("PASS test_vals01_single_feed_returns_empty")


def test_non_mixer_target_returns_empty() -> None:
    """Recycle targeting a Compressor directly (no Mixer): heuristic yields {}."""
    g = FlowsheetGraph()
    g.compounds = ["Methane"]
    g.property_package = "Peng-Robinson"

    g.add_unit(CompressorNode(tag="CP-01", params={"P_out": 5e6, "efficiency": 0.75}),
               strict=False)

    feed = EdgeIR(tag="FEED", composition={"Methane": 1.0})
    g.add_stream(feed, src_tag=None, dst_tag="CP-01", enforce_phase=False)

    vap = EdgeIR(tag="VAP", is_recycle=True, recycle_target="CP-01")
    g.add_stream(vap, src_tag="CP-01", dst_tag=None, enforce_phase=False)

    comp = _mixer_secondary_feed_composition(vap, g)
    assert comp == {}, f"Expected empty dict, got {comp!r}"
    print("PASS test_non_mixer_target_returns_empty")


def test_similar_secondary_feed_returns_empty() -> None:
    """Two feeds into a Mixer with L1 < 0.3: no secondary is selected."""
    g = FlowsheetGraph()
    g.compounds = ["Methane", "Ethane"]
    g.property_package = "Peng-Robinson"

    g.add_unit(MixerNode(tag="MX-01"), strict=False)

    feed1 = EdgeIR(tag="FEED1", composition={"Methane": 0.9, "Ethane": 0.1})
    g.add_stream(feed1, src_tag=None, dst_tag="MX-01", enforce_phase=False)

    feed2 = EdgeIR(tag="FEED2", composition={"Methane": 0.85, "Ethane": 0.15})
    g.add_stream(feed2, src_tag=None, dst_tag="MX-01", enforce_phase=False)

    rec = EdgeIR(tag="REC", is_recycle=True, recycle_target="MX-01")
    g.add_stream(rec, src_tag=None, dst_tag=None, enforce_phase=False)

    comp = _mixer_secondary_feed_composition(rec, g)
    # L1(FEED2, FEED1) = |0.85-0.9| + |0.15-0.1| = 0.05 + 0.05 = 0.1 < 0.3
    assert comp == {}, f"Expected empty dict (similar feeds), got {comp!r}"
    print("PASS test_similar_secondary_feed_returns_empty")


if __name__ == "__main__":
    test_val09_selects_solvent_feed()
    test_vals01_single_feed_returns_empty()
    test_non_mixer_target_returns_empty()
    test_similar_secondary_feed_returns_empty()
    print("\nAll tests passed.")
