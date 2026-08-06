"""ir/normalise.py deterministic topology tests (zero LLM).

Covers the substantial connectivity logic that turns an assembled graph into a
connected, correct one: mixer insertion for a multi-inlet unit, splitter insertion
for a multi-outlet unit (recycle-aware), separator-outlet completion for a
one-outlet Column, and the name-based port-repair heuristics for Vessel and Column.
Direct assertions on each pass, with no-op cases, plus one normalise() composition
check. Units are added with strict=False — we exercise normalisation, not
construction validation.

Run: PYTHONPATH=. python3.9 -m ir.test_normalise
     (run as a module — ir/types.py shadows stdlib `types` if run as a script)
"""
from ir.graph import FlowsheetGraph, EdgeIR, make_node
from ir.normalise import (
    normalise, _insert_mixers, _insert_splitters,
    _complete_separator_outlets, _repair_vessel_ports, _repair_column_ports,
)


def _graph(units, streams):
    g = FlowsheetGraph()
    g.compounds = ["Water"]
    for tag, ty in units:
        g.add_unit(make_node(ty, tag), strict=False)
    for edge, src, dst in streams:
        g.add_stream(edge, src, dst, enforce_phase=False)
    return g


# ── _insert_mixers ────────────────────────────────────────────────────────────

def test_insert_mixer_for_multi_inlet_unit():
    g = _graph([("HT-01", "Heater")],
               [(EdgeIR(tag="FEED-A"), None, "HT-01"),
                (EdgeIR(tag="FEED-B"), None, "HT-01")])
    out = _insert_mixers(g)
    mix = out.unit("MIX-HT-01")
    assert mix is not None and mix.unit_type == "Mixer"
    assert mix.metadata.get("auto_inserted") is True
    # both feeds now enter the mixer; the heater has a single (link) inlet
    assert {s.tag for s in out.inlet_streams("MIX-HT-01")} == {"FEED-A", "FEED-B"}
    assert [s.tag for s in out.inlet_streams("HT-01")] == ["S-MIX-HT-01-OUT"]
    print("OK: multi-inlet unit gets an upstream Mixer, feeds rerouted")


def test_no_mixer_for_single_inlet():
    g = _graph([("CL-01", "Cooler")], [(EdgeIR(tag="FEED"), None, "CL-01")])
    assert _insert_mixers(g).unit("MIX-CL-01") is None
    print("OK: single-inlet unit gets no Mixer")


# ── _insert_splitters ─────────────────────────────────────────────────────────

def test_insert_splitter_for_multi_outlet_unit():
    g = _graph([("CL-01", "Cooler")],
               [(EdgeIR(tag="FEED"), None, "CL-01"),
                (EdgeIR(tag="PROD-1"), "CL-01", None),
                (EdgeIR(tag="PROD-2"), "CL-01", None)])
    out = _insert_splitters(g)
    spl = out.unit("SPL-CL-01")
    assert spl is not None and spl.unit_type == "Splitter"
    assert spl.params["split_fractions"] == {"PROD-1": 0.5, "PROD-2": 0.5}
    # products now leave the splitter with distinct ports; cooler keeps one link outlet
    assert {s.tag for s in out.outlet_streams("SPL-CL-01")} == {"PROD-1", "PROD-2"}
    assert {s.src_port for s in out.outlet_streams("SPL-CL-01")} == {0, 1}
    assert [s.tag for s in out.outlet_streams("CL-01")] == ["S-CL-01-SPL"]
    print("OK: multi-outlet unit gets a downstream Splitter, equal split fractions")


def test_no_splitter_for_vessel_two_outlets():
    # a Vessel legitimately has 2 outlet ports (vapour + liquid) → no splitter
    g = _graph([("V-01", "Vessel")],
               [(EdgeIR(tag="FEED"), None, "V-01"),
                (EdgeIR(tag="VAP", src_port=0), "V-01", None),
                (EdgeIR(tag="LIQ", src_port=1), "V-01", None)])
    assert _insert_splitters(g).unit("SPL-V-01") is None
    print("OK: Vessel's two outlets do not trigger a Splitter")


def test_recycle_outlet_excluded_from_splitter_count():
    # one product + one recycle outlet → non-recycle count is 1 → no splitter
    g = _graph([("CL-01", "Cooler")],
               [(EdgeIR(tag="FEED"), None, "CL-01"),
                (EdgeIR(tag="PROD"), "CL-01", None),
                (EdgeIR(tag="REC", is_recycle=True, recycle_target="CL-01"),
                 "CL-01", None)])
    assert _insert_splitters(g).unit("SPL-CL-01") is None
    print("OK: recycle outlet is excluded from the splitter count")


# ── _complete_separator_outlets ───────────────────────────────────────────────

def test_complete_missing_column_bottoms():
    g = _graph([("COL-01", "Column")],
               [(EdgeIR(tag="FEED"), None, "COL-01"),
                (EdgeIR(tag="DIST", src_port=0), "COL-01", None)])
    out = _complete_separator_outlets(g)
    outs = out.outlet_streams("COL-01")
    assert len(outs) == 2
    synth = [s for s in outs if s.tag != "DIST"][0]
    assert "BOT" in synth.tag.upper()             # the missing product is the bottoms
    assert synth.src_port == 1                     # free port (DIST held 0)
    assert out.stream_dest(synth.tag) is None      # terminated boundary product
    print("OK: one-outlet Column gets its missing bottoms product synthesised")


def test_no_completion_when_column_complete():
    g = _graph([("COL-01", "Column")],
               [(EdgeIR(tag="FEED"), None, "COL-01"),
                (EdgeIR(tag="DIST", src_port=0), "COL-01", None),
                (EdgeIR(tag="BOT", src_port=1), "COL-01", None)])
    assert len(_complete_separator_outlets(g).outlet_streams("COL-01")) == 2
    print("OK: a complete Column is left untouched")


# ── port-repair heuristics ────────────────────────────────────────────────────

def test_vessel_ports_swapped_when_inverted():
    # liquid-named stream on port 0, vapour-named on port 1 → swap + phase set
    g = _graph([("V-01", "Vessel")],
               [(EdgeIR(tag="FEED"), None, "V-01"),
                (EdgeIR(tag="LIQUID", src_port=0), "V-01", None),
                (EdgeIR(tag="VAPOUR", src_port=1), "V-01", None)])
    out = _repair_vessel_ports(g)
    assert out.stream("LIQUID").src_port == 1 and out.stream("LIQUID").phase == "liquid"
    assert out.stream("VAPOUR").src_port == 0 and out.stream("VAPOUR").phase == "vapour"
    print("OK: Vessel ports swapped when names indicate inversion")


def test_vessel_ports_untouched_when_correct():
    g = _graph([("V-01", "Vessel")],
               [(EdgeIR(tag="FEED"), None, "V-01"),
                (EdgeIR(tag="VAPOUR", src_port=0), "V-01", None),
                (EdgeIR(tag="LIQUID", src_port=1), "V-01", None)])
    out = _repair_vessel_ports(g)
    assert out.stream("VAPOUR").src_port == 0 and out.stream("LIQUID").src_port == 1
    print("OK: correctly-ordered Vessel ports are left unchanged")


def test_column_ports_assigned_by_name():
    # bottoms emitted first (port 0), distillate second (port 1) → reassign by name
    g = _graph([("COL-01", "Column")],
               [(EdgeIR(tag="FEED"), None, "COL-01"),
                (EdgeIR(tag="BOTTOMS", src_port=0), "COL-01", None),
                (EdgeIR(tag="DISTILLATE", src_port=1), "COL-01", None)])
    out = _repair_column_ports(g)
    assert out.stream("DISTILLATE").src_port == 0
    assert out.stream("BOTTOMS").src_port == 1
    print("OK: Column products reassigned distillate→0, bottoms→1 by name")


# ── normalise() composition ───────────────────────────────────────────────────

def test_normalise_composes_passes():
    g = _graph([("CL-01", "Cooler")],
               [(EdgeIR(tag="FEED"), None, "CL-01"),
                (EdgeIR(tag="PROD-1"), "CL-01", None),
                (EdgeIR(tag="PROD-2"), "CL-01", None)])
    out = normalise(g)
    assert out.unit("SPL-CL-01") is not None      # splitter inserted end-to-end
    assert out.compounds == ["Water"]             # compounds survive the deepcopy
    print("OK: normalise() composes the passes (splitter inserted, compounds kept)")


def _run_all():
    test_insert_mixer_for_multi_inlet_unit()
    test_no_mixer_for_single_inlet()
    test_insert_splitter_for_multi_outlet_unit()
    test_no_splitter_for_vessel_two_outlets()
    test_recycle_outlet_excluded_from_splitter_count()
    test_complete_missing_column_bottoms()
    test_no_completion_when_column_complete()
    test_vessel_ports_swapped_when_inverted()
    test_vessel_ports_untouched_when_correct()
    test_column_ports_assigned_by_name()
    test_normalise_composes_passes()
    print("\nALL NORMALISE TESTS PASSED")


if __name__ == "__main__":
    _run_all()
