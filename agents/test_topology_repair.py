"""
Host-side unit tests for the deterministic topology-repair helpers in
agents/graph_pipeline.py (Fix 1 / Fix 2 / Fix 3).

These exercise the pure, module-level helpers directly against hand-built
FlowsheetGraphs — NO LangGraph, NO Docker/DWSIM, NO LLM — so the mechanical
correctness of the repair node can be verified locally.  The full pipeline
checkpoints (USE_LANGGRAPH=1 over the benchmark tiers) run on HPC.

Run: PYTHONPATH=. python3.9 agents/test_topology_repair.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ir.graph import FlowsheetGraph, EdgeIR, make_node
from ir import validate, to_dwsim
from agents.graph_pipeline import (
    _repair_vessel_outlets,
    _repropagate_recycles,
    _detect_missing_units,
    _is_vessel_outlet_issue,
    _is_cycle_issue,
    _feed_is_superheated_vapour,
    _vessel_feed_conditions,
    _dew_point_upper_bound,
    _SUPERHEAT_MARGIN_K,
)

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


# ── Minimal SemanticStream/Topology stand-ins for Fix 3 ────────────────────────

@dataclass
class _SemStream:
    tag: str
    src: Optional[str]
    dst: Optional[str]


@dataclass
class _SemTopo:
    streams: list = field(default_factory=list)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _vessel_one_outlet(outlet_port: int, phase: str) -> FlowsheetGraph:
    """Vessel with a single outlet on the given port/phase (missing the other).

    Feed carries no T, so the physical guard cannot confirm single-phase →
    the default behaviour (add the missing outlet) applies.
    """
    g = FlowsheetGraph()
    g.compounds = ["Water"]
    g.property_package = "Raoult's Law"
    g.add_unit(make_node("Vessel", "V-01"))
    g.add_stream(EdgeIR(tag="FEED", phase="mixed"), None, "V-01")
    g.add_stream(EdgeIR(tag="OUT1", src_port=outlet_port, phase=phase), "V-01", None)
    return g


def _heater_vessel(
    compounds: list[str],
    t_out_K:   Optional[float] = None,
    feed_T:    Optional[float] = 298.15,
) -> FlowsheetGraph:
    """Heater → Vessel with a single (vapour) outlet.

    Two modes:
      • t_out_K given   → Heater carries a T_out param (tests the param path).
      • t_out_K is None → REALISTIC topology-repair state: Heater params empty
        (ParamMapper has not run), the internal Heater→Vessel stream carries no T
        (StreamExtractor only sets T on FEED streams), only the FEED stream has T.
        The guard must then recover the feed T from the description.
    """
    g = FlowsheetGraph()
    g.compounds = list(compounds)
    g.property_package = "NRTL"
    params = {"T_out": t_out_K} if t_out_K is not None else {}
    g.add_unit(make_node("Heater", "HT-01", params=params))
    g.add_unit(make_node("Vessel", "V-01"))
    g.add_stream(EdgeIR(tag="FEED", phase="mixed", T=feed_T), None, "HT-01")
    g.add_stream(EdgeIR(tag="S_HV", phase="mixed"), "HT-01", "V-01")  # internal: no T
    g.add_stream(EdgeIR(tag="VAP", src_port=0, phase="vapour"), "V-01", None)
    return g


def _two_unit_cycle() -> FlowsheetGraph:
    """Heater A → Cooler B → (untagged recycle) → A.  A cycle until tagged."""
    g = FlowsheetGraph()
    g.compounds = ["Water"]
    g.property_package = "Raoult's Law"
    g.add_unit(make_node("Heater", "A"))
    g.add_unit(make_node("Cooler", "B"))
    g.add_stream(EdgeIR(tag="FEED", phase="mixed"), None, "A")
    g.add_stream(EdgeIR(tag="S_AB", phase="mixed"), "A", "B")
    # Recycle B → A, but is_recycle was dropped (=False) → creates a cycle.
    g.add_stream(EdgeIR(tag="S_REC", phase="mixed", is_recycle=False), "B", "A")
    return g


# ── Fix 1: Vessel single-outlet repair ─────────────────────────────────────────

def test_fix1_adds_liquid_when_vapour_present():
    print("\n[Fix 1] vapour outlet present (port 0) → add liquid (port 1)")
    g = _vessel_one_outlet(outlet_port=0, phase="vapour")

    rep = validate(g)
    check(not rep.valid, "graph invalid before repair (Vessel has 1 outlet)")
    check(any(_is_vessel_outlet_issue(i) for i in rep.errors()),
          "vessel-outlet pattern detected by _is_vessel_outlet_issue")

    changes, suppressed = _repair_vessel_outlets(g)
    check(any("added liquid product outlet" in c for c in changes),
          f"liquid outlet added; log={changes}")
    check(suppressed == [], "nothing suppressed (phase unknown → default add)")

    outs = g.outlet_streams("V-01")
    check(len(outs) == 2, "Vessel now has exactly 2 outlets")
    ports = sorted(s.src_port for s in outs)
    check(ports == [0, 1], f"outlets on distinct ports 0 and 1 (got {ports})")
    new = [s for s in outs if s.metadata.get("synthetic_outlet")][0]
    check(new.src_port == 1 and new.phase == "liquid",
          "synthetic outlet is liquid on port 1")
    check(g.stream_dest(new.tag) is None and g.stream_source(new.tag) == "V-01",
          "synthetic outlet is a terminal product stream (src=V-01, dst=None)")
    check(new in g.product_streams(), "synthetic outlet registered as a product stream")

    rep2 = validate(g)
    check(rep2.valid, "graph valid after Fix 1")


def test_fix1_adds_vapour_when_liquid_present():
    print("\n[Fix 1] liquid outlet present (port 1) → add vapour (port 0)")
    g = _vessel_one_outlet(outlet_port=1, phase="liquid")
    changes, _ = _repair_vessel_outlets(g)
    check(any("added vapour product outlet" in c for c in changes),
          f"vapour outlet added; log={changes}")
    new = [s for s in g.outlet_streams("V-01") if s.metadata.get("synthetic_outlet")][0]
    check(new.src_port == 0 and new.phase == "vapour",
          "synthetic outlet is vapour on port 0")
    check(validate(g).valid, "graph valid after Fix 1 (reverse direction)")


def test_fix1_dwsim_serialisation():
    print("\n[Fix 1] synthetic outlet serialises to DWSIM cleanly (no dangling)")
    g = _vessel_one_outlet(outlet_port=0, phase="vapour")
    _repair_vessel_outlets(g)
    outs = g.outlet_streams("V-01")
    new = [s for s in outs if s.metadata.get("synthetic_outlet")][0]
    d = to_dwsim(g)
    tags = {s["tag"] for s in d["streams"]}
    check(new.tag in tags, "synthetic outlet present in DWSIM streams")
    conns_with_new = [c for c in d["connections"] if new.tag in c]
    check(len(conns_with_new) == 1,
          f"exactly one connection references it (got {conns_with_new})")
    check(conns_with_new[0] == ["V-01", new.tag, 1, 0],
          f"connection is [V-01, {new.tag}, 1, 0] — unit→stream, no dangling end")


def test_fix1_idempotent():
    print("\n[Fix 1] idempotent — second entry is a no-op")
    g = _vessel_one_outlet(outlet_port=0, phase="vapour")
    _repair_vessel_outlets(g)
    again, _ = _repair_vessel_outlets(g)
    check(again == [], "no changes on second run")
    check(len(g.outlet_streams("V-01")) == 2, "still exactly 2 outlets (no duplicate)")


# ── Fix 1 physical guard: single-phase superheated vapour ──────────────────────
#
# The three core cases that pin the asymmetry (superheat → fail; two-phase and
# subcooled-near-bubble → add outlet) are driven from the DESCRIPTION on a
# realistic graph (Heater params empty, internal stream T unset) — exactly the
# state at topology_repair time, before ParamMapper/consistency run.

# PERT_HARD01_T+80 description fragment (heated 80 K above azeotrope BP).
_SUPERHEAT_DESC = ("Process an ethanol-water feed by heating to 158°C — 80K "
                   "above the azeotrope boiling point; entirely superheated vapour.")
_TWO_PHASE_DESC = "Heat the ethanol-water feed to 78°C and flash in a separator."
_SUBCOOLED_DESC = "Heat the ethanol-water feed to only 68°C and flash."  # T-10 analogue


def test_guard_dew_point_is_populated():
    print("\n[Fix 1 guard] dew-point upper bound is actually computable (not None)")
    dew = _dew_point_upper_bound(["Ethanol", "Water"], 101_325.0)
    check(dew is not None and 372.0 < dew < 374.5,
          f"dew upper bound for ethanol-water ≈ water BP (got {dew})")
    check(_dew_point_upper_bound(["Ethanol", "Unobtainium"], 101_325.0) is None,
          "returns None when a compound lacks vapour-pressure data")


def test_guard_suppresses_superheated_from_description():
    print("\n[Fix 1 guard] REALISTIC: superheat recovered from DESCRIPTION → suppress")
    # Heater params empty + internal stream has no T — the real topology_repair
    # state.  The guard must derive 431.15 K from the description, not params.
    g = _heater_vessel(["Ethanol", "Water"])  # no T_out param
    check(g.unit("HT-01").params == {}, "precondition: Heater T_out is NOT set yet")
    check(g.stream("S_HV").T is None, "precondition: internal stream carries no T")

    feed_T, _ = _vessel_feed_conditions(g, g.unit("V-01"), _SUPERHEAT_DESC)
    check(feed_T == 431.15, f"feed T (158°C) recovered from description (got {feed_T})")

    changes, suppressed = _repair_vessel_outlets(g, _SUPERHEAT_DESC)
    check(suppressed == ["V-01"], f"vessel suppressed; got {suppressed}")
    check(any("NOT adding a second outlet" in c for c in changes),
          f"suppression logged; log={changes}")
    outs = g.outlet_streams("V-01")
    check(len(outs) == 1 and not any(s.metadata.get("synthetic_outlet") for s in outs),
          "no liquid outlet fabricated; vessel keeps its single outlet")
    check(not validate(g).valid, "graph remains INVALID (→ routes to END)")


def test_guard_allows_two_phase_from_description():
    print("\n[Fix 1 guard] two-phase (78°C) from description → outlet added")
    g = _heater_vessel(["Ethanol", "Water"])
    feed_T, _ = _vessel_feed_conditions(g, g.unit("V-01"), _TWO_PHASE_DESC)
    check(feed_T == 351.15, f"feed T (78°C) recovered (got {feed_T})")
    changes, suppressed = _repair_vessel_outlets(g, _TWO_PHASE_DESC)
    check(suppressed == [], "nothing suppressed for a two-phase feed")
    check(len(g.outlet_streams("V-01")) == 2, "vessel now has 2 outlets")
    check(validate(g).valid, "graph valid after two-phase repair")


def test_guard_allows_subcooled_near_bubble():
    print("\n[Fix 1 guard] subcooled-near-bubble (68°C, T-10 analogue) → outlet added")
    # Pins the asymmetry: a subcooled feed is recoverable by the consistency pass
    # downstream, so it must NOT be suppressed — it gets its outlet like any
    # two-phase case.  A future change that suppresses this would break recovery.
    g = _heater_vessel(["Ethanol", "Water"])
    feed_T, _ = _vessel_feed_conditions(g, g.unit("V-01"), _SUBCOOLED_DESC)
    check(feed_T == 341.15, f"feed T (68°C) recovered (got {feed_T})")
    check(not _feed_is_superheated_vapour(g.compounds, feed_T, None),
          "subcooled feed is NOT treated as superheated vapour")
    changes, suppressed = _repair_vessel_outlets(g, _SUBCOOLED_DESC)
    check(suppressed == [], "subcooled feed not suppressed (stays recoverable)")
    check(len(g.outlet_streams("V-01")) == 2, "outlet added (recoverable case)")
    check(validate(g).valid, "graph valid (recoverable, proceeds downstream)")


def test_guard_margin_only_unambiguous_superheat():
    print(f"\n[Fix 1 guard] margin = {_SUPERHEAT_MARGIN_K} K — borderline superheat NOT suppressed")
    dew = _dew_point_upper_bound(["Ethanol", "Water"], 101_325.0)
    just_above = dew + (_SUPERHEAT_MARGIN_K - 2.0)   # within margin → not suppressed
    well_above = dew + (_SUPERHEAT_MARGIN_K + 5.0)   # clears margin → suppressed
    check(not _feed_is_superheated_vapour(["Ethanol", "Water"], just_above, None),
          f"T={just_above:.1f} K (within margin of dew {dew:.1f}) → not superheated")
    check(_feed_is_superheated_vapour(["Ethanol", "Water"], well_above, None),
          f"T={well_above:.1f} K (clears margin) → superheated")


def test_guard_surfaces_when_unevaluable():
    print("\n[Fix 1 guard] cannot evaluate (no feed T) → surfaced WARNING, outlet added")
    g = _vessel_one_outlet(outlet_port=0, phase="vapour")  # feed has no T, no desc
    changes, suppressed = _repair_vessel_outlets(g, description="")
    check(suppressed == [], "not suppressed when guard cannot evaluate")
    check(any("could not be evaluated" in c and "WARNING" in c for c in changes),
          f"inability-to-evaluate is surfaced as a WARNING; log={changes}")
    check(len(g.outlet_streams("V-01")) == 2, "outlet still added (default behaviour)")


def _heater_vessel_two_outlet(compounds: list[str]) -> FlowsheetGraph:
    """Heater → Vessel WITH BOTH outlets — the SAN03_T+80 shape.

    The LLM extracts two outlets for a 'flash' even when the feed turns out
    superheated; DWSIM then solves it with zero liquid (single_phase_vapor_ok).
    """
    g = FlowsheetGraph()
    g.compounds = list(compounds)
    g.property_package = "NRTL"
    g.add_unit(make_node("Heater", "HT-01", params={}))
    g.add_unit(make_node("Vessel", "V-01"))
    g.add_stream(EdgeIR(tag="FEED", phase="mixed", T=298.15), None, "HT-01")
    g.add_stream(EdgeIR(tag="S_HV", phase="mixed"), "HT-01", "V-01")
    g.add_stream(EdgeIR(tag="VAP", src_port=0, phase="vapour"), "V-01", None)
    g.add_stream(EdgeIR(tag="LIQ", src_port=1, phase="liquid"), "V-01", None)
    return g


def test_san03_two_outlet_superheated_passes_validation():
    print("\n[SAN03] 2-outlet superheated vessel → VALID → guard never involved")
    # PERT_SAN03_T+80 mirror: benzene-toluene, superheated, but extracted as a
    # 2-outlet flash.  It must stay VALID so _validate_node routes to thermo
    # (never topology_repair) and it converges with zero liquid — i.e. it passes,
    # exactly as in v2.  The distinguishing signal vs HARD01 is the outlet count.
    g = _heater_vessel_two_outlet(["Benzene", "Toluene"])
    rep = validate(g)
    check(rep.valid, "2-outlet vessel is VALID even with a superheated feed")
    check(not any(_is_vessel_outlet_issue(i) for i in rep.errors()),
          "no vessel-outlet issue → _validate_node routes to thermo, not topology_repair")
    # Even if the guard were invoked, it must be a no-op on a 2-outlet vessel.
    changes, suppressed = _repair_vessel_outlets(g, _SUPERHEAT_DESC)
    check(suppressed == [] and changes == [],
          "guard is a no-op on a 2-outlet vessel (len(outlets)!=1 → skipped)")
    check(len(g.outlet_streams("V-01")) == 2, "vessel keeps its two outlets")


def test_guard_surfaces_when_dew_unknown():
    print("\n[Fix 1 guard] cannot evaluate (no dew data) → surfaced WARNING, outlet added")
    # Compound absent from the vapour-pressure tables → dew bound is None even
    # though a feed T is available; must surface rather than silently default.
    g = _heater_vessel(["Ethanol", "Unobtainium"], t_out_K=500.0)
    changes, suppressed = _repair_vessel_outlets(g, description="")
    check(suppressed == [], "not suppressed when dew point is unknown")
    check(any("could not be evaluated" in c and "no dew-point data" in c
              for c in changes),
          f"missing dew data is surfaced as a WARNING; log={changes}")
    check(len(g.outlet_streams("V-01")) == 2, "outlet still added (default behaviour)")


# ── Fix 2: Cycle-vs-recycle flag repropagation ─────────────────────────────────

def test_fix2_repropagates_dropped_recycle():
    print("\n[Fix 2] repropagate a recycle flag a guard dropped → DAG valid")
    g = _two_unit_cycle()
    origin = {"S_REC": {"is_recycle": True, "recycle_target": "A",
                        "dropped_by": "guard3-phrase"}}

    rep = validate(g)
    check(not rep.valid, "graph invalid before repair (untagged cycle)")
    check(any(_is_cycle_issue(i) for i in rep.errors()),
          "cycle pattern detected by _is_cycle_issue")

    changes = _repropagate_recycles(g, origin)
    check(len(changes) == 1 and "repropagated is_recycle on S_REC" in changes[0]
          and "guard3-phrase" in changes[0],
          f"S_REC re-tagged with dropped stage logged; log={changes}")
    edge = g.stream("S_REC")
    check(edge.is_recycle and edge.recycle_target == "A",
          "S_REC.is_recycle=True, recycle_target='A'")
    check(validate(g).valid, "graph valid after Fix 2 (recycle excluded from DAG)")


def test_fix2_does_not_invent_recycle():
    print("\n[Fix 2] negative — never invent a recycle that was never set")
    g = _two_unit_cycle()
    origin = {"S_REC": {"is_recycle": False, "recycle_target": None,
                        "dropped_by": None}}
    changes = _repropagate_recycles(g, origin)
    check(changes == [], "no re-tag when original flag was False")
    check(not g.stream("S_REC").is_recycle, "S_REC remains non-recycle")
    check(not validate(g).valid, "graph still invalid (correctly not auto-fixed)")


def test_fix2_idempotent():
    print("\n[Fix 2] idempotent — second entry is a no-op")
    g = _two_unit_cycle()
    origin = {"S_REC": {"is_recycle": True, "recycle_target": "A",
                        "dropped_by": "guard3-phrase"}}
    _repropagate_recycles(g, origin)
    again = _repropagate_recycles(g, origin)
    check(again == [], "no changes on second run (already tagged)")


# ── Fix 3: Missing-unit reference (flag only) ──────────────────────────────────

def test_fix3_detects_missing_units():
    print("\n[Fix 3] detect (not repair) streams referencing a missing unit")
    g = FlowsheetGraph()
    g.compounds = ["Water"]
    g.add_unit(make_node("Heater", "A"))
    sem = _SemTopo(streams=[
        _SemStream("FEED", None, "A"),       # ok
        _SemStream("S1", "A", "GHOST"),       # dst missing
        _SemStream("S2", "PHANTOM", "A"),     # src missing
    ])
    missing = _detect_missing_units(g, sem)
    tags = {(m["missing_tag"], m["role"]) for m in missing}
    check(("GHOST", "dst") in tags, "missing dst 'GHOST' flagged")
    check(("PHANTOM", "src") in tags, "missing src 'PHANTOM' flagged")
    check(len(missing) == 2, f"exactly the two missing refs flagged (got {missing})")
    check(all("A" not in (m["missing_tag"],) for m in missing),
          "the real unit 'A' is never flagged")


def main():
    test_fix1_adds_liquid_when_vapour_present()
    test_fix1_adds_vapour_when_liquid_present()
    test_fix1_dwsim_serialisation()
    test_fix1_idempotent()
    test_guard_dew_point_is_populated()
    test_guard_suppresses_superheated_from_description()
    test_guard_allows_two_phase_from_description()
    test_guard_allows_subcooled_near_bubble()
    test_guard_margin_only_unambiguous_superheat()
    test_guard_surfaces_when_unevaluable()
    test_san03_two_outlet_superheated_passes_validation()
    test_guard_surfaces_when_dew_unknown()
    test_fix2_repropagates_dropped_recycle()
    test_fix2_does_not_invent_recycle()
    test_fix2_idempotent()
    test_fix3_detects_missing_units()
    print(f"\n{'='*60}\nRESULT: {_passed} passed, {_failed} failed\n{'='*60}")
    raise SystemExit(1 if _failed else 0)


if __name__ == "__main__":
    main()
