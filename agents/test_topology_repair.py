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
    """Vessel with a single outlet on the given port/phase (missing the other)."""
    g = FlowsheetGraph()
    g.compounds = ["Water"]
    g.property_package = "Raoult's Law"
    g.add_unit(make_node("Vessel", "V-01"))
    g.add_stream(EdgeIR(tag="FEED", phase="mixed"), None, "V-01")
    g.add_stream(EdgeIR(tag="OUT1", src_port=outlet_port, phase=phase), "V-01", None)
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

    changes = _repair_vessel_outlets(g)
    check(len(changes) == 1 and "added liquid product outlet" in changes[0],
          f"one liquid outlet added; log={changes}")

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
    changes = _repair_vessel_outlets(g)
    check(len(changes) == 1 and "added vapour product outlet" in changes[0],
          f"one vapour outlet added; log={changes}")
    new = [s for s in g.outlet_streams("V-01") if s.metadata.get("synthetic_outlet")][0]
    check(new.src_port == 0 and new.phase == "vapour",
          "synthetic outlet is vapour on port 0")
    check(validate(g).valid, "graph valid after Fix 1 (reverse direction)")


def test_fix1_dwsim_serialisation():
    print("\n[Fix 1] synthetic outlet serialises to DWSIM cleanly (no dangling)")
    g = _vessel_one_outlet(outlet_port=0, phase="vapour")
    _repair_vessel_outlets(g)
    new = [s for s in g.outlet_streams("V-01") if s.metadata.get("synthetic_outlet")][0]
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
    again = _repair_vessel_outlets(g)
    check(again == [], "no changes on second run")
    check(len(g.outlet_streams("V-01")) == 2, "still exactly 2 outlets (no duplicate)")


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
    test_fix2_repropagates_dropped_recycle()
    test_fix2_does_not_invent_recycle()
    test_fix2_idempotent()
    test_fix3_detects_missing_units()
    print(f"\n{'='*60}\nRESULT: {_passed} passed, {_failed} failed\n{'='*60}")
    raise SystemExit(1 if _failed else 0)


if __name__ == "__main__":
    main()
