"""
Offline verification that ablation activation counters fire correctly.

Constructs a minimal IR graph (CoolerNode → PumpNode, two CONDITION_FIX errors)
and runs one call to BeamRepairSearch.search(), then checks:

  n_bp_calls > 0            (bubble_point_K called: probe + per-step)
  coupling-map probe non-empty (the synthetic Cooler→Pump relation exists)

The beam-search counters are also printed.  A zero coupling-query count in this
small search is not treated as proof that coupling is broken because the search
may terminate before a second target-selection step.

Run on the host without Docker or LLM:
  PYTHONPATH=. python3.9 scripts/verification/verify_ablation_counters.py
"""
import json
import sys

PASS = True


def _fail(msg: str) -> None:
    global PASS
    print(f"  FAIL: {msg}", flush=True)
    PASS = False


def _ok(msg: str) -> None:
    print(f"  ok:   {msg}", flush=True)


# ── Minimal graph construction ────────────────────────────────────────────────

from ir.graph import FlowsheetGraph, EdgeIR, CoolerNode, PumpNode
from ir.types import (
    SimError, RepairStrategy, ErrorType, ErrorTarget,
    ErrorSeverity, TargetKind,
)


def _make_graph() -> FlowsheetGraph:
    g = FlowsheetGraph()
    g.compounds        = ["Ethanol", "Water"]
    g.property_package = "NRTL"

    g.add_unit(CoolerNode(tag="CL-01", params={"T_out": 273.15}), strict=False)
    g.add_unit(PumpNode(tag="PM-01",   params={"P_out": 506_625.0}), strict=False)

    # FEED → CL-01
    g.add_stream(
        EdgeIR(tag="FEED", T=298.15, P=101_325.0, flow=1.0,
               composition={"Ethanol": 0.5, "Water": 0.5}, phase="mixed"),
        src_tag=None, dst_tag="CL-01",
    )
    # CL-01 → PM-01  (liquid so PumpNode physics path activates)
    g.add_stream(
        EdgeIR(tag="S1", T=273.15, P=101_325.0, flow=1.0,
               composition={"Ethanol": 0.5, "Water": 0.5}, phase="liquid"),
        src_tag="CL-01", dst_tag="PM-01",
        enforce_phase=False,   # CoolerNode outlet is "any" — no mismatch, but skip check
    )
    # PM-01 → (outlet)
    g.add_stream(
        EdgeIR(tag="S2", T=273.15, P=506_625.0, flow=1.0,
               composition={"Ethanol": 0.5, "Water": 0.5}, phase="liquid"),
        src_tag="PM-01", dst_tag=None,
        enforce_phase=False,
    )
    return g


def _make_errors() -> list[SimError]:
    cooler_err = SimError(
        error_type      = ErrorType.INVALID_UNIT_CONFIG,
        target          = ErrorTarget(TargetKind.UNIT, "CL-01"),
        evidence        = "T_out=273.15 K is below bubble point of feed mixture",
        repair_strategy = RepairStrategy.CONDITION_FIX,
        severity        = ErrorSeverity.CRITICAL,
    )
    pump_err = SimError(
        error_type      = ErrorType.INVALID_UNIT_CONFIG,
        target          = ErrorTarget(TargetKind.UNIT, "PM-01"),
        evidence        = "P_out=506625.0 Pa is unphysical for current feed conditions",
        repair_strategy = RepairStrategy.CONDITION_FIX,
        severity        = ErrorSeverity.CRITICAL,
    )
    return [cooler_err, pump_err]


# ── Run the beam search ───────────────────────────────────────────────────────

print("Building minimal graph (CL-01 → PM-01, 2 CONDITION_FIX errors)...")
try:
    graph  = _make_graph()
    errors = _make_errors()
    print(f"  compounds={graph.compounds}  units={[u.tag for u in graph.units()]}")
except Exception as exc:
    print(f"Graph construction failed: {exc}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print("Running BeamRepairSearch.search() (depth=2, width=3, no LLM)...")
from agents.stage4.beam_search import BeamRepairSearch

searcher = BeamRepairSearch(width=3, depth=2, run_local_opt=False)
try:
    _, changes = searcher.search(graph, errors, llm_agent=None)
except Exception as exc:
    print(f"search() raised: {exc}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── Parse ABLATION_STATS_LOG ──────────────────────────────────────────────────

abl = None
for c in changes:
    if isinstance(c, str) and c.startswith("ABLATION_STATS_LOG:"):
        try:
            abl = json.loads(c[len("ABLATION_STATS_LOG:"):])
        except Exception as exc:
            _fail(f"ABLATION_STATS_LOG parse error: {exc}")
        break

if abl is None:
    _fail("ABLATION_STATS_LOG entry not found in returned changes")
    print(f"  changes returned ({len(changes)} items):")
    for i, c in enumerate(changes):
        print(f"    [{i}] {str(c)[:120]}")
    sys.exit(1)

print(f"Parsed ABLATION_STATS_LOG: {json.dumps(abl, indent=2)}")

# ── Assertions ────────────────────────────────────────────────────────────────

bp = abl.get("n_bp_calls", 0)
if bp > 0:
    _ok(f"n_bp_calls={bp} > 0  (bubble_point_K called at probe + per beam step)")
else:
    _fail(f"n_bp_calls={bp} — bubble_point_K was not called")

cq = abl.get("n_coupling_queries", 0)
_ok(f"n_coupling_queries={cq}  (observed beam-search queries)")

ph = abl.get("n_phase_candidates", 0)
_ok(f"n_phase_candidates={ph}  (physics path: Cooler→Pump bp-margin candidates)")

sel = abl.get("phase_candidate_selected", False)
_ok(f"phase_candidate_selected={sel}")

nonempty = abl.get("nonempty_boosts", [])
_ok(f"nonempty_boosts count={len(nonempty)}")

# Verify the coupling rule itself with a relation that must be non-empty.  This
# catches a missing/incorrect coupling map without pretending that this tiny
# beam search necessarily reaches a second target-selection step.
from ir.coupling import ParameterCouplingMap
probe = ParameterCouplingMap().get_coupled_boosts(
    graph, "CL-01", "T_out", {"PM-01"})
if probe.get("PM-01", 0.0) > 0:
    _ok(f"coupling-map probe={probe}  (Cooler.T_out→Pump is active)")
else:
    _fail(f"coupling-map probe returned {probe}; expected a PM-01 boost")

# ── All changes for audit ─────────────────────────────────────────────────────

print(f"\nAll changes ({len(changes)} items):")
for i, c in enumerate(changes):
    print(f"  [{i}] {str(c)[:120]}")

# ── Summary ───────────────────────────────────────────────────────────────────

print()
if PASS:
    print("PASS — all assertions satisfied")
else:
    print("FAIL — one or more assertions failed (see above)")
    sys.exit(1)
