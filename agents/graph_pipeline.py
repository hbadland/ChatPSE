"""
agents/graph_pipeline.py — LangGraph scaffold for the v2 flowsheet pipeline.

Phase 1: wraps existing agent calls in a StateGraph that reproduces
         orchestrator_v2.py results identically.  No new logic.

Phase 1 constraint: _topology_node always uses UnitExtractor + StreamExtractor.
TopologyChain (the 4-call LangChain path) is deliberately excluded here so the
checkpoint run is a clean apples-to-apples comparison against v2.  It will be
wired in during Phase 4 once the scaffold is proven equivalent.

Enable via: USE_LANGGRAPH=1 (checked in orchestrator_v2.py)
"""
from __future__ import annotations

import hashlib
import json
import operator
import os
import sys
import time
import traceback
from typing import Annotated, Any, Optional

import networkx as nx

# ── LangGraph availability ────────────────────────────────────────────────────
try:
    from langgraph.graph import StateGraph, END
    from typing import TypedDict
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    from typing import TypedDict  # still available in stdlib

# ── Agent imports ─────────────────────────────────────────────────────────────
from agents.basis import BasisAgent
from agents.executor import Executor
from agents.llm import DEFAULT_MODEL
from agents.stage1 import UnitExtractor, StreamExtractor
from agents.stage1.unit_extractor import SemanticUnits, SemanticUnit
from agents.stage1.stream_extractor import SemanticTopology, SemanticStream
# TopologyChain is NOT used in Phase 1 — imported only so GraphPipeline.__init__
# can log its availability, matching OrchestratorV2's startup message.
try:
    from agents.stage1.topology_chain import TopologyChain as _TopologyChain
    _TC_AVAILABLE = True
except ImportError:
    _TopologyChain = None  # type: ignore[assignment,misc]
    _TC_AVAILABLE = False
from agents.stage2 import GraphBuilder
from agents.stage3 import ThermoMapper, ParamMapper
# Reuse ParamMapper's description parsers so the topology_repair phase guard reads
# the SAME temperature/pressure signal ParamMapper will later use for T_out — the
# unit params themselves are not populated until _thermo_node (after this node).
from agents.stage3.param_mapper import _extract_temperatures, _extract_pressures
from agents.stage4 import ErrorClassifier, RepairAgent
from agents.stage4.repair_agent import RepairMemory
from agents.stage4.sim_hints import SimulationHints, EMPTY_HINTS
from agents.rule_store import FailureRuleStore, RULES_PATH
from ir import FlowsheetGraph, normalise, validate, to_dwsim
from ir.consistency import GlobalConsistencyPass
from ir.graph import SeparatorNode, EdgeIR, PORT_SPECS
from ir.thermo_estimation import boiling_point_K
from ir.to_dwsim import _best_ref_stream_by_composition
from ir.margin_model import get_global_margin_model
from ir.types import ErrorType as _ErrorType
from ir.types import RepairStrategy as _RepairStrategy
from rag.retriever import Retriever

# Pull shared helpers and types from orchestrator_v2 so we don't duplicate them.
from agents.orchestrator_v2 import (
    _RECYCLE_PHRASES,
    _SUMMARISER_SYSTEM_TIGHT,
    _inject_reference_reactor_params,
    _no_critic_failures,
    _record_repairs_in_store,
    _reference_guided_refinement,
    _resolve_recycle_target,
    _summarise_for_unit_extraction,
    IterationRecord,
    PipelineResult,
)


# ── Pipeline state ────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    # ── Inputs (set once by run()) ────────────────────────────────────────────
    description:    str
    tier:           str
    reference_file: Optional[str]
    max_iterations: int
    t_start:        float

    # ── Variant B (reference-assisted topology injection — diagnostic) ────────
    variant_b_active:      bool          # VARIANT_B=1 AND a reference_file exists
    topology_source:       Optional[str] # "reference-exact" | "reference-inferred-connections"
    reference_unit_params: dict          # tag → {T_out, P_out, …} from the reference
    variant_b_inferred_feed: bool        # the feed stream was synthesised, not given

    # ── Stage 0 ───────────────────────────────────────────────────────────────
    basis_result:   Optional[Any]
    norm_desc:      str
    compounds:      list[str]

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    sem_units:      Optional[Any]   # SemanticUnits
    sem_topo:       Optional[Any]   # SemanticTopology
    # Original is_recycle/recycle_target per stream tag, captured BEFORE the
    # recycle guards mutate the SemanticStreams.  Used by topology_repair (Fix 2)
    # to repropagate a recycle flag that a guard dropped.
    recycle_origin: dict            # tag → {is_recycle, recycle_target, dropped_by}

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    ir_graph:       Optional[Any]   # FlowsheetGraph
    ir_report:      Optional[Any]   # ValidationReport
    # Streams referencing a unit tag absent from the unit list (Fix 3).
    # Not deterministically repairable — flagged and routed to the LLM repair.
    missing_units:  list[dict]      # [{stream, missing_tag, role}]

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    dwsim_json:     Optional[dict]
    reference_data: Optional[dict]
    tried_packages: list[str]

    # ── Stage 4 loop ─────────────────────────────────────────────────────────
    iteration:      int
    eff_max_iter:   int
    beam_extended:  bool
    repair_memory:  Optional[Any]   # RepairMemory
    sim_hints:      Optional[Any]   # SimulationHints
    execution:      Optional[Any]   # ExecutionResult
    errors:         list[Any]       # list[ClassifiedError]
    prev_hash:      Optional[str]

    # ── Accumulating across nodes (operator.add reducer) ─────────────────────
    warnings:       Annotated[list[str], operator.add]
    iterations_log: Annotated[list[Any], operator.add]

    # ── Routing / output ─────────────────────────────────────────────────────
    outcome:        str


# ── Routing functions (module-level, stateless) ───────────────────────────────

def _route_entry(state: PipelineState) -> str:
    # Variant B (diagnostic): when active, skip the LLM basis+topology nodes
    # entirely and enter at reference_topology.  Otherwise the normal entry.
    return "reference_topology" if state.get("variant_b_active") else "basis"


def _route_basis(state: PipelineState) -> str:
    return END if state["outcome"] == "BASIS_FAILED" else "topology"


def _route_stage1(state: PipelineState) -> str:
    return END if state["outcome"] == "PLAN_FAILED" else "build"


def _route_validate(state: PipelineState) -> str:
    outcome = state["outcome"]
    if outcome == "INVALID_IR":
        return END
    if outcome == "INVALID_TOPOLOGY":
        return "topology_repair"
    return "thermo"


def _route_topology_repair(state: PipelineState) -> str:
    outcome = state["outcome"]
    if outcome == "TOPO_OK":            # graph now valid → continue normally
        return "thermo"
    if outcome == "TOPO_INFEASIBLE":    # single-phase vessel — correctly unrepairable
        return END
    # Unfixed deterministic patterns, or Fix-3 missing units → LLM repair.
    return "repair"


def _route_thermo(state: PipelineState) -> str:
    return END if state["outcome"] in ("PLAN_FAILED", "INVALID_JSON") else "execute"


def _route_execute(state: PipelineState) -> str:
    return END if state["outcome"] in ("PASS", "HUMAN", "MAX_ITER") else "repair"


# ── Deterministic topology-repair helpers (no LLM, pure, unit-testable) ────────
#
# Each helper takes a FlowsheetGraph (and supporting data) and returns a change
# log.  Fix 1 and Fix 2 mutate the graph in place; Fix 3 is detection-only.
# They are deliberately module-level so they can be exercised directly with a
# hand-built graph — without LangGraph, DWSIM, or an LLM.

# Validation evidence fragments that mark a *repairable* topology pattern.
_VESSEL_OUTLET_MARKERS = (
    "Vessel must have exactly 2 outlets",
    "Vessel needs ≥2 outlet(s)",
)
_CYCLE_MARKER = "Cycle detected in non-recycle streams"

# How far above the dew-point upper bound a feed must sit before it is treated as
# unambiguously single-phase superheated vapour.  Wide enough (≈ one flash stage)
# that only clear superheat suppresses the outlet — borderline cases still get it.
_SUPERHEAT_MARGIN_K = 12.0


def _is_vessel_outlet_issue(issue: Any) -> bool:
    """True when the issue is a Vessel that is missing one of its two outlets."""
    err = issue.error
    if err.error_type != _ErrorType.INVALID_TOPOLOGY:
        return False
    return any(m in err.evidence for m in _VESSEL_OUTLET_MARKERS)


def _is_cycle_issue(issue: Any) -> bool:
    err = issue.error
    return (err.error_type == _ErrorType.INVALID_TOPOLOGY
            and _CYCLE_MARKER in err.evidence)


def _dew_point_upper_bound(
    compounds: list[str],
    P:         float,
) -> Optional[float]:
    """
    Rigorous UPPER bound on a mixture's dew point at pressure P: the highest
    pure-component boiling point.  Above it every component's K-value > 1, so no
    liquid phase can exist.  Returns None when the boiling point is unknown for
    ANY compound — in that case the bound is untrustworthy and the superheat
    guard CANNOT be evaluated (the caller must surface this, not silently pass).
    """
    if not compounds:
        return None
    bps = [boiling_point_K(c, P) for c in compounds]
    if any(b is None for b in bps):
        return None
    return max(bps) if bps else None


def _vessel_feed_conditions(
    graph:       FlowsheetGraph,
    vessel:      Any,
    description: str = "",
) -> tuple[Optional[float], Optional[float]]:
    """
    Best-effort (T, P) of a Vessel's feed at topology_repair time.

    Critical ordering note: this node runs BEFORE _thermo_node, so neither
    ParamMapper (which sets upstream Heater T_out) nor the GlobalConsistencyPass
    has run, and StreamExtractor only sets T/P on FEED streams — the Vessel's
    inlet (an internal stream) carries no T.  So we source feed T/P in order:
      1. the inlet stream's own T/P (usually None for internal streams),
      2. the immediate upstream unit's T_out / P_out param (usually unset here),
      3. the DESCRIPTION — replicating ParamMapper's heater/cooler/reactor target
         rule on the immediate upstream unit, so the guard reads the same signal
         ParamMapper will later turn into T_out.
    Returns (None, …) only when no source yields a temperature at all.
    """
    inlets = graph.inlet_streams(vessel.tag)
    feed_T = next((s.T for s in inlets if s.T is not None), None)
    feed_P = next((s.P for s in inlets if s.P is not None), None)

    # 2. immediate upstream unit params (rarely set this early, but cheap to try)
    if feed_T is None or feed_P is None:
        for inlet in inlets:
            src = graph.stream_source(inlet.tag)
            u   = graph.unit(src) if src else None
            if u is None:
                continue
            if feed_T is None and u.params.get("T_out") is not None:
                feed_T = float(u.params["T_out"])
            if feed_P is None and u.params.get("P_out") is not None:
                feed_P = float(u.params["P_out"])

    # 3. description fallback via the immediate upstream unit (no LLM)
    if feed_T is None and description:
        desc_temps = _extract_temperatures(description)
        if desc_temps:
            for inlet in inlets:
                src = graph.stream_source(inlet.tag)
                u   = graph.unit(src) if src else None
                if u is None:
                    continue
                up_feed_T = next(
                    (s.T for s in graph.inlet_streams(u.tag) if s.T is not None),
                    None)
                if u.unit_type == "Heater":
                    cands = [t for t in desc_temps
                             if up_feed_T is None or t > up_feed_T + 1.0]
                    if cands:
                        feed_T = min(cands)   # matches ParamMapper Heater rule
                        break
                elif u.unit_type == "Cooler":
                    cands = [t for t in desc_temps
                             if up_feed_T is None or t < up_feed_T - 1.0]
                    if cands:
                        feed_T = max(cands)
                        break
                elif u.unit_type == "ConversionReactor":
                    feed_T = max(desc_temps)
                    break

    # Pressure fallback from the description (max → never UNDER-estimate P, which
    # would lower the dew bound and risk over-suppression).
    if feed_P is None and description:
        desc_press = _extract_pressures(description)
        if desc_press:
            feed_P = max(desc_press)

    return feed_T, feed_P


def _feed_is_superheated_vapour(
    compounds: list[str],
    feed_T:    Optional[float],
    feed_P:    Optional[float],
) -> bool:
    """
    Positively confirm the feed is single-phase superheated vapour: T clears the
    dew-point upper bound by a wide margin (only unambiguous superheat).  Returns
    False for subcooled / two-phase / unknown / un-evaluable feeds, so the caller
    keeps the default behaviour (add the outlet).
    """
    if feed_T is None or not compounds:
        return False
    P = feed_P if feed_P is not None else 101_325.0
    dew_upper = _dew_point_upper_bound(compounds, P)
    if dew_upper is None:
        return False
    return feed_T > dew_upper + _SUPERHEAT_MARGIN_K


def _repair_vessel_outlets(
    graph:       FlowsheetGraph,
    description: str = "",
) -> tuple[list[str], list[str]]:
    """
    Fix 1 — add the missing phase outlet to any Vessel that has exactly one.

    The missing phase is read from the existing outlet's PORT SPEC (port 0 =
    vapour, port 1 = liquid): if the existing outlet is the vapour port, add the
    liquid port, and vice versa.  The new outlet is a real connected terminal
    product stream (src = vessel, dst = None) — never a dangling port — which
    to_dwsim serialises as a single [vessel, stream, port, 0] connection and
    which is skipped by the mass-balance check (Vessels are excluded there).

    PHYSICAL GUARD: a missing outlet is only added when a second phase can
    actually exist.  If the feed is positively single-phase superheated vapour
    (T above the dew-point upper bound), NO liquid outlet is fabricated — the
    single outlet is physically correct and the Vessel is left as-is (the case
    is then correctly unrepairable and routed to END).  Subcooled / two-phase /
    unknown feeds keep the default behaviour (add the outlet), since an
    under-heated feed is recoverable by the consistency pass downstream.

    This guard fires ONLY on a Vessel that was extracted with exactly ONE outlet
    — the intended-outlet-count signal.  A superheated separator that the LLM
    extracted with TWO outlets (e.g. PERT_SAN03_T+80, "flash the superheated
    vapour") is already VALID, never reaches this node, and converges with zero
    liquid (single_phase_vapor_ok).  A superheated separator extracted with ONE
    outlet (PERT_HARD01_T+80, "produces no liquid") is INVALID and unrepairable
    → END.  This matches v2 exactly: v2 aborts on any 1-outlet Vessel, so
    suppressing here can never turn a v2-pass into a v3-fail.

    Returns (changes, suppressed_vessel_tags).  Idempotent: a Vessel that does
    not have exactly one outlet is left untouched, so re-entering after the fix
    (which yields two outlets) is a no-op.
    """
    changes:    list[str] = []
    suppressed: list[str] = []
    existing_tags = set(graph.stream_tags())
    for node in graph.units():
        if not isinstance(node, SeparatorNode):
            continue
        outlets = graph.outlet_streams(node.tag)
        if len(outlets) != 1:
            continue  # 0 outlets or already-repaired 2 outlets → not our case

        # Physical guard: do not fabricate a second phase that cannot exist.
        feed_T, feed_P = _vessel_feed_conditions(graph, node, description)
        P_eval    = feed_P if feed_P is not None else 101_325.0
        dew_upper = _dew_point_upper_bound(graph.compounds, P_eval)

        if feed_T is None or dew_upper is None:
            # The guard could not be evaluated.  Surface it loudly rather than
            # silently defaulting — a guard reading a None feed-T / dew point is
            # a guard that is not actually protecting anything for this case.
            if dew_upper is None:
                _missing = [c for c in graph.compounds
                            if boiling_point_K(c, P_eval) is None]
                reason = f"no dew-point data for compounds {_missing}"
            else:
                reason = "feed temperature unavailable at topology_repair time"
            warn = (f"[TOPO_REPAIR] WARNING: Vessel {node.tag}: superheat guard "
                    f"could not be evaluated ({reason}) — adding outlet WITHOUT "
                    f"phase confirmation")
            print(warn, flush=True, file=sys.stderr)
            changes.append(warn)
            # fall through to the default add below
        elif feed_T > dew_upper + _SUPERHEAT_MARGIN_K:
            suppressed.append(node.tag)
            changes.append(
                f"[TOPO_REPAIR] Vessel {node.tag}: feed is single-phase "
                f"superheated vapour (T={feed_T:.1f} K > dew≈{dew_upper:.1f} K "
                f"+ {_SUPERHEAT_MARGIN_K:.0f} K) — NOT adding a second outlet "
                f"(no liquid phase exists); case left unrepairable")
            continue

        existing = outlets[0]
        existing_phase = node.outlet_phase(existing.src_port)  # per port spec
        if existing_phase == "vapour":
            missing_port, missing_phase = 1, "liquid"
        else:
            missing_port, missing_phase = 0, "vapour"

        # Idempotency guard: never duplicate a port that is already present.
        if any(s.src_port == missing_port for s in outlets):
            continue

        new_tag = f"{node.tag}_{missing_phase.upper()}OUT"
        _n = 1
        while new_tag in existing_tags:
            _n += 1
            new_tag = f"{node.tag}_{missing_phase.upper()}OUT{_n}"

        edge = EdgeIR(
            tag      = new_tag,
            src_port = missing_port,
            phase    = missing_phase,
            metadata = {"synthetic_outlet": True, "is_product": True},
        )
        # dst=None → terminal product/sink stream (graph.product_streams()).
        graph.add_stream(edge, node.tag, None)
        existing_tags.add(new_tag)
        changes.append(
            f"[TOPO_REPAIR] Vessel {node.tag}: added {missing_phase} "
            f"product outlet {new_tag}")
    return changes, suppressed


def _repropagate_recycles(
    graph:          FlowsheetGraph,
    recycle_origin: dict,
) -> list[str]:
    """
    Fix 2 — restore an is_recycle flag that a recycle guard dropped.

    Traces the cycle that validate_dag() detected (via the unit→unit graph) and,
    for each edge in it whose ORIGINAL SemanticStream was is_recycle=True (per
    recycle_origin, captured before the guards ran), re-tags the EdgeIR as a
    recycle so validate_dag() will exclude it and the DAG becomes valid.

    Logs the stage at which the flag was dropped.  Idempotent: an edge already
    flagged is_recycle is skipped, and if no cycle remains nothing happens.
    """
    changes: list[str] = []
    ug = graph.unit_graph()
    try:
        cycle = nx.find_cycle(ug)
    except nx.NetworkXNoCycle:
        return changes

    for u, v in cycle:
        stream_tag = ug.edges[u, v].get("stream_tag")
        if not stream_tag:
            continue
        edge = graph.stream(stream_tag)
        if edge is None or edge.is_recycle:
            continue  # idempotent — already a recycle
        origin = recycle_origin.get(stream_tag)
        if not (origin and origin.get("is_recycle")):
            continue  # flag was never set originally — Fix 2 does not invent one
        edge.is_recycle = True
        # recycle_target must be a valid unit tag (validate.py enforces this).
        edge.recycle_target = (
            origin.get("recycle_target") or edge.recycle_target or v)
        stage = origin.get("dropped_by") or "unknown"
        changes.append(
            f"[TOPO_REPAIR] repropagated is_recycle on {stream_tag} "
            f"(lost at {stage})")
    return changes


def _detect_missing_units(graph: FlowsheetGraph, sem_topo: Any) -> list[dict]:
    """
    Fix 3 — detect (do NOT repair) streams that reference a unit tag absent from
    the unit list.  Returns [{stream, missing_tag, role}].  A missing unit's
    type cannot be invented deterministically, so the caller routes these to the
    LLM repair node for a real re-extraction attempt.
    """
    missing: list[dict] = []
    if sem_topo is None:
        return missing
    unit_tags = graph.unit_tags()
    seen: set[tuple[str, str]] = set()
    for s in getattr(sem_topo, "streams", []):
        for role, tag in (("src", s.src), ("dst", s.dst)):
            if tag and tag not in unit_tags and (s.tag, tag) not in seen:
                seen.add((s.tag, tag))
                missing.append(
                    {"stream": s.tag, "missing_tag": tag, "role": role})
    return missing


# ── Variant B: reference-assisted topology injection (diagnostic) ──────────────
#
# Purpose: isolate the thermo-mapping + convergence half of the pipeline from LLM
# topology extraction by feeding known topology from the reference flowsheets.
# Activated by VARIANT_B=1 AND a reference_file; off by default (no-op).
#
# IMPORTANT REALITY: every reference file ships an EMPTY connections array (only
# per-unit params and per-stream conditions were annotated).  So connectivity is
# always INFERRED here, never read — which contaminates the diagnostic and is
# flagged loudly (topology_source="reference-inferred-connections").

def _variant_b_enabled() -> bool:
    return os.environ.get("VARIANT_B", "").strip().lower() in ("1", "true", "yes")


# Cases whose underlying process is REACTIVE (a stable property of the case, not
# of the reference data).  The reference-generation systematically dropped reactor
# units, so for these a reference with no reactor unit is INCOMPLETE and its
# convergence/MAPE numbers are not meaningful ground truth.  See the project note
# "reference reactor bug".  The untrusted flag is computed structurally (reactive
# case AND no reactor in the reference), so it AUTO-CLEARS once a reference is
# re-extracted with its reactor.
_REACTIVE_CASES = {"VAL_03", "VAL_04", "VAL_05", "VAL_10"}


def _reference_has_reactor(reference: Optional[dict]) -> bool:
    for u in (reference or {}).get("units", []):
        t = str(u.get("type", "")).lower()
        if "react" in t or "gibbs" in t:
            return True
    return False


def _reference_trust(reference: Optional[dict]) -> Optional[str]:
    """Return an untrusted-reason string for a reference that can't serve as
    ground truth, else None (trusted).  Precedence: an explicit
    excluded-invalid-reference marker (e.g. mass-balance violation) first, then
    a reactive case whose reference is still missing its reactor."""
    if "excluded-invalid-reference" in str((reference or {}).get("reference_validity", "")):
        return "excluded-invalid-reference"
    case_id = (reference or {}).get("case_id")
    if case_id in _REACTIVE_CASES and not _reference_has_reactor(reference):
        return "missing reactor"
    return None


def _load_reference_json(reference_file: Optional[str]) -> Optional[dict]:
    """Load a reference flowsheet dict (mirrors _thermo_node's loader paths)."""
    if not reference_file:
        return None
    from pathlib import Path as _Path
    for _p in (_Path(reference_file),
               _Path(__file__).resolve().parent.parent / reference_file):
        try:
            with open(_p) as _rf:
                return json.load(_rf)
        except FileNotFoundError:
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"[VARIANT_B] error reading reference_file {reference_file}: {exc}",
                  flush=True, file=sys.stderr)
            return None
    return None


def _required_outlets(unit_type: str) -> int:
    specs = PORT_SPECS.get(unit_type, [])
    n = len([s for s in specs if s.direction == "outlet" and s.required])
    return max(n, 1)


def _required_inlet_phase(unit_type: str) -> str:
    """Required inlet phase for a unit, or 'any' (Pump→liquid, Compressor/Expander→vapour)."""
    for s in PORT_SPECS.get(unit_type, []):
        if s.direction == "inlet" and s.required and s.phase in ("liquid", "vapour"):
            return s.phase
    return "any"


def _infer_sequential_topology(
    sem_units: SemanticUnits,
    compounds: list[str],
) -> tuple[SemanticTopology, bool]:
    """
    Deterministic MINIMAL connectivity for references that lack connections.

    Builds a linear backbone in unit-tag order (unit[i] → unit[i+1]), a single
    synthesised feed into unit[0], and fills each unit's REQUIRED extra outlets
    (Vessel/Splitter need 2) with terminal product streams so the IR can satisfy
    port constraints.  No recycles are inferred.

    Phase-aware ordering: a Vessel's two outlets are port 0 = vapour, port 1 =
    liquid (assigned by GraphBuilder in creation order).  When a Vessel feeds a
    unit that requires a liquid inlet (e.g. a Pump), the backbone is created
    SECOND so it lands on the liquid port — avoiding a spurious phase mismatch
    introduced purely by the inferred ordering.  This is still inference, not
    data — the caller flags it as such.  Returns (topology, inferred_feed=True).
    """
    units = sem_units.units
    streams: list[SemanticStream] = []
    if not units:
        return SemanticTopology(streams=streams), True

    # Synthesised feed (equimolar) into the first unit — guarantees a valid feed.
    feed_comp = ({c: round(1.0 / len(compounds), 6) for c in compounds}
                 if compounds else {})
    streams.append(SemanticStream(
        tag="VB-FEED", src=None, dst=units[0].tag, is_feed=True,
        T=298.15, P=101_325.0, flow=1.0, composition=feed_comp))

    # Per-unit outlet streams, created in phase-aware order so port assignment
    # (GraphBuilder assigns ports in creation order) respects downstream needs.
    sidx = prod = 0
    for i, u in enumerate(units):
        succ   = units[i + 1].tag if i < len(units) - 1 else None
        n_out  = _required_outlets(u.type)
        # By default the backbone (to successor) is the first outlet (port 0).
        # For a Vessel feeding a liquid-only unit, place it on port 1 (liquid).
        backbone_slot = 0
        if (succ is not None and u.type == "Vessel" and n_out >= 2
                and _required_inlet_phase(units[i + 1].type) == "liquid"):
            backbone_slot = 1
        for slot in range(n_out):
            if succ is not None and slot == backbone_slot:
                streams.append(SemanticStream(
                    tag=f"VB-S{sidx:02d}", src=u.tag, dst=succ, is_feed=False))
                sidx += 1
            else:
                streams.append(SemanticStream(
                    tag=f"VB-P{prod:02d}", src=u.tag, dst=None, is_feed=False))
                prod += 1

    return SemanticTopology(streams=streams), True


def _sem_stream_from_ref(
    tag:  str,
    src:  Optional[str],
    dst:  Optional[str],
    cond: dict,
) -> SemanticStream:
    """A SemanticStream carrying the reference stream's conditions verbatim."""
    return SemanticStream(
        tag=tag, src=src, dst=dst, is_feed=(src is None),
        T=cond.get("T_K"), P=cond.get("P_Pa"), flow=cond.get("flow_mol_s"),
        composition=dict(cond.get("composition", {}) or {}))


def _topology_from_connections(
    connections: list,
    sem_units:   SemanticUnits,
    reference:   Optional[dict] = None,
) -> SemanticTopology:
    """
    Build SemanticTopology from a populated reference connections array, used
    VERBATIM (no inference).  Handles the canonical DWSIM/schema format
    [src, dst, src_port, dst_port] where each entry pairs a unit with a stream:
        [unit, stream, …]  → that unit is the stream's SOURCE
        [stream, unit, …]  → that unit is the stream's DEST
    Entries are paired by stream tag to recover each unit→unit edge with its real
    stream identity, and each stream's T/P/flow/composition is carried over from
    reference["streams"].  A plain unit→unit entry (both tags are units) is also
    honoured as a fallback.  Not exercised by the current references (all ship
    empty connections); covered by a unit test against a synthetic populated ref.

    Recycle edges are declared via an optional reference["recycles"] array of
    {"stream": <tag>, "target_unit": <unit tag>}; matching streams are marked
    is_recycle=True with recycle_target set, so the downstream recycle machinery
    (GraphBuilder, to_dwsim, the executor's recycle blocks) closes the loop the
    same way it does for LLM-extracted recycles.  This is what lets a *complete*
    reference seed a recycle topology — connectivity is taken verbatim, no
    inference and no LLM.
    """
    unit_tags   = {u.tag for u in sem_units.units}
    ref_streams = (reference or {}).get("streams", {}) or {}

    # Optional recycle annotations: stream tag → target (upstream) unit tag.
    recycle_map: dict[str, str] = {}
    for r in (reference or {}).get("recycles", []) or []:
        st, tu = r.get("stream"), r.get("target_unit")
        if st and tu in unit_tags:
            recycle_map[st] = tu
        elif st:
            print(f"[VARIANT_B] WARNING: recycle annotation for stream '{st}' has "
                  f"target_unit={tu!r} which is not a unit tag — ignored",
                  flush=True, file=sys.stderr)

    stream_src:   dict[str, str]            = {}   # stream tag → source unit
    stream_dst:   dict[str, str]            = {}   # stream tag → dest unit
    direct_edges: list[tuple[str, str]]     = []   # unit → unit (fallback)
    seen_streams: list[str]                 = []   # preserve first-seen order

    for c in connections:
        if not (isinstance(c, (list, tuple)) and len(c) >= 2):
            continue
        a, b = c[0], c[1]
        a_is_unit, b_is_unit = a in unit_tags, b in unit_tags
        if a_is_unit and not b_is_unit:           # unit → stream
            if b not in stream_src and b not in seen_streams:
                seen_streams.append(b)
            stream_src[b] = a
        elif b_is_unit and not a_is_unit:         # stream → unit
            if a not in stream_dst and a not in seen_streams:
                seen_streams.append(a)
            stream_dst[a] = b
        elif a_is_unit and b_is_unit:             # unit → unit (fallback)
            direct_edges.append((a, b))
        # stream→stream / unknown tags: skip

    streams: list[SemanticStream] = []
    for st in seen_streams:
        sem = _sem_stream_from_ref(
            st, stream_src.get(st), stream_dst.get(st), ref_streams.get(st, {}))
        if st in recycle_map:
            sem.is_recycle     = True
            sem.recycle_target = recycle_map[st]
            # A recycle edge feeds back to its target; honour the annotation even
            # if the connections array only recorded the producing side.
            if sem.dst is None:
                sem.dst = recycle_map[st]
            print(f"[VARIANT_B] recycle edge from reference: stream '{st}' "
                  f"→ {recycle_map[st]} (is_recycle=True)",
                  flush=True, file=sys.stderr)
        streams.append(sem)
    for i, (src, dst) in enumerate(direct_edges):
        streams.append(SemanticStream(
            tag=f"REF-D{i:02d}", src=src, dst=dst, is_feed=False))

    return SemanticTopology(streams=streams)


def _reference_unit_params(units_json: list[dict]) -> dict:
    """Extract non-null per-unit params (T_out, P_out, dP, efficiency …)."""
    out: dict[str, dict] = {}
    for u in units_json:
        params = {k: v for k, v in (u.get("params") or {}).items() if v is not None}
        if params:
            out[u["tag"]] = params
    return out


def _variant_b_mape(execution: Any, reference_data: Optional[dict]) -> Optional[dict]:
    """MAPE (%) of converged stream T/P/vf vs the closest reference stream by
    composition.  None unless the solve converged and the reference has streams."""
    if execution is None or not getattr(execution, "solved", False):
        return None
    ref_streams = (reference_data or {}).get("streams", {})
    if not ref_streams:
        return None
    errs: dict[str, list[float]] = {"T": [], "P": [], "vf": []}
    for _tag, sr in getattr(execution, "stream_results", {}).items():
        comp = getattr(sr, "composition", {}) or {}
        best, _dist = _best_ref_stream_by_composition(comp, ref_streams)
        if best is None:
            continue
        ref = ref_streams[best]
        for key, got, refv in (("T",  getattr(sr, "T_K", None),          ref.get("T_K")),
                               ("P",  getattr(sr, "P_Pa", None),         ref.get("P_Pa")),
                               ("vf", getattr(sr, "vapor_fraction", None), ref.get("vapor_fraction"))):
            if got is None or refv is None:
                continue
            denom = abs(refv) if abs(refv) > 1e-9 else 1.0
            errs[key].append(abs(got - refv) / denom)
    def _m(xs: list[float]) -> Optional[float]:
        return round(100.0 * sum(xs) / len(xs), 2) if xs else None
    return {"T": _m(errs["T"]), "P": _m(errs["P"]), "vf": _m(errs["vf"])}


def compute_variant_b_ladder(state: PipelineState) -> dict:
    """Build the per-case diagnostic ladder from final pipeline state."""
    ir_graph  = state.get("ir_graph")
    execution = state.get("execution")
    outcome   = state.get("outcome")
    ref       = state.get("reference_data") or {}

    built_valid_ir = False
    if ir_graph is not None:
        try:
            built_valid_ir = validate(ir_graph).valid
        except Exception:  # noqa: BLE001
            built_valid_ir = False
    reached_dwsim = execution is not None
    converged     = bool(getattr(execution, "solved", False))

    if outcome == "PASS":
        failure_stage = None
    elif not built_valid_ir:
        failure_stage = "ir_build"
    elif not reached_dwsim:
        failure_stage = "ir_build"
    elif outcome == "MAX_ITER":
        failure_stage = "max_iter"
    elif not converged:
        failure_stage = "dwsim_no_converge"
    else:
        failure_stage = "other"

    mape = _variant_b_mape(execution, ref) if converged else None
    return {
        "case":                ref.get("case_id") or state.get("reference_file") or "?",
        "topology_source":     state.get("topology_source"),
        "inferred_feed":       state.get("variant_b_inferred_feed", False),
        # Interim integrity gate: reactive case whose reference dropped its
        # reactor → ground truth is incomplete, so converged/MAPE here are NOT
        # meaningful.  None when trusted; auto-clears if the reference is fixed.
        "untrusted_reference": _reference_trust(ref),
        "built_valid_ir":      built_valid_ir,
        "reached_dwsim":       reached_dwsim,
        "converged":           converged,
        "n_repair_iterations": state.get("iteration", 0),
        "reference_mape_T":    (mape or {}).get("T"),
        "reference_mape_P":    (mape or {}).get("P"),
        "reference_mape_vf":   (mape or {}).get("vf"),
        "failure_stage":       failure_stage,
        "outcome":             outcome,
    }


def variant_b_summary(diags: list[dict]) -> str:
    """Aggregate per-case ladders into a headline summary table.

    Untrusted cases (reactive reference missing its reactor) are flagged with '*'
    and their converged/MAPE values are NOT counted toward the headline numbers —
    so an incomplete reference can't make a reactive convergence look meaningful.
    """
    if not diags:
        return "[VARIANT_B] no cases."
    cols = [("case", 9), ("topo_src", 13), ("trust", 11), ("valid_ir", 8),
            ("dwsim", 6), ("converged", 10), ("iters", 5), ("MAPE_T", 7),
            ("failure", 17)]
    def _row(vals):
        return "  ".join(str(v)[:w].ljust(w) for v, w in zip(vals, [c[1] for c in cols]))
    lines = ["", "=" * 100, "VARIANT B — reference-assisted topology diagnostic",
             "=" * 100, _row([c[0] for c in cols]), "-" * 100]
    for d in diags:
        ts  = "ref-inferred" if d["topology_source"] == "reference-inferred-connections" else "ref-exact"
        unt = d.get("untrusted_reference")
        trust   = f"UNTRUSTED*" if unt else "ok"
        conv    = f"{d['converged']}*" if unt else str(d["converged"])
        mape    = "*" if unt else (d["reference_mape_T"] if d["reference_mape_T"] is not None else "-")
        lines.append(_row([
            d["case"], ts, trust, d["built_valid_ir"], d["reached_dwsim"],
            conv, d["n_repair_iterations"], mape, d["failure_stage"] or "—",
        ]))
    n          = len(diags)
    trusted    = [d for d in diags if not d.get("untrusted_reference")]
    untrusted  = [d for d in diags if d.get("untrusted_reference")]
    m          = len(trusted)
    _cnt  = lambda k: sum(1 for d in diags if d.get(k))             # noqa: E731
    _cntT = lambda k: sum(1 for d in trusted if d.get(k))           # noqa: E731
    lines += [
        "-" * 100,
        f"of {n} cases — built valid IR: {_cnt('built_valid_ir')}/{n}"
        f" | reached DWSIM: {_cnt('reached_dwsim')}/{n}"
        f" | converged (TRUSTED only): {_cntT('converged')}/{m}",
        f"UNTRUSTED: {len(untrusted)}/{n} reactive reference(s) missing a reactor "
        f"{sorted(d['case'] for d in untrusted)} — their converged/MAPE are NOT meaningful",
        f"connectivity: {sum(1 for d in diags if d['topology_source']=='reference-inferred-connections')}/{n} INFERRED"
        f" (reference connections were empty — diagnostic is contaminated by inference)",
        "* untrusted reference: reactive case whose reference dropped its reactor; "
        "convergence/MAPE are not valid ground truth (see project 'reference reactor bug').",
        "=" * 100, "",
    ]
    return "\n".join(lines)


# ── Pipeline class ────────────────────────────────────────────────────────────

class GraphPipeline:
    """
    LangGraph-based flowsheet pipeline.  Identical results to OrchestratorV2;
    only the control flow is expressed as a StateGraph.

    Variant B (VARIANT_B=1) adds a diagnostic entry path that injects reference
    topology in place of the LLM basis+topology nodes; off by default.
    """

    def __init__(
        self,
        model:          str = DEFAULT_MODEL,
        max_iterations: int = 10,
        rule_store:     Optional[FailureRuleStore] = None,
    ):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "langgraph is not installed — pip install langgraph")

        self._model    = model
        self._max_iter = max_iterations

        retriever = Retriever()

        self._basis       = BasisAgent(model=model)
        self._unit_ext    = UnitExtractor(model=model)
        self._stream_ext  = StreamExtractor(model=model)
        self._builder     = GraphBuilder()

        # Log TopologyChain availability (matches OrchestratorV2 startup messages)
        # but do NOT instantiate it — Phase 1 uses UnitExtractor + StreamExtractor only.
        if not _TC_AVAILABLE:
            print("[GP] LangChain not installed (TopologyChain excluded from Phase 1 anyway)",
                  flush=True)

        self._thermo      = ThermoMapper(model=model, retriever=retriever)
        self._params      = ParamMapper(model=model, retriever=retriever)
        self._consistency = GlobalConsistencyPass()
        self._executor    = Executor()
        self._classifier  = ErrorClassifier(model=model)
        self._repair      = RepairAgent(model=model, retriever=retriever)

        if rule_store is not None:
            self._rule_store = rule_store
        else:
            self._rule_store = FailureRuleStore()
            self._rule_store.load(RULES_PATH)
            if self._rule_store.num_patterns() > 0:
                print(f"[GP] rule_store loaded: {self._rule_store.num_patterns()} patterns",
                      flush=True)

        self._app = self._build_graph()

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self):
        g = StateGraph(PipelineState)

        g.add_node("basis",             self._basis_node)
        g.add_node("topology",          self._topology_node)
        g.add_node("reference_topology", self._reference_topology_node)  # Variant B
        g.add_node("build",             self._build_node)
        g.add_node("validate",          self._validate_node)
        g.add_node("topology_repair",   self._topology_repair_node)
        g.add_node("thermo",            self._thermo_node)
        g.add_node("execute",           self._execute_node)
        g.add_node("repair",            self._repair_node)

        # Conditional entry: Variant B enters at reference_topology, skipping the
        # LLM basis+topology nodes entirely; otherwise the normal entry at basis.
        g.set_conditional_entry_point(
            _route_entry,
            {"reference_topology": "reference_topology", "basis": "basis"})
        g.add_edge("reference_topology", "build")

        g.add_conditional_edges("basis",    _route_basis,    {"topology": "topology", END: END})
        g.add_conditional_edges("topology", _route_stage1,   {"build": "build",       END: END})
        g.add_edge("build", "validate")
        # validate → topology_repair (repairable INVALID_TOPOLOGY) | thermo | END
        g.add_conditional_edges("validate", _route_validate,
                                {"thermo": "thermo",
                                 "topology_repair": "topology_repair",
                                 END: END})
        # topology_repair → thermo (valid) | repair (LLM) | END (infeasible)
        g.add_conditional_edges("topology_repair", _route_topology_repair,
                                {"thermo": "thermo", "repair": "repair", END: END})
        g.add_conditional_edges("thermo",   _route_thermo,   {"execute": "execute",   END: END})
        g.add_conditional_edges("execute",  _route_execute,  {"repair": "repair",     END: END})
        g.add_edge("repair", "execute")

        return g.compile()

    # ── Public interface ──────────────────────────────────────────────────────

    def run(
        self,
        description:    str,
        reference_file: Optional[str] = None,
        tier:           str = "standard",
    ) -> PipelineResult:
        # ── Variant B activation (diagnostic; off by default) ─────────────────
        variant_b_active = False
        if _variant_b_enabled():
            if reference_file:
                variant_b_active = True
                print(f"[VARIANT_B] ACTIVE — injecting reference topology from "
                      f"{reference_file} (LLM basis/topology bypassed)",
                      flush=True, file=sys.stderr)
            else:
                print("[VARIANT_B] requested but SKIPPED for this case — no "
                      "reference_file; falling back to normal extraction",
                      flush=True, file=sys.stderr)

        initial: PipelineState = {
            "description":    description,
            "tier":           tier,
            "reference_file": reference_file,
            "max_iterations": self._max_iter,
            "t_start":        time.time(),
            "variant_b_active":        variant_b_active,
            "topology_source":         None,
            "reference_unit_params":   {},
            "variant_b_inferred_feed": False,
            "basis_result":   None,
            "norm_desc":      "",
            "compounds":      [],
            "sem_units":      None,
            "sem_topo":       None,
            "recycle_origin": {},
            "ir_graph":       None,
            "ir_report":      None,
            "missing_units":  [],
            "dwsim_json":     None,
            "reference_data": None,
            "tried_packages": [],
            "iteration":      0,
            "eff_max_iter":   self._max_iter,
            "beam_extended":  False,
            "repair_memory":  None,
            "sim_hints":      None,
            "execution":      None,
            "errors":         [],
            "prev_hash":      None,
            "warnings":       [],
            "iterations_log": [],
            "outcome":        "MAX_ITER",
        }
        final = self._app.invoke(initial)
        return self._to_result(final)

    def _to_result(self, state: PipelineState) -> PipelineResult:
        result = PipelineResult(
            description  = state["description"],
            outcome      = state["outcome"],
            basis_result = state.get("basis_result"),
            ir_report    = state.get("ir_report"),
            warnings     = list(state.get("warnings", [])),
            total_time_s = time.time() - state["t_start"],
        )
        result.iterations      = list(state.get("iterations_log", []))
        result.final_graph     = state.get("ir_graph")
        result.final_flowsheet = state.get("dwsim_json")
        result.final_execution = state.get("execution")
        result.margin_snapshot = get_global_margin_model().snapshot()
        # Variant B diagnostic ladder (attached only when the mode was active).
        if state.get("variant_b_active"):
            ladder = compute_variant_b_ladder(state)
            result.variant_b_diag = ladder  # type: ignore[attr-defined]
            print(f"[VARIANT_B] ladder: {ladder}", flush=True, file=sys.stderr)
        return result

    # ── Node implementations ──────────────────────────────────────────────────

    def _reference_topology_node(self, state: PipelineState) -> dict:
        """
        Variant B — construct SemanticUnits + SemanticTopology from the reference
        flowsheet, bypassing BasisAgent/UnitExtractor/StreamExtractor entirely.
        Units, compounds and per-unit setpoints are taken EXACTLY from the
        reference; connectivity is INFERRED (references ship empty connections)
        and flagged loudly.  Everything from build_node onward runs unchanged.
        """
        ref = _load_reference_json(state["reference_file"])
        if ref is None:
            print("[VARIANT_B] reference load FAILED — cannot inject topology",
                  flush=True, file=sys.stderr)
            return {"outcome": "PLAN_FAILED",
                    "warnings": ["[VARIANT_B] reference_file could not be loaded"]}

        compounds  = list(ref.get("compounds", []))
        units_json = ref.get("units", [])
        sem_units  = SemanticUnits(units=[
            SemanticUnit(tag=u["tag"], type=u["type"], role="reference")
            for u in units_json])
        ref_params = _reference_unit_params(units_json)

        if ref.get("connections"):
            # Populated connectivity present → use it VERBATIM (no inference).
            # (Does not occur in the current refs, but honoured if a future
            # reference is hand-annotated.)
            topology_source = "reference-exact"
            sem_topo = _topology_from_connections(
                ref["connections"], sem_units, ref)
            inferred_feed = False
            print(f"[VARIANT_B] reference connections GIVEN ({len(ref['connections'])} "
                  f"edges) for {ref.get('case_id','?')} — using topology VERBATIM "
                  f"({len(sem_topo.streams)} streams, no inference)",
                  flush=True, file=sys.stderr)
        else:
            topology_source = "reference-inferred-connections"
            sem_topo, inferred_feed = _infer_sequential_topology(sem_units, compounds)
            print(f"[VARIANT_B] *** CONNECTIONS INFERRED (not given) *** for "
                  f"{ref.get('case_id','?')}: reference connections array is "
                  f"empty — built a deterministic sequential backbone over "
                  f"{len(sem_units.units)} units. This inference contaminates the "
                  f"diagnostic; treat downstream results accordingly.",
                  flush=True, file=sys.stderr)

        print(f"[VARIANT_B] reference topology: case={ref.get('case_id','?')} "
              f"units={[u.tag for u in sem_units.units]} compounds={compounds} "
              f"topology_source={topology_source}", flush=True, file=sys.stderr)

        return {
            "compounds":               compounds,
            "norm_desc":               state["description"],
            "sem_units":               sem_units,
            "sem_topo":                sem_topo,
            "reference_data":          ref,
            "reference_unit_params":   ref_params,
            "topology_source":         topology_source,
            "variant_b_inferred_feed": inferred_feed,
            "warnings": [f"[VARIANT_B] topology_source={topology_source}"],
        }

    def _basis_node(self, state: PipelineState) -> dict:
        # Integrity guard: Variant B must NEVER reach the LLM basis node.  If it
        # does, the diagnostic is invalid — fail loudly rather than silently run.
        if state.get("variant_b_active"):
            raise AssertionError(
                "[VARIANT_B] integrity violation: BasisAgent (basis node) was "
                "entered while Variant B is active — LLM extraction must not run.")
        print("[GP] step: basis.identify START", flush=True)
        basis = self._basis.identify(state["description"])
        print(f"[GP] step: basis.identify END  success={basis.success}", flush=True)
        if not basis.success:
            return {"basis_result": basis, "outcome": "BASIS_FAILED"}
        print(f"[GP] compounds={basis.dwsim_compounds}", flush=True)
        return {
            "basis_result": basis,
            "norm_desc":    basis.normalised_description,
            "compounds":    list(basis.dwsim_compounds),
        }

    def _topology_node(self, state: PipelineState) -> dict:
        # Integrity guard: Variant B must NEVER reach the LLM topology node.
        if state.get("variant_b_active"):
            raise AssertionError(
                "[VARIANT_B] integrity violation: UnitExtractor/StreamExtractor "
                "(topology node) was entered while Variant B is active — LLM "
                "extraction must not run.")
        desc      = state["norm_desc"]
        compounds = state["compounds"]
        tier      = state["tier"]
        basis     = state["basis_result"]
        new_warns: list[str] = []

        # ── Optional summarisation (mirrors orchestrator_v2 exactly) ─────────
        _n_words = len(desc.split())
        if _n_words > 200 and tier != "validation":
            print(f"[GP] description length={_n_words} words — summarising for UnitExtractor",
                  flush=True, file=sys.stderr)
            try:
                desc_for_units = _summarise_for_unit_extraction(desc, self._model)
                print(f"[GP] unit summary ({len(desc_for_units.split())} words): "
                      f"{desc_for_units[:150]!r}",
                      flush=True, file=sys.stderr)
                new_warns.append(
                    f"[summary] description condensed "
                    f"({_n_words}→{len(desc_for_units.split())} words) for UnitExtractor")
                if len(desc_for_units.split()) > 150:
                    _first_words = len(desc_for_units.split())
                    print(f"[GP] first summary still {_first_words} words — "
                          "running tighter second-pass summarisation",
                          flush=True, file=sys.stderr)
                    try:
                        from agents.llm import chat as _chat
                        _tight = _chat(
                            f"Equipment list:\n{desc_for_units}\n\n"
                            "List only the equipment tags and types, one per line."
                            " Maximum 20 words total.",
                            system=_SUMMARISER_SYSTEM_TIGHT,
                            model=self._model,
                            temperature=0.0,
                            max_tokens=256,
                        )
                        if _tight.strip():
                            desc_for_units = _tight.strip()
                            new_warns.append(
                                f"[summary2] second-pass condensed to "
                                f"{len(desc_for_units.split())} words")
                    except Exception as _sum2_exc:
                        print(f"[GP] second-pass summariser failed ({_sum2_exc})",
                              flush=True, file=sys.stderr)
            except Exception as _sum_exc:
                print(f"[GP] summariser failed ({_sum_exc}) — using full description",
                      flush=True, file=sys.stderr)
                desc_for_units = desc
        else:
            if tier == "validation" and _n_words > 200:
                print(f"[GP] validation tier: skipping summarisation ({_n_words} words)",
                      flush=True, file=sys.stderr)
            desc_for_units = desc

        # ── Stage 1 extraction (Phase 1: always UnitExtractor + StreamExtractor) ─
        # TopologyChain is excluded from Phase 1 to keep the checkpoint a clean
        # reproduction of v2.  It will be wired in during Phase 4.
        try:
            print("[GP] step: unit_ext.extract START", flush=True)
            sem_units = self._unit_ext.extract(desc_for_units, compounds, tier=tier)
            print(f"[GP] step: unit_ext.extract END  "
                  f"units={[u.tag for u in sem_units.units]}", flush=True)

            print("[GP] step: stream_ext.extract START", flush=True)
            sem_topo = self._stream_ext.extract(
                desc, compounds,
                unit_tags              = [u.tag  for u in sem_units.units],
                unit_roles             = {u.tag: u.role for u in sem_units.units},
                concentration_hints    = basis.concentration_hints or [],
                suggested_compositions = basis.suggested_compositions or {},
            )
            print("[GP] step: stream_ext.extract END", flush=True)

            # Capture original recycle flags BEFORE the guards mutate the
            # SemanticStreams, so topology_repair (Fix 2) can repropagate a flag
            # that a guard later drops — and report which guard dropped it.
            recycle_origin: dict[str, dict] = {
                _s.tag: {
                    "is_recycle":     bool(getattr(_s, "is_recycle", False)),
                    "recycle_target": getattr(_s, "recycle_target", None),
                    "dropped_by":     None,
                }
                for _s in sem_topo.streams
            }

            # ── Recycle guard 1: target-validity + fuzzy resolution ───────────
            _unit_tag_set = {u.tag for u in sem_units.units}
            for _s in sem_topo.streams:
                if not getattr(_s, "is_recycle", False):
                    continue
                if _s.recycle_target and _s.recycle_target in _unit_tag_set:
                    print(f"[GP] recycle detected: stream '{_s.tag}' → {_s.recycle_target}",
                          flush=True, file=sys.stderr)
                    continue
                _resolved = _resolve_recycle_target(
                    _s.recycle_target or "", sem_units.units)
                if _resolved:
                    print(f"[GP] recycle guard: resolved target "
                          f"{_s.recycle_target!r} → '{_resolved}' for stream '{_s.tag}'",
                          flush=True, file=sys.stderr)
                    new_warns.append(
                        f"[recycle] '{_s.tag}' target resolved: "
                        f"{_s.recycle_target!r} → '{_resolved}'")
                    _s.recycle_target = _resolved
                else:
                    print(f"[GP] recycle guard: cleared unresolvable recycle on "
                          f"'{_s.tag}' (target={_s.recycle_target!r})",
                          flush=True, file=sys.stderr)
                    new_warns.append(
                        f"[recycle] '{_s.tag}' recycle_target={_s.recycle_target!r} "
                        "unresolvable — cleared to avoid false positive")
                    recycle_origin[_s.tag]["dropped_by"] = "guard1-target"
                    _s.is_recycle     = False
                    _s.recycle_target = None

            # ── Recycle guard 2: multi-recycle deduplication ─────────────────
            _unit_order = {u.tag: i for i, u in enumerate(sem_units.units)}
            _recycle_by_src: dict[str, list] = {}
            for _s in sem_topo.streams:
                if getattr(_s, "is_recycle", False) and _s.src:
                    _recycle_by_src.setdefault(_s.src, []).append(_s)
            for _src_tag, _candidates in _recycle_by_src.items():
                if len(_candidates) <= 1:
                    continue
                _src_idx = _unit_order.get(_src_tag, len(sem_units.units))
                def _target_rank(_s, _src_idx=_src_idx):
                    t_idx = _unit_order.get(_s.recycle_target, len(sem_units.units))
                    return (0 if t_idx < _src_idx else 1, t_idx)
                _keep = min(_candidates, key=_target_rank)
                for _s in _candidates:
                    if _s is not _keep:
                        print(f"[GP] multi-recycle guard: cleared extra recycle on "
                              f"'{_s.tag}' from src='{_src_tag}'; "
                              f"keeping '{_keep.tag}' → {_keep.recycle_target!r}",
                              flush=True, file=sys.stderr)
                        new_warns.append(
                            f"[recycle] '{_s.tag}' cleared — multiple recycles from "
                            f"'{_src_tag}', kept '{_keep.tag}' → '{_keep.recycle_target}'")
                        recycle_origin[_s.tag]["dropped_by"] = "guard2-multi"
                        _s.is_recycle     = False
                        _s.recycle_target = None

            # ── Recycle guard 3: phrase guard ─────────────────────────────────
            _desc_lower = desc.lower()
            for _s in sem_topo.streams:
                if getattr(_s, "is_recycle", False):
                    if not any(p in _desc_lower for p in _RECYCLE_PHRASES):
                        print(f"[GP] phrase guard: suppressed false-positive recycle "
                              f"on stream '{_s.tag}'",
                              flush=True, file=sys.stderr)
                        new_warns.append(
                            f"[recycle] '{_s.tag}' is_recycle=True but no trigger "
                            "phrase in description — cleared to avoid false positive")
                        recycle_origin[_s.tag]["dropped_by"] = "guard3-phrase"
                        _s.is_recycle     = False
                        _s.recycle_target = None

        except (RuntimeError, ValueError, TimeoutError) as exc:
            # ValueError covers an exhausted empty/whitespace-only LLM response
            # (e.g. StreamExtractor's parser); TimeoutError covers an exhausted
            # wall-clock timeout on a hung LLM call.  Degrade this one case to
            # PLAN_FAILED with a real diagnostic instead of crashing the graph.
            print(f"[GP] Stage 1 EXCEPTION: {type(exc).__name__}: {exc}", flush=True)
            return {"outcome": "PLAN_FAILED", "warnings": [f"Stage 1 failed: {exc}"]}

        return {
            "sem_units":      sem_units,
            "sem_topo":       sem_topo,
            "recycle_origin": recycle_origin,
            "warnings":       new_warns,
        }

    def _build_node(self, state: PipelineState) -> dict:
        sem_units = state["sem_units"]
        sem_topo  = state["sem_topo"]
        compounds = state["compounds"]
        new_warns: list[str] = []

        print("[GP] step: builder.build START", flush=True)
        graph = self._builder.build(sem_units, sem_topo, compounds)
        print("[GP] step: builder.build END", flush=True)

        # Compound reconciliation: augment graph.compounds with any found in
        # stream compositions that BasisAgent missed.
        _known_lower = {c.lower() for c in graph.compounds}
        for _stream in graph.streams():
            for _name in _stream.composition:
                if _name.lower() not in _known_lower:
                    graph.compounds.append(_name)
                    _known_lower.add(_name.lower())
                    new_warns.append(
                        f"[compounds] '{_name}' found in feed composition "
                        f"but missing from basis list — added automatically")
        if graph.compounds != list(compounds):
            print(f"[GP] compounds reconciled: {graph.compounds}", flush=True)

        # Back-fill: every stream with composition data must have all compounds.
        for _stream in graph.streams():
            if _stream.composition:
                for _name in graph.compounds:
                    if _name not in _stream.composition:
                        _stream.composition[_name] = 0.0

        print("[GP] step: normalise(graph) #1 START", flush=True)
        graph = normalise(graph)
        print("[GP] step: normalise(graph) #1 END", flush=True)

        return {"ir_graph": graph, "warnings": new_warns}

    def _validate_node(self, state: PipelineState) -> dict:
        graph = state["ir_graph"]
        print(f"[GP] step: validate(graph) #1 START  "
              f"graph.compounds={graph.compounds}", flush=True)
        ir_report = validate(graph)
        print(f"[GP] step: validate(graph) #1 END  valid={ir_report.valid}", flush=True)

        if not ir_report.valid:
            new_warns = [str(e) for e in ir_report.errors()]
            for e in ir_report.errors():
                print(f"[GP] INVALID_IR: {e}", flush=True, file=sys.stderr)

            crit       = ir_report.errors()
            has_vessel = any(_is_vessel_outlet_issue(i) for i in crit)   # Fix 1
            has_cycle  = any(_is_cycle_issue(i) for i in crit)           # Fix 2
            missing    = _detect_missing_units(graph, state.get("sem_topo"))  # Fix 3

            # Route to the deterministic topology_repair node only when the
            # invalidity matches one of the three repairable patterns; every
            # other INVALID_IR dead-ends at END exactly as before.
            if has_vessel or has_cycle or missing:
                print(f"[GP] INVALID_TOPOLOGY (repairable): "
                      f"vessel_outlet={has_vessel} cycle={has_cycle} "
                      f"missing_units={len(missing)}", flush=True, file=sys.stderr)
                return {
                    "ir_report":     ir_report,
                    "outcome":       "INVALID_TOPOLOGY",
                    "missing_units": missing,
                    "warnings":      new_warns,
                }
            return {"ir_report": ir_report, "outcome": "INVALID_IR", "warnings": new_warns}

        new_warns = [str(w) for w in ir_report.warnings()]
        return {"ir_report": ir_report, "warnings": new_warns}

    def _topology_repair_node(self, state: PipelineState) -> dict:
        """
        Deterministic topology repair — NO LLM calls.  Reached only when
        _validate_node flagged a repairable INVALID_TOPOLOGY pattern.

          Fix 1: add a Vessel's missing phase outlet (terminal product stream).
          Fix 2: repropagate an is_recycle flag a guard dropped, breaking a cycle.
          Fix 3: flag streams referencing a missing unit (NOT repairable here) and
                 hand off to the LLM repair node for a real re-extraction attempt.

        After Fix 1/2 the graph is re-validated: if valid → thermo; otherwise
        (including any Fix-3 missing-unit case) → LLM repair node.  Safe to enter
        twice — the helpers are idempotent (no duplicate outlet, no re-tag).
        """
        graph          = state["ir_graph"]
        recycle_origin = state.get("recycle_origin") or {}
        missing_units  = list(state.get("missing_units") or [])
        changes: list[str] = []

        # ── Fix 1 + Fix 2 (deterministic, in-place) ───────────────────────────
        vessel_changes, suppressed = _repair_vessel_outlets(
            graph, state.get("norm_desc", ""))
        changes += vessel_changes
        changes += _repropagate_recycles(graph, recycle_origin)

        # ── Fix 3 (flag only — cannot invent a missing unit's type) ───────────
        if not missing_units:
            missing_units = _detect_missing_units(graph, state.get("sem_topo"))
        for m in missing_units:
            print(f"[TOPO_REPAIR] flagged missing unit {m['missing_tag']} "
                  f"referenced by {m['stream']} — routing to LLM repair",
                  flush=True, file=sys.stderr)
        for c in changes:
            print(c, flush=True, file=sys.stderr)

        # ── Re-validate (validate() includes validate_dag) ────────────────────
        report = validate(graph)
        print(f"[GP] step: topology_repair re-validate  valid={report.valid}  "
              f"fixes={len(changes)}  missing_units={len(missing_units)}",
              flush=True, file=sys.stderr)

        if report.valid and not missing_units:
            return {
                "ir_graph":  graph,
                "ir_report": report,
                "outcome":   "TOPO_OK",
                "warnings":  changes,
            }

        # Single-phase Vessel(s) whose missing outlet was deliberately NOT
        # fabricated: the flowsheet is physically infeasible and correctly
        # unrepairable — route to END rather than the LLM repair node.
        if suppressed:
            print(f"[GP] topology_repair: single-phase vessel(s) {suppressed} — "
                  f"INVALID (correctly unrepairable) → END",
                  flush=True, file=sys.stderr)
            return {
                "ir_graph":  graph,
                "ir_report": report,
                "outcome":   "TOPO_INFEASIBLE",
                "warnings":  changes + [str(e) for e in report.errors()],
            }

        # Hand off to the LLM repair node with the residual validation errors.
        return {
            "ir_graph":      graph,
            "ir_report":     report,
            "errors":        [i.error for i in report.errors()],
            "missing_units": missing_units,
            "outcome":       "TOPO_REPAIR_LLM",
            "warnings":      changes
                             + [f"[missing_unit] {m}" for m in missing_units]
                             + [str(e) for e in report.errors()],
        }

    def _thermo_node(self, state: PipelineState) -> dict:
        graph     = state["ir_graph"]
        desc      = state["norm_desc"]
        compounds = state["compounds"]
        new_warns: list[str] = []

        # Variant B only: seed each unit's reference setpoints (T_out/P_out/…)
        # before ParamMapper runs.  ParamMapper preserves existing node.params
        # (node.params take priority over its estimates), so the reference
        # setpoints survive — letting thermo/execute run against correct targets.
        # Guarded: reference_unit_params is empty on the normal path → no-op.
        _ref_params = state.get("reference_unit_params") or {}
        if _ref_params:
            for _u in graph.units():
                for _k, _v in _ref_params.get(_u.tag, {}).items():
                    _u.params.setdefault(_k, _v)
            print(f"[VARIANT_B] seeded reference setpoints for "
                  f"{sorted(_ref_params)}", flush=True, file=sys.stderr)

        try:
            print("[GP] step: thermo.assign START", flush=True)
            graph = self._thermo.assign(graph, description=desc)
            print(f"[GP] step: thermo.assign END  "
                  f"pkg={getattr(graph, 'property_package', '?')}", flush=True)

            print("[GP] step: params.assign START", flush=True)
            graph = self._params.assign(graph, description=desc)
            print("[GP] step: params.assign END", flush=True)

            print("[GP] step: consistency.apply START", flush=True)
            graph, consistency_changes = self._consistency.apply(graph)
            print(f"[GP] step: consistency.apply END  "
                  f"changes={len(consistency_changes)}", flush=True)
            new_warns += [f"[consistency] {c}" for c in consistency_changes]

            print("[GP] step: rule_store.apply_to_graph START", flush=True)
            graph, rule_changes = self._rule_store.apply_to_graph(graph, compounds)
            print(f"[GP] step: rule_store.apply_to_graph END  "
                  f"changes={len(rule_changes)}", flush=True)
            new_warns += [f"[rule] {c}" for c in rule_changes]

        except Exception as exc:
            print(f"[GP] Stage 3 EXCEPTION: {type(exc).__name__}: {exc}", flush=True)
            return {"outcome": "PLAN_FAILED", "warnings": [f"Stage 3 failed: {exc}"]}

        print("[GP] step: normalise(graph) #2 START", flush=True)
        graph = normalise(graph)
        print("[GP] step: normalise(graph) #2 END", flush=True)

        print(f"[GP] step: validate(graph) #2 START  "
              f"graph.compounds={graph.compounds}", flush=True)
        post_report = validate(graph)
        print(f"[GP] step: validate(graph) #2 END  valid={post_report.valid}", flush=True)
        if not post_report.valid:
            new_warns += [str(e) for e in post_report.errors()]
            return {"ir_graph": graph, "outcome": "INVALID_JSON", "warnings": new_warns}

        # Load reference data (Stage 3→4 bridge)
        reference_data: Optional[dict] = None
        reference_file = state["reference_file"]
        if reference_file:
            import json as _json
            from pathlib import Path as _Path
            _candidates = [
                _Path(reference_file),
                _Path(__file__).resolve().parent.parent / reference_file,
            ]
            for _p in _candidates:
                try:
                    with open(_p) as _rf:
                        reference_data = _json.load(_rf)
                    print(f"[GP] loaded reference data from '{_p}'",
                          flush=True, file=sys.stderr)
                    break
                except FileNotFoundError:
                    continue
                except Exception as _ref_exc:
                    print(f"[GP] WARNING: error reading reference_file: {_ref_exc}",
                          flush=True, file=sys.stderr)
                    break
            else:
                print(f"[GP] WARNING: reference_file '{reference_file}' not found",
                      flush=True, file=sys.stderr)

        # Validation tier: seed ConversionReactor reaction + conversion from the
        # reference before serialising (mirrors orchestrator_v2). ParamMapper
        # inserts reaction="" when LLM extraction fails; this overrides it with
        # the consistency-verified reference stoichiometry.
        if state["tier"] == "validation" and reference_data is not None:
            _inject_reference_reactor_params(graph, reference_data)

        print("[GP] step: to_dwsim(graph) START", flush=True)
        dwsim_json = to_dwsim(graph, reference_data=reference_data)
        print("[GP] step: to_dwsim(graph) END", flush=True)

        return {
            "ir_graph":       graph,
            "dwsim_json":     dwsim_json,
            "reference_data": reference_data,
            "tried_packages": [graph.property_package],
            "warnings":       new_warns,
        }

    def _execute_node(self, state: PipelineState) -> dict:
        iteration    = state["iteration"]
        eff_max_iter = state["eff_max_iter"]

        # Iteration ceiling check (mirrors the for-loop break in orchestrator_v2)
        if iteration >= eff_max_iter:
            return {"outcome": "MAX_ITER"}

        # Defensive: a topology_repair → repair hand-off can reach the execute
        # loop before thermo built the flowsheet.  With no flowsheet there is
        # nothing to run — escalate rather than crash the executor on None.
        if state["dwsim_json"] is None:
            print("[GP] execute: no dwsim_json (flowsheet not built) — HUMAN",
                  flush=True, file=sys.stderr)
            return {"outcome": "HUMAN"}

        # RepairMemory: init on first call, carry across iterations
        repair_memory: RepairMemory = state["repair_memory"] or RepairMemory()
        repair_memory.tick()

        # Hash tracking for loop-detection logging
        _cur_hash = hashlib.md5(
            json.dumps(state["dwsim_json"], sort_keys=True).encode()
        ).hexdigest()[:8]
        _prev = state["prev_hash"]
        _note = "UNCHANGED" if _cur_hash == _prev else f"changed  prev={_prev}"
        print(f"[GP] iteration={iteration} flowsheet_hash={_cur_hash} ({_note})",
              flush=True, file=sys.stderr)

        print(f"[GP] step: executor.run iteration={iteration} START", flush=True)
        execution = self._executor.run(state["dwsim_json"])
        print(f"[GP] step: executor.run iteration={iteration} END  "
              f"solved={getattr(execution, 'solved', '?')}", flush=True)

        sim_hints = SimulationHints.from_execution(execution, iteration=iteration)

        # ── Solved path ───────────────────────────────────────────────────────
        if getattr(execution, "solved", False) and _no_critic_failures(execution):
            dwsim_json = state["dwsim_json"]
            ref_data   = state["reference_data"]
            new_warns: list[str] = []
            if ref_data is not None:
                _ref_json, _ref_exec, _ref_changes = _reference_guided_refinement(
                    state["ir_graph"], execution, ref_data, self._executor)
                if _ref_exec is not None and getattr(_ref_exec, "solved", False):
                    dwsim_json = _ref_json
                    execution  = _ref_exec
                    new_warns  = list(_ref_changes)
                    print(f"[GP] reference-guided refinement: "
                          f"{len(_ref_changes)} correction(s)",
                          flush=True, file=sys.stderr)

            iter_record = IterationRecord(
                iteration=iteration, errors=[], changes=["PASS"],
                flowsheet=dwsim_json, execution=execution, elapsed_s=0.0)
            return {
                "outcome":        "PASS",
                "execution":      execution,
                "dwsim_json":     dwsim_json,
                "repair_memory":  repair_memory,
                "sim_hints":      sim_hints,
                "prev_hash":      _cur_hash,
                "warnings":       new_warns,
                "iterations_log": [iter_record],
            }

        # ── Classify errors ───────────────────────────────────────────────────
        errors = self._classifier.classify(execution, state["ir_graph"])
        print(f"[REPAIR] iteration={iteration} "
              f"n_errors={len(errors)} "
              f"strategies={[e.repair_strategy.value for e in errors]} "
              f"targets={[str(e.target) for e in errors]}",
              flush=True, file=sys.stderr)
        for _e in errors:
            print(f"[REPAIR]   error: {_e}", flush=True, file=sys.stderr)

        # ── Terminal path ─────────────────────────────────────────────────────
        if any(e.is_terminal for e in errors):
            iter_record = IterationRecord(
                iteration=iteration, errors=errors, changes=[],
                flowsheet=state["dwsim_json"], execution=execution, elapsed_s=0.0)
            return {
                "outcome":        "HUMAN",
                "execution":      execution,
                "errors":         errors,
                "repair_memory":  repair_memory,
                "sim_hints":      sim_hints,
                "prev_hash":      _cur_hash,
                "iterations_log": [iter_record],
            }

        # ── Beam search extension ─────────────────────────────────────────────
        beam_extended   = state["beam_extended"]
        eff_max_iter_new = eff_max_iter
        _BEAM_MAX_ITER  = 15
        if not beam_extended:
            n_cond = sum(
                1 for e in errors
                if e.repair_strategy == _RepairStrategy.CONDITION_FIX)
            if n_cond > 1:
                beam_extended    = True
                eff_max_iter_new = _BEAM_MAX_ITER
                print(f"[GP] beam search active (n_cond_errors={n_cond}): "
                      f"extending max_iter {eff_max_iter} → {_BEAM_MAX_ITER}",
                      flush=True, file=sys.stderr)

        # Continue to repair_node
        return {
            "execution":     execution,
            "errors":        errors,
            "repair_memory": repair_memory,
            "sim_hints":     sim_hints,
            "prev_hash":     _cur_hash,
            "eff_max_iter":  eff_max_iter_new,
            "beam_extended": beam_extended,
            "outcome":       "CONTINUE",
        }

    def _repair_node(self, state: PipelineState) -> dict:
        iteration     = state["iteration"]
        graph         = state["ir_graph"]
        errors        = state["errors"]
        tried_pkgs    = list(state["tried_packages"])
        repair_memory = state["repair_memory"]
        sim_hints     = state["sim_hints"] or EMPTY_HINTS
        execution     = state["execution"]
        ref_data      = state["reference_data"]
        new_warns: list[str] = []

        try:
            _params_before = {u.tag: dict(u.params) for u in graph.units()}

            graph, changes = self._repair.repair(
                graph, errors, tried_pkgs,
                description=state["norm_desc"],
                memory=repair_memory,
                sim_hints=sim_hints,
            )

            _params_after = {u.tag: dict(u.params) for u in graph.units()}
            _param_diffs  = {
                tag: {p: (_params_before[tag].get(p), v)
                      for p, v in _params_after[tag].items()
                      if _params_before.get(tag, {}).get(p) != v}
                for tag in _params_after
                if _params_before.get(tag) != _params_after.get(tag)
            }
            print(f"[REPAIR] iteration={iteration} "
                  f"changes={changes[:6]} param_diffs={_param_diffs}",
                  flush=True, file=sys.stderr)
            if not _param_diffs:
                print(f"[REPAIR] WARNING: iteration={iteration} "
                      "no unit params changed — repair loop is stuck",
                      flush=True, file=sys.stderr)

            _record_repairs_in_store(
                errors, changes, graph, self._rule_store,
                compounds=state["compounds"])
            try:
                self._rule_store.save(RULES_PATH)
            except Exception as _save_exc:
                print(f"[GP] rule_store.save failed: {_save_exc}",
                      flush=True, file=sys.stderr)

            graph = normalise(graph)
            post  = validate(graph)
            if not post.valid:
                new_warns += [str(e) for e in post.errors()]

            if graph.property_package not in tried_pkgs:
                tried_pkgs = tried_pkgs + [graph.property_package]

            # Validation tier: re-seed reactor reaction/conversion from the
            # reference after repair, since the repair agent may have rebuilt
            # or perturbed the reactor node's params.
            if state["tier"] == "validation" and ref_data is not None:
                _inject_reference_reactor_params(graph, ref_data)

            dwsim_json = to_dwsim(graph, reference_data=ref_data)

            iter_record = IterationRecord(
                iteration=iteration, errors=errors, changes=changes,
                flowsheet=dwsim_json, execution=execution, elapsed_s=0.0)

            return {
                "ir_graph":       graph,
                "dwsim_json":     dwsim_json,
                "tried_packages": tried_pkgs,
                "iteration":      iteration + 1,
                "warnings":       new_warns,
                "iterations_log": [iter_record],
                "outcome":        "CONTINUE",
            }

        except Exception as _repair_exc:
            print(f"[GP] repair/normalise error iter={iteration}: {_repair_exc}",
                  flush=True, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            iter_record = IterationRecord(
                iteration=iteration, errors=errors, changes=[],
                flowsheet=state["dwsim_json"], execution=execution, elapsed_s=0.0)
            # Force MAX_ITER to exit the loop on the next execute_node call by
            # setting iteration to eff_max_iter — mirrors the orchestrator's break.
            return {
                "iteration":      state["eff_max_iter"],
                "warnings":       [f"repair error iter={iteration}: {_repair_exc}"],
                "iterations_log": [iter_record],
                "outcome":        "CONTINUE",
            }
