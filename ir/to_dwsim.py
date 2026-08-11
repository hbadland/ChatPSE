"""
Deterministic IR → DWSIM JSON mapping.

Converts a FlowsheetGraph into the dict format consumed by agents/schema.py
and agents/executor.py.  No LLM calls.  Raises ValueError if the graph
contains units or streams that cannot be unambiguously mapped.

Connection format (DWSIM): [src_tag, dst_tag, src_port, dst_port]
  - When src is a unit and dst is a stream: [unit_tag, stream_tag, src_port, 0]
  - When src is a stream and dst is a unit: [stream_tag, unit_tag, 0, dst_port]

The executor expects connections as (src, dst, src_port, dst_port) where src/dst
are always stream tags or unit tags — we emit the flattened DWSIM convention used
by the existing executor:  [src_unit, stream_tag, src_port, 0] for unit outputs
and the stream itself encodes the dst implicitly via its appearance as a unit input.

Actually the existing schema uses:  [src_tag, dst_tag, src_port, dst_port]
where all four elements can be stream or unit tags.  We follow that exactly.
"""
from __future__ import annotations

from typing import Optional

from ir.graph import FlowsheetGraph, EdgeIR, NodeIR


def to_dwsim(graph: FlowsheetGraph,
             reference_data: Optional[dict] = None) -> dict:
    """
    Convert FlowsheetGraph to the DWSIM JSON dict (agents/schema.py format).
    Raises ValueError if any unit type is unsupported or required params are missing.

    When the graph contains recycle edges (is_recycle=True), each recycle stream
    is split into two MaterialStreams:
      - <tag>        : the calculated stream (outlet of downstream unit → Recycle block)
      - <tag>-INIT   : the assumed stream   (Recycle block → upstream unit, initial guess)
    A "recycle_blocks" list is added so agents/executor.py can call add_recycle_block().

    reference_data : parsed reference flowsheet dict (from a reference JSON file).
      When provided and the graph has recycles, the INIT stream conditions are
      seeded from the reference stream whose composition most closely matches the
      recycle edge — giving DWSIM the correct starting point for convergence.
    """
    dwsim: dict = {
        "compounds":        graph.compounds,
        "property_package": graph.property_package,
    }

    if graph.binary_parameters:
        dwsim["binary_parameters"] = graph.binary_parameters

    dwsim["streams"]     = _build_streams(graph, reference_data)
    dwsim["units"]       = [_unit_to_dict(u) for u in graph.units()]
    dwsim["connections"] = _build_connections(graph)

    # DWSIM's ConversionReactor exposes two material outlet ports (vapour +
    # liquid); both must carry a material stream or solve() fails. The IR models
    # a reactor with one product outlet, so add a dead-end stream for any
    # unconnected material outlet port.
    _extra_s, _extra_c = _ensure_reactor_outlets(graph, dwsim["connections"])
    dwsim["streams"].extend(_extra_s)
    dwsim["connections"].extend(_extra_c)

    recycle_blocks = _build_recycle_blocks(graph)
    if recycle_blocks:
        dwsim["recycle_blocks"] = recycle_blocks

    return dwsim


# ── Stream ─────────────────────────────────────────────────────────────────────

def _build_streams(graph: FlowsheetGraph,
                   reference_data: Optional[dict] = None) -> list[dict]:
    """Build the stream list, appending a synthetic init stream for each recycle edge."""
    result = []
    for s in graph.streams():
        result.append(_stream_to_dict(s))
        if s.is_recycle:
            result.append(_recycle_init_stream_dict(s, graph, reference_data))
    return result


def _stream_to_dict(edge: EdgeIR) -> dict:
    d: dict = {"tag": edge.tag}
    if edge.T           is not None: d["T"]           = edge.T
    if edge.P           is not None: d["P"]           = edge.P
    if edge.flow        is not None: d["flow"]        = edge.flow
    if edge.composition:             d["composition"] = edge.composition
    return d


def _recycle_init_stream_dict(edge: EdgeIR,
                               graph: FlowsheetGraph,
                               reference_data: Optional[dict] = None) -> dict:
    """Create the assumed/initial stream for a recycle block outlet.

    Estimation priority:
      1. Reference file  — find the reference stream whose composition best
         matches the resolved composition and seed T/P/flow from it.
      2. Physics-based   — derive T from the source unit's T_out param,
         P from P_out or inlet stream pressure, flow from any feed stream.
      3. Fallback        — 300 K / 101325 Pa / 1 mol/s (last resort only).

    Composition priority:
      edge.composition (if non-empty) → feed stream composition → omitted.
    """
    d: dict = {"tag": f"{edge.tag}-INIT"}

    # Resolve composition: use edge's own first, then the Mixer-adjacency heuristic
    # (secondary entrainer/solvent feed when recycle targets a Mixer with a
    # materially distinct second feed), then the first feed stream as a final fallback.
    _comp = (edge.composition
             or _mixer_secondary_feed_composition(edge, graph)
             or _feed_stream_composition(graph))

    import sys as _sys

    # ── 1. Reference file ──────────────────────────────────────────────────────
    if reference_data and _comp:
        ref_streams = reference_data.get("streams", {})
        if isinstance(ref_streams, dict) and ref_streams:
            best_tag, best_dist = _best_ref_stream_by_composition(
                _comp, ref_streams)
            if best_tag is not None:
                ref_s = ref_streams[best_tag]
                src = "edge comp" if edge.composition else "feed fallback"
                print(
                    f"[TO_DWSIM] recycle INIT '{edge.tag}-INIT': seeded from "
                    f"reference stream '{best_tag}' (comp_dist={best_dist:.4f}, "
                    f"comp_src={src})",
                    flush=True, file=_sys.stderr)
                d["T"]    = float(ref_s.get(
                    "T_K",       edge.T    if edge.T    is not None
                                 else _estimate_recycle_T(edge, graph)))
                d["P"]    = float(ref_s.get(
                    "P_Pa",      edge.P    if edge.P    is not None
                                 else _estimate_recycle_P(edge, graph)))
                d["flow"] = float(ref_s.get(
                    "flow_mol_s", edge.flow if edge.flow is not None
                                  else _estimate_recycle_flow(edge, graph)))
                d["composition"] = _comp
                return d

    # ── 2. Physics-based estimate ──────────────────────────────────────────────
    T_est    = edge.T    if edge.T    is not None else _estimate_recycle_T(edge, graph)
    P_est    = edge.P    if edge.P    is not None else _estimate_recycle_P(edge, graph)
    flow_est = edge.flow if edge.flow is not None else _estimate_recycle_flow(edge, graph)

    src = "edge comp" if edge.composition else ("feed fallback" if _comp else "none")
    print(
        f"[TO_DWSIM] recycle INIT '{edge.tag}-INIT': physics estimate "
        f"T={T_est:.1f} K  P={P_est:.0f} Pa  flow={flow_est:.4f} mol/s  "
        f"comp_src={src}",
        flush=True, file=_sys.stderr)

    d["T"]    = T_est
    d["P"]    = P_est
    d["flow"] = flow_est
    if _comp:
        d["composition"] = _comp
    return d


# ── Recycle INIT estimation helpers ───────────────────────────────────────────

def _feed_stream_composition(graph: FlowsheetGraph) -> dict:
    """Return the composition of the first non-recycle feed stream (no source unit).

    Used as a fallback when a recycle edge carries no composition, ensuring
    DWSIM always receives a non-empty composition dict for VLE initialisation.
    Returns {} when no feed stream with a composition can be found.
    """
    for s in graph.streams():
        if not s.is_recycle and graph.stream_source(s.tag) is None and s.composition:
            return s.composition
    return {}


def _mixer_secondary_feed_composition(edge: EdgeIR, graph: FlowsheetGraph) -> dict:
    """Mixer-adjacency heuristic: seed recycle INIT from a secondary entrainer feed.

    When the recycle target is a Mixer and a second non-recycle feed stream enters
    that Mixer with L1 distance > 0.3 from the primary feed, return the secondary
    feed's composition for use as the recycle INIT composition.

    Handles two topologies:
      - Explicit Mixer target: edge.recycle_target names an existing Mixer unit
        (graph.stream_dest returns None because no graph edge exists for the recycle).
      - Auto-inserted Mixer: normalise()._insert_mixers wired the recycle stream to
        an auto-generated Mixer; graph.stream_dest resolves to that Mixer's tag.

    Returns {} when the heuristic does not apply: target is not a Mixer, only one
    feed stream is present, or no secondary feed differs materially from the primary.
    """
    target_tag = graph.stream_dest(edge.tag) or edge.recycle_target
    if not target_tag:
        return {}
    target_unit = graph.unit(target_tag)
    if not target_unit or target_unit.unit_type != "Mixer":
        return {}

    feeds = [
        s for s in graph.inlet_streams(target_tag)
        if not s.is_recycle and graph.stream_source(s.tag) is None and s.composition
    ]
    if len(feeds) < 2:
        return {}

    primary_comp = feeds[0].composition

    def _l1(a: dict, b: dict) -> float:
        keys = set(a) | set(b)
        return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)

    for s in feeds[1:]:
        if _l1(s.composition, primary_comp) > 0.3:
            return s.composition

    return {}


def _best_ref_stream_by_composition(
        composition: dict,
        ref_streams: dict) -> tuple[Optional[str], float]:
    """Return (tag, L1_dist) for the reference stream whose mole-fraction
    composition is closest to the given composition dict.  Case-insensitive."""
    comp_lower = {k.lower(): float(v) for k, v in composition.items()}
    best_tag   = None
    best_dist  = float("inf")
    for tag, ref_s in ref_streams.items():
        ref_comp = ref_s.get("composition", {})
        if not ref_comp:
            continue
        ref_lower  = {k.lower(): float(v) for k, v in ref_comp.items()}
        all_keys   = set(comp_lower) | set(ref_lower)
        dist       = sum(abs(comp_lower.get(k, 0.0) - ref_lower.get(k, 0.0))
                         for k in all_keys)
        if dist < best_dist:
            best_dist = dist
            best_tag  = tag
    return best_tag, best_dist


def _estimate_recycle_T(edge: EdgeIR, graph: FlowsheetGraph) -> float:
    """Estimate T from source unit T_out, then source unit's inlets, then any stream."""
    src_tag = graph.stream_source(edge.tag)
    if src_tag:
        src_unit = graph.unit(src_tag)
        if src_unit:
            t = src_unit.params.get("T_out")
            if t is not None:
                return float(t)
        for inlet in graph.inlet_streams(src_tag):
            if inlet.T is not None:
                return float(inlet.T)
    # Any non-recycle stream that has T
    for s in graph.streams():
        if s.T is not None and not s.is_recycle:
            return float(s.T)
    return 300.0


def _estimate_recycle_P(edge: EdgeIR, graph: FlowsheetGraph) -> float:
    """Estimate P from source unit P_out, then source unit's inlets, then any stream."""
    src_tag = graph.stream_source(edge.tag)
    if src_tag:
        src_unit = graph.unit(src_tag)
        if src_unit:
            p = src_unit.params.get("P_out")
            if p is not None:
                return float(p)
        for inlet in graph.inlet_streams(src_tag):
            if inlet.P is not None:
                return float(inlet.P)
    for s in graph.streams():
        if s.P is not None and not s.is_recycle:
            return float(s.P)
    return 101325.0


def _estimate_recycle_flow(edge: EdgeIR, graph: FlowsheetGraph) -> float:
    """Estimate flow from feed streams (no source unit), then any stream."""
    for s in graph.streams():
        if not s.is_recycle and graph.stream_source(s.tag) is None and s.flow is not None:
            return float(s.flow)
    for s in graph.streams():
        if s.flow is not None and not s.is_recycle:
            return float(s.flow)
    return 1.0


# DWSIM Reactor_Conversion material outlet ports (port 2 is the energy stream).
_REACTOR_MATERIAL_OUTLET_PORTS = (0, 1)


def _ensure_reactor_outlets(graph: FlowsheetGraph,
                            connections: list[list]) -> tuple[list[dict], list[list]]:
    """Synthesize a dead-end product stream for each unconnected reactor material
    outlet port.  Returns (extra_streams, extra_connections) to append.

    DWSIM rejects a ConversionReactor at solve() unless BOTH material outlet
    ports (0 and 1) have a stream attached, even when one phase carries zero
    flow.  The IR only ever wires the single product outlet, so we fill the gap
    here at the DWSIM boundary rather than polluting the IR with a phantom edge.
    """
    extra_streams: list[dict] = []
    extra_conns:   list[list] = []
    for node in graph.units():
        if node.unit_type != "ConversionReactor":
            continue
        used_ports = {c[2] for c in connections if c[0] == node.tag}
        for port in _REACTOR_MATERIAL_OUTLET_PORTS:
            if port not in used_ports:
                tag = f"{node.tag}-OUT{port}"
                extra_streams.append({"tag": tag})
                extra_conns.append([node.tag, tag, port, 0])
    return extra_streams, extra_conns


def _build_recycle_blocks(graph: FlowsheetGraph) -> list[dict]:
    """Return one recycle block entry per recycle edge, or [] if none."""
    blocks = []
    for i, edge in enumerate(graph.recycle_edges(), start=1):
        blocks.append({
            "tag":           f"REC-{i:02d}",
            "inlet_stream":  edge.tag,
            "outlet_stream": f"{edge.tag}-INIT",
        })
    return blocks


# ── Unit ───────────────────────────────────────────────────────────────────────

_UNIT_PARAM_KEYS: dict[str, list[str]] = {
    "Heater":            ["T_out", "dP"],
    "Cooler":            ["T_out", "dP"],
    "Vessel":            ["dP"],
    "Mixer":             ["dP"],
    "Splitter":          ["split_fractions", "dP"],
    "Pump":              ["P_out", "efficiency", "dP"],
    "Compressor":        ["P_out", "efficiency", "dP"],
    "Expander":          ["P_out", "efficiency", "dP"],
    "ConversionReactor": ["temperature_K", "pressure_Pa", "conversion", "reaction"],
    "Column":            ["light_key", "heavy_key", "light_key_frac_bottoms",
                          "heavy_key_frac_distillate", "reflux_ratio",
                          "condenser_pressure_Pa", "boiler_pressure_Pa"],
    "Decanter":          ["dP"],
}

def _unit_to_dict(node: NodeIR) -> dict:
    d: dict = {"tag": node.tag, "type": node.unit_type}

    if node.property_package:
        d["property_package"] = node.property_package

    allowed_keys = _UNIT_PARAM_KEYS.get(node.unit_type, [])
    for key in allowed_keys:
        val = node.params.get(key)
        if val is not None:
            d[key] = val

    return d


# ── Connections ────────────────────────────────────────────────────────────────

def _build_connections(graph: FlowsheetGraph) -> list[list]:
    """
    Build the connections list in DWSIM format:
      [src_tag, dst_tag, src_port, dst_port]

    For each stream in the graph we emit up to two connection entries:
      1. upstream unit → stream     (src=unit, dst=stream, src_port=stream.src_port, dst_port=0)
      2. stream → downstream unit   (src=stream, dst=unit, src_port=0, dst_port=stream.dst_port)

    For recycle streams the routing changes:
      - The calc stream  (<tag>)      connects src_unit → calc_stream only.
        The calc_stream → Recycle block connection is handled by add_recycle_block().
      - The init stream  (<tag>-INIT) connects init_stream → dst_unit only.
        The Recycle block → init_stream connection is also handled by add_recycle_block().
    """
    connections: list[list] = []

    # Per-destination-unit inlet-port allocator.  Multi-inlet units (only the
    # Mixer, max_inlets > 1) need a distinct DWSIM inlet index for each incoming
    # stream — DWSIM rejects two connections to the same inlet port.  Single-inlet
    # units keep stream.dst_port (0), so serialization is unchanged for every
    # non-mixer unit and for mixers that have only one inlet.
    _inlet_next: dict[str, int] = {}

    def _dst_port(dst_unit_tag: str, default_port: int) -> int:
        node = graph.unit(dst_unit_tag)
        if node is not None and node.max_inlets() > 1:
            port = _inlet_next.get(dst_unit_tag, 0)
            _inlet_next[dst_unit_tag] = port + 1
            return port
        return default_port

    for stream in graph.streams():
        src_unit = graph.stream_source(stream.tag)
        dst_unit = graph.stream_dest(stream.tag)

        if stream.is_recycle:
            # Calc stream: downstream unit → calc stream (recycle block inlet)
            if src_unit:
                connections.append([src_unit, stream.tag, stream.src_port, 0])
            # Init stream: recycle block outlet → upstream unit
            if dst_unit:
                connections.append([f"{stream.tag}-INIT", dst_unit, 0,
                                    _dst_port(dst_unit, stream.dst_port)])
        else:
            if src_unit:
                connections.append([src_unit, stream.tag, stream.src_port, 0])
            if dst_unit:
                connections.append([stream.tag, dst_unit, 0,
                                    _dst_port(dst_unit, stream.dst_port)])

    return connections
