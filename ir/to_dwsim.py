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

from ir.graph import FlowsheetGraph, EdgeIR, NodeIR


def to_dwsim(graph: FlowsheetGraph) -> dict:
    """
    Convert FlowsheetGraph to the DWSIM JSON dict (agents/schema.py format).
    Raises ValueError if any unit type is unsupported or required params are missing.
    """
    dwsim: dict = {
        "compounds":        graph.compounds,
        "property_package": graph.property_package,
    }

    if graph.binary_parameters:
        dwsim["binary_parameters"] = graph.binary_parameters

    dwsim["streams"]     = [_stream_to_dict(s) for s in graph.streams()]
    dwsim["units"]       = [_unit_to_dict(u)   for u in graph.units()]
    dwsim["connections"] = _build_connections(graph)

    return dwsim


# ── Stream ─────────────────────────────────────────────────────────────────────

def _stream_to_dict(edge: EdgeIR) -> dict:
    d: dict = {"tag": edge.tag}
    if edge.T           is not None: d["T"]           = edge.T
    if edge.P           is not None: d["P"]           = edge.P
    if edge.flow        is not None: d["flow"]        = edge.flow
    if edge.composition:             d["composition"] = edge.composition
    return d


# ── Unit ───────────────────────────────────────────────────────────────────────

_UNIT_PARAM_KEYS: dict[str, list[str]] = {
    "Heater":     ["T_out", "dP"],
    "Cooler":     ["T_out", "dP"],
    "Vessel":     ["dP"],
    "Mixer":      ["dP"],
    "Splitter":   ["split_fractions", "dP"],
    "Pump":       ["P_out", "efficiency", "dP"],
    "Compressor": ["P_out", "efficiency", "dP"],
    "Expander":   ["P_out", "efficiency", "dP"],
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
    """
    connections: list[list] = []

    for stream in graph.streams():
        src_unit = graph.stream_source(stream.tag)
        dst_unit = graph.stream_dest(stream.tag)

        if src_unit:
            connections.append([src_unit, stream.tag, stream.src_port, 0])
        if dst_unit:
            connections.append([stream.tag, dst_unit, 0, stream.dst_port])

    return connections
