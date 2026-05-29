"""
Deterministic graph normalisation. Zero LLM calls.

Transforms applied in order:
  1. Mixer insertion   — unit with >1 inlets gets an auto-inserted Mixer upstream
  2. Splitter insertion — unit with >1 outlets gets an auto-inserted Splitter downstream
  3. Vessel port repair — swap src_port 0/1 when stream name implies inversion
  4. Phase label propagation — set stream.phase from downstream unit port spec

Each transform returns a new FlowsheetGraph; the input is never mutated.
"""
from __future__ import annotations

from ir.graph import FlowsheetGraph, NodeIR, EdgeIR, PORT_SPECS, make_node

_VAP_KEYWORDS = frozenset({"VAP", "VAPOR", "VAPOUR", "GAS", "TOP", "OVER", "DIST"})
_LIQ_KEYWORDS = frozenset({"LIQ", "LIQUID", "BOT", "BOTTOM", "BOTT", "BASE"})


def normalise(graph: FlowsheetGraph) -> FlowsheetGraph:
    _compounds        = list(graph.compounds)
    _property_package = graph.property_package
    g = graph.copy()
    g = _insert_mixers(g)
    g = _insert_splitters(g)
    g = _repair_vessel_ports(g)
    g = _propagate_phases(g)
    # Defensive: deepcopy of NetworkX graph can silently drop plain-attribute fields
    # if the graph has no nodes at copy time; re-stamp them from the pre-copy values.
    if not g.compounds and _compounds:
        g.compounds = _compounds
    if not g.property_package and _property_package:
        g.property_package = _property_package
    return g


# ── Mixer insertion ────────────────────────────────────────────────────────────

def _insert_mixers(graph: FlowsheetGraph) -> FlowsheetGraph:
    """
    For any unit (except Mixer) that receives more than one inlet stream,
    insert an auto-generated Mixer upstream and route all inlets through it.
    """
    g = graph.copy()
    changed = True
    while changed:
        changed = False
        for node in g.units():
            if node.unit_type == "Mixer":
                continue
            inlets = g.inlet_streams(node.tag)
            if len(inlets) <= 1:
                continue

            mixer_tag  = f"MIX-{node.tag}"
            link_tag   = f"S-{mixer_tag}-OUT"

            # Avoid duplicate auto-insertion on repeated passes
            if g.unit(mixer_tag):
                continue

            mixer = make_node("Mixer", mixer_tag,
                              params={"dP": 0.0},
                              metadata={"auto_inserted": True})
            g.add_unit(mixer)

            # Detach each inlet from node, re-attach to mixer
            for stream in inlets:
                src = g.stream_source(stream.tag)
                # Remove existing edge: src → stream → node
                g._g.remove_edge(stream.tag, node.tag)
                # Attach stream → mixer instead
                g._g.add_edge(stream.tag, mixer_tag)

            # Add new stream: mixer → node
            link = EdgeIR(tag=link_tag, metadata={"auto_inserted": True})
            g.add_stream(link, mixer_tag, node.tag)

            changed = True
            break  # restart scan after topology change
    return g


# ── Splitter insertion ─────────────────────────────────────────────────────────

def _insert_splitters(graph: FlowsheetGraph) -> FlowsheetGraph:
    """
    For any unit whose outlet stream count exceeds the number of outlet ports
    defined in PORT_SPECS, insert an auto-generated Splitter downstream.

    Vessel has 2 outlet port specs (vapour + liquid) so 2 outlets is correct
    and must NOT trigger insertion.  Only a Heater/Cooler/etc. with >1 outlet
    needs a Splitter.
    """
    g = graph.copy()
    changed = True
    while changed:
        changed = False
        for node in g.units():
            if node.unit_type == "Splitter":
                continue
            specs = PORT_SPECS.get(node.unit_type, [])
            max_outlets = len([s for s in specs if s.direction == "outlet"])
            outlets = g.outlet_streams(node.tag)
            if len(outlets) <= max_outlets:
                continue

            splitter_tag = f"SPL-{node.tag}"
            link_tag     = f"S-{node.tag}-SPL"

            if g.unit(splitter_tag):
                continue

            n = len(outlets)
            equal_frac = round(1.0 / n, 6)
            split_fracs = {s.tag: equal_frac for s in outlets}

            splitter = make_node(
                "Splitter", splitter_tag,
                params={"dP": 0.0, "split_fractions": split_fracs},
                metadata={"auto_inserted": True},
            )
            g.add_unit(splitter)

            # Detach each outlet from node, attach to splitter.
            # Assign incremental src_ports so they are distinct on the Splitter
            # (all streams had src_port=0 from the original single-outlet unit).
            for port_idx, stream in enumerate(outlets):
                dst = g.stream_dest(stream.tag)
                g._g.remove_edge(node.tag, stream.tag)
                g._g.add_edge(splitter_tag, stream.tag)
                stream.src_port = port_idx

            # Add new stream: node → splitter
            link = EdgeIR(tag=link_tag, metadata={"auto_inserted": True})
            g.add_stream(link, node.tag, splitter_tag)

            changed = True
            break
    return g


# ── Vessel port repair ─────────────────────────────────────────────────────────

def _repair_vessel_ports(graph: FlowsheetGraph) -> FlowsheetGraph:
    """
    Swap Vessel src_port 0 and 1 when stream-name heuristics indicate inversion:
      - port 0 carries a stream whose name contains a liquid keyword
      - port 1 carries a stream whose name contains a vapour keyword
    """
    g = graph.copy()
    for node in g.units():
        if node.unit_type != "Vessel":
            continue

        outlets = g.outlet_streams(node.tag)
        if len(outlets) != 2:
            continue

        # Identify the two outlet streams by src_port
        by_port: dict[int, EdgeIR] = {}
        for s in outlets:
            by_port[s.src_port] = s

        s0 = by_port.get(0)
        s1 = by_port.get(1)
        if s0 is None or s1 is None:
            continue

        p0_upper = s0.tag.upper()
        p1_upper = s1.tag.upper()
        p0_is_liq = any(k in p0_upper for k in _LIQ_KEYWORDS)
        p1_is_vap = any(k in p1_upper for k in _VAP_KEYWORDS)

        if p0_is_liq or p1_is_vap:
            # Swap ports in place
            s0.src_port = 1
            s1.src_port = 0
            s0.phase    = "liquid"
            s1.phase    = "vapour"

    return g


# ── Phase label propagation ────────────────────────────────────────────────────

def _propagate_phases(graph: FlowsheetGraph) -> FlowsheetGraph:
    """
    Assign stream.phase from the downstream unit's inlet PortSpec.
    Only overwrites phase="mixed" (the default) to avoid clobbering
    explicit assignments from _repair_vessel_ports.
    """
    g = graph.copy()
    for node in g.units():
        specs = PORT_SPECS.get(node.unit_type, [])
        outlet_specs = [s for s in specs if s.direction == "outlet"]
        if not outlet_specs:
            continue
        for stream in g.outlet_streams(node.tag):
            if stream.phase != "mixed":
                continue
            # Match by src_port
            matching = [s for s in outlet_specs if s.port_id == stream.src_port]
            if matching and matching[0].phase != "any":
                stream.phase = matching[0].phase
    return g
