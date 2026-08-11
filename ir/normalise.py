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
# Column outlet naming: distillate → port 0, bottoms → port 1.
_DIST_KEYWORDS = frozenset({"DIST", "DISTILLATE", "TOP", "OVER", "OVHD",
                            "OVERHEAD", "LIGHT"})
_BOT_KEYWORDS  = frozenset({"BOT", "BOTTOM", "BOTT", "BTMS", "BASE", "HEAVY",
                            "RESIDUE"})
# Decanter vapour outlet naming (the two liquids take ports 1 and 2).
_DEC_VAP_KEYWORDS = frozenset({"VAP", "VAPOR", "VAPOUR", "GAS"})


def normalise(graph: FlowsheetGraph) -> FlowsheetGraph:
    _compounds        = list(graph.compounds)
    _property_package = graph.property_package
    g = graph.copy()
    g = _insert_mixers(g)
    g = _insert_splitters(g)
    g = _complete_separator_outlets(g)   # before port repair: synthesise missing outlets
    g = _repair_vessel_ports(g)
    g = _repair_column_ports(g)
    g = _repair_decanter_ports(g)
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
    For any unit (except Mixer) that would receive more than one physical inlet,
    insert an auto-generated Mixer upstream and route all inlets through it.

    "Effective inlets" include:
      (a) streams already wired to the unit via graph edges (inlet_streams)
      (b) is_recycle streams whose recycle_target equals the unit but that have
          dst=None — their INIT stream lands on this unit during to_dwsim
          connection-building and acts as an additional inlet.  On a single-inlet
          unit (Compressor, Heater, Cooler, etc.) both the feed and the INIT
          stream would target port 0, causing a port collision in DWSIM.
          Inserting a Mixer here routes the INIT stream to the Mixer (which
          accepts multiple ports) instead of directly to the single-inlet unit.
    """
    g = graph.copy()
    changed = True
    while changed:
        changed = False
        for node in g.units():
            if node.unit_type == "Mixer":
                continue
            inlets = g.inlet_streams(node.tag)
            # Recycle streams that target this unit via recycle_target but have
            # no graph edge yet (dst=None).  Their INIT stream will become a
            # physical inlet in DWSIM, so they must be counted here.
            unlinked_recycles = [
                s for s in g.streams()
                if s.is_recycle
                and s.recycle_target == node.tag
                and g.stream_dest(s.tag) is None
            ]
            if len(inlets) + len(unlinked_recycles) <= 1:
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

            # Detach each graph-wired inlet from node, re-attach to mixer
            for stream in inlets:
                src = g.stream_source(stream.tag)
                # Remove existing edge: src → stream → node
                g._g.remove_edge(stream.tag, node.tag)
                # Attach stream → mixer instead
                g._g.add_edge(stream.tag, mixer_tag)

            # Wire unlinked recycle streams into the mixer so that
            # stream_dest() returns mixer_tag.  _build_connections will then
            # route the INIT stream to the Mixer (multi-inlet) rather than
            # directly to node (single-inlet, port 0 already taken by the feed).
            for stream in unlinked_recycles:
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

    Recycle streams are excluded from both the outlet count and the Splitter
    rerouting: they are not product splits and must remain connected directly
    to their source unit so that validate_dag() and to_dwsim() can identify
    them correctly.
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
            # Recycle streams are not physical product splits — exclude from count
            # so a unit with one product outlet and one recycle outlet does not
            # incorrectly get a Splitter inserted.
            non_recycle_outlets = [s for s in outlets if not s.is_recycle]
            if len(non_recycle_outlets) <= max_outlets:
                continue

            splitter_tag = f"SPL-{node.tag}"
            link_tag     = f"S-{node.tag}-SPL"

            if g.unit(splitter_tag):
                continue

            n = len(non_recycle_outlets)
            equal_frac = round(1.0 / n, 6)
            split_fracs = {s.tag: equal_frac for s in non_recycle_outlets}

            splitter = make_node(
                "Splitter", splitter_tag,
                params={"dP": 0.0, "split_fractions": split_fracs},
                metadata={"auto_inserted": True},
            )
            g.add_unit(splitter)

            # Detach only non-recycle outlets from node, attach to splitter.
            # Assign incremental src_ports so they are distinct on the Splitter
            # (all streams had src_port=0 from the original single-outlet unit).
            # Recycle outlets stay connected to node unchanged.
            for port_idx, stream in enumerate(non_recycle_outlets):
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


def _complete_separator_outlets(graph: FlowsheetGraph) -> FlowsheetGraph:
    """
    Synthesise missing product outlets so a Column (needs distillate + bottoms)
    or Decanter (needs two liquid outlets) is structurally valid. Extraction
    sometimes wires only one product stream to a column ("Column needs ≥2
    outlets, has 1"); the missing outlet is added as a TERMINATED product stream
    (dst=None, a boundary), never left dangling. Runs BEFORE port repair so the
    name-based repair then assigns the correct ports across all outlets.
    """
    g = graph.copy()
    for node in g.units():
        outs = g.outlet_streams(node.tag)
        if not outs:
            # No outlets at all — a degenerate emission; leave for validation to
            # flag rather than fabricating an entire product set.
            continue

        if node.unit_type == "Column":
            if len(outs) >= 2:
                continue
            # Exactly one outlet: synthesise the MISSING semantic role by name, on
            # a free port. _repair_column_ports then assigns the correct ports.
            existing = outs[0]
            is_bot = any(k in existing.tag.upper() for k in _BOT_KEYWORDS)
            label = "DIST" if is_bot else "BOT"   # add the other product
            free_port = 1 if existing.src_port == 0 else 0
            tag = _unique_stream_tag(g, f"{node.tag}-{label}")
            g.add_stream(EdgeIR(tag=tag, src_port=free_port, phase="mixed"),
                         node.tag, None, enforce_phase=False)

        elif node.unit_type == "Decanter":
            # Needs two LIQUID outlets (vapour port 0 optional). Count non-vapour
            # outlets; add liquid product streams until there are two.
            liq = [s for s in outs
                   if not any(k in s.tag.upper() for k in _DEC_VAP_KEYWORDS)]
            used = {s.src_port for s in outs}
            nxt = 1
            while len(liq) < 2:
                while nxt in used:
                    nxt += 1
                tag = _unique_stream_tag(g, f"{node.tag}-L{len(liq) + 1}")
                g.add_stream(EdgeIR(tag=tag, src_port=nxt, phase="liquid"),
                             node.tag, None, enforce_phase=False)
                used.add(nxt)
                liq.append(tag)
    return g


def _unique_stream_tag(graph: FlowsheetGraph, base: str) -> str:
    existing = set(graph.stream_tags())
    if base not in existing:
        return base
    i = 2
    while f"{base}-{i}" in existing:
        i += 1
    return f"{base}-{i}"


def _repair_column_ports(graph: FlowsheetGraph) -> FlowsheetGraph:
    """
    Column outlet ports MUST be distillate → 0, bottoms → 1 (the DWSIM
    ShortcutColumn distillate/bottoms port order). The GraphBuilder assigns ports
    by stream ORDER, which is not semantically meaningful; reassign by name so the
    products wire to the correct ports regardless of emission order.
    """
    g = graph.copy()
    for node in g.units():
        if node.unit_type != "Column":
            continue
        outlets = g.outlet_streams(node.tag)
        if len(outlets) != 2:
            continue
        a, b = outlets
        au, bu = a.tag.upper(), b.tag.upper()
        a_bot  = any(k in au for k in _BOT_KEYWORDS)
        b_bot  = any(k in bu for k in _BOT_KEYWORDS)
        a_dist = any(k in au for k in _DIST_KEYWORDS)
        b_dist = any(k in bu for k in _DIST_KEYWORDS)
        dist = bot = None
        if   a_bot and not b_bot:   bot, dist = a, b
        elif b_bot and not a_bot:   bot, dist = b, a
        elif a_dist and not b_dist: dist, bot = a, b
        elif b_dist and not a_dist: dist, bot = b, a
        if dist is not None and bot is not None:
            dist.src_port = 0
            bot.src_port  = 1
            bot.phase     = "liquid"
    return g


def _repair_decanter_ports(graph: FlowsheetGraph) -> FlowsheetGraph:
    """
    Decanter outlet ports: vapour → 0, the two liquid phases → 1 and 2 (DWSIM
    Vessel VLLE port order). The vapour outlet (often zero-flow) is identified by
    name; the remaining outlets take the liquid ports. Handles the common
    liquid-only decanter (2 outlets) by leaving port 0 for the vapour.
    """
    g = graph.copy()
    for node in g.units():
        if node.unit_type != "Decanter":
            continue
        outlets = g.outlet_streams(node.tag)
        vap = None
        liqs: list[EdgeIR] = []
        for s in outlets:
            u = s.tag.upper()
            if any(k in u for k in _DEC_VAP_KEYWORDS) and not any(
                    k in u for k in _LIQ_KEYWORDS):
                vap = s
            else:
                liqs.append(s)
        if vap is not None:
            vap.src_port = 0
            vap.phase    = "vapour"
        for i, s in enumerate(liqs[:2]):
            s.src_port = 1 + i
            s.phase    = "liquid"
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
