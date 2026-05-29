"""
Agent C — Graph Builder.

Input : SemanticUnits + SemanticTopology (from Stage 1) + compound list
Output: FlowsheetGraph (the canonical IR)

This is a deterministic assembly step — no LLM call.
It translates the flat semantic dicts into the graph IR, then hands off
to ir/normalise.py and ir/validate.py.

The graph is NOT yet simulation-ready: property_package and unit params
are populated by Stage 3 agents (ThermoMapper, ParamMapper).
"""
from __future__ import annotations

from ir.graph import FlowsheetGraph, EdgeIR, make_node
from agents.stage1.unit_extractor import SemanticUnits
from agents.stage1.stream_extractor import SemanticTopology


class GraphBuilder:
    """Assembles a FlowsheetGraph from Stage 1 semantic output. Zero LLM calls."""

    def build(
        self,
        units:     SemanticUnits,
        topology:  SemanticTopology,
        compounds: list[str],
    ) -> FlowsheetGraph:
        graph = FlowsheetGraph()
        graph.compounds = list(compounds)

        # Add all unit nodes
        for sem_unit in units.units:
            node = make_node(
                unit_type = sem_unit.type,
                tag       = sem_unit.tag,
                params    = {},
                metadata  = {"role": sem_unit.role},
            )
            graph.add_unit(node)

        # Case-insensitive lookup so LLM-written "water" matches canonical "Water".
        _compound_canon: dict[str, str] = {c.lower(): c for c in compounds}

        # Add all stream nodes and edges.
        # Track next available outlet port per unit so multi-outlet units
        # (Vessel: port 0 = vapour, port 1 = liquid) get distinct src_ports.
        # _repair_vessel_ports in normalise.py then swaps them if naming heuristics
        # indicate the LLM listed liquid before vapour.
        _next_outlet_port: dict[str, int] = {}

        for sem_stream in topology.streams:
            src_port = 0
            if sem_stream.src is not None:
                src_node = graph.unit(sem_stream.src)
                if src_node is not None and src_node.max_outlets() > 1:
                    src_port = _next_outlet_port.get(sem_stream.src, 0)
                    _next_outlet_port[sem_stream.src] = src_port + 1

            raw_comp = dict(sem_stream.composition)
            composition = {
                _compound_canon.get(k.lower(), k): v for k, v in raw_comp.items()
            }

            edge = EdgeIR(
                tag         = sem_stream.tag,
                T           = sem_stream.T,
                P           = sem_stream.P,
                flow        = sem_stream.flow,
                composition = composition,
                src_port    = src_port,
                metadata    = {"is_feed": sem_stream.is_feed},
            )
            graph.add_stream(edge, sem_stream.src, sem_stream.dst)

        return graph
