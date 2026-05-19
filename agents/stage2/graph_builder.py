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

from ir.graph import FlowsheetGraph, NodeIR, EdgeIR
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
            node = NodeIR(
                tag       = sem_unit.tag,
                unit_type = sem_unit.type,
                params    = {},
                metadata  = {"role": sem_unit.role},
            )
            graph.add_unit(node)

        # Add all stream nodes and edges
        for sem_stream in topology.streams:
            edge = EdgeIR(
                tag         = sem_stream.tag,
                T           = sem_stream.T,
                P           = sem_stream.P,
                flow        = sem_stream.flow,
                composition = dict(sem_stream.composition),
                metadata    = {"is_feed": sem_stream.is_feed},
            )
            graph.add_stream(edge, sem_stream.src, sem_stream.dst)

        return graph
