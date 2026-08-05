"""
Deterministic repair operations on FlowsheetGraph.

DeterministicRepair handles every repair that requires no LLM:
  PARAM_INJECT   — inject BIPs from corpus
  TOPOLOGY_FIX   — re-run normaliser
  UNIT_CONVERSION — °C→K, bar→Pa (streams and unit params)
  DEFAULT_FILL   — fill missing params with spec defaults
  PORT_REPAIR    — reassign src/dst port numbers on streams
  THERMO_SWITCH  — select next best package via ThermoRetriever

CONDITION_FIX is intentionally NOT handled here — it requires
understanding of process context and is delegated to LLMRepair.
"""
from __future__ import annotations

from typing import Optional

from ir.graph import FlowsheetGraph, SeparatorNode, Source
from ir.normalise import normalise
from ir.types import RepairStrategy, SimError


class DeterministicRepair:
    """
    Stateless repair primitives.  Each method returns (patched_graph, change_log).
    The graph is copied before modification; the original is never mutated.
    """

    def apply(
        self,
        graph:          FlowsheetGraph,
        error:          SimError,
        retriever,
        tried_packages: set[str] | None = None,
    ) -> tuple[FlowsheetGraph, list[str]]:
        """
        Dispatch on error.repair_strategy.  Returns (graph, changes).
        Raises ValueError for CONDITION_FIX — caller must route to LLMRepair.
        """
        tried_packages = set(tried_packages) if tried_packages else set()
        s = error.repair_strategy

        if s == RepairStrategy.PARAM_INJECT:
            return self.inject_bips(graph, retriever)

        if s == RepairStrategy.TOPOLOGY_FIX:
            return self.fix_topology(graph)

        if s == RepairStrategy.UNIT_CONVERSION:
            return self.fix_unit_conversions(graph, error)

        if s == RepairStrategy.DEFAULT_FILL:
            return self.apply_defaults(graph, retriever)

        if s == RepairStrategy.PORT_REPAIR:
            return self.fix_port_violations(graph)

        if s == RepairStrategy.THERMO_SWITCH:
            return self.switch_package(graph, retriever, tried_packages)

        if s == RepairStrategy.CONDITION_FIX:
            raise ValueError(
                "CONDITION_FIX cannot be handled deterministically — "
                "delegate to LLMRepair")

        # RepairStrategy.HUMAN — nothing to do
        return graph, [f"HUMAN: {error.target} — {error.evidence[:80]}"]

    # ── Individual repair methods ──────────────────────────────────────────────

    def inject_bips(
        self, graph: FlowsheetGraph, retriever
    ) -> tuple[FlowsheetGraph, list[str]]:
        pkg = graph.property_package
        if pkg not in ("NRTL", "UNIQUAC"):
            return graph, []
        if graph.binary_parameters:
            return graph, []
        bips, missing = retriever.query_bips(graph.compounds, pkg)
        if not missing:
            g = graph.copy()
            g.binary_parameters = bips
            return g, [f"PARAM_INJECT: {pkg} BIPs injected for {graph.compounds}"]
        return graph, [f"PARAM_INJECT: BIPs missing for pairs {missing}"]

    def fix_topology(self, graph: FlowsheetGraph) -> tuple[FlowsheetGraph, list[str]]:
        return normalise(graph), ["TOPOLOGY_FIX: graph re-normalised"]

    def fix_unit_conversions(
        self, graph: FlowsheetGraph, error: SimError
    ) -> tuple[FlowsheetGraph, list[str]]:
        """
        Apply deterministic unit conversions to the target named in error.
        Handles: stream T (°C→K), stream P (bar→Pa), unit T_out/P_out.
        """
        from ir.types import TargetKind
        g       = graph.copy()
        changes: list[str] = []

        if error.target.kind == TargetKind.STREAM:
            stream = g.stream(error.target.tag)
            if stream is not None:
                if stream.T is not None and stream.T < 100:
                    old   = stream.T
                    stream.T = round(stream.T + 273.15, 2)
                    changes.append(
                        f"UNIT_CONVERSION: stream {stream.tag} T {old}→{stream.T} K")
                if stream.P is not None and stream.P < 500:
                    old   = stream.P
                    stream.P = round(stream.P * 1e5, 0)
                    changes.append(
                        f"UNIT_CONVERSION: stream {stream.tag} P {old}→{stream.P} Pa")

        elif error.target.kind == TargetKind.UNIT:
            node = g.unit(error.target.tag)
            if node is not None:
                if "T_out" in node.params and node.params["T_out"] < 100:
                    old = node.params["T_out"]
                    # Sanctioned physical correction: override + honest retag.
                    node.correct_param("T_out", round(old + 273.15, 2), Source.COMPUTED)
                    changes.append(
                        f"UNIT_CONVERSION: {node.tag} T_out {old}→{node.params['T_out']} K")
                if "P_out" in node.params and node.params["P_out"] < 500:
                    old = node.params["P_out"]
                    # Sanctioned physical correction: override + honest retag.
                    node.correct_param("P_out", round(old * 1e5, 0), Source.COMPUTED)
                    changes.append(
                        f"UNIT_CONVERSION: {node.tag} P_out {old}→{node.params['P_out']} Pa")

        return g, changes

    def apply_defaults(
        self, graph: FlowsheetGraph, retriever
    ) -> tuple[FlowsheetGraph, list[str]]:
        """Fill missing optional params from unit_specs.json defaults."""
        g       = graph.copy()
        changes: list[str] = []
        for node in g.units():
            defaults = retriever.units.defaults(node.unit_type)
            filled   = {}
            for param, default_val in defaults.items():
                if param not in node.params:
                    node.params[param] = default_val
                    filled[param]      = default_val
            if filled:
                changes.append(
                    f"DEFAULT_FILL: {node.tag} ({node.unit_type}) → {filled}")
        return g, changes

    def fix_port_violations(
        self, graph: FlowsheetGraph
    ) -> tuple[FlowsheetGraph, list[str]]:
        """
        Ensure Vessel outlet streams use canonical port assignments:
          port 0 → vapour, port 1 → liquid.
        For other units, reset all src/dst ports to 0 when they exceed max.
        """
        g       = graph.copy()
        changes: list[str] = []

        for node in g.units():
            outlets = g.outlet_streams(node.tag)
            if isinstance(node, SeparatorNode):
                _fix_vessel_ports(g, node, outlets, changes)
            else:
                max_out = node.max_outlets()
                for s in outlets:
                    if s.src_port >= max_out:
                        old       = s.src_port
                        s.src_port = 0
                        changes.append(
                            f"PORT_REPAIR: {node.tag} stream {s.tag} src_port {old}→0")

        return g, changes

    def switch_package(
        self,
        graph:     FlowsheetGraph,
        retriever,
        tried:     set[str],
    ) -> tuple[FlowsheetGraph, list[str]]:
        candidates = retriever.select_package(
            graph.compounds, exclude=set(tried) | {graph.property_package})
        if not candidates:
            return graph, ["THERMO_SWITCH: all packages exhausted"]
        g       = graph.copy()
        old_pkg = g.property_package
        g.property_package  = candidates[0]
        g.binary_parameters = []
        if g.property_package in ("NRTL", "UNIQUAC"):
            bips, missing = retriever.query_bips(g.compounds, g.property_package)
            if not missing:
                g.binary_parameters = bips
        return g, [f"THERMO_SWITCH: {old_pkg} → {g.property_package}"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fix_vessel_ports(graph, node, outlets, changes):
    """Sort Vessel outlets: vapour→port 0, liquid→port 1."""
    vapour_streams = [s for s in outlets if s.phase == "vapour"]
    liquid_streams = [s for s in outlets if s.phase == "liquid"]
    mixed_streams  = [s for s in outlets
                      if s.phase not in ("vapour", "liquid")]

    for s in vapour_streams:
        if s.src_port != 0:
            old        = s.src_port
            s.src_port = 0
            changes.append(
                f"PORT_REPAIR: {node.tag} vapour stream {s.tag} src_port {old}→0")
    for s in liquid_streams:
        if s.src_port != 1:
            old        = s.src_port
            s.src_port = 1
            changes.append(
                f"PORT_REPAIR: {node.tag} liquid stream {s.tag} src_port {old}→1")
    for i, s in enumerate(mixed_streams):
        target_port = i
        if s.src_port != target_port:
            old        = s.src_port
            s.src_port = target_port
            changes.append(
                f"PORT_REPAIR: {node.tag} mixed stream {s.tag} src_port {old}→{target_port}")
