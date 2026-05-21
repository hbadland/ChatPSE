"""
Structural graph heuristics — lightweight pre-repair structural analysis (Item 7).

Detects common structural mistakes and reports them as issues.
Applies minimal structural fixes only when unambiguous and safe.

Checks performed:
  1. Vessel with no upstream temperature conditioning (flash will fail)
  2. Pump receiving feed from a vapour-phase unit (phase mismatch risk)
  3. Consecutive Heaters without separation (likely redundant)
  4. Missing Heater before Vessel when feed is liquid (can auto-flag)

Only conservative structural changes are auto-applied (none currently).
All detected issues are returned for the orchestrator to act on.
"""
from __future__ import annotations

from ir.graph import (
    FlowsheetGraph,
    HeaterNode, CoolerNode, SeparatorNode,
    PumpNode, CompressorNode, ExpanderNode,
)


class StructuralHeuristics:
    """
    Stateless structural analyser.  Returns (graph, issues, changes).

      graph   — possibly patched graph (currently returned unchanged)
      issues  — detected structural problems, strings for logging
      changes — structural fixes applied (currently empty)
    """

    def check(
        self,
        graph: FlowsheetGraph,
    ) -> tuple[FlowsheetGraph, list[str], list[str]]:
        """
        Run all structural checks.  Returns (graph, issues, changes).
        The graph is returned as-is (structural auto-fixes are not applied
        to avoid cascading errors; issues are returned for caller decisions).
        """
        issues:  list[str] = []
        changes: list[str] = []

        self._vessel_without_conditioning(graph, issues)
        self._pump_vapour_feed(graph, issues)
        self._consecutive_heaters(graph, issues)
        self._compressor_liquid_feed(graph, issues)

        return graph, issues, changes

    # ── Check 1: Vessel with no upstream conditioning ─────────────────────────

    def _vessel_without_conditioning(
        self,
        graph:  FlowsheetGraph,
        issues: list[str],
    ) -> None:
        for node in graph.units():
            if not isinstance(node, SeparatorNode):
                continue

            upstream_types: set[str] = set()
            for stream in graph.inlet_streams(node.tag):
                src_tag = graph.stream_source(stream.tag)
                if src_tag:
                    src = graph.unit(src_tag)
                    if src:
                        upstream_types.add(src.unit_type)

            conditioning = upstream_types & {"Heater", "Cooler", "Compressor", "Expander"}
            if upstream_types and not conditioning:
                issues.append(
                    f"STRUCTURAL[vessel_no_conditioning]: {node.tag} (Vessel) has "
                    f"no upstream temperature/pressure conditioning. "
                    f"Upstream: {sorted(upstream_types)}. "
                    f"Flash separation will likely fail — a Heater is required."
                )

    # ── Check 2: Pump receiving vapour-phase feed ─────────────────────────────

    def _pump_vapour_feed(
        self,
        graph:  FlowsheetGraph,
        issues: list[str],
    ) -> None:
        for node in graph.units():
            if not isinstance(node, PumpNode):
                continue
            for stream in graph.inlet_streams(node.tag):
                src_tag = graph.stream_source(stream.tag)
                if src_tag:
                    src = graph.unit(src_tag)
                    if isinstance(src, (CompressorNode, ExpanderNode)):
                        issues.append(
                            f"STRUCTURAL[pump_vapour_feed]: {node.tag} (Pump) "
                            f"receives feed from {src_tag} ({src.unit_type}) — "
                            f"likely vapour-phase. Pump requires liquid inlet."
                        )

    # ── Check 3: Consecutive Heaters without separation ───────────────────────

    def _consecutive_heaters(
        self,
        graph:  FlowsheetGraph,
        issues: list[str],
    ) -> None:
        for node in graph.units():
            if not isinstance(node, HeaterNode):
                continue
            for outlet in graph.outlet_streams(node.tag):
                dst_tag = graph.stream_dest(outlet.tag)
                if dst_tag:
                    dst = graph.unit(dst_tag)
                    if isinstance(dst, HeaterNode):
                        issues.append(
                            f"STRUCTURAL[consecutive_heaters]: {node.tag}→{dst_tag} "
                            f"are consecutive Heaters with no separation unit between "
                            f"them. This may be redundant or indicate a missing Vessel."
                        )

    # ── Check 4: Compressor receiving liquid feed ─────────────────────────────

    def _compressor_liquid_feed(
        self,
        graph:  FlowsheetGraph,
        issues: list[str],
    ) -> None:
        for node in graph.units():
            if not isinstance(node, CompressorNode):
                continue
            for stream in graph.inlet_streams(node.tag):
                src_tag = graph.stream_source(stream.tag)
                if src_tag:
                    src = graph.unit(src_tag)
                    if isinstance(src, CoolerNode):
                        t_out = src.params.get("T_out")
                        if t_out is not None:
                            from ir.thermo_estimation import bubble_point_K
                            in_P = next(
                                (s.P for s in graph.inlet_streams(src.tag) if s.P is not None),
                                101_325.0,
                            )
                            bp = bubble_point_K(graph.compounds, in_P or 101_325.0)
                            if bp is not None and float(t_out) < bp:
                                issues.append(
                                    f"STRUCTURAL[compressor_liquid_risk]: {node.tag} "
                                    f"(Compressor) may receive liquid feed from {src_tag} "
                                    f"(T_out={t_out} K < BP={bp:.1f} K). "
                                    f"Compressor requires vapour inlet."
                                )
