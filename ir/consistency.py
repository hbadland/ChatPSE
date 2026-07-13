"""
Global Consistency Pass — deterministic, zero LLM calls.

Two-pass algorithm:
  Forward pass  (1–3): propagate T/P → enforce monotonic → fill gaps
  Backward pass (4):   propagate downstream requirements back to upstream units
  Coupling check (5):  final Heater→Vessel flash-feasibility enforcement

Returns (patched_graph, change_log). The input graph is never mutated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import networkx as nx

from ir.graph import (
    FlowsheetGraph, NodeIR,
    HeaterNode, CoolerNode, SeparatorNode,
    PumpNode, CompressorNode, ExpanderNode, ConversionReactorNode,
)
from ir.thermo_estimation import bubble_point_K  # noqa: F401 — re-exported for callers
from ir.constraint_solver import (
    Constraint, ConstraintPriority, ConstraintSolver,
)

_solver = ConstraintSolver()

# ── desc_T_out plausibility window ────────────────────────────────────────────
# When T_out was extracted from the description (_desc_T_out sentinel), we trust
# it only if the implied temperature rise sits within this window.  Outside it,
# the extraction is likely wrong (too small → LLM read feed temp; too large →
# LLM returned °C value instead of K, e.g. "80" instead of "353.15").
_DESC_T_MIN_DELTA: float = 5.0    # K — minimum plausible rise above feed_T
_DESC_T_MAX_DELTA: float = 200.0  # K — maximum plausible rise above feed_T


def _desc_t_plausible(t_out: float, feed_T: float) -> bool:
    """True when a description-extracted T_out is in the plausible operating range."""
    delta = t_out - feed_T
    return _DESC_T_MIN_DELTA <= delta <= _DESC_T_MAX_DELTA


# ── Propagated stream conditions (ephemeral — not stored in IR) ───────────────

@dataclass
class _PropCond:
    T: Optional[float] = None  # K
    P: Optional[float] = None  # Pa


# ── Public entry point ─────────────────────────────────────────────────────────

class GlobalConsistencyPass:
    """
    Deterministic graph-level physical consistency enforcement.
    Zero LLM calls. Runs after ParamMapper, before the execution loop.

    Usage:
        gcp = GlobalConsistencyPass()
        graph, changes = gcp.apply(graph)
    """

    def apply(
        self,
        graph: FlowsheetGraph,
    ) -> tuple[FlowsheetGraph, list[str]]:
        g       = graph.copy()
        changes: list[str] = []

        # Forward pass: propagate, enforce monotonic, fill gaps
        conds = self._propagate(g)
        self._enforce_monotonic(g, conds, changes)
        self._fill_missing(g, conds, changes)

        # Backward pass: propagate downstream requirements upstream
        self._backward_propagate(g, conds, changes)

        # Final coupling check (catches anything the backward pass missed)
        self._check_coupling(g, conds, changes)

        # ── Write propagated T/P back onto streams that have none ──────────────
        # Re-propagate with the now-corrected unit params, then fill ONLY missing
        # stream conditions (never override an extracted/specified value) so a
        # downstream Vessel inherits its true feed T (the reactor outlet) instead
        # of being flagged "no feed T" and solved from a stale default.
        final = self._propagate(g)
        for stream in g.streams():
            c = final.get(stream.tag)
            if c is None:
                continue
            if stream.T is None and c.T is not None:
                stream.T = round(float(c.T), 2)
                changes.append(
                    f"CONSISTENCY[fill_stream_T]: {stream.tag} T={stream.T} K "
                    f"(propagated from upstream unit outlet)")
            if stream.P is None and c.P is not None:
                stream.P = round(float(c.P), 2)

        # Vessel/Separator feed-T inheritance. An adiabatic flash operates at its
        # inlet temperature. If propagation still left a Separator's inlet
        # stream(s) with no T (a feed with unspecified T, or an upstream chain
        # that carried none), inherit the graph's feed temperature so the vessel
        # flashes at a real T instead of being flagged "no feed T — may produce
        # zero vapour" and solved from a stale default.
        _feed_T = next((s.T for s in g.feed_streams() if s.T is not None), None)
        if _feed_T is not None:
            for node in g.units():
                if not isinstance(node, SeparatorNode):
                    continue
                inlets = g.inlet_streams(node.tag)
                if inlets and all(s.T is None for s in inlets):
                    for s in inlets:
                        s.T = round(float(_feed_T), 2)
                    changes.append(
                        f"CONSISTENCY[vessel_inherit_T]: {node.tag} inlet "
                        f"T={round(float(_feed_T), 2)} K (inherited feed T)")

        return g, changes

    # ── 1. Topological forward propagation ───────────────────────────────────

    def _propagate(self, graph: FlowsheetGraph) -> dict[str, _PropCond]:
        conds: dict[str, _PropCond] = {}

        # Seed from feed streams (have explicit T/P)
        for stream in graph.streams():
            if stream.T is not None or stream.P is not None:
                conds[stream.tag] = _PropCond(T=stream.T, P=stream.P)

        try:
            order = list(nx.topological_sort(graph.unit_graph()))
        except nx.NetworkXUnfeasible:
            order = [u.tag for u in graph.units()]

        for unit_tag in order:
            node = graph.unit(unit_tag)
            if node is None:
                continue

            inlets = graph.inlet_streams(unit_tag)
            in_T   = next((conds[s.tag].T for s in inlets
                           if s.tag in conds and conds[s.tag].T is not None), None)
            in_P   = next((conds[s.tag].P for s in inlets
                           if s.tag in conds and conds[s.tag].P is not None), None)

            out_T, out_P = self._unit_outlet(node, in_T, in_P)

            for stream in graph.outlet_streams(unit_tag):
                prev = conds.get(stream.tag, _PropCond())
                conds[stream.tag] = _PropCond(
                    T = out_T if out_T is not None else prev.T,
                    P = out_P if out_P is not None else prev.P,
                )

        return conds

    @staticmethod
    def _unit_outlet(
        node: NodeIR,
        in_T: Optional[float],
        in_P: Optional[float],
    ) -> tuple[Optional[float], Optional[float]]:
        if isinstance(node, (HeaterNode, CoolerNode)):
            t_out = node.params.get("T_out")
            return (float(t_out) if t_out is not None else None), in_P

        if isinstance(node, SeparatorNode):
            return in_T, in_P

        if isinstance(node, (PumpNode, CompressorNode, ExpanderNode)):
            p_out = node.params.get("P_out")
            return in_T, (float(p_out) if p_out is not None else None)

        if isinstance(node, ConversionReactorNode):
            # A reactor SETS its outlet temperature (temperature_K); without this
            # the reactor's operating T never propagates and every downstream unit
            # inherits the reactor INLET T (the "no feed T" flood).
            t = node.params.get("temperature_K")
            p = node.params.get("pressure_Pa")
            return (float(t) if t is not None else in_T,
                    float(p) if p is not None else in_P)

        return in_T, in_P  # Mixer, Splitter: pass-through

    # ── 2. Monotonic constraint enforcement ───────────────────────────────────

    def _enforce_monotonic(
        self,
        graph: FlowsheetGraph,
        conds: dict[str, _PropCond],
        changes: list[str],
    ) -> None:
        for node in graph.units():
            inlets = graph.inlet_streams(node.tag)
            in_T   = next((conds[s.tag].T for s in inlets
                           if s.tag in conds and conds[s.tag].T is not None), None)
            in_P   = next((conds[s.tag].P for s in inlets
                           if s.tag in conds and conds[s.tag].P is not None), None)

            if isinstance(node, HeaterNode):
                self._fix_heater(node, in_T, graph.compounds, in_P, changes)
            elif isinstance(node, CoolerNode):
                self._fix_cooler(node, in_T, changes)
            elif isinstance(node, (PumpNode, CompressorNode)):
                self._fix_p_raiser(node, in_P, changes)
            elif isinstance(node, ExpanderNode):
                self._fix_expander(node, in_P, changes)

    def _fix_heater(
        self,
        node: HeaterNode,
        in_T: Optional[float],
        compounds: list[str],
        in_P: Optional[float],
        changes: list[str],
    ) -> None:
        t_out = node.params.get("T_out")
        if t_out is None or in_T is None:
            return
        t_out = float(t_out)

        if node.params.get("_desc_T_out"):
            # Tier 1 — monotonic: a heater must raise temperature by at least 1 K.
            if t_out > in_T + 1.0 and _desc_t_plausible(t_out, in_T):
                # Tier 2 — plausibility window passed: trust the description.
                return
            # Outside the plausibility window: likely extraction error — apply
            # bubble-point correction and warn.
            bp    = bubble_point_K(compounds, in_P or 101_325.0)
            new_T = round((bp + 15.0) if (bp and bp > in_T) else (in_T + 30.0), 2)
            node.params["T_out"] = new_T
            changes.append(
                f"CONSISTENCY[desc_T_out overridden]: {node.tag} "
                f"T_out {t_out}→{new_T} K "
                f"appears implausible (feed_T={in_T:.1f} K); bubble-point correction applied")
            return

        if t_out > in_T:
            return
        bp    = bubble_point_K(compounds, in_P or 101_325.0)
        new_T = round((bp + 15.0) if (bp and bp > in_T) else (in_T + 30.0), 2)
        node.params["T_out"] = new_T
        changes.append(
            f"CONSISTENCY[monotonic]: {node.tag} T_out {t_out}→{new_T} K "
            f"(must be > feed T={in_T:.1f} K)")

    def _fix_cooler(
        self,
        node: CoolerNode,
        in_T: Optional[float],
        changes: list[str],
    ) -> None:
        t_out = node.params.get("T_out")
        if t_out is None or in_T is None:
            return
        t_out = float(t_out)
        if t_out < in_T:
            return
        new_T = max(round(in_T - 25.0, 2), 273.15)
        node.params["T_out"] = new_T
        changes.append(
            f"CONSISTENCY[monotonic]: {node.tag} T_out {t_out}→{new_T} K "
            f"(must be < feed T={in_T:.1f} K)")

    def _fix_p_raiser(
        self,
        node: NodeIR,
        in_P: Optional[float],
        changes: list[str],
    ) -> None:
        p_out = node.params.get("P_out")
        if p_out is None or in_P is None:
            return
        p_out = float(p_out)
        if p_out > in_P:
            return
        new_P = round(in_P * 5.0, 0)
        node.params["P_out"] = new_P
        changes.append(
            f"CONSISTENCY[monotonic]: {node.tag} P_out {p_out}→{new_P} Pa "
            f"(must be > feed P={in_P:.0f} Pa)")

    def _fix_expander(
        self,
        node: ExpanderNode,
        in_P: Optional[float],
        changes: list[str],
    ) -> None:
        p_out = node.params.get("P_out")
        if p_out is None or in_P is None:
            return
        p_out = float(p_out)
        if p_out < in_P:
            return
        new_P = max(round(in_P / 3.0, 0), 101_325.0)
        node.params["P_out"] = new_P
        changes.append(
            f"CONSISTENCY[monotonic]: {node.tag} P_out {p_out}→{new_P} Pa "
            f"(must be < feed P={in_P:.0f} Pa)")

    # ── 3. Fill missing required params ──────────────────────────────────────

    def _fill_missing(
        self,
        graph: FlowsheetGraph,
        conds: dict[str, _PropCond],
        changes: list[str],
    ) -> None:
        for node in graph.units():
            inlets = graph.inlet_streams(node.tag)
            in_T   = next((conds[s.tag].T for s in inlets
                           if s.tag in conds and conds[s.tag].T is not None), None)
            in_P   = next((conds[s.tag].P for s in inlets
                           if s.tag in conds and conds[s.tag].P is not None), 101_325.0)

            if isinstance(node, HeaterNode) and "T_out" not in node.params:
                bp    = bubble_point_K(graph.compounds, in_P)
                new_T = round((bp + 20.0) if bp else ((in_T or 298.15) + 50.0), 2)
                node.params["T_out"] = new_T
                changes.append(
                    f"CONSISTENCY[fill]: {node.tag} T_out={new_T} K (missing)")

            elif isinstance(node, CoolerNode) and "T_out" not in node.params:
                new_T = max(round((in_T or 373.15) - 30.0, 2), 273.15)
                node.params["T_out"] = new_T
                changes.append(
                    f"CONSISTENCY[fill]: {node.tag} T_out={new_T} K (missing)")

            elif isinstance(node, (PumpNode, CompressorNode)) \
                    and "P_out" not in node.params:
                new_P = round(in_P * 5.0, 0)
                node.params["P_out"] = new_P
                changes.append(
                    f"CONSISTENCY[fill]: {node.tag} P_out={new_P} Pa (missing)")

            elif isinstance(node, ExpanderNode) \
                    and "P_out" not in node.params:
                new_P = max(round(in_P / 3.0, 0), 101_325.0)
                node.params["P_out"] = new_P
                changes.append(
                    f"CONSISTENCY[fill]: {node.tag} P_out={new_P} Pa (missing)")

    # ── 4. Backward propagation of downstream requirements ───────────────────

    def _backward_propagate(
        self,
        graph: FlowsheetGraph,
        conds: dict[str, _PropCond],
        changes: list[str],
    ) -> None:
        """
        Walk units in reverse topological order, propagating requirements
        from downstream back to upstream unit parameters.

        Rules enforced:
          • Vessel  → upstream Heater: T_out ≥ BP + 5 K (two-phase inlet needed)
          • Pump    → upstream Cooler: T_out ≤ BP − 10 K (liquid feed required)
          • Pump    → upstream Heater: flag if T_out > BP (phase mismatch)
        """
        try:
            order = list(nx.topological_sort(graph.unit_graph()))
        except nx.NetworkXUnfeasible:
            order = [u.tag for u in graph.units()]

        for unit_tag in reversed(order):
            node = graph.unit(unit_tag)
            if node is None:
                continue

            for outlet_stream in graph.outlet_streams(unit_tag):
                dst_tag = graph.stream_dest(outlet_stream.tag)
                if dst_tag is None:
                    continue
                dst = graph.unit(dst_tag)
                if dst is None:
                    continue

                in_P = next(
                    (conds[s.tag].P for s in graph.inlet_streams(unit_tag)
                     if s.tag in conds and conds[s.tag].P is not None),
                    101_325.0,
                )
                bp = bubble_point_K(graph.compounds, in_P)

                # Rule: unit feeding a Vessel must deliver T ≥ BP + margin.
                # Two-tier _desc_T_out check: trust description only if plausible.
                if isinstance(dst, SeparatorNode) and bp is not None:
                    if isinstance(node, HeaterNode):
                        t_out = node.params.get("T_out")
                        if t_out is not None:
                            t_out_f = float(t_out)
                            in_T = next(
                                (conds[s.tag].T for s in graph.inlet_streams(unit_tag)
                                 if s.tag in conds and conds[s.tag].T is not None),
                                None,
                            )
                            if node.params.get("_desc_T_out"):
                                if in_T is not None and _desc_t_plausible(t_out_f, in_T):
                                    pass  # Trust the description
                                else:
                                    new_T = round(bp + 15.0, 2)
                                    old   = node.params["T_out"]
                                    node.params["T_out"] = new_T
                                    feed_str = f"{in_T:.1f}" if in_T is not None else "unknown"
                                    changes.append(
                                        f"CONSISTENCY[desc_T_out overridden]: {node.tag}→{dst_tag}: "
                                        f"T_out {t_out_f}→{new_T} K "
                                        f"appears implausible (feed_T={feed_str} K); bubble-point correction applied")
                            else:
                                result = _solver.resolve(
                                    "T_out",
                                    t_out_f,
                                    [Constraint(
                                        param    = "T_out",
                                        priority = ConstraintPriority.PHYSICAL_FEASIBILITY,
                                        source   = f"Vessel flash: T > BP+5 (BP={bp:.1f} K)",
                                        min_val  = bp + 5.0,
                                    )],
                                )
                                if abs(result.resolved_value - t_out_f) > 0.1:
                                    new_T = round(max(result.resolved_value, bp + 15.0), 2)
                                    old   = node.params["T_out"]
                                    node.params["T_out"] = new_T
                                    tag_str = f"CONFLICT:{result.dropped_sources}" if result.conflict else "ok"
                                    changes.append(
                                        f"CONSISTENCY[backward/{tag_str}]: {node.tag}→{dst_tag}: "
                                        f"T_out {old}→{new_T} K "
                                        f"(Vessel needs T > BP={bp:.1f} K)")

                # Rule: unit feeding a Pump must deliver liquid (T ≤ BP − 10)
                if isinstance(dst, PumpNode) and bp is not None:
                    if isinstance(node, CoolerNode):
                        t_out = node.params.get("T_out")
                        if t_out is not None:
                            result = _solver.resolve(
                                "T_out",
                                float(t_out),
                                [Constraint(
                                    param    = "T_out",
                                    priority = ConstraintPriority.PHYSICAL_FEASIBILITY,
                                    source   = f"Pump needs liquid: T < BP-10 (BP={bp:.1f} K)",
                                    max_val  = bp - 10.0,
                                )],
                            )
                            if abs(result.resolved_value - float(t_out)) > 0.1:
                                new_T = max(round(result.resolved_value, 2), 273.15)
                                old   = node.params["T_out"]
                                node.params["T_out"] = new_T
                                changes.append(
                                    f"CONSISTENCY[backward]: {node.tag}→{dst_tag}: "
                                    f"T_out {old}→{new_T} K "
                                    f"(Pump needs liquid feed: BP={bp:.1f} K)")

    # ── 5. Cross-unit coupling (final check) ─────────────────────────────────

    def _check_coupling(
        self,
        graph: FlowsheetGraph,
        conds: dict[str, _PropCond],
        changes: list[str],
    ) -> None:
        """
        Final catch: Heater→Vessel coupling must place the stream in the
        two-phase region. The backward pass handles most cases; this catches
        any that slipped through or were introduced by monotonic enforcement.
        """
        for node in graph.units():
            if not isinstance(node, HeaterNode):
                continue
            for outlet_stream in graph.outlet_streams(node.tag):
                dst_tag = graph.stream_dest(outlet_stream.tag)
                if dst_tag is None:
                    continue
                dst = graph.unit(dst_tag)
                if not isinstance(dst, SeparatorNode):
                    continue

                t_out = node.params.get("T_out")
                if t_out is None:
                    continue
                t_out = float(t_out)

                inlets = graph.inlet_streams(node.tag)
                in_P   = next((conds[s.tag].P for s in inlets
                               if s.tag in conds and conds[s.tag].P is not None),
                              101_325.0)
                bp = bubble_point_K(graph.compounds, in_P)
                if bp is None:
                    continue

                # Two-tier _desc_T_out check: trust description only if plausible.
                if node.params.get("_desc_T_out"):
                    in_T = next(
                        (conds[s.tag].T for s in inlets
                         if s.tag in conds and conds[s.tag].T is not None),
                        None,
                    )
                    if in_T is not None and _desc_t_plausible(t_out, in_T):
                        continue  # Plausibility window passed: trust the description
                    if t_out < bp + 5.0:
                        new_T = round(bp + 15.0, 2)
                        old   = node.params["T_out"]
                        node.params["T_out"] = new_T
                        feed_str = f"{in_T:.1f}" if in_T is not None else "unknown"
                        changes.append(
                            f"CONSISTENCY[desc_T_out overridden]: {node.tag}→{dst_tag}: "
                            f"T_out {old}→{new_T} K "
                            f"appears implausible (feed_T={feed_str} K); bubble-point correction applied")
                elif t_out < bp + 5.0:
                    new_T = round(bp + 15.0, 2)
                    old   = node.params["T_out"]
                    node.params["T_out"] = new_T
                    changes.append(
                        f"CONSISTENCY[coupling]: {node.tag}→{dst_tag}: "
                        f"T_out {old}→{new_T} K "
                        f"(bubble pt ≈ {bp:.1f} K; need two-phase region for flash)")
