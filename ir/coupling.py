"""
Parameter coupling map for constraint-coupled sequential optimisation (Item 1 — Issue 1 hardened).

After fixing one parameter in a beam step, the next step prioritises parameters
that are physically or constraint-linked to it.  This achieves joint-optimisation
behaviour without combinatorial explosion.

Coupling rules (directed, graph-topology-aware):
  Heater.T_out   → downstream Vessel : fixing T affects flash feasibility
  Heater.T_out   → upstream Pump/Compressor : P determines bubble point
  Cooler.T_out   → downstream Pump : liquid-feed quality changes
  Pump.P_out     → upstream Cooler : higher P raises BP → cooler target changes
  Compressor.P_out → downstream Heater/Cooler : phase-change window shifts

Issue-1 mitigation — CoupledSettler:
  After any P_out change, the bubble point at that pressure changes.  If an
  upstream Heater/Cooler was previously set to a margin based on the old BP, it
  is now potentially wrong — causing the next iteration to re-fix it, creating
  oscillation.

  CoupledSettler.settle() runs deterministically (no LLM) after each parameter
  fix to propagate BP changes to physically coupled upstream parameters.  It uses
  the learned MarginModel for the offset rather than a fixed constant, so the
  correction improves with experience.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ir.graph import FlowsheetGraph
    from ir.types import SimError

logger = logging.getLogger(__name__)

_MAX_SETTLE_PASSES = 3   # convergence limit for multi-pass settling


@dataclass(frozen=True)
class CoupledTarget:
    unit_tag: str
    param:    str
    boost:    float   # additive priority boost — higher = fix sooner
    reason:   str


class ParameterCouplingMap:
    """
    Graph-aware coupling: given a just-fixed (unit_tag, param), returns a
    dict {error_target_tag: boost} for errors that are coupled to it.

    Used by beam_search._pick_target to bias which unfixed error to address
    next.  Deterministic, zero LLM calls.
    """

    def get_coupled_boosts(
        self,
        graph:          "FlowsheetGraph",
        fixed_tag:      str,
        fixed_param:    str,
        unfixed_tags:   set[str],
    ) -> dict[str, float]:
        """
        Returns {error_target_tag: boost} for unfixed errors that are
        coupled to the parameter just fixed at fixed_tag.
        """
        from ir.graph import (
            HeaterNode, CoolerNode, SeparatorNode,
            PumpNode, CompressorNode, ExpanderNode,
        )

        boosts: dict[str, float] = {}
        node = graph.unit(fixed_tag)
        if node is None:
            return boosts

        # ── Rule 1: Heater.T_out fixed → prioritise downstream Vessel flash ──
        if isinstance(node, HeaterNode) and fixed_param == "T_out":
            for outlet in graph.outlet_streams(fixed_tag):
                dst_tag = graph.stream_dest(outlet.tag)
                if dst_tag and dst_tag in unfixed_tags:
                    dst = graph.unit(dst_tag)
                    if isinstance(dst, SeparatorNode):
                        boosts[dst_tag] = 15.0

            # Also boost upstream pressure units (P shifts bubble point)
            for inp in graph.inlet_streams(fixed_tag):
                src_tag = graph.stream_source(inp.tag)
                if src_tag and src_tag in unfixed_tags:
                    src = graph.unit(src_tag)
                    if isinstance(src, (PumpNode, CompressorNode)):
                        boosts[src_tag] = 8.0

        # ── Rule 2: Pump.P_out fixed → prioritise upstream Cooler ───────────
        elif isinstance(node, PumpNode) and fixed_param == "P_out":
            for inp in graph.inlet_streams(fixed_tag):
                src_tag = graph.stream_source(inp.tag)
                if src_tag and src_tag in unfixed_tags:
                    src = graph.unit(src_tag)
                    if isinstance(src, CoolerNode):
                        boosts[src_tag] = 20.0  # strong: P changes BP directly

        # ── Rule 3: Cooler.T_out fixed → prioritise downstream Pump ─────────
        elif isinstance(node, CoolerNode) and fixed_param == "T_out":
            for outlet in graph.outlet_streams(fixed_tag):
                dst_tag = graph.stream_dest(outlet.tag)
                if dst_tag and dst_tag in unfixed_tags:
                    dst = graph.unit(dst_tag)
                    if isinstance(dst, PumpNode):
                        boosts[dst_tag] = 18.0

        # ── Rule 4: Compressor.P_out fixed → prioritise downstream conditioning
        elif isinstance(node, CompressorNode) and fixed_param == "P_out":
            for outlet in graph.outlet_streams(fixed_tag):
                dst_tag = graph.stream_dest(outlet.tag)
                if dst_tag and dst_tag in unfixed_tags:
                    dst = graph.unit(dst_tag)
                    if isinstance(dst, (HeaterNode, CoolerNode)):
                        boosts[dst_tag] = 12.0

        # ── Rule 5: Expander.P_out fixed → prioritise downstream conditioning
        elif isinstance(node, ExpanderNode) and fixed_param == "P_out":
            for outlet in graph.outlet_streams(fixed_tag):
                dst_tag = graph.stream_dest(outlet.tag)
                if dst_tag and dst_tag in unfixed_tags:
                    dst = graph.unit(dst_tag)
                    if isinstance(dst, (HeaterNode, CoolerNode)):
                        boosts[dst_tag] = 10.0

        return boosts


# ── Coupled settler ────────────────────────────────────────────────────────────

class CoupledSettler:
    """
    Deterministic joint settling after a parameter change.

    When P_out changes on a Pump or Compressor, the bubble point at that
    pressure changes.  Any upstream Heater or Cooler whose T_out was set
    relative to the old bubble point is now potentially wrong.

    settle() recomputes the correct target using the learned MarginModel and
    adjusts the coupled parameter deterministically.  This breaks the
    oscillation cycle where fixing P triggers a T re-fix on the next iteration.

    Zero LLM calls.  Never touches the one parameter just fixed (no regression).
    """

    def settle(
        self,
        graph:      "FlowsheetGraph",
        fixed_tag:  str,
        fixed_param: str,
    ) -> tuple["FlowsheetGraph", list[str]]:
        """
        Returns (possibly patched graph, change_log).  Input is never mutated.
        """
        from ir.graph import (
            PumpNode, CompressorNode, ExpanderNode,
            HeaterNode, CoolerNode, SeparatorNode,
        )
        from ir.thermo_estimation import bubble_point_K
        from ir.margin_model import get_global_margin_model
        from agents.rule_store import classify_compounds

        changes: list[str] = []
        node = graph.unit(fixed_tag)
        if node is None:
            return graph, changes

        # ── Case 1: P_out changed → BP changes → settle upstream T conditioning
        if fixed_param == "P_out" and isinstance(node, (PumpNode, CompressorNode)):
            new_p = node.params.get("P_out")
            if new_p is None:
                return graph, changes

            new_bp = bubble_point_K(graph.compounds, float(new_p))
            if new_bp is None:
                return graph, changes

            margin_model = get_global_margin_model()
            compound_classes = classify_compounds(graph.compounds)
            g = graph.copy()

            for pass_num in range(_MAX_SETTLE_PASSES):
                changed_this_pass = False

                for inp in graph.inlet_streams(fixed_tag):
                    src_tag = graph.stream_source(inp.tag)
                    if src_tag is None or src_tag == fixed_tag:
                        continue
                    src = g.unit(src_tag)

                    if isinstance(src, CoolerNode):
                        t_out = src.params.get("T_out")
                        if t_out is not None:
                            margin = margin_model.get_margin(
                                "Cooler", node.unit_type, compound_classes,
                                "T_out", default=15.0)
                            target = max(new_bp - margin, 273.15)
                            if float(t_out) > target + 5.0:  # meaningful violation
                                old = src.params["T_out"]
                                src.params["T_out"] = round(target, 2)
                                changes.append(
                                    f"COUPLED_SETTLE: {src_tag}.T_out "
                                    f"{old:.1f}→{target:.1f} K "
                                    f"(P_out@{fixed_tag} changed → "
                                    f"BP={new_bp:.1f} K, need T<BP-{margin:.0f})"
                                )
                                changed_this_pass = True

                    elif isinstance(src, HeaterNode):
                        t_out = src.params.get("T_out")
                        if t_out is not None:
                            heater_downstream = {
                                graph.stream_dest(o.tag)
                                for o in graph.outlet_streams(src_tag)
                            }
                            feeds_vessel = any(
                                isinstance(graph.unit(d), SeparatorNode)
                                for d in heater_downstream if d
                            )
                            if feeds_vessel:
                                margin = margin_model.get_margin(
                                    "Heater", "Vessel", compound_classes,
                                    "T_out", default=20.0)
                                target = new_bp + margin
                                if float(t_out) < target - 5.0:  # below target
                                    old = src.params["T_out"]
                                    src.params["T_out"] = round(target, 2)
                                    changes.append(
                                        f"COUPLED_SETTLE: {src_tag}.T_out "
                                        f"{old:.1f}→{target:.1f} K "
                                        f"(P_out@{fixed_tag} changed → "
                                        f"BP={new_bp:.1f} K, need T>BP+{margin:.0f})"
                                    )
                                    changed_this_pass = True

                if not changed_this_pass:
                    if pass_num > 0:
                        logger.debug(
                            "COUPLED_SETTLE %s: converged after %d passes",
                            fixed_tag, pass_num + 1,
                        )
                    break

            return g, changes

        return graph, changes
