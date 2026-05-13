"""
Assembler — deterministic, zero-LLM final step of the Planner pipeline.

Combines TopologyPlan + ConnectionPlan + ConditionPlan into the canonical
flowsheet dict that agents/schema.py validates and the Executor runs.

No LLM call.  No retries.  If inputs are valid, output is always valid.
"""
from __future__ import annotations

from agents.planner_types import (
    TopologyPlan, ConnectionPlan, ConditionPlan
)


class Assembler:
    """Combines the three sub-plans into a complete flowsheet dict."""

    def assemble(
            self,
            compounds:        list[str],
            property_package: str,
            topology:         TopologyPlan,
            connections:      ConnectionPlan,
            conditions:       ConditionPlan,
    ) -> dict:
        """
        Return a flowsheet dict ready for schema.validate() and the Executor.

        Feed streams get full T/P/flow/composition from ConditionPlan.
        Intermediate and product streams get only a tag (simulator fills them).
        Units get their type-specific parameters merged from ConditionPlan.
        """
        # ── Streams ────────────────────────────────────────────────────────────
        streams: list[dict] = []

        for tag in connections.feed_tags:
            cond = conditions.feed_conditions.get(tag)
            if cond is not None:
                streams.append({
                    "tag":         tag,
                    "T":           cond.T,
                    "P":           cond.P,
                    "flow":        cond.flow,
                    "composition": dict(cond.composition),
                })
            else:
                streams.append({"tag": tag})   # fallback

        for tag in connections.intermediate_tags + connections.product_tags:
            streams.append({"tag": tag})

        # ── Units ──────────────────────────────────────────────────────────────
        units: list[dict] = []
        for unit_spec in topology.units:
            params = conditions.unit_parameters.get(unit_spec.tag, {})
            unit = {"tag": unit_spec.tag, "type": unit_spec.type}
            unit.update(params)
            units.append(unit)

        # ── Assemble ───────────────────────────────────────────────────────────
        return {
            "compounds":        compounds,
            "property_package": property_package,
            "streams":          streams,
            "units":            units,
            "connections":      [list(c) for c in connections.connections],
        }
