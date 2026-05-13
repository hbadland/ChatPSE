"""
ConnectionAgent — builds the stream/connection graph for a fixed topology.

Given: unit sequence from TopologyAgent.
Produces: feed stream tags, intermediate stream tags, product stream tags,
          and the full connections list.

This is a graph-wiring task, not a thermodynamics task.  The LLM only needs
to connect a known list of boxes in the correct order — tractable for small
open-source models.
"""
from __future__ import annotations

import json
import re

from agents.llm           import chat, DEFAULT_MODEL
from agents.planner_types import TopologyPlan, ConnectionPlan

# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a process flowsheet wiring expert.
Given an ordered list of unit operations and the number of feed streams,
output the stream connection graph.

Output ONLY this JSON object (no markdown, no explanation):
{
  "feed_tags":         ["<tag>", ...],
  "intermediate_tags": ["<tag>", ...],
  "product_tags":      ["<tag>", ...],
  "connections": [
    ["<src_tag>", "<dst_tag>", <src_port>, <dst_port>],
    ...
  ]
}

Connection rules:
- Every unit-to-unit link needs an intermediate stream between them.
- Feed streams enter the first unit (or Mixer inlets).
- Product streams exit the last unit's outlets.
- Intermediate streams need only a tag — no conditions.
- Connection format: [src_tag, dst_tag, src_port, dst_port]
    stream → unit : src_port=0, dst_port=inlet_index
    unit → stream : src_port=outlet_index, dst_port=0
- Vessel outlets: src_port=0 → vapour, src_port=1 → liquid.
- Mixer inlets:   dst_port increments per inlet (0, 1, 2, …).
- Splitter outlets: src_port increments per outlet (0, 1, 2, …).
- All other units: single outlet on src_port=0.

Stream naming conventions:
- Feeds    : FEED (single), FEED1/FEED2 (multiple), or compound-named (MEOH, H2O)
- Vapour products   : VAP
- Liquid products   : LIQ
- Intermediates     : descriptive short tag (COMP, HOT, COOL, MIX, PUMP, …)
- If no flash separation: single product stream named PRODUCT
"""

_FEW_SHOT = """\
Example 1 — Units: [HT-01 (Heater), V-01 (Vessel)], n_feeds=1, compounds=[Methanol, Water]:
{"feed_tags": ["FEED"], "intermediate_tags": ["HOT"], "product_tags": ["VAP", "LIQ"],
 "connections": [["FEED","HT-01",0,0],["HT-01","HOT",0,0],["HOT","V-01",0,0],["V-01","VAP",0,0],["V-01","LIQ",1,0]]}

Example 2 — Units: [K-01 (Compressor), CL-01 (Cooler), V-01 (Vessel)], n_feeds=1, compounds=[Methane, Ethane]:
{"feed_tags": ["FEED"], "intermediate_tags": ["COMP", "COOL"], "product_tags": ["VAP", "LIQ"],
 "connections": [["FEED","K-01",0,0],["K-01","COMP",0,0],["COMP","CL-01",0,0],["CL-01","COOL",0,0],["COOL","V-01",0,0],["V-01","VAP",0,0],["V-01","LIQ",1,0]]}

Example 3 — Units: [MX-01 (Mixer), HT-01 (Heater), V-01 (Vessel)], n_feeds=2, compounds=[Methanol, Water]:
{"feed_tags": ["MEOH", "H2O"], "intermediate_tags": ["MIX", "HOT"], "product_tags": ["VAP", "LIQ"],
 "connections": [["MEOH","MX-01",0,0],["H2O","MX-01",0,1],["MX-01","MIX",0,0],["MIX","HT-01",0,0],["HT-01","HOT",0,0],["HOT","V-01",0,0],["V-01","VAP",0,0],["V-01","LIQ",1,0]]}

Example 4 — Units: [K-01 (Compressor), CL-01 (Cooler)], n_feeds=1, compounds=[Propane] (no flash):
{"feed_tags": ["FEED"], "intermediate_tags": ["COMP"], "product_tags": ["PRODUCT"],
 "connections": [["FEED","K-01",0,0],["K-01","COMP",0,0],["COMP","CL-01",0,0],["CL-01","PRODUCT",0,0]]}

Example 5 — Units: [V-01 (Vessel)], n_feeds=1, compounds=[Nitrogen, Oxygen]:
{"feed_tags": ["FEED"], "intermediate_tags": [], "product_tags": ["VAP", "LIQ"],
 "connections": [["FEED","V-01",0,0],["V-01","VAP",0,0],["V-01","LIQ",1,0]]}

Example 6 — Units: [MX-01 (Mixer), HT-01 (Heater), V-01 (Vessel)], n_feeds=3, compounds=[A, B, C]:
{"feed_tags": ["FEED1", "FEED2", "FEED3"], "intermediate_tags": ["MIX", "HOT"], "product_tags": ["VAP", "LIQ"],
 "connections": [["FEED1","MX-01",0,0],["FEED2","MX-01",0,1],["FEED3","MX-01",0,2],["MX-01","MIX",0,0],["MIX","HT-01",0,0],["HT-01","HOT",0,0],["HOT","V-01",0,0],["V-01","VAP",0,0],["V-01","LIQ",1,0]]}
"""


class ConnectionAgent:
    """
    Wires up the stream and connection graph for a fixed unit topology.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self._model = model

    def plan(
            self,
            description: str,
            topology:    TopologyPlan,
            compounds:   list[str],
            max_retries: int = 2,
    ) -> ConnectionPlan:
        """
        Return a ConnectionPlan for the given topology.

        Args:
            description : original process description (for context)
            topology    : unit sequence from TopologyAgent
            compounds   : compound names (used for stream naming context)
            max_retries : LLM retry limit
        """
        units_str = ", ".join(
            f"{u.tag} ({u.type})" for u in topology.units
        )
        prompt = (
            f"{_FEW_SHOT}\n"
            f"Now wire the connections for:\n"
            f"  Units    : [{units_str}]\n"
            f"  n_feeds  : {topology.n_feeds}\n"
            f"  Compounds: {compounds}\n"
            f"  Description: {description}\n"
        )

        last_err = ""
        for _ in range(max_retries):
            raw = chat(prompt, system=_SYSTEM, model=self._model, temperature=0)
            plan, err = _parse_connections(raw, topology)
            if plan is not None:
                return plan
            last_err = err
            prompt = (
                f"{prompt}\n\n"
                f"Previous output was invalid: {err}\n"
                f"Fix and return valid JSON only."
            )

        raise ValueError(
            f"ConnectionAgent failed after {max_retries} attempts. Last error: {last_err}")


# ── Parsing ────────────────────────────────────────────────────────────────────

def _parse_connections(raw: str, topology: TopologyPlan) -> tuple[ConnectionPlan | None, str]:
    """Extract and validate a ConnectionPlan from raw LLM output."""
    try:
        text = raw.strip()
        m = re.search(r'\{.*\}', text, re.DOTALL)
        data = json.loads(m.group(0) if m else text)
    except (json.JSONDecodeError, AttributeError) as e:
        return None, f"JSON parse error: {e}"

    for key in ("feed_tags", "intermediate_tags", "product_tags", "connections"):
        if key not in data:
            return None, f"Missing required key '{key}'"

    feed_tags         = [str(t) for t in data["feed_tags"]]
    intermediate_tags = [str(t) for t in data["intermediate_tags"]]
    product_tags      = [str(t) for t in data["product_tags"]]

    if len(feed_tags) != topology.n_feeds:
        return None, (
            f"Expected {topology.n_feeds} feed stream(s), "
            f"got {len(feed_tags)}: {feed_tags}"
        )

    if not feed_tags:
        return None, "feed_tags must not be empty"

    # Validate connections are lists of 4 elements
    connections = []
    for i, conn in enumerate(data.get("connections", [])):
        if not isinstance(conn, list) or len(conn) != 4:
            return None, f"Connection {i} must be a 4-element list, got: {conn}"
        connections.append(list(conn))

    # All stream tags that appear in connections must be declared
    all_tags = set(feed_tags) | set(intermediate_tags) | set(product_tags)
    unit_tags = {u.tag for u in topology.units}
    for conn in connections:
        for tag in (conn[0], conn[1]):
            if tag not in all_tags and tag not in unit_tags:
                return None, f"Tag '{tag}' in connections is not declared in any stream list"

    return ConnectionPlan(
        feed_tags=feed_tags,
        intermediate_tags=intermediate_tags,
        product_tags=product_tags,
        connections=connections,
    ), ""
