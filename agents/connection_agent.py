"""
ConnectionAgent — builds the stream/connection graph for a fixed topology.

Given: unit sequence from TopologyAgent.
Produces: feed stream tags, intermediate stream tags, product stream tags,
          and the full connections list.

Two-stage design:
  Stage 1 (zero LLM): deterministic wiring for standard linear topologies.
    Handles any single-feed chain and multi-feed chains with a leading Mixer.
    Falls through for Splitter topologies and mid-chain Vessels.
  Stage 2 (LLM): called only when Stage 1 cannot determine the wiring.
"""
from __future__ import annotations

import json
import re

from agents.llm           import chat, DEFAULT_MODEL, retry_temperature
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

Example 7 — Units: [HT-01 (Heater), SP-01 (Splitter)], n_feeds=1, compounds=[Ethanol, Water]:
{"feed_tags": ["FEED"], "intermediate_tags": ["HOT"], "product_tags": ["SPLIT1", "SPLIT2"],
 "connections": [["FEED","HT-01",0,0],["HT-01","HOT",0,0],["HOT","SP-01",0,0],["SP-01","SPLIT1",0,0],["SP-01","SPLIT2",1,0]]}

Example 8 — Units: [E-01 (Expander)], n_feeds=1, compounds=[Methane, Propane] (no flash):
{"feed_tags": ["FEED"], "intermediate_tags": [], "product_tags": ["PRODUCT"],
 "connections": [["FEED","E-01",0,0],["E-01","PRODUCT",0,0]]}
"""


# ── Stage 1: deterministic wiring ─────────────────────────────────────────────

# Canonical intermediate stream tag for each unit type's primary outlet.
_OUTSTREAM_NAME: dict[str, str] = {
    "Heater":     "HOT",
    "Cooler":     "COOL",
    "Mixer":      "MIX",
    "Compressor": "COMP",
    "Pump":       "PUMP",
    "Expander":   "EXP",
}


def _outstream_name(unit_type: str, occurrence: int, total_of_type: int) -> str:
    """
    Return the intermediate stream tag for a unit's primary outlet.

    When more than one unit of the same type appears in the chain, append
    the 1-based occurrence index (HOT1/HOT2) to avoid duplicate tags.
    """
    base = _OUTSTREAM_NAME.get(unit_type, "S")
    return f"{base}{occurrence}" if total_of_type > 1 else base


def _build_deterministic_connections(topology: TopologyPlan) -> ConnectionPlan | None:
    """
    Zero-LLM Stage 1: deterministic wiring for standard linear topologies.

    Handles:
    - n_feeds == 1  with any ordered chain of supported units
    - n_feeds >  1  only when units[0].type == "Mixer"

    Falls through (returns None) when:
    - topology is empty
    - any unit is a Splitter (outlet count is ambiguous)
    - n_feeds > 1 and the first unit is not a Mixer
    - a Vessel appears in a non-terminal position

    Stream naming:
    - Single feed         : "FEED"
    - Multiple feeds      : "FEED1", "FEED2", ...
    - Intermediate outlet : unit-type-derived name (HOT, COOL, COMP, PUMP, EXP, MIX)
                            suffixed with occurrence index when more than one of same type
    - Terminal Vessel     : "VAP" (port 0) and "LIQ" (port 1)
    - Terminal non-Vessel : "PRODUCT"
    """
    units   = topology.units
    n_feeds = topology.n_feeds

    if not units:
        return None
    if any(u.type == "Splitter" for u in units):
        return None
    if n_feeds > 1 and units[0].type != "Mixer":
        return None

    feed_tags = ["FEED"] if n_feeds == 1 else [f"FEED{i + 1}" for i in range(n_feeds)]

    # Pre-count each unit type for duplicate-aware stream naming
    type_total: dict[str, int] = {}
    for u in units:
        type_total[u.type] = type_total.get(u.type, 0) + 1
    type_seen: dict[str, int] = {}

    connections:       list      = []
    intermediate_tags: list[str] = []
    product_tags:      list[str] = []

    # Wire feeds into the first unit
    if n_feeds == 1:
        connections.append([feed_tags[0], units[0].tag, 0, 0])
    else:
        for i, ft in enumerate(feed_tags):
            connections.append([ft, units[0].tag, 0, i])

    # Walk the chain unit by unit
    for i, unit in enumerate(units):
        is_last = (i == len(units) - 1)
        type_seen[unit.type] = type_seen.get(unit.type, 0) + 1

        if unit.type == "Vessel":
            if not is_last:
                return None  # Vessel mid-chain → ambiguous downstream wiring
            product_tags += ["VAP", "LIQ"]
            connections.append([unit.tag, "VAP", 0, 0])
            connections.append([unit.tag, "LIQ", 1, 0])

        elif is_last:
            product_tags.append("PRODUCT")
            connections.append([unit.tag, "PRODUCT", 0, 0])

        else:
            name = _outstream_name(
                unit.type, type_seen[unit.type], type_total[unit.type])
            intermediate_tags.append(name)
            connections.append([unit.tag, name, 0, 0])
            connections.append([name, units[i + 1].tag, 0, 0])

    return ConnectionPlan(
        feed_tags=feed_tags,
        intermediate_tags=intermediate_tags,
        product_tags=product_tags,
        connections=connections,
    )


def _build_retry_prompt(
        err: str,
        topology: "TopologyPlan",
) -> str:
    units_str = ", ".join(f"{u.tag} ({u.type})" for u in topology.units)
    return (
        f"CORRECTION REQUIRED — fix EXACTLY this error before anything else:\n"
        f"  {err}\n\n"
        f"Key rules:\n"
        f"  - stream → unit : src_port=0, dst_port=inlet_index\n"
        f"  - unit → stream : src_port=outlet_index, dst_port=0\n"
        f"  - Vessel: src_port=0 → vapour outlet, src_port=1 → liquid outlet\n"
        f"  - Mixer:  dst_port increments per inlet (0, 1, 2, …)\n"
        f"  - Splitter: src_port increments per outlet (0, 1, 2, …)\n"
        f"  - All other units: single outlet on src_port=0\n"
        f"  - Every tag used in connections must be declared in feed/intermediate/product lists.\n\n"
        f"Reference examples:\n"
        f"  HT+Vessel flash: "
        f'{{"feed_tags": ["FEED"], "intermediate_tags": ["S-HT"], "product_tags": ["VAP", "LIQ"], '
        f'"connections": [["FEED","HT-01",0,0],["HT-01","S-HT",0,0],["S-HT","V-01",0,0],'
        f'["V-01","VAP",0,0],["V-01","LIQ",1,0]]}}\n'
        f"  HT only (no flash): "
        f'{{"feed_tags": ["FEED"], "intermediate_tags": [], "product_tags": ["PRODUCT"], '
        f'"connections": [["FEED","HT-01",0,0],["HT-01","PRODUCT",0,0]]}}\n\n'
        f"Units: [{units_str}]\n"
        f"n_feeds: {topology.n_feeds}\n\n"
        f"Return ONLY the JSON object:\n"
        f'{{"feed_tags": [...], "intermediate_tags": [...], "product_tags": [...], "connections": [...]}}'
    )


class ConnectionAgent:
    """
    Wires up the stream and connection graph for a fixed unit topology.

    Stage 1 is free (deterministic); Stage 2 uses one LLM call.
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

        Stage 1 (zero LLM): deterministic wiring for linear topologies.
          Covers single-feed chains and multi-feed chains with a leading Mixer.
          Falls through for Splitter topologies and any non-linear wiring.
        Stage 2 (LLM): called only when Stage 1 returns None.

        Args:
            description : original process description (for LLM context)
            topology    : unit sequence from TopologyAgent
            compounds   : compound names (used for LLM stream naming context)
            max_retries : LLM retry limit (Stage 2 only)
        """
        # ── Stage 1: deterministic wiring ─────────────────────────────────
        result = _build_deterministic_connections(topology)
        if result is not None:
            return result

        # ── Stage 2: LLM ──────────────────────────────────────────────────
        units_str = ", ".join(
            f"{u.tag} ({u.type})" for u in topology.units
        )
        _PORT_REMINDER = (
            "\nREMINDER — port rules (apply these exactly):\n"
            "  stream → unit : src_port=0, dst_port=inlet_index\n"
            "  unit → stream : src_port=outlet_index, dst_port=0\n"
            "  Vessel: src_port=0 → vapour outlet, src_port=1 → liquid outlet\n"
            "  Mixer:  dst_port increments per inlet (0, 1, 2, …)\n"
            "  All other units: single outlet on src_port=0\n"
        )
        prompt = (
            f"{_FEW_SHOT}\n"
            f"Now wire the connections for:\n"
            f"  Units    : [{units_str}]\n"
            f"  n_feeds  : {topology.n_feeds}\n"
            f"  Compounds: {compounds}\n"
            f"  Description: {description}\n"
            f"{_PORT_REMINDER}"
        )

        last_err = ""
        for attempt in range(max_retries):
            try:
                raw = chat(prompt, system=_SYSTEM, model=self._model,
                           temperature=retry_temperature(attempt), thinking=False)
                plan, err = _parse_connections(raw, topology)
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                prompt = _build_retry_prompt(last_err, topology)
                continue
            if plan is not None:
                return plan
            last_err = err
            prompt = _build_retry_prompt(err, topology)

        raise ValueError(
            f"ConnectionAgent failed after {max_retries} attempts. Last error: {last_err}")


# ── Parsing ────────────────────────────────────────────────────────────────────

def _parse_connections(raw: str, topology: TopologyPlan) -> tuple[ConnectionPlan | None, str]:
    """Extract and validate a ConnectionPlan from raw LLM output."""
    try:
        text = raw.strip()
        m = re.search(r'\{.*\}', text, re.DOTALL)
        data = json.loads(m.group(0) if m else text)
        if not isinstance(data, dict):
            return None, f"Expected JSON object, got {type(data).__name__}: {str(data)[:80]}"
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as e:
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
