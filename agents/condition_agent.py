"""
ConditionAgent — assigns numerical values to feed streams and unit parameters.

Given: compound list, property package, unit topology, connection graph,
       bubble-point estimate, suggested compositions.
Produces: T/P/flow/composition for every feed stream, and type-specific
          parameters for every unit (T_out, P_out, efficiency, dP, …).

This is the most numerically demanding step.  By this point the package is
pre-selected, the topology is fixed, and connections are wired — the LLM
only needs to fill in numbers with clear physical constraints.  This focused
scope makes the task tractable for open-source models.
"""
from __future__ import annotations

import json
import re

from agents.llm           import chat, DEFAULT_MODEL, retry_temperature
from agents.planner_types import (
    TopologyPlan, ConnectionPlan, ConditionPlan, StreamCondition
)

# ── System prompts (one per LLM call) ─────────────────────────────────────────

_SYSTEM_FEEDS = """\
You are a chemical process feed conditions expert.
Set the temperature, pressure, molar flow, and composition for each feed stream.

Output ONLY this JSON object (no markdown, no explanation):
{
  "<stream_tag>": {
    "T": <K>, "P": <Pa>, "flow": <mol/s>,
    "composition": {"<compound>": <mole_fraction>, ...}
  },
  ...
}

SI units — no exceptions:
  Temperature : Kelvin   (25 °C = 298.15 K,  80 °C = 353.15 K, 100 °C = 373.15 K)
  Pressure    : Pascals  (1 atm = 101 325 Pa, 1 bar = 100 000 Pa, 10 bar = 1 000 000 Pa)
  Flow        : mol/s    (use 1.0 if not specified)

MUST:
1. ALL compounds must appear in EVERY stream composition — use 0.0 if a compound
   is absent from that particular feed stream.
2. Mole fractions in each stream must sum to EXACTLY 1.0.
3. Temperature MUST be in Kelvin. 25 °C → 298.15 K, NOT 25.
4. Pressure MUST be in Pascals. 1 bar → 100 000 Pa, NOT 1.

Feed temperature guide:
  - Liquid feed at ambient conditions : T ≈ 298 K, P = 101 325 Pa
  - Hot vapour feed (Cooler topology) : T = bubble_point + 50–70 K
  - High-pressure gas feed             : use the stated pressure in Pa

Bubble point reference at 1 atm (use CONDITION ESTIMATE if provided instead):
  Methane 112 K | Ethane 185 K | Propane 231 K | n-Butane 273 K
  n-Pentane 309 K | Diethyl Ether 308 K | DCM 313 K | Acetone 329 K
  Chloroform 334 K | MeOH 338 K | THF 339 K | n-Hexane 342 K
  EtOAc 350 K | EtOH 352 K | Benzene 353 K | Cyclohexane 354 K
  IPA 356 K | n-PrOH 370 K | Water 373 K | Toluene 384 K | n-Butanol 391 K
  Mixture ≈ Σ(xᵢ × T_bub_i)\
"""

_FEW_SHOT_FEEDS = """\
Example A — single feed, 50/50 methanol/water liquid at ambient:
Feeds: [FEED]  Compounds: [Methanol, Water]  Package: NRTL
{"FEED": {"T": 298.15, "P": 101325.0, "flow": 1.0,
          "composition": {"Methanol": 0.5, "Water": 0.5}}}

Example B — two pure feeds for a Mixer:
Feeds: [MEOH, H2O]  Compounds: [Methanol, Water]  Package: NRTL
{"MEOH": {"T": 298.15, "P": 101325.0, "flow": 1.0,
          "composition": {"Methanol": 1.0, "Water": 0.0}},
 "H2O":  {"T": 298.15, "P": 101325.0, "flow": 1.0,
          "composition": {"Methanol": 0.0, "Water": 1.0}}}

Example C — high-pressure gas feed for a Compressor topology:
Feeds: [FEED]  Compounds: [Methane, Ethane]  Package: Peng-Robinson
{"FEED": {"T": 250.0, "P": 500000.0, "flow": 1.0,
          "composition": {"Methane": 0.7, "Ethane": 0.3}}}

Example D — hot vapour feed for a Cooler topology (T above bubble point):
Feeds: [FEED]  Compounds: [Benzene, Toluene]  Package: Raoult's Law
  (Benzene bp ≈ 353 K, Toluene bp ≈ 384 K; 50/50 blend ≈ 369 K → hot vapour T ≈ 420 K)
{"FEED": {"T": 420.0, "P": 101325.0, "flow": 1.0,
          "composition": {"Benzene": 0.5, "Toluene": 0.5}}}

Example E — non-equimolar feed with composition hint (70/30 ethanol/water):
Feeds: [FEED]  Compounds: [Ethanol, Water]  Package: NRTL
  (Bubble point rule: 0.7×352K + 0.3×373K = 358.5K → ambient liquid feed at 298 K)
{"FEED": {"T": 298.15, "P": 101325.0, "flow": 1.0,
          "composition": {"Ethanol": 0.7, "Water": 0.3}}}
"""

_SYSTEM_UNITS = """\
You are a chemical process unit operation expert.
Given confirmed feed conditions, set the operating parameters for each unit.

Output ONLY this JSON object (no markdown, no explanation):
{
  "<unit_tag>": {<type-specific parameters>},
  ...
}

Unit parameter formats:
  Heater     : {"T_out": <K>,  "dP": 0.0}
  Cooler     : {"T_out": <K>,  "dP": 0.0}
  Vessel     : {"dP": 0.0}
  Mixer      : {"dP": 0.0}
  Splitter   : {"split_fractions": {"<outlet_tag>": <fraction>, ...}, "dP": 0.0}
  Pump       : {"P_out": <Pa>, "efficiency": 0.75}
  Compressor : {"P_out": <Pa>, "efficiency": 0.75}
  Expander   : {"P_out": <Pa>, "efficiency": 0.75}

MUST — read every rule carefully:
1. Every unit in the list MUST have an entry. Do not omit any.
2. Heater T_out MUST be ABOVE the bubble point.
   WHY: below the bubble point the stream is all-liquid; the Vessel vapour outlet
        has ZERO flow, which is a simulation failure.
   TARGET: bubble_point + 15–25 K.
3. Cooler T_out MUST be in the two-phase region — ABOVE the bubble point.
   WHY: same as above. Cooling BELOW the bubble point = all-liquid = zero vapour.
   TARGET: bubble_point + 10–20 K.
4. Compressor P_out MUST be GREATER than the feed stream pressure.
   FORBIDDEN: P_out ≤ feed P (compressor would do nothing).
5. Pump P_out MUST be GREATER than the feed stream pressure.
   FORBIDDEN: P_out ≤ feed P (pump would do nothing).
6. Expander P_out MUST be LESS than the feed stream pressure.
   FORBIDDEN: P_out ≥ feed P (expander would do nothing).
7. Splitter split_fractions must sum to exactly 1.0 across all outlet streams.\
"""

_FEW_SHOT_UNITS = """\
Example A — Heater → Vessel (bubble point computed from Step 1):
Units: [HT-01 (Heater), V-01 (Vessel)]
Feed T=298.15K  P=101325Pa  — Estimated bubble point: 355K
→ HT-01 T_out must be 355+15=370K to 355+25=380K
{"HT-01": {"T_out": 372.0, "dP": 0.0}, "V-01": {"dP": 0.0}}

Example B — Compressor → Cooler → Vessel:
Units: [K-01 (Compressor), CL-01 (Cooler), V-01 (Vessel)]
Feed T=250K  P=500000Pa (5 bar)  — Estimated bubble point at 5MPa: ~220K
→ K-01 P_out must be > 500000Pa (description says 50 bar = 5000000Pa)
→ CL-01 T_out must be above estimated bubble point at 5MPa
{"K-01": {"P_out": 5000000.0, "efficiency": 0.75},
 "CL-01": {"T_out": 235.0, "dP": 0.0},
 "V-01": {"dP": 0.0}}

Example C — Mixer → Heater → Vessel (two feeds merged):
Units: [MX-01 (Mixer), HT-01 (Heater), V-01 (Vessel)]
Blended feed T=298.15K  P=101325Pa — Estimated bubble point: 355K
{"MX-01": {"dP": 0.0}, "HT-01": {"T_out": 370.0, "dP": 0.0}, "V-01": {"dP": 0.0}}

Example D — Pump → Vessel (liquid pressurisation):
Units: [P-01 (Pump), V-01 (Vessel)]
Feed T=250K  P=101325Pa (1 atm) — description says pump to 10 bar
→ P-01 P_out = 10 bar = 1000000 Pa  (must be > 101325 Pa)
{"P-01": {"P_out": 1000000.0, "efficiency": 0.75}, "V-01": {"dP": 0.0}}

Example E — Expander (pressure reduction, work recovery):
Units: [E-01 (Expander)]
Feed T=300K  P=5000000Pa (50 bar) — description says expand to 5 bar
→ E-01 P_out = 5 bar = 500000 Pa  (must be < 5000000 Pa)
{"E-01": {"P_out": 500000.0, "efficiency": 0.75}}

Example F — Heater → Splitter (two equal product streams):
Units: [HT-01 (Heater), SP-01 (Splitter)]  Products: [SPLIT1, SPLIT2]
Feed T=298.15K  P=101325Pa — no flash; heat then split equally
{"HT-01": {"T_out": 350.0, "dP": 0.0},
 "SP-01": {"split_fractions": {"SPLIT1": 0.5, "SPLIT2": 0.5}, "dP": 0.0}}
"""


class ConditionAgent:
    """
    Assigns numerical operating conditions to feed streams and unit parameters.

    Stage 1 (zero LLM): deterministic conditions for simple flash topologies.
    Stage 2 (two sequential LLM calls):
      Call 1 — feed conditions (T, P, flow, composition per feed stream)
      Call 2 — unit parameters (T_out / P_out / etc. per unit),
               with locked feed conditions and computed bubble point as context.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self._model = model

    def plan(
            self,
            description:            str,
            compounds:              list[str],
            property_package:       str,
            topology:               TopologyPlan,
            connections:            ConnectionPlan,
            condition_estimate:     str | None = None,
            suggested_compositions: dict | None = None,
            condition_feedback:     str | None = None,
            max_retries:            int = 4,
    ) -> ConditionPlan:
        """
        Return a ConditionPlan with feed conditions and unit parameters.

        Stage 1 bypassed when condition_feedback is set (prior attempt failed).
        Stage 2: two focused LLM calls — feeds first, units second.
        """
        # ── Stage 1: deterministic (zero LLM) ─────────────────────────────────
        if condition_feedback is None:
            simple = _build_simple_conditions(
                compounds, topology, connections, suggested_compositions)
            if simple is not None:
                return simple

        # ── Stage 2a: feed conditions ──────────────────────────────────────────
        feed_conditions = self._plan_feeds(
            description, compounds, property_package,
            connections, condition_estimate, suggested_compositions,
            condition_feedback, max_retries,
        )

        # ── Stage 2b: unit parameters (with locked feed conditions) ────────────
        unit_parameters = self._plan_units(
            description, compounds, property_package,
            topology, connections, feed_conditions,
            condition_estimate, max_retries,
        )

        return ConditionPlan(
            feed_conditions=feed_conditions,
            unit_parameters=unit_parameters,
        )

    # ── Call 1: feed conditions ────────────────────────────────────────────────

    def _plan_feeds(
            self,
            description:            str,
            compounds:              list[str],
            property_package:       str,
            connections:            ConnectionPlan,
            condition_estimate:     str | None,
            suggested_compositions: dict | None,
            condition_feedback:     str | None,
            max_retries:            int,
    ) -> dict[str, StreamCondition]:
        """Call 1: set T, P, flow, composition for every feed stream."""
        feeds_str = ", ".join(connections.feed_tags)
        lines = []
        if condition_feedback:
            lines += [
                "CORRECTION FROM PRIOR FAILURE — address this BEFORE reading the examples:",
                condition_feedback,
                "",
            ]
        lines += [
            _FEW_SHOT_FEEDS,
            "━━━ NOW SET FEED CONDITIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  Feed streams  : [{feeds_str}]",
            f"  Compounds     : {compounds}",
            f"  Package       : {property_package}",
            f"  Description   : {description}",
        ]
        if condition_estimate:
            lines += ["", "CONDITION ESTIMATE (use these T/P values):", condition_estimate]
        if suggested_compositions:
            lines += ["", "SUGGESTED COMPOSITIONS from compound identification:"]
            for alias, comp in suggested_compositions.items():
                frac_str = ", ".join(f"{k}: {v:.4f}" for k, v in comp.items())
                lines.append(f"  '{alias}' → {{{frac_str}}}")
        lines += [
            "",
            f"Return ONLY the JSON object for feed streams {connections.feed_tags}.",
        ]
        prompt = "\n".join(lines)

        last_err = ""
        for attempt in range(max_retries):
            try:
                raw = chat(prompt, system=_SYSTEM_FEEDS, model=self._model,
                           temperature=retry_temperature(attempt), thinking=False)
                feed_conds, err = _parse_feed_conditions(raw, compounds, connections)
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                prompt = _build_feeds_retry_prompt(
                    last_err, connections.feed_tags, compounds)
                continue
            if feed_conds is not None:
                return feed_conds
            last_err = err
            prompt = _build_feeds_retry_prompt(err, connections.feed_tags, compounds)

        raise ValueError(
            f"ConditionAgent (feeds) failed after {max_retries} attempts. "
            f"Last error: {last_err}")

    # ── Call 2: unit parameters ────────────────────────────────────────────────

    def _plan_units(
            self,
            description:        str,
            compounds:          list[str],
            property_package:   str,
            topology:           TopologyPlan,
            connections:        ConnectionPlan,
            feed_conditions:    dict[str, StreamCondition],
            condition_estimate: str | None,
            max_retries:        int,
    ) -> dict[str, dict]:
        """Call 2: set unit parameters using the locked feed conditions as context."""
        context = _build_unit_context(
            compounds, feed_conditions, topology, connections)
        units_str = ", ".join(f"{u.tag} ({u.type})" for u in topology.units)
        prompt = "\n".join([
            _FEW_SHOT_UNITS,
            "━━━ NOW SET UNIT PARAMETERS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            context,
            "",
            f"Description   : {description}",
            f"Package       : {property_package}",
            "",
            f"Return ONLY the JSON object for units [{units_str}].",
        ])

        last_err = ""
        for attempt in range(max_retries):
            try:
                raw = chat(prompt, system=_SYSTEM_UNITS, model=self._model,
                           temperature=retry_temperature(attempt), thinking=False)
                unit_params, err = _parse_unit_parameters(raw, topology)
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                prompt = _build_units_retry_prompt(
                    last_err, topology, feed_conditions, compounds)
                continue
            if unit_params is not None:
                return unit_params
            last_err = err
            prompt = _build_units_retry_prompt(
                err, topology, feed_conditions, compounds)

        raise ValueError(
            f"ConditionAgent (units) failed after {max_retries} attempts. "
            f"Last error: {last_err}")


# ── Context builder for Call 2 ─────────────────────────────────────────────────

def _build_unit_context(
        compounds:       list[str],
        feed_conditions: dict[str, StreamCondition],
        topology:        "TopologyPlan",
        connections:     "ConnectionPlan",
) -> str:
    """
    Build the rich context block for Call 2.

    Shows the LOCKED feed conditions from Call 1, the computed mixture bubble
    point, and per-unit T_out/P_out targets derived from those values.
    This eliminates the need for the model to re-derive bubble points.
    """
    from agents.chem_data import estimate_bubble_point

    lines = ["LOCKED FEED CONDITIONS (from Step 1 — confirmed, do not change):"]
    for tag, cond in feed_conditions.items():
        comp_str = ", ".join(f"{k}: {v}" for k, v in cond.composition.items())
        lines.append(
            f"  {tag}: T={cond.T}K  P={cond.P}Pa  flow={cond.flow}mol/s")
        lines.append(f"         composition={{{comp_str}}}")

    # Compute bubble point from blended feed composition
    if feed_conditions:
        total_flow = sum(c.flow for c in feed_conditions.values()) or 1.0
        blend: dict[str, float] = {}
        for cond in feed_conditions.values():
            for compound, frac in cond.composition.items():
                blend[compound] = blend.get(compound, 0.0) + (cond.flow / total_flow) * frac
        ref_feed = next(iter(feed_conditions.values()))
        t_bub = estimate_bubble_point(compounds, blend, ref_feed.P)
        feed_p = ref_feed.P

        if t_bub is not None:
            lines.append(f"\nESTIMATED MIXTURE BUBBLE POINT: {t_bub}K at {feed_p:.0f}Pa")
            has_vessel = any(u.type == "Vessel" for u in topology.units)
            for u in topology.units:
                if u.type == "Heater" and has_vessel:
                    lo, hi = round(t_bub + 15, 0), round(t_bub + 25, 0)
                    lines.append(
                        f"→ {u.tag} (Heater) T_out target: {lo}–{hi}K "
                        f"(bubble_point + 15–25K)")
                elif u.type == "Cooler" and has_vessel:
                    lo, hi = round(t_bub + 10, 0), round(t_bub + 20, 0)
                    lines.append(
                        f"→ {u.tag} (Cooler) T_out target: {lo}–{hi}K "
                        f"(two-phase region, above bubble_point)")
        else:
            lines.append(
                "\nBUBBLE POINT: unknown compound — set Heater/Cooler "
                "T_out = feed T + 50–70K to ensure two-phase flow")

        # Per-unit pressure guidance for pressure-changing equipment
        feed_p_min = min(c.P for c in feed_conditions.values())
        feed_p_max = max(c.P for c in feed_conditions.values())
        for u in topology.units:
            if u.type in ("Compressor", "Pump"):
                lines.append(
                    f"→ {u.tag} ({u.type}) P_out must be > {feed_p_min:.0f}Pa "
                    f"(feed pressure)")
            elif u.type == "Expander":
                lines.append(
                    f"→ {u.tag} (Expander) P_out must be < {feed_p_max:.0f}Pa "
                    f"(feed pressure)")

    # Splitter outlet tags from connection graph
    outlet_map: dict[str, list[str]] = {}
    for conn in connections.connections:
        src, dst = conn[0], conn[1]
        for u in topology.units:
            if u.type == "Splitter" and src == u.tag:
                outlet_map.setdefault(u.tag, []).append(dst)

    lines.append("\nUNITS TO CONFIGURE:")
    for u in topology.units:
        if u.type == "Splitter":
            outlets = outlet_map.get(u.tag, ["OUT1", "OUT2"])
            outlets_str = ", ".join(f'"{o}": <fraction>' for o in outlets)
            lines.append(
                f"  {u.tag} (Splitter): split_fractions must sum to 1.0 "
                f"→ {{{outlets_str}}}")
        elif u.type in ("Heater", "Cooler"):
            lines.append(f"  {u.tag} ({u.type}): set T_out [K] and dP [Pa, = 0.0]")
        elif u.type in ("Compressor", "Pump", "Expander"):
            lines.append(
                f"  {u.tag} ({u.type}): set P_out [Pa] and efficiency [0–1, default 0.75]")
        else:
            lines.append(f"  {u.tag} ({u.type}): set dP [Pa, = 0.0]")

    return "\n".join(lines)


# ── Retry prompts (error at top) ───────────────────────────────────────────────

def _build_feeds_retry_prompt(
        err: str,
        feed_tags: list[str],
        compounds: list[str],
) -> str:
    return (
        f"CORRECTION REQUIRED — fix EXACTLY this error:\n"
        f"  {err}\n\n"
        f"Key rules:\n"
        f"  - Temperature in Kelvin (25°C = 298.15K, NOT 25).\n"
        f"  - Pressure in Pascals (1 atm = 101325Pa, 1 bar = 100000Pa).\n"
        f"  - ALL compounds must appear in every composition (0.0 if absent).\n"
        f"  - Mole fractions must sum to exactly 1.0 per stream.\n\n"
        f"Required feed streams: {feed_tags}\n"
        f"Compounds: {compounds}\n\n"
        f"Return ONLY the JSON object:\n"
        + "{\n"
        + "".join(f'  "{t}": {{"T": <K>, "P": <Pa>, "flow": <mol/s>, '
                 f'"composition": {{...}}}},\n'
                 for t in feed_tags)
        + "}"
    )


def _build_units_retry_prompt(
        err: str,
        topology: "TopologyPlan",
        feed_conditions: dict[str, StreamCondition],
        compounds: list[str],
) -> str:
    from agents.chem_data import estimate_bubble_point
    units_str = ", ".join(f"{u.tag} ({u.type})" for u in topology.units)

    # Re-compute bubble point using flow-weighted blend (same as _build_unit_context)
    bubble_hint = ""
    if feed_conditions:
        total_flow = sum(c.flow for c in feed_conditions.values()) or 1.0
        blend: dict[str, float] = {}
        for cond in feed_conditions.values():
            for compound, frac in cond.composition.items():
                blend[compound] = blend.get(compound, 0.0) + (cond.flow / total_flow) * frac
        ref = next(iter(feed_conditions.values()))
        t_bub = estimate_bubble_point(compounds, blend, ref.P)
        if t_bub is not None:
            bubble_hint = (
                f"  - Estimated bubble point: {t_bub}K at {ref.P:.0f}Pa.\n"
                f"  - Heater/Cooler T_out must be > {t_bub}K.\n"
            )

    return (
        f"CORRECTION REQUIRED — fix EXACTLY this error:\n"
        f"  {err}\n\n"
        f"Key rules:\n"
        f"  - Every unit must have an entry.\n"
        f"  - Compressor/Pump P_out > feed pressure. Expander P_out < feed pressure.\n"
        + bubble_hint
        + f"\nRequired units: [{units_str}]\n\n"
        f"Return ONLY the JSON object:\n"
        + "{\n"
        + "".join(f'  "{u.tag}": {{...}},\n' for u in topology.units)
        + "}"
    )


# ── Stage 1: deterministic conditions for simple flash topologies ─────────────

_SIMPLE_UNIT_TYPES = {"Heater", "Cooler", "Vessel", "Mixer", "Splitter"}
_PRESSURE_CHANGERS = {"Compressor", "Pump", "Expander"}


def _build_simple_conditions(
        compounds:              list[str],
        topology:               "TopologyPlan",
        connections:            "ConnectionPlan",
        suggested_compositions: dict | None,
) -> "ConditionPlan | None":
    """
    Zero-LLM Stage 1: deterministic conditions for simple flash topologies.

    Returns a ConditionPlan when:
    - All units are in {Heater, Cooler, Vessel, Mixer, Splitter}
    - All compounds are in the NBP table (bubble point estimable)
    - n_feeds is 1 or equals the number of compounds (pure-component feeds)

    Feed temperature heuristics:
    - Heater present  → feed is ambient liquid  (298.15 K)
    - Cooler present, no Heater → feed is hot vapour  (bubble_pt + 60 K)
    - Vessel only     → feed is ambient (pre-flight Fix C raises T if needed)

    Unit T_out = bubble_point + 20 K for Heater, + 8 K for Cooler
    (safely within the two-phase region for the first iteration).
    """
    from agents.chem_data import estimate_bubble_point

    # Only handle pressure-static topologies
    if any(u.type in _PRESSURE_CHANGERS for u in topology.units):
        return None

    # Validate all unit types are known
    if any(u.type not in _SIMPLE_UNIT_TYPES for u in topology.units):
        return None

    # Infer an equimolar or suggested composition for the blended feed
    blend_comp = _blend_composition(compounds, suggested_compositions)

    # Estimate bubble point at 1 atm
    t_bub = estimate_bubble_point(compounds, blend_comp, 101_325.0)
    if t_bub is None:
        return None  # Unknown compound — fall through to LLM

    # Choose feed temperature based on topology
    has_heater = any(u.type == "Heater" for u in topology.units)
    has_cooler = any(u.type == "Cooler" for u in topology.units)
    if has_cooler and not has_heater:
        feed_T = round(t_bub + 60.0, 1)   # hot vapour above dew point
    else:
        feed_T = 298.15                     # ambient liquid

    n_feeds = len(connections.feed_tags)

    # Build per-feed compositions
    if n_feeds == 1:
        per_feed = {connections.feed_tags[0]: blend_comp}
    elif n_feeds == len(compounds):
        # Assume one pure-component feed per compound (common mixer pattern)
        per_feed = {}
        for tag, compound in zip(connections.feed_tags, compounds):
            per_feed[tag] = {c: (1.0 if c == compound else 0.0) for c in compounds}
    elif suggested_compositions and len(suggested_compositions) == n_feeds:
        # Match by position
        per_feed = {}
        for tag, comp_dict in zip(connections.feed_tags, suggested_compositions.values()):
            if not isinstance(comp_dict, dict):
                return None
            full = {c: float(comp_dict.get(c, 0.0)) for c in compounds}
            total = sum(full.values())
            if total < 0.01:
                return None
            per_feed[tag] = {c: round(v / total, 6) for c, v in full.items()}
    else:
        return None  # Can't determine per-feed compositions

    feed_conditions: dict[str, StreamCondition] = {}
    for tag, comp in per_feed.items():
        feed_conditions[tag] = StreamCondition(
            T=feed_T, P=101_325.0, flow=1.0, composition=comp)

    # Build unit parameters
    unit_parameters: dict[str, dict] = {}
    for unit in topology.units:
        if unit.type == "Heater":
            unit_parameters[unit.tag] = {"T_out": round(t_bub + 20.0, 1), "dP": 0.0}
        elif unit.type == "Cooler":
            unit_parameters[unit.tag] = {"T_out": round(t_bub + 8.0, 1), "dP": 0.0}
        elif unit.type in ("Vessel", "Mixer", "Splitter"):
            unit_parameters[unit.tag] = {"dP": 0.0}

    return ConditionPlan(
        feed_conditions=feed_conditions,
        unit_parameters=unit_parameters,
    )


def _blend_composition(
        compounds:              list[str],
        suggested_compositions: dict | None,
) -> dict[str, float]:
    """
    Return a normalised composition dict for the blended feed mixture.
    Uses the first entry in suggested_compositions that covers all compounds,
    otherwise falls back to equimolar.
    """
    if suggested_compositions:
        for comp_dict in suggested_compositions.values():
            if not isinstance(comp_dict, dict):
                continue
            full = {c: float(comp_dict.get(c, 0.0)) for c in compounds}
            total = sum(full.values())
            if total > 0.01:
                return {c: round(v / total, 6) for c, v in full.items()}
    # Equimolar fallback
    n = max(len(compounds), 1)
    return {c: round(1.0 / n, 6) for c in compounds}


# ── Parsing ────────────────────────────────────────────────────────────────────

def _parse_feed_conditions(
        raw:         str,
        compounds:   list[str],
        connections: ConnectionPlan,
) -> tuple[dict[str, StreamCondition] | None, str]:
    """Parse Call 1 output: {tag: {T, P, flow, composition}} → feed_conditions dict."""
    try:
        text = raw.strip()
        m = re.search(r'\{.*\}', text, re.DOTALL)
        data = json.loads(m.group(0) if m else text)
        if not isinstance(data, dict):
            return None, f"Expected JSON object, got {type(data).__name__}: {str(data)[:80]}"
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as e:
        return None, f"JSON parse error: {e}"

    feed_conditions: dict[str, StreamCondition] = {}
    for tag, cond in data.items():
        if not isinstance(cond, dict):
            return None, (
                f"Stream '{tag}' must be a dict with keys T/P/flow/composition, "
                f"got {type(cond).__name__}: {str(cond)[:40]}"
            )
        missing = [k for k in ("T", "P", "flow", "composition") if k not in cond]
        if missing:
            return None, f"Stream '{tag}' missing keys: {missing}"
        comp = cond["composition"]
        if not isinstance(comp, dict):
            return None, f"Stream '{tag}' composition must be a dict"
        missing_c = [c for c in compounds if c not in comp]
        if missing_c:
            return None, (
                f"Stream '{tag}' missing compounds in composition: {missing_c}. "
                f"Set them to 0.0 if absent."
            )
        total = sum(comp.values())
        if abs(total - 1.0) > 0.01:
            return None, (
                f"Stream '{tag}' composition sums to {total:.4f}, must be 1.0."
            )
        feed_conditions[tag] = StreamCondition(
            T=float(cond["T"]),
            P=float(cond["P"]),
            flow=float(cond["flow"]),
            composition={k: float(v) for k, v in comp.items()},
        )

    missing_feeds = [t for t in connections.feed_tags if t not in feed_conditions]
    if missing_feeds:
        return None, f"No conditions provided for feed stream(s): {missing_feeds}"

    return feed_conditions, ""


def _parse_unit_parameters(
        raw:      str,
        topology: "TopologyPlan",
) -> tuple[dict[str, dict] | None, str]:
    """Parse Call 2 output: {unit_tag: {params}} → unit_parameters dict."""
    try:
        text = raw.strip()
        m = re.search(r'\{.*\}', text, re.DOTALL)
        data = json.loads(m.group(0) if m else text)
        if not isinstance(data, dict):
            return None, f"Expected JSON object, got {type(data).__name__}: {str(data)[:80]}"
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as e:
        return None, f"JSON parse error: {e}"

    unit_tags = {u.tag for u in topology.units}
    missing_units = [u.tag for u in topology.units if u.tag not in data]
    if missing_units:
        return None, f"Missing parameter entries for unit(s): {missing_units}"

    unit_parameters: dict[str, dict] = {}
    for tag, params in data.items():
        if tag not in unit_tags:
            continue  # ignore extra keys silently
        if not isinstance(params, dict):
            return None, f"Unit '{tag}' parameters must be a dict, got {type(params).__name__}"
        unit_parameters[tag] = {
            k: (float(v) if isinstance(v, (int, float)) else v)
            for k, v in params.items()
        }

    return unit_parameters, ""
