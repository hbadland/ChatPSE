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

from agents.llm           import chat, DEFAULT_MODEL
from agents.planner_types import (
    TopologyPlan, ConnectionPlan, ConditionPlan, StreamCondition
)

# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a chemical process numerical conditions expert.
Given a fixed topology and connection graph, assign T/P/flow/composition to
each feed stream and the required parameters to each unit operation.

Output ONLY this JSON object (no markdown, no explanation):
{
  "feed_conditions": {
    "<stream_tag>": {
      "T": <K>, "P": <Pa>, "flow": <mol/s>,
      "composition": {"<compound>": <mole_fraction>, ...}
    },
    ...
  },
  "unit_parameters": {
    "<unit_tag>": {<type-specific params>},
    ...
  }
}

SI units throughout — no exceptions:
  Temperature : Kelvin   (25 °C = 298.15 K, 80 °C = 353.15 K, 100 °C = 373.15 K)
  Pressure    : Pascals  (1 atm = 101 325 Pa, 1 bar = 100 000 Pa, 10 bar = 1 000 000 Pa)
  Flow        : mol/s    (default 1.0 if not specified)

Unit parameter reference:
  Heater     : {"T_out": <K>,  "dP": 0.0}
  Cooler     : {"T_out": <K>,  "dP": 0.0}
  Vessel     : {"dP": 0.0}
  Mixer      : {"dP": 0.0}
  Splitter   : {"split_fractions": {"<stream>": <fraction>, ...}, "dP": 0.0}
  Pump       : {"P_out": <Pa>, "efficiency": 0.75}
  Compressor : {"P_out": <Pa>, "efficiency": 0.75}
  Expander   : {"P_out": <Pa>, "efficiency": 0.75}

Critical rules:
1. Mole fractions in EVERY feed stream must sum to exactly 1.0.
2. All compounds must appear in every feed stream composition (use 0.0 for absent).
3. Heater T_out MUST be above the mixture bubble point to produce vapour.
4. Cooler T_out MUST be above the bubble point to produce a two-phase mixture.
   (Below the bubble point → all liquid, zero vapour in flash vessel.)
5. Compressor/Pump P_out MUST be greater than feed stream pressure.
6. Expander P_out MUST be less than feed stream pressure.
"""

_FEW_SHOT = """\
Example 1 — Units: [HT-01 (Heater), V-01 (Vessel)], feeds: [FEED]
  Compounds: [Methanol, Water], package: NRTL, bubble point ~355 K, 50/50 at 1 atm:

{"feed_conditions": {"FEED": {"T": 298.15, "P": 101325.0, "flow": 1.0,
   "composition": {"Methanol": 0.5, "Water": 0.5}}},
 "unit_parameters": {"HT-01": {"T_out": 370.0, "dP": 0.0}, "V-01": {"dP": 0.0}}}

Example 2 — Units: [K-01 (Compressor), CL-01 (Cooler), V-01 (Vessel)], feeds: [FEED]
  Compounds: [Methane, Ethane], package: Peng-Robinson, 70/30, feed at 5 bar 250 K, target 50 bar:

{"feed_conditions": {"FEED": {"T": 250.0, "P": 500000.0, "flow": 1.0,
   "composition": {"Methane": 0.7, "Ethane": 0.3}}},
 "unit_parameters": {"K-01": {"P_out": 5000000.0, "efficiency": 0.75},
                     "CL-01": {"T_out": 220.0, "dP": 0.0},
                     "V-01": {"dP": 0.0}}}

Example 3 — Units: [MX-01 (Mixer), HT-01 (Heater), V-01 (Vessel)], feeds: [MEOH, H2O]
  Compounds: [Methanol, Water], package: NRTL, two pure feed streams at 298 K 1 atm:

{"feed_conditions": {
   "MEOH": {"T": 298.15, "P": 101325.0, "flow": 1.0, "composition": {"Methanol": 1.0, "Water": 0.0}},
   "H2O":  {"T": 298.15, "P": 101325.0, "flow": 1.0, "composition": {"Methanol": 0.0, "Water": 1.0}}},
 "unit_parameters": {"MX-01": {"dP": 0.0}, "HT-01": {"T_out": 360.0, "dP": 0.0}, "V-01": {"dP": 0.0}}}

Example 4 — Units: [CL-01 (Cooler), V-01 (Vessel)], feeds: [FEED]
  Compounds: [Benzene, Toluene], package: Raoult's Law, hot vapour feed at 400 K 1 atm, 40/60:

{"feed_conditions": {"FEED": {"T": 400.0, "P": 101325.0, "flow": 1.0,
   "composition": {"Benzene": 0.4, "Toluene": 0.6}}},
 "unit_parameters": {"CL-01": {"T_out": 360.0, "dP": 0.0}, "V-01": {"dP": 0.0}}}

Example 5 — Units: [V-01 (Vessel)], feeds: [FEED]
  Compounds: [Nitrogen, Oxygen], package: Lee-Kesler-Plöcker, cryogenic feed at 80 K 10 bar, 79/21:

{"feed_conditions": {"FEED": {"T": 80.0, "P": 1000000.0, "flow": 1.0,
   "composition": {"Nitrogen": 0.79, "Oxygen": 0.21}}},
 "unit_parameters": {"V-01": {"dP": 0.0}}}
"""


class ConditionAgent:
    """
    Assigns numerical operating conditions to feed streams and unit parameters.
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
            max_retries:            int = 3,
    ) -> ConditionPlan:
        """
        Return a ConditionPlan with feed conditions and unit parameters.

        Args:
            description            : original process description
            compounds              : exact DWSIM compound names
            property_package       : pre-selected thermodynamic package
            topology               : unit sequence from TopologyAgent
            connections            : stream graph from ConnectionAgent
            condition_estimate     : bubble-point estimate from orchestrator
            suggested_compositions : feed compositions from BasisAgent
            condition_feedback     : correction from a prior failed attempt
            max_retries            : LLM retry limit (each retry gives more guidance)
        """
        base_prompt = self._build_prompt(
            description, compounds, property_package,
            topology, connections,
            condition_estimate, suggested_compositions,
        )
        prompt = base_prompt

        if condition_feedback:
            prompt += f"\n\nCORRECTION FROM PREVIOUS ATTEMPT:\n{condition_feedback}\n"

        last_err = ""
        for attempt in range(max_retries):
            raw = chat(prompt, system=_SYSTEM, model=self._model, temperature=0)
            plan, err = _parse_conditions(raw, compounds, connections)
            if plan is not None:
                return plan
            last_err = err
            prompt = (
                f"{base_prompt}\n\n"
                f"ATTEMPT {attempt + 1} ERROR — fix this specific issue:\n"
                f"  {err}\n"
                f"Return corrected JSON only."
            )

        raise ValueError(
            f"ConditionAgent failed after {max_retries} attempts. Last error: {last_err}")

    def _build_prompt(
            self,
            description:            str,
            compounds:              list[str],
            property_package:       str,
            topology:               TopologyPlan,
            connections:            ConnectionPlan,
            condition_estimate:     str | None,
            suggested_compositions: dict | None,
    ) -> str:
        units_str = ", ".join(f"{u.tag} ({u.type})" for u in topology.units)
        feeds_str = ", ".join(connections.feed_tags)

        lines = [
            _FEW_SHOT,
            "Now assign conditions for:",
            f"  Units            : [{units_str}]",
            f"  Feed streams     : [{feeds_str}]",
            f"  Compounds        : {compounds}",
            f"  Property package : {property_package}",
            f"  Description      : {description}",
        ]

        if condition_estimate:
            lines += ["", "CONDITION ESTIMATE (use as a starting point):", condition_estimate]

        if suggested_compositions:
            lines += ["", "SUGGESTED FEED COMPOSITIONS from compound identification:"]
            for alias, comp in suggested_compositions.items():
                frac_str = ", ".join(f"{k}: {v:.4f}" for k, v in comp.items())
                lines.append(f"  '{alias}' → {{{frac_str}}}")

        return "\n".join(lines)


# ── Parsing ────────────────────────────────────────────────────────────────────

def _parse_conditions(
        raw:         str,
        compounds:   list[str],
        connections: ConnectionPlan,
) -> tuple[ConditionPlan | None, str]:
    """Extract and validate a ConditionPlan from raw LLM output."""
    try:
        text = raw.strip()
        m = re.search(r'\{.*\}', text, re.DOTALL)
        data = json.loads(m.group(0) if m else text)
    except (json.JSONDecodeError, AttributeError) as e:
        return None, f"JSON parse error: {e}"

    if "feed_conditions" not in data or "unit_parameters" not in data:
        return None, 'Missing "feed_conditions" or "unit_parameters"'

    # ── Parse feed conditions ──────────────────────────────────────────────────
    feed_conditions: dict[str, StreamCondition] = {}
    for tag, cond in data["feed_conditions"].items():
        missing = [k for k in ("T", "P", "flow", "composition") if k not in cond]
        if missing:
            return None, f"Feed stream '{tag}' missing keys: {missing}"
        comp = cond["composition"]
        if not isinstance(comp, dict):
            return None, f"Feed stream '{tag}' composition must be a dict"
        # Check all compounds are present
        missing_c = [c for c in compounds if c not in comp]
        if missing_c:
            return None, (
                f"Feed stream '{tag}' missing compounds in composition: {missing_c}. "
                f"Set them to 0.0 if absent from this stream."
            )
        total = sum(comp.values())
        if abs(total - 1.0) > 0.01:
            return None, (
                f"Feed stream '{tag}' composition sums to {total:.4f}, must be 1.0."
            )
        feed_conditions[tag] = StreamCondition(
            T=float(cond["T"]),
            P=float(cond["P"]),
            flow=float(cond["flow"]),
            composition={k: float(v) for k, v in comp.items()},
        )

    # Verify all declared feed streams have conditions
    missing_feeds = [t for t in connections.feed_tags if t not in feed_conditions]
    if missing_feeds:
        return None, f"No conditions provided for feed stream(s): {missing_feeds}"

    # ── Parse unit parameters ──────────────────────────────────────────────────
    unit_parameters: dict[str, dict] = {}
    for tag, params in data["unit_parameters"].items():
        if not isinstance(params, dict):
            return None, f"Unit '{tag}' parameters must be a dict"
        unit_parameters[tag] = {k: (float(v) if isinstance(v, (int, float)) else v)
                                 for k, v in params.items()}

    return ConditionPlan(
        feed_conditions=feed_conditions,
        unit_parameters=unit_parameters,
    ), ""
