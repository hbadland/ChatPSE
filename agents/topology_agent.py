"""
TopologyAgent — determines the unit-operation sequence for a process description.

Stage 1 (zero LLM): topology_library pattern match.
  If the description scores above the confidence threshold, the library's
  template is used directly — free, instant, reliable for known patterns.

Stage 2 (LLM): called only when Stage 1 returns no match.
  The LLM receives a narrow task: output a JSON list of units with tags and
  types. No stream conditions, no connections, no thermodynamics. This makes
  the task tractable for open-source models.
"""
from __future__ import annotations

import json
import re

from agents.llm           import chat, DEFAULT_MODEL, retry_temperature
from agents.planner_types import TopologyPlan, UnitSpec
from agents.topology_library import match as topology_match, TopologyHint

# ── Tag prefix conventions ─────────────────────────────────────────────────────
_TAG_PREFIX: dict[str, str] = {
    "Heater":     "HT",
    "Cooler":     "CL",
    "Vessel":     "V",
    "Mixer":      "MX",
    "Splitter":   "SP",
    "Pump":       "P",
    "Compressor": "K",
    "Expander":   "E",
}
_SUPPORTED_TYPES = set(_TAG_PREFIX)


def _make_tags(unit_types: list[str]) -> list[UnitSpec]:
    """Assign canonical tags to an ordered list of unit types."""
    counts: dict[str, int] = {}
    specs = []
    for utype in unit_types:
        counts[utype] = counts.get(utype, 0) + 1
        prefix = _TAG_PREFIX.get(utype, "U")
        specs.append(UnitSpec(tag=f"{prefix}-{counts[utype]:02d}", type=utype))
    return specs


def _n_feeds_from_template(hint: TopologyHint) -> int:
    """Infer number of feed streams from template name."""
    name = hint.template.name.lower()
    if "3-stream" in name or "three" in name:
        return 3
    if "2-stream" in name or "two stream" in name or "two feed" in name:
        return 2
    return 1


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a chemical process topology expert.
Given a process description and compound list, output the SEQUENCE of unit
operations needed — nothing else.

Output ONLY this JSON object (no markdown, no explanation):
{
  "units": [{"tag": "<TAG>", "type": "<type>"}, ...],
  "n_feeds": <integer>,
  "reasoning": "<one sentence>"
}

MUST:
- "units" lists units in processing order (feed enters the first unit).
- "n_feeds" is the number of SEPARATE external feed streams (usually 1;
  use 2 for a Mixer with two feeds, 3 for a three-stream Mixer).
- Supported types: Heater, Cooler, Vessel, Mixer, Splitter, Pump, Compressor, Expander.
- Tag conventions: HT=Heater, CL=Cooler, V=Vessel, MX=Mixer, SP=Splitter,
  P=Pump, K=Compressor, E=Expander.  Number sequentially: HT-01, HT-02, …

INCLUDE a Vessel when:
- The description says "flash", "separate", "drum", "two-phase", or "vapour/liquid split".
- The feed will enter the two-phase region under the stated or implied conditions.
- Separation of a vapour phase from a liquid phase is the goal of the process.

FORBIDDEN:
- Do NOT include a Vessel if the description says "no flash", "no phase separation",
  or only mentions compression/heating/cooling without separation.
- Do NOT include a Heater before a Vessel if the description says the feed is already
  in the two-phase region at the stated conditions.
- Do NOT include more units than the description requires — minimal topology preferred.
"""

_FEW_SHOT = """\
Example 1 — "Heat a methanol/water liquid at 298 K to partial vaporisation then flash it":
{"units": [{"tag": "HT-01", "type": "Heater"}, {"tag": "V-01", "type": "Vessel"}], "n_feeds": 1, "reasoning": "Heater raises feed into the two-phase region; Vessel separates vapour from liquid."}

Example 2 — "Cool a hot benzene/toluene vapour at 400 K into the two-phase region then flash":
{"units": [{"tag": "CL-01", "type": "Cooler"}, {"tag": "V-01", "type": "Vessel"}], "n_feeds": 1, "reasoning": "Cooler brings hot vapour into two-phase region; Vessel separates phases."}

Example 3 — "Compress a methane/ethane gas from 5 bar to 50 bar, cool the compressed stream, then flash":
{"units": [{"tag": "K-01", "type": "Compressor"}, {"tag": "CL-01", "type": "Cooler"}, {"tag": "V-01", "type": "Vessel"}], "n_feeds": 1, "reasoning": "Compressor raises pressure; Cooler brings stream into two-phase region; Vessel separates."}

Example 4 — "Mix a methanol stream and a water stream, heat the blend, then flash":
{"units": [{"tag": "MX-01", "type": "Mixer"}, {"tag": "HT-01", "type": "Heater"}, {"tag": "V-01", "type": "Vessel"}], "n_feeds": 2, "reasoning": "Two feeds blended in Mixer; Heater raises blend into two-phase region; Vessel separates."}

Example 5 — "Flash a nitrogen/oxygen feed at 80 K and 10 bar directly in a drum":
{"units": [{"tag": "V-01", "type": "Vessel"}], "n_feeds": 1, "reasoning": "Feed is already in the two-phase region at stated conditions; a single Vessel suffices."}

Example 6 — "Compress propane from 1 bar to 10 bar, then cool to 320 K. No phase separation required.":
{"units": [{"tag": "K-01", "type": "Compressor"}, {"tag": "CL-01", "type": "Cooler"}], "n_feeds": 1, "reasoning": "Compressor raises pressure; Cooler reduces temperature. No Vessel — description explicitly excludes separation."}

Example 7 — "Pump a propane/butane liquid to 10 bar then flash it":
{"units": [{"tag": "P-01", "type": "Pump"}, {"tag": "V-01", "type": "Vessel"}], "n_feeds": 1, "reasoning": "Pump pressurises liquid; Vessel flash-separates phases at elevated pressure."}

Example 8 — "Flash separate a 30/70 CO2/methane mixture at 80 bar and 250 K":
{"units": [{"tag": "V-01", "type": "Vessel"}], "n_feeds": 1, "reasoning": "Feed is stated to be at flash conditions; no pre-conditioning needed."}

Example 9 — "Preheat a methane stream from 25 °C to 200 °C before entering a reactor. No separation needed.":
{"units": [{"tag": "HT-01", "type": "Heater"}], "n_feeds": 1, "reasoning": "Heater raises feed to reactor inlet temperature; no flash or phase separation is required."}

Example 10 — "Expand a high-pressure methane/propane gas from 50 bar to 5 bar to recover work":
{"units": [{"tag": "E-01", "type": "Expander"}], "n_feeds": 1, "reasoning": "Single Expander reduces pressure and recovers work; no separation or heat exchange mentioned."}

Example 11 — "Heat a benzene/toluene liquid and split it equally into two product streams":
{"units": [{"tag": "HT-01", "type": "Heater"}, {"tag": "SP-01", "type": "Splitter"}], "n_feeds": 1, "reasoning": "Heater raises feed temperature; Splitter divides the heated stream into two equal outputs."}
"""


def _build_retry_prompt(
        err: str,
        description: str,
        compounds: list[str],
        property_package: str,
) -> str:
    types_list = ", ".join(sorted(_SUPPORTED_TYPES))
    return (
        f"CORRECTION REQUIRED — fix EXACTLY this error before anything else:\n"
        f"  {err}\n\n"
        f"Key rules:\n"
        f"  - Supported types: {types_list}\n"
        f"  - Tag format: HT-01, CL-01, V-01, MX-01, SP-01, P-01, K-01, E-01\n"
        f"  - Include a Vessel ONLY when separation (flash/phase split) is needed.\n"
        f"  - Minimal topology: do not add units not required by the description.\n\n"
        f"Reference examples:\n"
        f'  Flash topology: {{"units": [{{"tag": "HT-01", "type": "Heater"}}, '
        f'{{"tag": "V-01", "type": "Vessel"}}], "n_feeds": 1, '
        f'"reasoning": "Heater raises feed into two-phase region; Vessel separates."}}\n'
        f'  Heater only (no separation): {{"units": [{{"tag": "HT-01", "type": "Heater"}}], '
        f'"n_feeds": 1, "reasoning": "Heater raises temperature; no flash required."}}\n\n'
        f"Description      : {description}\n"
        f"Compounds        : {compounds}\n"
        f"Property package : {property_package}\n\n"
        f"Return ONLY the JSON object:\n"
        f'{{"units": [...], "n_feeds": <int>, "reasoning": "<one sentence>"}}'
    )


class TopologyAgent:
    """
    Determines the unit-operation sequence for a process description.

    Stage 1 is free (topology_library); Stage 2 uses one LLM call.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self._model = model

    def plan(
            self,
            description:       str,
            compounds:         list[str],
            property_package:  str,
            hint:              TopologyHint | None = None,
            max_retries:       int = 2,
            topology_feedback: str | None = None,
    ) -> TopologyPlan:
        """
        Return a TopologyPlan for the given description.

        Args:
            description     : normalised process description
            compounds       : exact DWSIM compound names
            property_package: pre-selected package (used as context for LLM)
            hint            : topology_library match result (None = no library match)
            max_retries     : LLM retry limit
        """
        # ── Stage 1: topology library ──────────────────────────────────────────
        # If no hint was pre-computed by the orchestrator, try matching now.
        if hint is None:
            hint = topology_match(description, compounds)

        if hint is not None:
            units = _make_tags(hint.template.units)
            return TopologyPlan(
                units=units,
                n_feeds=_n_feeds_from_template(hint),
                reasoning=f"Pattern-matched: {hint.template.name} (score={hint.score:.2f})",
                source="topology_library",
            )

        # ── Stage 2: LLM ──────────────────────────────────────────────────────
        feedback_block = ""
        if topology_feedback:
            feedback_block = (
                f"\nMUST NOT: use the same unit sequence — it failed with:\n"
                f"{topology_feedback.strip()}\n"
                f"Choose a fundamentally different unit sequence that avoids this failure.\n"
            )
        prompt = (
            f"{_FEW_SHOT}\n"
            f"Now generate the topology for:\n"
            f"  Description      : {description}\n"
            f"  Compounds        : {compounds}\n"
            f"  Property package : {property_package}\n"
            f"{feedback_block}"
        )

        last_err = ""
        for attempt in range(max_retries):
            raw = chat(prompt, system=_SYSTEM, model=self._model,
                       temperature=retry_temperature(attempt), thinking=False)
            plan, err = _parse_topology(raw)
            if plan is not None:
                return plan
            last_err = err
            prompt = _build_retry_prompt(err, description, compounds, property_package)

        raise ValueError(
            f"TopologyAgent failed after {max_retries} attempts. Last error: {last_err}")


# ── Parsing ────────────────────────────────────────────────────────────────────

def _parse_topology(raw: str) -> tuple[TopologyPlan | None, str]:
    """Extract and validate a TopologyPlan from raw LLM output."""
    try:
        text = raw.strip()
        m = re.search(r'\{.*\}', text, re.DOTALL)
        data = json.loads(m.group(0) if m else text)
        if not isinstance(data, dict):
            return None, f"Expected JSON object, got {type(data).__name__}: {str(data)[:80]}"
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as e:
        return None, f"JSON parse error: {e}"

    if "units" not in data or not isinstance(data["units"], list):
        return None, 'Missing or invalid "units" list'

    units = []
    for item in data["units"]:
        if not isinstance(item, dict) or "tag" not in item or "type" not in item:
            return None, f"Unit item missing 'tag' or 'type': {item}"
        if item["type"] not in _SUPPORTED_TYPES:
            return None, f"Unsupported unit type '{item['type']}'"
        units.append(UnitSpec(tag=item["tag"], type=item["type"]))

    if not units:
        return None, "Unit list is empty"

    n_feeds = int(data.get("n_feeds", 1))
    reasoning = str(data.get("reasoning", ""))

    return TopologyPlan(units=units, n_feeds=n_feeds, reasoning=reasoning), ""
