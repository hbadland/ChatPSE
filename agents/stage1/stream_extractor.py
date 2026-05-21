"""
Agent B — Stream Extractor.

Input : NL description + SemanticUnits from Agent A
Output: SemanticTopology — streams with source/destination and feed conditions

Responsibility: define HOW units are connected and what the feed conditions are.
Does NOT assign unit parameters (T_out, P_out, etc.) — that is Agent E.

Each stream entry declares:
  - tag: stream name
  - src: source unit tag (null for feed streams)
  - dst: destination unit tag (null for product streams)
  - is_feed: true if this is a process feed stream (carries T, P, composition)
  - T, P, flow, composition: only for feed streams

Prompt is short and structured for small models.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from agents.llm import chat, DEFAULT_MODEL, retry_temperature

_SYSTEM = """\
Define the process streams connecting the unit operations.
Return ONLY a JSON object — no explanation, no markdown.

Schema:
{
  "streams": [
    {
      "tag": "FEED",
      "src": null,
      "dst": "HT-01",
      "is_feed": true,
      "T": 298.15,
      "P": 101325.0,
      "flow": 1.0,
      "composition": {"Ethanol": 0.5, "Water": 0.5}
    },
    {
      "tag": "HOT",
      "src": "HT-01",
      "dst": "V-01",
      "is_feed": false
    },
    {
      "tag": "VAP",
      "src": "V-01",
      "dst": null,
      "is_feed": false
    }
  ]
}

Rules:
- Every unit must have at least one inlet and one outlet stream
- Feed streams (src=null) MUST include T [K], P [Pa], flow [mol/s], composition (mole fractions summing to 1.0)
- Intermediate and product streams carry tag, src, dst only — do NOT guess T/P/composition
- Units never connect directly — every unit-to-unit link needs a stream between them
- A Vessel has TWO outlet streams: vapour (name containing VAP/VAPOR/GAS) and liquid (name containing LIQ/LIQUID)
- All temperatures in Kelvin, all pressures in Pascals
- Reasonable defaults: ambient feed at T=298.15 K, P=101325 Pa unless described otherwise

Examples:

Process: "Heat ethanol-water feed, then flash separate"
Units: HT-01 (Heater), V-01 (Vessel)
Output:
{"streams": [
  {"tag": "FEED", "src": null,   "dst": "HT-01", "is_feed": true,
   "T": 298.15, "P": 101325.0, "flow": 1.0, "composition": {"Ethanol": 0.5, "Water": 0.5}},
  {"tag": "HOT",  "src": "HT-01","dst": "V-01",  "is_feed": false},
  {"tag": "VAP",  "src": "V-01", "dst": null,    "is_feed": false},
  {"tag": "LIQ",  "src": "V-01", "dst": null,    "is_feed": false}
]}

Process: "Compress methane feed, cool it, then expand"
Units: CP-01 (Compressor), CL-01 (Cooler), EX-01 (Expander)
Output:
{"streams": [
  {"tag": "FEED",  "src": null,    "dst": "CP-01", "is_feed": true,
   "T": 298.15, "P": 101325.0, "flow": 1.0, "composition": {"Methane": 1.0}},
  {"tag": "COMP",  "src": "CP-01", "dst": "CL-01", "is_feed": false},
  {"tag": "COOL",  "src": "CL-01", "dst": "EX-01", "is_feed": false},
  {"tag": "EXPND", "src": "EX-01", "dst": null,    "is_feed": false}
]}"""


@dataclass
class SemanticStream:
    tag:         str
    src:         Optional[str]
    dst:         Optional[str]
    is_feed:     bool
    T:           Optional[float] = None
    P:           Optional[float] = None
    flow:        Optional[float] = None
    composition: dict            = field(default_factory=dict)

    def validate(self, unit_tags: set[str]) -> list[str]:
        errors = []
        if not self.tag:
            errors.append("Stream has empty tag")
        if self.src and self.src not in unit_tags:
            errors.append(f"Stream '{self.tag}' src='{self.src}' not in unit list")
        if self.dst and self.dst not in unit_tags:
            errors.append(f"Stream '{self.tag}' dst='{self.dst}' not in unit list")
        if self.src is None and self.dst is None:
            errors.append(f"Stream '{self.tag}' has no src and no dst (isolated)")
        if self.is_feed:
            if self.T is None:
                errors.append(f"Feed stream '{self.tag}' missing T")
            elif not (50 < self.T < 2000):
                errors.append(f"Feed stream '{self.tag}' T={self.T} K out of range — use Kelvin")
            if self.P is None:
                errors.append(f"Feed stream '{self.tag}' missing P")
            elif not (100 < self.P < 1e8):
                errors.append(f"Feed stream '{self.tag}' P={self.P} Pa out of range — use Pascals")
            if not self.composition:
                errors.append(f"Feed stream '{self.tag}' missing composition")
            else:
                total = sum(self.composition.values())
                if abs(total - 1.0) > 0.01:
                    errors.append(
                        f"Feed stream '{self.tag}' composition sums to {total:.4f}, not 1.0")
        return errors


@dataclass
class SemanticTopology:
    streams:  list[SemanticStream]
    raw_json: dict = field(default_factory=dict)

    def validate(self, unit_tags: set[str]) -> list[str]:
        errors: list[str] = []
        tags: set[str] = set()
        for s in self.streams:
            errors += s.validate(unit_tags)
            if s.tag in tags:
                errors.append(f"Duplicate stream tag '{s.tag}'")
            tags.add(s.tag)
        return errors

    def feed_streams(self) -> list[SemanticStream]:
        return [s for s in self.streams if s.is_feed]


class StreamExtractor:
    def __init__(self, model: str = DEFAULT_MODEL):
        self._model = model

    def extract(
        self,
        description:           str,
        compounds:             list[str],
        unit_tags:             list[str],
        unit_roles:            dict[str, str],
        max_retries:           int = 3,
        concentration_hints:   list[str] | None = None,
        suggested_compositions: dict[str, dict[str, float]] | None = None,
    ) -> SemanticTopology:
        prompt = _build_prompt(description, compounds, unit_tags, unit_roles,
                               concentration_hints, suggested_compositions)
        unit_tag_set = set(unit_tags)
        last_error = ""
        for attempt in range(max_retries):
            raw = chat(
                prompt + (f"\n\nPrevious error: {last_error}" if last_error else ""),
                system=_SYSTEM,
                model=self._model,
                temperature=retry_temperature(attempt),
                max_tokens=2048,
            )
            try:
                data  = _parse_json(raw)
                streams = [_parse_stream(s) for s in data.get("streams", [])]
                result  = SemanticTopology(streams=streams, raw_json=data)
                errors  = result.validate(unit_tag_set)
                if not errors:
                    return result
                last_error = "; ".join(errors)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                last_error = str(e)

        raise RuntimeError(
            f"StreamExtractor failed after {max_retries} attempts. "
            f"Last error: {last_error}\nLast response: {raw[:300]}")


def _build_prompt(
    description:            str,
    compounds:              list[str],
    unit_tags:              list[str],
    unit_roles:             dict[str, str],
    concentration_hints:    list[str] | None = None,
    suggested_compositions: dict[str, dict[str, float]] | None = None,
) -> str:
    unit_lines = "\n".join(
        f"  {tag}: {unit_roles.get(tag, '')}" for tag in unit_tags)
    parts = [
        f"Process description: {description}",
        f"Compounds: {', '.join(compounds)}",
        f"Units (in flow order):\n{unit_lines}",
    ]
    # Inject structured composition information from BasisAgent — avoids
    # the model having to re-infer feed composition from raw prose.
    if concentration_hints:
        hint_block = "\n".join(f"  - {h}" for h in concentration_hints)
        parts.append(f"Concentration information (use for feed composition):\n{hint_block}")
    if suggested_compositions:
        comp_lines = []
        for alias, comp in suggested_compositions.items():
            fracs = ", ".join(f"{k}: {v:.3f}" for k, v in comp.items())
            comp_lines.append(f"  {alias}: {{{fracs}}}")
        parts.append(
            "Suggested mole fractions (use directly for feed composition "
            "if consistent with description):\n" + "\n".join(comp_lines))
    parts.append(
        "Define the streams connecting these units. "
        "Include the feed stream(s) with T, P, flow, and composition.")
    return "\n\n".join(parts)


def _parse_stream(s: dict) -> SemanticStream:
    return SemanticStream(
        tag         = s["tag"],
        src         = s.get("src"),
        dst         = s.get("dst"),
        is_feed     = bool(s.get("is_feed", False)),
        T           = s.get("T"),
        P           = s.get("P"),
        flow        = s.get("flow"),
        composition = s.get("composition", {}),
    )


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    return json.loads(text)
