"""
Agent A — Unit Extractor.

Input : normalised NL description + compound list from BasisAgent
Output: SemanticUnits — ordered list of unit operations with roles

Responsibility: identify WHAT unit operations are needed and WHY.
Does NOT assign parameters, stream connections, or thermodynamics.

Prompt is short and structured to work reliably on small models.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from agents.llm import chat, DEFAULT_MODEL, retry_temperature

SUPPORTED_UNIT_TYPES = [
    "Heater", "Cooler", "Vessel", "Mixer",
    "Splitter", "Pump", "Compressor", "Expander",
]

_EMPTY_ERRORS = ("empty response", "line 1 column 1", "only markdown")

# Stripped fallback used on the final retry when the model returns nothing —
# avoids the full prompt triggering a long thinking block that exhausts max_tokens.
_MINIMAL_SYSTEM = (
    'Return ONLY: {"units": [{"tag": "HT-01", "type": "Heater", "role": "..."}]}'
    "\nTypes: Heater Cooler Vessel Mixer Splitter Pump Compressor Expander"
)


_SYSTEM = """\
/no_think
Extract unit operations from a chemical process description.
Return ONLY a JSON object — no explanation, no markdown, no <think> blocks.

Schema:
{
  "units": [
    {
      "tag": "HT-01",
      "type": "<one of: Heater Cooler Vessel Mixer Splitter Pump Compressor Expander>",
      "role": "<one-line purpose, e.g. 'heat feed to flash temperature'>"
    }
  ]
}

Rules:
- Tags: use type abbreviation + 2-digit index (HT-01, V-01, MX-01, etc.)
  Abbreviations: Heater=HT, Cooler=CL, Vessel=V, Mixer=MX, Splitter=SP, Pump=PM, Compressor=CP, Expander=EX
- List units in process flow order (feed to product)
- Include only units explicitly needed — do not add units not implied by the description
- A Vessel performs flash separation (vapour + liquid); use it when phase separation is needed
- Include a Mixer only when the description explicitly mentions combining or mixing two feed streams
- Include a Splitter only when the description explicitly mentions splitting a stream into two fractions
- IGNORE any preamble, commentary, or metadata about property packages, thermodynamic models,
  configuration validity, or simulation settings — extract only the physical unit operations

Examples:

Input: "Heat a feed of ethanol and water to 80°C, then flash it to separate the vapour"
Compounds: ethanol, water
Output:
{"units": [
  {"tag": "HT-01", "type": "Heater", "role": "heat ethanol-water feed to flash temperature"},
  {"tag": "V-01",  "type": "Vessel", "role": "flash separation of ethanol and water"}
]}

Input: "Compress a natural gas stream, cool it, then expand through a turbine"
Compounds: methane, ethane, propane
Output:
{"units": [
  {"tag": "CP-01", "type": "Compressor", "role": "compress natural gas feed"},
  {"tag": "CL-01", "type": "Cooler",     "role": "cool compressed gas before expansion"},
  {"tag": "EX-01", "type": "Expander",   "role": "expand gas through turbine to recover work"}
]}

Input: "Pump liquid acetone to high pressure, heat it, then flash to recover acetone vapour"
Compounds: acetone, water
Output:
{"units": [
  {"tag": "PM-01", "type": "Pump",   "role": "raise liquid feed pressure"},
  {"tag": "HT-01", "type": "Heater", "role": "heat pressurised feed to flash temperature"},
  {"tag": "V-01",  "type": "Vessel", "role": "flash to separate acetone vapour from liquid"}
]}

Input: "Invalid package: near-azeotrope with Raoult's Law. Heat a 1-propanol/water feed to 90°C then flash to separate vapour."
Compounds: 1-Propanol, Water
Output:
{"units": [
  {"tag": "HT-01", "type": "Heater", "role": "heat 1-propanol/water feed to flash temperature"},
  {"tag": "V-01",  "type": "Vessel", "role": "flash separation of 1-propanol and water"}
]}"""


@dataclass
class SemanticUnit:
    tag:  str
    type: str
    role: str

    def validate(self) -> list[str]:
        errors = []
        if self.type not in SUPPORTED_UNIT_TYPES:
            errors.append(f"Unsupported unit type '{self.type}' for tag '{self.tag}'")
        if not self.tag:
            errors.append("Unit has empty tag")
        return errors


@dataclass
class SemanticUnits:
    units:    list[SemanticUnit]
    raw_json: dict = field(default_factory=dict)

    def validate(self) -> list[str]:
        errors: list[str] = []
        tags: set[str] = set()
        for u in self.units:
            errors += u.validate()
            if u.tag in tags:
                errors.append(f"Duplicate unit tag '{u.tag}'")
            tags.add(u.tag)
        return errors


class UnitExtractor:
    def __init__(self, model: str = DEFAULT_MODEL):
        self._model = model

    def extract(
        self,
        description: str,
        compounds:   list[str],
        max_retries: int = 3,
    ) -> SemanticUnits:
        prompt = _build_prompt(description, compounds)
        last_error = ""
        for attempt in range(max_retries):
            use_minimal = (
                attempt == max_retries - 1
                and any(e in last_error for e in _EMPTY_ERRORS)
            )
            if use_minimal:
                current_prompt = (
                    f"Process: {description}\n"
                    f"Compounds: {', '.join(compounds)}\n"
                    "List the unit operations needed."
                )
                current_system = _MINIMAL_SYSTEM
            else:
                current_prompt = prompt + (
                    f"\n\nPrevious error: {last_error}" if last_error else "")
                current_system = _SYSTEM
            raw = chat(
                current_prompt,
                system=current_system,
                model=self._model,
                temperature=retry_temperature(attempt),
                max_tokens=4096,
            )
            try:
                data = _parse_json(raw)
                units = [
                    SemanticUnit(tag=u["tag"], type=u["type"], role=u.get("role", ""))
                    for u in data.get("units", [])
                ]
                result = SemanticUnits(units=units, raw_json=data)
                errors = result.validate()
                if not errors:
                    return result
                last_error = "; ".join(errors)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                last_error = str(e)

        raise RuntimeError(
            f"UnitExtractor failed after {max_retries} attempts. "
            f"Last error: {last_error}\nLast response: {raw[:300]}")


def _build_prompt(description: str, compounds: list[str]) -> str:
    return (
        f"Process description: {description}\n"
        f"Compounds present: {', '.join(compounds)}\n\n"
        "List the unit operations needed for this process in flow order."
    )


def _parse_json(text: str) -> dict:
    text = text.strip()
    if not text:
        raise ValueError("model returned an empty response")
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    text = text.strip()
    if not text:
        raise ValueError("response contained only markdown fences")
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    return json.loads(text)
