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
    "ConversionReactor",
]

_EMPTY_ERRORS = ("empty response", "line 1 column 1", "only markdown")

# Stripped fallback used on the final retry when the model returns nothing —
# avoids the full prompt triggering a long thinking block that exhausts max_tokens.
_MINIMAL_SYSTEM = (
    'Return ONLY: {"units": [{"tag": "HT-01", "type": "Heater", "role": "..."}]}'
    "\nTypes: Heater Cooler Vessel Mixer Splitter Pump Compressor Expander ConversionReactor"
)

# Compact system prompt for cases with >10 compounds — identical rules but no
# few-shot examples, saving ~300 tokens to avoid exceeding Ollama context.
_SYSTEM_COMPACT = """\
/no_think
Extract unit operations from a chemical process description.
Return ONLY a JSON object — no explanation, no markdown, no <think> blocks.

Schema:
{
  "units": [
    {
      "tag": "HT-01",
      "type": "<one of: Heater Cooler Vessel Mixer Splitter Pump Compressor Expander ConversionReactor>",
      "role": "<one-line purpose>"
    }
  ]
}

Rules:
- Tags: type abbreviation + 2-digit index (HT-01, V-01, MX-01, SP-01, PM-01, CP-01, EX-01, CL-01, RX-01)
- List units in process flow order (feed to product)
- Include only units explicitly needed — do not add units not implied by the description
- A Vessel performs flash separation (vapour + liquid); use it when phase separation is needed
- Include a Mixer only when combining two or more feed streams is described
- Include a Splitter only when splitting a stream into two fractions is described
- Use ConversionReactor for reactor, reformer, converter, shift reactor, methanator, or furnace used for reaction
- IGNORE preamble about property packages, thermodynamic models, or simulation settings\
"""


_SYSTEM = """\
/no_think
Extract unit operations from a chemical process description.
Return ONLY a JSON object — no explanation, no markdown, no <think> blocks.

Schema:
{
  "units": [
    {
      "tag": "HT-01",
      "type": "<one of: Heater Cooler Vessel Mixer Splitter Pump Compressor Expander ConversionReactor>",
      "role": "<one-line purpose, e.g. 'heat feed to flash temperature'>",
      "reaction": "<ONLY for ConversionReactor: 'reactants -> products', else omit>"
    }
  ]
}

Rules:
- Tags: use type abbreviation + 2-digit index (HT-01, V-01, MX-01, etc.)
  Abbreviations: Heater=HT, Cooler=CL, Vessel=V, Mixer=MX, Splitter=SP, Pump=PM, Compressor=CP, Expander=EX, ConversionReactor=RX
- List units in process flow order (feed to product)
- Include only units explicitly needed — do not add units not implied by the description
- A Vessel performs flash separation (vapour + liquid); use it when phase separation is needed
- Include a Mixer only when the description explicitly mentions combining or mixing two feed streams
- Include a Splitter only when the description explicitly mentions splitting a stream into two fractions
- Use ConversionReactor for any unit described as: reactor, reformer, converter, shift reactor, methanator,
  or a furnace/burner used for chemical reaction (not purely for heating)
- For a ConversionReactor ONLY, also add a "reaction" field giving the stoichiometry as
  "reactants -> products" using the EXACT compound names from the Compounds list, with integer
  coefficients where needed, e.g. "Methane + Water -> Carbon monoxide + 3 Hydrogen". Do NOT add a
  "reaction" field to any other unit type. A reactor with no reaction is useless — always fill it.
- IGNORE any preamble, commentary, or metadata about property packages, thermodynamic models,
  configuration validity, or simulation settings — extract only the physical unit operations

Examples:

Input: "Mix toluene and hydrogen, heat to reaction temperature, and dealkylate toluene to benzene and methane in a reactor"
Compounds: Hydrogen, Toluene, Benzene, Methane
Output:
{"units": [
  {"tag": "MX-01", "type": "Mixer",  "role": "mix toluene and hydrogen feeds"},
  {"tag": "HT-01", "type": "Heater", "role": "heat feed to reaction temperature"},
  {"tag": "RX-01", "type": "ConversionReactor", "role": "hydrodealkylate toluene to benzene",
   "reaction": "Toluene + Hydrogen -> Benzene + Methane"}
]}

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
]}

Input: "Preheat a methane and steam feed to 800°C, reform it over a nickel catalyst at 25 bar, then cool the syngas and separate the condensate."
Compounds: Methane, Water, Carbon Monoxide, Hydrogen, Carbon Dioxide
Output:
{"units": [
  {"tag": "HT-01", "type": "Heater",            "role": "preheat methane/steam feed to reforming temperature"},
  {"tag": "RX-01", "type": "ConversionReactor",  "role": "steam methane reforming at 800°C and 25 bar"},
  {"tag": "CL-01", "type": "Cooler",             "role": "cool syngas before condensate separation"},
  {"tag": "V-01",  "type": "Vessel",             "role": "separate condensed water from syngas"}
]}

Input: "Feed a butanol-water mixture to a decanter. The two liquid phases undergo liquid-liquid phase splitting."
Compounds: n-Butanol, Water
Output:
{"units": [
  {"tag": "V-01", "type": "Vessel", "role": "liquid-liquid decanter to split butanol-rich and water-rich phases"}
]}

Input: "Distil an ethanol-water feed in a distillation column. Overhead vapour is condensed and bottoms are reboiled."
Compounds: Ethanol, Water
Output:
{"units": [
  {"tag": "HT-01", "type": "Heater", "role": "column reboiler to vaporise bottoms"},
  {"tag": "CL-01", "type": "Cooler", "role": "column condenser to condense overhead vapour"},
  {"tag": "V-01",  "type": "Vessel", "role": "column separator for vapour-liquid equilibrium"}
]}

Input: "Rich TEG from the absorber column is pumped to a regenerator where it is heated to drive off absorbed water. Lean TEG is recycled back."
Compounds: Triethylene glycol, Water
Output:
{"units": [
  {"tag": "HT-01", "type": "Heater", "role": "absorber column reboiler"},
  {"tag": "CL-01", "type": "Cooler", "role": "absorber column condenser"},
  {"tag": "V-01",  "type": "Vessel", "role": "absorber column separator"},
  {"tag": "PM-01", "type": "Pump",   "role": "pump rich TEG to regenerator pressure"},
  {"tag": "HT-02", "type": "Heater", "role": "regenerator reboiler to drive off water from rich TEG"},
  {"tag": "V-02",  "type": "Vessel", "role": "regenerator separator for lean TEG and water vapour"}
]}"""


_TAG_ABBREV: dict[str, str] = {
    "Heater":            "HT",
    "Cooler":            "CL",
    "Vessel":            "V",
    "Mixer":             "MX",
    "Splitter":          "SP",
    "Pump":              "PM",
    "Compressor":        "CP",
    "Expander":          "EX",
    "ConversionReactor": "RX",
}

# Each tuple: (regex_pattern, [(unit_type, role)]).
# Applied in order — one match emits all listed unit types (e.g. "column" → Heater+Cooler).
_KW_UNITS: list[tuple[str, list[tuple[str, str]]]] = [
    (r"\breactor\b|\breformer\b|\bconverter\b|\bshift\s+reactor\b|\bmethanat\w*",
     [("ConversionReactor", "chemical reaction")]),
    (r"\bcolumn\b",
     [("Heater", "column reboiler"), ("Cooler", "column condenser")]),
    (r"\bdecanter\b",
     [("Vessel", "liquid-liquid decanter")]),
    (r"\bvessel\b|\bflash\s+drum\b",
     [("Vessel", "flash separation")]),
    (r"\bheater\b|\breboiler\b|\bfurnace\b",
     [("Heater", "heat duty")]),
    (r"\bcooler\b|\bcondenser\b|\bchiller\b",
     [("Cooler", "cooling duty")]),
    (r"\bmixer\b",
     [("Mixer", "stream mixing")]),
    (r"\bsplitter\b",
     [("Splitter", "stream splitting")]),
    (r"\bpump\b",
     [("Pump", "pressure increase")]),
    (r"\bcompressor\b",
     [("Compressor", "gas compression")]),
    (r"\bexpander\b|\bturbine\b",
     [("Expander", "work extraction")]),
]


@dataclass
class SemanticUnit:
    tag:  str
    type: str
    role: str
    # Stoichiometry for a ConversionReactor, "reactants -> products" with exact
    # compound names, e.g. "Methane + Water -> Carbon monoxide + 3 Hydrogen".
    # Empty for all other unit types.  Without it, to_dwsim emits reaction="" and
    # DWSIM performs no conversion (the reactor does nothing).
    reaction: str = ""

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
        tier:        str = "standard",
    ) -> SemanticUnits:
        prompt = _build_prompt(description, compounds)
        last_error = ""
        for attempt in range(max_retries):
            use_minimal = (
                attempt > 0
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
            n_words     = len(description.split())
            n_compounds = len(compounds)
            if tier == "validation":
                max_tokens = 16384
            elif n_compounds > 10:
                max_tokens = 12288
            elif n_words > 300 or n_compounds > 5:
                max_tokens = 8192
            else:
                max_tokens = 4096
            # For large compound lists, swap to the compact system prompt (no
            # examples) to save ~300 tokens and avoid Ollama context overflow.
            # Compounds are passed as a plain comma-separated list only.
            if not use_minimal and n_compounds > 10:
                current_system = _SYSTEM_COMPACT
                current_prompt = (
                    f"Process description: {description}\n"
                    f"Compounds: {', '.join(compounds)}\n\n"
                    "List ALL unit operations needed for this process in flow order."
                    + (f"\n\nPrevious error: {last_error}" if last_error else "")
                )
            import sys as _sys
            print(f"[UNIT_EXT] attempt={attempt} tier={tier} "
                  f"prompt={'minimal' if use_minimal else ('compact' if n_compounds > 10 else 'full')} "
                  f"max_tokens={max_tokens} words={n_words} compounds={n_compounds} "
                  f"last_error={last_error!r}",
                  flush=True, file=_sys.stderr)
            raw = chat(
                current_prompt,
                system=current_system,
                model=self._model,
                temperature=retry_temperature(attempt),
                max_tokens=max_tokens,
            )
            try:
                data = _parse_json(raw)
                units = [
                    SemanticUnit(tag=u["tag"], type=u["type"], role=u.get("role", ""),
                                 reaction=u.get("reaction", ""))
                    for u in data.get("units", [])
                ]
                result = SemanticUnits(units=units, raw_json=data)
                errors = result.validate()
                if not errors:
                    return result
                last_error = "; ".join(errors)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                last_error = str(e)

        _fallback = _keyword_fallback(description)
        if _fallback is not None:
            return _fallback
        raise RuntimeError(
            f"UnitExtractor failed after {max_retries} attempts. "
            f"Last error: {last_error}\nLast response: {raw[:300]}")


def _keyword_fallback(description: str) -> Optional[SemanticUnits]:
    """Deterministic last-resort extraction: scan description for equipment keywords."""
    import sys as _sys
    desc_lower = description.lower()
    units_out: list[SemanticUnit] = []
    counters: dict[str, int] = {}
    for pattern, type_roles in _KW_UNITS:
        n = len(re.findall(pattern, desc_lower))
        if n == 0:
            continue
        for _ in range(n):
            for utype, role in type_roles:
                abbrev = _TAG_ABBREV.get(utype, "U")
                counters[abbrev] = counters.get(abbrev, 0) + 1
                units_out.append(SemanticUnit(
                    tag=f"{abbrev}-{counters[abbrev]:02d}",
                    type=utype,
                    role=role,
                ))
    if not units_out:
        return None
    print(
        f"[UNIT_EXT] keyword fallback activated: "
        f"{[u.tag + '/' + u.type for u in units_out]}",
        flush=True, file=_sys.stderr)
    return SemanticUnits(units=units_out)


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
