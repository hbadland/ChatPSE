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

from agents.llm import chat, DEFAULT_MODEL, retry_temperature, retry_seed

_EMPTY_ERRORS = ("empty response", "line 1 column 1", "only markdown")

_MINIMAL_SYSTEM = (
    'Return ONLY: {"streams": [{"tag": "FEED", "src": null, "dst": "HT-01", '
    '"is_feed": true, "T": 298.15, "P": 101325.0, "flow": 1.0, '
    '"composition": {"Ethanol": 0.5, "Water": 0.5}}]}'
    "\nFeed streams need T[K], P[Pa], flow[mol/s], composition. "
    "Other streams need only tag/src/dst."
)


_SYSTEM = """\
/no_think
Define the process streams connecting the unit operations.
Return ONLY a JSON object — no explanation, no markdown, no <think> blocks.

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

Optional recycle fields — include ONLY for streams explicitly described as recycled:
  "is_recycle": true,          (default: false — omit for all normal streams)
  "recycle_target": "MX-01"   (must be an exact unit tag from the Units list)

Rules:
- Every unit must have at least one inlet and one outlet stream
- Feed streams (src=null) MUST include T [K], P [Pa], flow [mol/s], composition (mole fractions summing to 1.0)
- Intermediate and product streams carry tag, src, dst only — do NOT guess T/P/composition
- Units never connect directly — every unit-to-unit link needs a stream between them
- A Vessel has TWO outlet streams: vapour (name containing VAP/VAPOR/GAS) and liquid (name containing LIQ/LIQUID)
- All temperatures in Kelvin, all pressures in Pascals — never use atm, bar, or kPa strings
- Qualitative pressure defaults (use these exact Pa values when pressure is described in words):
    "sub-atmospheric" / "below atmospheric" / "below ambient"  → P = 50000
    "near-atmospheric" / "approximately atmospheric"            → P = 101325
    "atmospheric"                                               → P = 101325
    "vacuum"                                                    → P = 10000
    "moderate pressure" / "medium pressure"                     → P = 300000
    "high pressure" / "elevated pressure"                       → P = 1000000
- Reasonable defaults: T=298.15 K, P=101325 Pa unless the description states otherwise
- Recycle streams: set "is_recycle": true and "recycle_target": "<unit_tag>" ONLY when the
  description uses one of these exact phrases: "recycled back to", "returned to",
  "fed back to", "recirculated to", "recycled to". recycle_target MUST be an exact unit
  tag from the Units list. DEFAULT for all streams: is_recycle=false — when in doubt,
  do NOT tag as recycle.

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
]}

Process: "Feed an acetone-water-chloroform mixture to an extractive distillation column operating at sub-atmospheric pressure. Overhead is condensed; bottoms leave as liquid product."
Units: HT-01 (Heater — column reboiler), CL-01 (Cooler — column condenser), V-01 (Vessel — column separator)
Output:
{"streams": [
  {"tag": "FEED", "src": null,    "dst": "HT-01", "is_feed": true,
   "T": 298.15, "P": 50000.0, "flow": 1.0,
   "composition": {"Acetone": 0.4, "Water": 0.3, "Chloroform": 0.3}},
  {"tag": "HOT",  "src": "HT-01", "dst": "CL-01", "is_feed": false},
  {"tag": "COOL", "src": "CL-01", "dst": "V-01",  "is_feed": false},
  {"tag": "VAP",  "src": "V-01",  "dst": null,    "is_feed": false},
  {"tag": "LIQ",  "src": "V-01",  "dst": null,    "is_feed": false}
]}

Process: "Pump a toluene feed to high pressure, heat it in a heat exchanger, then flash in a knockout drum"
Units: PM-01 (Pump), HT-01 (Heater), V-01 (Vessel)
Output:
{"streams": [
  {"tag": "FEED",   "src": null,    "dst": "PM-01", "is_feed": true,
   "T": 298.15, "P": 101325.0, "flow": 1.0, "composition": {"Toluene": 1.0}},
  {"tag": "PUMP",   "src": "PM-01", "dst": "HT-01", "is_feed": false},
  {"tag": "HOT",    "src": "HT-01", "dst": "V-01",  "is_feed": false},
  {"tag": "VAP",    "src": "V-01",  "dst": null,    "is_feed": false},
  {"tag": "LIQ",    "src": "V-01",  "dst": null,    "is_feed": false}
]}"""


# Ordered most-specific first so the first match wins.
_PRESSURE_QUALIFIERS: list[tuple[str, float]] = [
    (r"\bsub[- ]?atmospheric\b|\bbelow\s+atmospheric\b|\bbelow\s+ambient\b", 50_000.0),
    (r"\bnear[- ]?atmospheric\b|\bapproximately\s+atmospheric\b",            101_325.0),
    (r"\batmospheric\b",                                                      101_325.0),
    (r"\bvacuum\b",                                                            10_000.0),
    (r"\bhigh[- ]?pressure\b|\belevated\s+pressure\b",                     1_000_000.0),
    (r"\bmoderate[- ]?pressure\b|\bmedium[- ]?pressure\b",                   300_000.0),
    (r"\blow[- ]?pressure\b",                                                  50_000.0),
]


@dataclass
class SemanticStream:
    tag:            str
    src:            Optional[str]
    dst:            Optional[str]
    is_feed:        bool
    T:              Optional[float] = None
    P:              Optional[float] = None
    flow:           Optional[float] = None
    composition:    dict            = field(default_factory=dict)
    is_recycle:     bool            = False
    recycle_target: Optional[str]  = None

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
        if self.is_recycle:
            if not self.recycle_target:
                errors.append(
                    f"Stream '{self.tag}' is_recycle=true but recycle_target is null")
            elif self.recycle_target not in unit_tags:
                errors.append(
                    f"Stream '{self.tag}' recycle_target='{self.recycle_target}' "
                    "not in unit list")
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
            use_minimal = (
                attempt == max_retries - 1
                and any(e in last_error for e in _EMPTY_ERRORS)
            )
            if use_minimal:
                unit_summary = ", ".join(
                    f"{t}({unit_roles.get(t, '?')})" for t in unit_tags)
                current_prompt = (
                    f"Process: {description}\n"
                    f"Units: {unit_summary}\n"
                    "Define the connecting streams."
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
                seed=retry_seed(attempt, description),
                max_tokens=8192,
            )
            try:
                data    = _parse_json(raw)
                streams = [_parse_stream(s) for s in data.get("streams", [])]
                streams = _reconcile_unit_refs(streams, unit_tag_set)
                _resolve_qualitative_pressure(description, streams)
                result  = SemanticTopology(streams=streams, raw_json=data)
                errors  = result.validate(unit_tag_set)
                if not errors:
                    return result
                last_error = "; ".join(errors)
            except (ValueError, KeyError, TypeError) as e:
                # ValueError covers _parse_json's empty-response / markdown-only
                # raises AND json.JSONDecodeError (a ValueError subclass).  These
                # MUST be caught here so the loop retries and, on the final
                # attempt, switches to the minimal-prompt fallback for empty
                # responses (the `use_minimal` branch keys off last_error).
                # Catching only JSONDecodeError previously let a bare empty
                # ValueError escape on attempt 0, bypassing all retries/fallback.
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
    exact_tags = ", ".join(unit_tags)
    parts = [
        f"Process description: {description}",
        f"Compounds: {', '.join(compounds)}",
        (
            f"Units (in flow order):\n{unit_lines}\n"
            f"  IMPORTANT: src and dst must be EXACTLY one of these tags "
            f"(or null): {exact_tags}"
        ),
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


def _best_match(ref: str, unit_tags: set[str]) -> Optional[str]:
    """
    Try to snap a hallucinated unit reference to the closest known tag.

    Strategy 1 — normalise: strip hyphens/underscores and compare case-insensitively.
      "HT01" → "ht01" matches "HT-01" → "ht01"
    Strategy 2 — numeric index: if exactly one known tag shares the same numeric
      suffix, return it.  "HEATER-01" → "01" → only "HT-01" has "01" → snap.
    """
    def _norm(s: str) -> str:
        return re.sub(r'[-_\s]', '', s).lower()

    ref_norm = _norm(ref)
    for tag in unit_tags:
        if _norm(tag) == ref_norm:
            return tag

    m = re.search(r'\d+', ref)
    if m:
        idx = m.group()
        candidates = [
            t for t in unit_tags
            if (nm := re.search(r'\d+', t)) and nm.group() == idx
        ]
        if len(candidates) == 1:
            return candidates[0]

    return None


def _reconcile_unit_refs(
    streams:   list["SemanticStream"],
    unit_tags: set[str],
) -> list["SemanticStream"]:
    """
    Silently fix src/dst references that don't match the known unit tag list.
    Leaves unresolvable references untouched so validation can still report them.
    """
    for stream in streams:
        if stream.src and stream.src not in unit_tags:
            fixed = _best_match(stream.src, unit_tags)
            if fixed:
                stream.src = fixed
        if stream.dst and stream.dst not in unit_tags:
            fixed = _best_match(stream.dst, unit_tags)
            if fixed:
                stream.dst = fixed
    return streams


def _parse_stream(s: dict) -> SemanticStream:
    return SemanticStream(
        tag            = s["tag"],
        src            = s.get("src"),
        dst            = s.get("dst"),
        is_feed        = bool(s.get("is_feed", False)),
        T              = s.get("T"),
        P              = s.get("P"),
        flow           = s.get("flow"),
        composition    = s.get("composition", {}),
        is_recycle     = bool(s.get("is_recycle", False)),
        recycle_target = s.get("recycle_target"),
    )


def _resolve_qualitative_pressure(description: str,
                                   streams: list["SemanticStream"]) -> None:
    """Set P on feed streams whose P is None when the description uses a qualitative term."""
    desc_lower = description.lower()
    for pattern, pa_value in _PRESSURE_QUALIFIERS:
        if re.search(pattern, desc_lower):
            import sys as _sys
            for s in streams:
                if s.is_feed and s.P is None:
                    s.P = pa_value
                    print(
                        f"[STREAM_EXT] qualitative pressure '{pattern}' → "
                        f"stream '{s.tag}' P={pa_value} Pa",
                        flush=True, file=_sys.stderr)
            return  # first (most specific) match only


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
