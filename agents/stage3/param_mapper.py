"""
Hybrid ParamMapper: deterministic-first with minimal LLM fallback.

Assignment pipeline per unit (in priority order):
  1. DescriptionParser   — regex extraction of explicit T/P values from description text
  2. PhysicalEstimator   — bubble-point estimates, process heuristics, monotonic defaults
  3. LLM fallback        — ONE focused call per unknown param, only when 1+2 fail

The LLM is never responsible for primary numerical reasoning.
GlobalConsistencyPass (ir/consistency.py) runs AFTER this module and enforces
cross-unit constraints that per-unit assignment cannot see.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from ir.graph import FlowsheetGraph, NodeIR
from ir.thermo_estimation import bubble_point_K
from agents.llm import chat, DEFAULT_MODEL, retry_temperature
from rag.retriever import Retriever

# Units whose required params cannot be defaulted without some input
_REQUIRES_ASSIGNMENT = {"Heater", "Cooler", "Pump", "Compressor", "Expander"}
# Units that are fully handled by defaults + normaliser
_DEFAULTS_ONLY       = {"Vessel", "Mixer", "Splitter"}

# ── LLM system prompt (minimal — only invoked when deterministic fails) ────────

_LLM_SYSTEM = """\
Assign ONE missing numerical parameter for a chemical process unit.
Return ONLY a JSON object: {"param": "<name>", "value": <number>}
No explanation, no markdown. Temperatures in Kelvin (K). Pressures in Pascals (Pa).

━━━ EXAMPLE 1 — Heater ━━━
Unit: HT-01 (Heater) — heat feed to flash temperature
Feed: T=298.15 K, P=101325 Pa
Missing: T_out [K]  Constraint: must be > 298.15 K
Bubble point estimate: 354 K → flash target 362–374 K
{"param": "T_out", "value": 368.15}

━━━ EXAMPLE 2 — Compressor ━━━
Unit: CP-01 (Compressor) — compress methane to high pressure
Feed: T=298.15 K, P=101325 Pa
Missing: P_out [Pa]  Constraint: must be > 101325 Pa
Description says: "compress to 5 bar"
{"param": "P_out", "value": 500000.0}

━━━ EXAMPLE 3 — Cooler ━━━
Unit: CL-01 (Cooler) — cool hot gas before expansion
Feed: T=423.15 K, P=500000 Pa
Missing: T_out [K]  Constraint: must be < 423.15 K
{"param": "T_out", "value": 323.15}

━━━ EXAMPLE 4 — Pump ━━━
Unit: PM-01 (Pump) — raise liquid acetone pressure
Feed: T=298.15 K, P=101325 Pa
Missing: P_out [Pa]  Constraint: must be > 101325 Pa
{"param": "P_out", "value": 1013250.0}"""


class ParamMapper:
    def __init__(self, model: str = DEFAULT_MODEL,
                 retriever: Optional[Retriever] = None):
        self._model     = model
        self._retriever = retriever or Retriever()

    def assign(
        self,
        graph:       FlowsheetGraph,
        description: str = "",
        max_retries: int = 2,
    ) -> FlowsheetGraph:
        g = graph.copy()

        feed_T, feed_P, _   = _feed_conditions(g)
        desc_temps           = _extract_temperatures(description)
        desc_pressures       = _extract_pressures(description)
        bp                   = bubble_point_K(g.compounds, feed_P or 101_325.0)
        downstream_map       = _downstream_unit_types(g)

        for node in g.units():
            feeds_vessel = "Vessel" in downstream_map.get(node.tag, set())
            feeds_pump   = "Pump"   in downstream_map.get(node.tag, set())
            params = self._assign_unit(
                node, description, feed_T, feed_P,
                desc_temps, desc_pressures, bp, max_retries,
                feeds_vessel=feeds_vessel, feeds_pump=feeds_pump)
            node.params = params

        return g

    # ── Per-unit assignment ────────────────────────────────────────────────────

    def _assign_unit(
        self,
        node:           NodeIR,
        description:    str,
        feed_T:         Optional[float],
        feed_P:         Optional[float],
        desc_temps:     list[float],
        desc_pressures: list[float],
        bp:             Optional[float],
        max_retries:    int,
        feeds_vessel:   bool = False,
        feeds_pump:     bool = False,
    ) -> dict:
        defaults = self._retriever.units.defaults(node.unit_type)
        params   = dict(defaults)

        if node.unit_type in _DEFAULTS_ONLY:
            # Preserve existing params (e.g. split_fractions on auto-inserted Splitter).
            # node.params takes priority over retriever defaults.
            params.update(node.params)
            return params

        # ── Step 1: Description parser ─────────────────────────────────────────
        parsed = _parse_params_from_description(
            node.unit_type, description, desc_temps, desc_pressures, feed_T, feed_P)
        params.update(parsed)

        # ── Step 2: Physical estimator (constraint-aware) ──────────────────────
        estimated = _estimate_params(
            node, params, feed_T, feed_P, bp,
            feeds_vessel=feeds_vessel, feeds_pump=feeds_pump)
        params.update(estimated)

        # ── Step 3: LLM fallback (only if required param is still missing) ─────
        required = _required_params(node.unit_type)
        still_missing = [r for r in required if r not in params]
        if still_missing:
            for param_name in still_missing:
                llm_val = self._llm_fallback(
                    node, param_name, description, feed_T, feed_P, bp,
                    params, max_retries)
                if llm_val is not None:
                    params[param_name] = llm_val

        return params

    # ── LLM fallback (single-param, highly constrained) ───────────────────────

    def _llm_fallback(
        self,
        node:        NodeIR,
        param_name:  str,
        description: str,
        feed_T:      Optional[float],
        feed_P:      Optional[float],
        bp:          Optional[float],
        current:     dict,
        max_retries: int,
    ) -> Optional[float]:
        param_unit, constraint = _param_constraint(node.unit_type, param_name,
                                                    feed_T, feed_P)
        role   = node.metadata.get("role", "")
        prompt = (
            f"Unit: {node.tag} ({node.unit_type}) — {role}\n"
            f"Feed: T={feed_T} K, P={feed_P} Pa\n"
            f"Missing: {param_name} [{param_unit}]  Constraint: {constraint}\n"
        )
        if bp is not None:
            prompt += f"Bubble point estimate: {bp} K\n"
        # Inject relevant sentence from description (keep prompt short)
        relevant = _relevant_sentence(description, node.unit_type)
        if relevant:
            prompt += f"Description context: {relevant}\n"
        prompt += f'\nReturn: {{"param": "{param_name}", "value": <number>}}'

        last_error = ""
        for attempt in range(max_retries):
            raw = chat(
                prompt + (f"\n\nPrevious error: {last_error}" if last_error else ""),
                system=_LLM_SYSTEM,
                model=self._model,
                temperature=retry_temperature(attempt),
                max_tokens=64,
            )
            try:
                data  = _parse_json(raw)
                value = data.get("value")
                if value is None:
                    raise ValueError("missing 'value' key")
                value = float(value)
                # Clamp to valid range before validating
                value = _clamp_param(node.unit_type, param_name, value, feed_T, feed_P)
                if value is None:
                    last_error = "value out of hard physical range"
                    continue
                errs  = _validate_single_param(node.unit_type, param_name, value)
                if not errs:
                    return value
                last_error = "; ".join(errs)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                last_error = str(e)
        return None


# ── Step 1: Description parser ─────────────────────────────────────────────────

def _extract_temperatures(text: str) -> list[float]:
    """Return all temperatures from text, converted to Kelvin, sorted ascending."""
    temps: list[float] = []

    # Celsius
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*°C', text, re.IGNORECASE):
        temps.append(float(m.group(1)) + 273.15)
    # Fahrenheit
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*°F', text, re.IGNORECASE):
        temps.append((float(m.group(1)) - 32.0) * 5.0 / 9.0 + 273.15)
    # Kelvin (must be followed by space or end, not "Pa"/"bar"/etc.)
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*K\b(?!\s*Pa|\s*bar|\s*atm)', text):
        val = float(m.group(1))
        if 100.0 < val < 2000.0:
            temps.append(val)

    return sorted(set(round(t, 2) for t in temps))


def _extract_pressures(text: str) -> list[float]:
    """Return all pressures from text, converted to Pascals, sorted ascending."""
    pressures: list[float] = []

    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*bar\b', text, re.IGNORECASE):
        pressures.append(float(m.group(1)) * 1e5)
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*atm\b', text, re.IGNORECASE):
        pressures.append(float(m.group(1)) * 101_325.0)
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*MPa\b', text, re.IGNORECASE):
        pressures.append(float(m.group(1)) * 1e6)
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*kPa\b', text, re.IGNORECASE):
        pressures.append(float(m.group(1)) * 1e3)
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*psi(?:a|g)?\b', text, re.IGNORECASE):
        pressures.append(float(m.group(1)) * 6_894.76)

    # "X bar abs" / "X bara"
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*bara\b', text, re.IGNORECASE):
        pressures.append(float(m.group(1)) * 1e5)

    return sorted(set(round(p, 0) for p in pressures))


def _parse_params_from_description(
    unit_type:      str,
    description:    str,
    desc_temps:     list[float],
    desc_pressures: list[float],
    feed_T:         Optional[float],
    feed_P:         Optional[float],
) -> dict:
    """
    Extract unit parameters directly from the description using extracted values.

    Logic per unit type:
      Heater  : T_out = highest extracted T above feed_T
      Cooler  : T_out = lowest extracted T below feed_T
      Pump/CP : P_out = highest extracted P above feed_P (if any)
      Expander: P_out = lowest extracted P below feed_P (if any)
    """
    params: dict = {}

    if unit_type in ("Heater", "Cooler"):
        if unit_type == "Heater":
            candidates = [t for t in desc_temps
                          if feed_T is None or t > feed_T + 1.0]
            if candidates:
                params["T_out"] = round(min(candidates), 2)
        else:
            candidates = [t for t in desc_temps
                          if feed_T is None or t < feed_T - 1.0]
            if candidates:
                params["T_out"] = round(max(candidates), 2)

    elif unit_type in ("Pump", "Compressor"):
        candidates = [p for p in desc_pressures
                      if feed_P is None or p > feed_P * 1.05]
        if candidates:
            params["P_out"] = round(min(candidates), 0)

    elif unit_type == "Expander":
        candidates = [p for p in desc_pressures
                      if feed_P is None or p < feed_P * 0.95]
        if candidates:
            params["P_out"] = round(max(candidates), 0)

    return params


# ── Step 2: Physical estimator (constraint-aware) ────────────────────────────

def _estimate_params(
    node:         NodeIR,
    params:       dict,
    feed_T:       Optional[float],
    feed_P:       Optional[float],
    bp:           Optional[float],
    feeds_vessel: bool = False,
    feeds_pump:   bool = False,
) -> dict:
    """
    Fill missing required params with physics-grounded estimates.
    Constraint-aware: uses a larger bubble-point margin when the unit
    directly feeds a downstream Vessel (requires two-phase inlet).
    Does not overwrite values already set by the description parser.
    """
    est: dict = {}

    # Pressure-changers must always have efficiency; the retriever default may be
    # absent if the spec file is incomplete.
    if node.unit_type in ("Pump", "Compressor", "Expander") and "efficiency" not in params:
        est["efficiency"] = 0.75

    if node.unit_type == "Heater" and "T_out" not in params:
        # Use a larger margin when feeding a flash vessel to reduce
        # the likelihood of GlobalConsistencyPass having to correct it.
        margin = 25.0 if feeds_vessel else 15.0
        if bp is not None:
            est["T_out"] = round(bp + margin, 2)
        elif feed_T is not None:
            est["T_out"] = round(feed_T + (margin + 25.0), 2)
        else:
            est["T_out"] = 373.15

    elif node.unit_type == "Cooler" and "T_out" not in params:
        if feeds_pump and bp is not None:
            # Must deliver liquid to the pump
            est["T_out"] = max(round(bp - 20.0, 2), 273.15)
        elif feed_T is not None:
            est["T_out"] = max(round(feed_T - 30.0, 2), 273.15)
        else:
            est["T_out"] = 298.15

    elif node.unit_type in ("Pump", "Compressor") and "P_out" not in params:
        base = feed_P or 101_325.0
        est["P_out"] = round(base * 5.0, 0)

    elif node.unit_type == "Expander" and "P_out" not in params:
        base = feed_P or 506_625.0
        est["P_out"] = max(round(base / 3.0, 0), 101_325.0)

    return est


# ── Helpers ────────────────────────────────────────────────────────────────────

def _downstream_unit_types(graph: FlowsheetGraph) -> dict[str, set[str]]:
    """Map each unit tag → set of unit types it directly feeds."""
    result: dict[str, set[str]] = {}
    for node in graph.units():
        types: set[str] = set()
        for s in graph.outlet_streams(node.tag):
            dst_tag = graph.stream_dest(s.tag)
            if dst_tag:
                dst = graph.unit(dst_tag)
                if dst:
                    types.add(dst.unit_type)
        result[node.tag] = types
    return result


def _clamp_param(
    unit_type:  str,
    param:      str,
    value:      float,
    feed_T:     Optional[float],
    feed_P:     Optional[float],
) -> Optional[float]:
    """
    Soft-clamp an LLM-generated value. Returns None only when completely
    outside the hard physical range (likely wrong units). Unit-specific
    constraint violations (heater T_out < feed_T) are corrected, not rejected.
    """
    if param == "T_out":
        if not (50.0 < value < 2000.0):
            return None
        if unit_type == "Heater" and feed_T is not None and value <= feed_T:
            return round(feed_T + 10.0, 2)
        if unit_type == "Cooler" and feed_T is not None and value >= feed_T:
            return round(feed_T - 10.0, 2)
        return value
    if param == "P_out":
        if not (100.0 < value < 1e8):
            return None
        if unit_type in ("Pump", "Compressor") and feed_P is not None and value <= feed_P:
            return round(feed_P * 2.0, 0)
        if unit_type == "Expander" and feed_P is not None and value >= feed_P:
            return round(feed_P / 2.0, 0)
        return value
    if param == "efficiency":
        if 0 < value <= 1.0:
            return value
        if 1.0 < value <= 100.0:
            return round(value / 100.0, 4)
        return 0.75
    return value


def _feed_conditions(
    graph: FlowsheetGraph,
) -> tuple[Optional[float], Optional[float], dict]:
    for stream in graph.streams():
        if stream.metadata.get("is_feed") and stream.T is not None:
            return stream.T, stream.P, stream.composition
    for stream in graph.streams():
        if stream.T is not None:
            return stream.T, stream.P, stream.composition
    return None, None, {}


def _required_params(unit_type: str) -> list[str]:
    _MAP = {
        "Heater":     ["T_out"],
        "Cooler":     ["T_out"],
        "Pump":       ["P_out"],
        "Compressor": ["P_out"],
        "Expander":   ["P_out"],
    }
    return _MAP.get(unit_type, [])


def _param_constraint(
    unit_type:  str,
    param_name: str,
    feed_T:     Optional[float],
    feed_P:     Optional[float],
) -> tuple[str, str]:
    """Return (unit_label, constraint_description) for the LLM prompt."""
    if param_name == "T_out":
        unit_label = "K"
        if unit_type == "Heater" and feed_T is not None:
            return unit_label, f"must be > {feed_T:.2f} K (feed temperature)"
        if unit_type == "Cooler" and feed_T is not None:
            return unit_label, f"must be < {feed_T:.2f} K (feed temperature)"
        return unit_label, "must be in range 100–1500 K"
    if param_name == "P_out":
        unit_label = "Pa"
        if unit_type in ("Pump", "Compressor") and feed_P is not None:
            return unit_label, f"must be > {feed_P:.0f} Pa (feed pressure)"
        if unit_type == "Expander" and feed_P is not None:
            return unit_label, f"must be < {feed_P:.0f} Pa (feed pressure)"
        return unit_label, "must be in range 1000–1e8 Pa"
    return "?", "must be physically reasonable"


def _validate_single_param(unit_type: str, param: str, value: float) -> list[str]:
    errors: list[str] = []
    if param == "T_out" and not (50.0 < value < 2000.0):
        errors.append(f"T_out={value} K out of range (50–2000 K)")
    if param == "P_out" and not (100.0 < value < 1e8):
        errors.append(f"P_out={value} Pa out of range (100–1e8 Pa)")
    return errors


def _relevant_sentence(description: str, unit_type: str) -> str:
    """Extract the most relevant sentence from description for this unit type."""
    keywords = {
        "Heater":     ["heat", "warm", "raise temperature", "preheat"],
        "Cooler":     ["cool", "chill", "condense", "reduce temperature"],
        "Pump":       ["pump", "pressurize", "raise pressure"],
        "Compressor": ["compress", "pressurize", "compress"],
        "Expander":   ["expand", "turbine", "let down", "reduce pressure"],
    }
    words = keywords.get(unit_type, [])
    sentences = re.split(r'[.;,]', description)
    for sentence in sentences:
        if any(w in sentence.lower() for w in words):
            return sentence.strip()[:120]
    return ""


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$",       "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group() if m else text)
