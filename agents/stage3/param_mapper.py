"""
Agent E — Param Mapper.

Input : FlowsheetGraph (units have no params yet) + Retriever
Output: FlowsheetGraph with unit params set (T_out, P_out, efficiency, dP, etc.)

Stage 1 (deterministic): apply defaults from unit_specs.json via Retriever.
Stage 2 (LLM): one call per unit that has required params — short focused prompt.

The LLM receives:
  - The description
  - The unit type + role (from metadata)
  - Feed stream conditions (T, P, composition)
  - A unit spec snippet from RAG (required params, notes)
  - The bubble-point estimate (if computable)

LLMs must output a flat JSON dict of param_name → value. Nothing else.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from ir.graph import FlowsheetGraph, NodeIR
from agents.llm import chat, DEFAULT_MODEL, retry_temperature
from rag.retriever import Retriever

# Units whose required params cannot be defaulted — must call LLM
_REQUIRES_LLM = {"Heater", "Cooler", "Pump", "Compressor", "Expander"}

# Units where defaults are sufficient — no LLM needed
_DEFAULTS_ONLY = {"Vessel", "Mixer", "Splitter"}

_SYSTEM = """\
Assign numerical parameters for one unit operation in a chemical process.
Return ONLY a JSON object of {param_name: value} — no explanation, no markdown.

All temperatures in Kelvin (K). All pressures in Pascals (Pa).
Examples: 25°C = 298.15 K | 80°C = 353.15 K | 1 atm = 101325 Pa | 5 bar = 500000 Pa"""


class ParamMapper:
    def __init__(self, model: str = DEFAULT_MODEL, retriever: Optional[Retriever] = None):
        self._model     = model
        self._retriever = retriever or Retriever()

    def assign(
        self,
        graph:       FlowsheetGraph,
        description: str = "",
        max_retries: int = 3,
    ) -> FlowsheetGraph:
        g = graph.copy()

        # Pre-compute feed conditions once (shared across all unit prompts)
        feed_T, feed_P, feed_comp = _feed_conditions(g)
        bubble_hint = _bubble_point_hint(g.compounds, feed_comp, feed_P)

        for node in g.units():
            params = self._assign_unit(
                node, g, description, feed_T, feed_P, bubble_hint, max_retries)
            node.params = params

        return g

    def _assign_unit(
        self,
        node:        NodeIR,
        graph:       FlowsheetGraph,
        description: str,
        feed_T:      Optional[float],
        feed_P:      Optional[float],
        bubble_hint: Optional[str],
        max_retries: int,
    ) -> dict:
        # Apply defaults unconditionally
        defaults = self._retriever.units.defaults(node.unit_type)
        params   = dict(defaults)

        if node.unit_type in _DEFAULTS_ONLY:
            return params

        # LLM fills required params
        llm_params = self._llm_assign(
            node, description, feed_T, feed_P, bubble_hint, max_retries)
        params.update(llm_params)
        return params

    def _llm_assign(
        self,
        node:        NodeIR,
        description: str,
        feed_T:      Optional[float],
        feed_P:      Optional[float],
        bubble_hint: Optional[str],
        max_retries: int,
    ) -> dict:
        unit_context = self._retriever.unit_context(node.unit_type)
        role         = node.metadata.get("role", "")

        prompt_lines = [
            f"Process: {description or 'not specified'}",
            f"Unit: {node.tag} ({node.unit_type}) — {role}",
            f"Feed conditions: T={feed_T} K, P={feed_P} Pa",
        ]
        if bubble_hint:
            prompt_lines.append(f"Bubble point estimate: {bubble_hint}")
        prompt_lines += [
            "",
            "Unit specification:",
            unit_context,
            "",
            f"Return a JSON object with the required parameters for {node.tag}.",
        ]
        prompt     = "\n".join(prompt_lines)
        last_error = ""
        raw        = ""
        for attempt in range(max_retries):
            raw = chat(
                prompt + (f"\n\nPrevious error: {last_error}" if last_error else ""),
                system=_SYSTEM,
                model=self._model,
                temperature=retry_temperature(attempt),
                max_tokens=256,
            )
            try:
                data = _parse_json(raw)
                errors = _validate_params(node.unit_type, data)
                if not errors:
                    return data
                last_error = "; ".join(errors)
            except (json.JSONDecodeError, TypeError) as e:
                last_error = str(e)

        # Return whatever was last parsed — Stage 4 repair loop handles bad values
        try:
            return _parse_json(raw)
        except Exception:
            return {}


# ── Helpers ────────────────────────────────────────────────────────────────────

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


def _bubble_point_hint(
    compounds: list[str],
    composition: dict,
    pressure_pa: Optional[float],
) -> Optional[str]:
    """Return a short bubble-point advisory string, or None if not computable."""
    import math
    _NBP_K: dict[str, float] = {
        "Methanol":337.85,"Ethanol":351.44,"n-Propanol":370.35,"1-Propanol":370.35,
        "Isopropanol":355.39,"2-Propanol":355.39,"n-Butanol":390.81,"1-Butanol":390.81,
        "Water":373.15,"Acetone":329.15,"Benzene":353.25,"Toluene":383.78,
        "n-Hexane":341.88,"n-Heptane":371.58,"Methane":111.66,"Ethane":184.55,
        "Propane":231.11,"n-Butane":272.65,"Carbon Dioxide":194.65,
        "Chloroform":334.35,"Dichloromethane":312.95,"Ethyl Acetate":350.26,
        "Acetonitrile":354.75,
    }
    if not composition or not compounds:
        return None
    if any(c not in _NBP_K for c in compounds):
        return None
    p = pressure_pa or 101_325.0
    if not (20_000 < p < 600_000):
        return None
    total = sum(composition.values())
    if total <= 0:
        return None
    t_bub = sum((composition.get(c, 0.0) / total) * _NBP_K[c] for c in compounds)
    if abs(p - 101_325.0) > 5_000:
        lnP  = math.log(p / 101_325.0)
        dHvap = 88.0 * t_bub
        t_bub = t_bub / (1.0 - 8.314 * t_bub * lnP / dHvap)
    t_lo = round(t_bub + 8,  0)
    t_hi = round(t_bub + 35, 0)
    return (
        f"Mixture bubble point ≈ {round(t_bub,1)} K at {p:.0f} Pa. "
        f"For flash separation: set T_out to {t_lo}–{t_hi} K."
    )


def _validate_params(unit_type: str, params: dict) -> list[str]:
    errors: list[str] = []
    if unit_type in ("Heater", "Cooler"):
        t = params.get("T_out")
        if t is None:
            errors.append("T_out is required")
        elif not (50 < float(t) < 2000):
            errors.append(f"T_out={t} K out of range — must be in Kelvin")
    if unit_type in ("Pump", "Compressor", "Expander"):
        p = params.get("P_out")
        if p is None:
            errors.append("P_out is required")
        elif not (100 < float(p) < 1e8):
            errors.append(f"P_out={p} Pa out of range — must be in Pascals")
    return errors


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    return json.loads(text)
