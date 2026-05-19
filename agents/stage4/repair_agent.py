"""
Agent G — Repair Agent.

Input : FlowsheetGraph + list[ClassifiedError] + Retriever
Output: (FlowsheetGraph, list[str]) — patched graph + change log

Repair strategy dispatch:
  PARAM_INJECT   — deterministic: inject BIPs from corpus
  TOPOLOGY_FIX   — deterministic: re-run normaliser
  THERMO_SWITCH  — deterministic: RAG selects next best package
  CONDITION_FIX  — LLM: one focused call, one unit, one param
  HUMAN          — not repaired; caller must escalate
"""
from __future__ import annotations

import json
import re
from typing import Optional

from ir.graph import FlowsheetGraph
from ir.normalise import normalise
from agents.llm import chat, DEFAULT_MODEL, retry_temperature
from agents.stage4.error_classifier import ClassifiedError
from rag.retriever import Retriever

_SYSTEM = """\
Fix a single unit operation parameter in a chemical process flowsheet.
Return ONLY a JSON object of {param_name: value} — no explanation, no markdown.
All temperatures in Kelvin. All pressures in Pascals."""


class RepairAgent:
    def __init__(self, model: str = DEFAULT_MODEL, retriever: Optional[Retriever] = None):
        self._model     = model
        self._retriever = retriever or Retriever()

    def repair(
        self,
        graph:       FlowsheetGraph,
        errors:      list[ClassifiedError],
        tried_packages: set[str] | None = None,
        description: str = "",
    ) -> tuple[FlowsheetGraph, list[str]]:
        """
        Apply all repairs in priority order. Returns (patched_graph, change_log).
        Terminal errors (HUMAN) are logged but not repaired.
        """
        g       = graph.copy()
        changes: list[str] = []
        tried_packages = tried_packages or set()

        for error in errors:
            if error.repair_strategy == "HUMAN":
                changes.append(f"HUMAN: {error.location} — {error.evidence[:80]}")
                continue

            if error.repair_strategy == "PARAM_INJECT":
                g, msg = self._param_inject(g)
                if msg:
                    changes.append(msg)

            elif error.repair_strategy == "TOPOLOGY_FIX":
                g = normalise(g)
                changes.append(f"TOPOLOGY_FIX: graph re-normalised")

            elif error.repair_strategy == "THERMO_SWITCH":
                g, msg = self._thermo_switch(g, tried_packages, description)
                if msg:
                    changes.append(msg)
                tried_packages.add(g.property_package)

            elif error.repair_strategy == "CONDITION_FIX":
                g, msg = self._condition_fix(g, error, description)
                if msg:
                    changes.append(msg)

        return g, changes

    # ── Strategy implementations ───────────────────────────────────────────────

    def _param_inject(self, graph: FlowsheetGraph) -> tuple[FlowsheetGraph, str]:
        pkg = graph.property_package
        if pkg not in ("NRTL", "UNIQUAC"):
            return graph, ""
        if graph.binary_parameters:
            return graph, ""
        bips, missing = self._retriever.query_bips(graph.compounds, pkg)
        if not missing:
            g = graph.copy()
            g.binary_parameters = bips
            return g, f"PARAM_INJECT: {pkg} BIPs injected for {graph.compounds}"
        return graph, f"PARAM_INJECT: BIPs missing for pairs {missing}"

    def _thermo_switch(
        self,
        graph:         FlowsheetGraph,
        tried:         set[str],
        description:   str,
    ) -> tuple[FlowsheetGraph, str]:
        candidates = self._retriever.select_package(
            graph.compounds, description, exclude=tried | {graph.property_package})
        if not candidates:
            return graph, "THERMO_SWITCH: all packages exhausted"
        g = graph.copy()
        old_pkg = g.property_package
        g.property_package  = candidates[0]
        g.binary_parameters = []  # clear stale BIPs
        # Re-inject BIPs if new package needs them
        if g.property_package in ("NRTL", "UNIQUAC"):
            bips, missing = self._retriever.query_bips(g.compounds, g.property_package)
            if not missing:
                g.binary_parameters = bips
        return g, f"THERMO_SWITCH: {old_pkg} → {g.property_package}"

    def _condition_fix(
        self,
        graph:       FlowsheetGraph,
        error:       ClassifiedError,
        description: str,
    ) -> tuple[FlowsheetGraph, str]:
        node = graph.unit(error.location)
        if node is None:
            # Location is a stream — adjust the stream value deterministically
            return self._fix_stream_condition(graph, error)

        unit_context = self._retriever.unit_context(node.unit_type)
        current_params = json.dumps(node.params)
        prompt = (
            f"Process: {description or 'not specified'}\n"
            f"Unit: {node.tag} ({node.unit_type})\n"
            f"Current params: {current_params}\n"
            f"Error: [{error.error_type}] {error.evidence}\n\n"
            f"Unit specification:\n{unit_context}\n\n"
            f"Return corrected params for {node.tag} as a JSON object."
        )
        raw        = ""
        last_error = ""
        for attempt in range(3):
            raw = chat(
                prompt + (f"\n\nPrevious error: {last_error}" if last_error else ""),
                system=_SYSTEM,
                model=self._model,
                temperature=retry_temperature(attempt),
                max_tokens=256,
            )
            try:
                new_params = _parse_json(raw)
                if new_params:
                    g = graph.copy()
                    target = g.unit(node.tag)
                    old = dict(target.params)
                    target.params.update(new_params)
                    changed = {k: (old.get(k), v)
                               for k, v in new_params.items() if old.get(k) != v}
                    return g, f"CONDITION_FIX: {node.tag} params updated: {changed}"
            except (json.JSONDecodeError, TypeError) as e:
                last_error = str(e)
        return graph, f"CONDITION_FIX: failed for {node.tag} — {last_error}"

    def _fix_stream_condition(
        self, graph: FlowsheetGraph, error: ClassifiedError
    ) -> tuple[FlowsheetGraph, str]:
        """Deterministic unit-conversion fix for stream UNPHYSICAL_T/P."""
        g = graph.copy()
        tag = error.location.replace("stream:", "").strip()
        stream = g.stream(tag)
        if stream is None:
            return g, ""
        if error.error_type == "UNPHYSICAL_VALUES" and stream.T is not None:
            if stream.T < 100:  # likely °C — convert to K
                old_t      = stream.T
                stream.T   = round(stream.T + 273.15, 2)
                return g, f"CONDITION_FIX: stream {tag} T {old_t}→{stream.T} K (°C→K)"
        if error.error_type == "UNPHYSICAL_VALUES" and stream.P is not None:
            if stream.P < 500:  # likely bar — convert to Pa
                old_p    = stream.P
                stream.P = round(stream.P * 1e5, 0)
                return g, f"CONDITION_FIX: stream {tag} P {old_p}→{stream.P} Pa (bar→Pa)"
        return g, ""


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    return json.loads(text)
