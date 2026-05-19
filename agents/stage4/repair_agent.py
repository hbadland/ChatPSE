"""
Agent G — Repair Agent (two-layer architecture).

Layer 1 — DeterministicRepair (ir/repair.py):
  Always attempted first.  Zero LLM calls.  Handles:
    PARAM_INJECT, TOPOLOGY_FIX, UNIT_CONVERSION, DEFAULT_FILL,
    PORT_REPAIR, THERMO_SWITCH

Layer 2 — LLMRepair (this file):
  Only called for CONDITION_FIX after deterministic fixes are
  insufficient.  One focused LLM call per unit, per param.

HUMAN errors are logged and returned unchanged.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from ir.graph import FlowsheetGraph
from ir.repair import DeterministicRepair
from ir.types import RepairStrategy, SimError
from agents.llm import chat, DEFAULT_MODEL, retry_temperature
from rag.retriever import Retriever

_SYSTEM = """\
Fix a single unit operation parameter in a chemical process flowsheet.
Return ONLY a JSON object of {param_name: value} — no explanation, no markdown.
All temperatures in Kelvin. All pressures in Pascals."""

_det_repair = DeterministicRepair()


class RepairAgent:
    def __init__(self, model: str = DEFAULT_MODEL, retriever: Optional[Retriever] = None):
        self._model     = model
        self._retriever = retriever or Retriever()

    def repair(
        self,
        graph:          FlowsheetGraph,
        errors:         list[SimError],
        tried_packages: set[str] | None = None,
        description:    str = "",
    ) -> tuple[FlowsheetGraph, list[str]]:
        """
        Apply all repairs in priority order.
        Deterministic repairs run first; LLM is called only for CONDITION_FIX.
        Returns (patched_graph, change_log).
        """
        g              = graph.copy()
        changes:list[str] = []
        tried_packages  = tried_packages or set()

        for error in errors:
            if error.repair_strategy == RepairStrategy.HUMAN:
                changes.append(f"HUMAN: {error.target} — {error.evidence[:80]}")
                continue

            if error.repair_strategy == RepairStrategy.CONDITION_FIX:
                g, msgs = self._llm_condition_fix(g, error, description)
                changes.extend(msgs)
            else:
                try:
                    g, msgs = _det_repair.apply(
                        g, error, self._retriever, tried_packages)
                    changes.extend(msgs)
                    if error.repair_strategy == RepairStrategy.THERMO_SWITCH:
                        tried_packages.add(g.property_package)
                except ValueError as exc:
                    changes.append(f"REPAIR_ERROR: {exc}")

        return g, changes

    # ── LLM repair ────────────────────────────────────────────────────────────

    def _llm_condition_fix(
        self,
        graph:       FlowsheetGraph,
        error:       SimError,
        description: str,
    ) -> tuple[FlowsheetGraph, list[str]]:
        from ir.types import TargetKind

        if error.target.kind == TargetKind.STREAM:
            # Streams: try deterministic unit conversion first
            det_error = error.__class__(
                error_type      = error.error_type,
                target          = error.target,
                evidence        = error.evidence,
                repair_strategy = RepairStrategy.UNIT_CONVERSION,
                severity        = error.severity,
            )
            g, msgs = _det_repair.fix_unit_conversions(graph, det_error)
            if msgs:
                return g, msgs
            return graph, []

        node = graph.unit(error.target.tag)
        if node is None:
            return graph, [f"CONDITION_FIX: unit {error.target.tag} not found"]

        unit_context   = self._retriever.unit_context(node.unit_type)
        current_params = json.dumps(node.params)
        prompt = (
            f"Process: {description or 'not specified'}\n"
            f"Unit: {node.tag} ({node.unit_type})\n"
            f"Current params: {current_params}\n"
            f"Error: [{error.error_type.value}] {error.evidence}\n\n"
            f"Unit specification:\n{unit_context}\n\n"
            f"Return corrected params for {node.tag} as a JSON object."
        )
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
                    g      = graph.copy()
                    target = g.unit(node.tag)
                    old    = dict(target.params)
                    target.params.update(new_params)
                    changed = {k: (old.get(k), v)
                               for k, v in new_params.items()
                               if old.get(k) != v}
                    return g, [f"CONDITION_FIX: {node.tag} params updated: {changed}"]
            except (json.JSONDecodeError, TypeError) as e:
                last_error = str(e)

        return graph, [f"CONDITION_FIX: failed for {node.tag} — {last_error}"]


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    return json.loads(text)
