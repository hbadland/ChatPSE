"""
Agent D — Thermo Mapper.

Input : FlowsheetGraph + Retriever
Output: FlowsheetGraph with property_package and binary_parameters set

Stage 1 (deterministic): ThermoRetriever selects the best package via hard rules.
Stage 2 (LLM, only when Stage 1 is ambiguous): short targeted prompt confirms or
  overrides. The LLM receives a pre-computed context snippet — NOT the full rules.

LLMs must not invent packages. The allowed list is injected explicitly.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from ir.graph import FlowsheetGraph
from agents.llm import chat, DEFAULT_MODEL, retry_temperature
from rag.retriever import Retriever

_ALLOWED_PACKAGES = [
    "Raoult's Law", "NRTL", "UNIQUAC",
    "Peng-Robinson", "Soave-Redlich-Kwong", "Lee-Kesler-Plöcker",
]

_SYSTEM = """\
Select the thermodynamic property package for this chemical process.
Return ONLY a JSON object — no explanation, no markdown.

Schema: {"package": "<package name>", "reason": "<one sentence>"}

Allowed package names (use exactly):
  Raoult's Law | NRTL | UNIQUAC | Peng-Robinson | Soave-Redlich-Kwong | Lee-Kesler-Plöcker"""


class ThermoMapper:
    def __init__(self, model: str = DEFAULT_MODEL, retriever: Optional[Retriever] = None):
        self._model     = model
        self._retriever = retriever or Retriever()

    def assign(
        self,
        graph:       FlowsheetGraph,
        description: str = "",
        exclude:     set[str] | None = None,
        max_retries: int = 2,
    ) -> FlowsheetGraph:
        """
        Assign property_package and binary_parameters to the graph.
        Returns a new graph (input not mutated).
        """
        g = graph.copy()
        exclude = exclude or set()

        # ── Stage 1: deterministic selection via RAG ───────────────────────────
        feed_T = _feed_temperature(g)
        feed_P = _feed_pressure(g)
        candidates = self._retriever.select_package(
            g.compounds, description, feed_P, feed_T, exclude)

        if len(candidates) == 1:
            pkg = candidates[0]
        else:
            # ── Stage 2: short LLM confirmation ───────────────────────────────
            pkg = self._llm_select(g, description, candidates, max_retries) or candidates[0]

        g.property_package = pkg

        # Inject BIPs if package requires them and corpus has coverage
        if pkg in ("NRTL", "UNIQUAC"):
            bips, missing = self._retriever.query_bips(g.compounds, pkg, feed_T)
            if not missing:
                g.binary_parameters = bips

        return g

    def _llm_select(
        self,
        graph:       FlowsheetGraph,
        description: str,
        candidates:  list[str],
        max_retries: int,
    ) -> Optional[str]:
        context = self._retriever.thermo_context(graph.compounds)
        prompt  = (
            f"Process: {description or 'not specified'}\n"
            f"Compounds: {', '.join(graph.compounds)}\n"
            f"Compound analysis:\n{context}\n"
            f"Candidate packages (best first): {', '.join(candidates)}\n\n"
            "Select the single best package."
        )
        last_error = ""
        raw = ""
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
                pkg  = data.get("package", "").strip()
                if pkg in _ALLOWED_PACKAGES:
                    return pkg
                last_error = f"'{pkg}' is not an allowed package name"
            except (json.JSONDecodeError, KeyError) as e:
                last_error = str(e)
        return None


def _feed_temperature(graph: FlowsheetGraph) -> float:
    for stream in graph.streams():
        if stream.T is not None and stream.metadata.get("is_feed"):
            return stream.T
    return 300.0


def _feed_pressure(graph: FlowsheetGraph) -> float:
    for stream in graph.streams():
        if stream.P is not None and stream.metadata.get("is_feed"):
            return stream.P
    return 101_325.0


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    return json.loads(text)
