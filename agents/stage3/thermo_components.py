"""
Decomposed ThermoMapper components (Item 6).

Three single-responsibility classes replace the monolithic ThermoMapper:

  PackageSelector    — deterministic rule-based selection, zero LLM
  BIPInjector        — pure corpus lookup, zero LLM
  ThermoLLMFallback  — LLM confirmation only when PackageSelector is ambiguous

ThermoMapper (thermo_mapper.py) is preserved for backward compatibility;
it now delegates to these three components internally.  Callers that want
finer control can use the components directly.
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

_SELECTION_SYSTEM = """\
Select the thermodynamic property package for this chemical process.
Return ONLY a JSON object — no explanation, no markdown.

Schema: {"package": "<package name>", "reason": "<one sentence>"}

Allowed package names (use exactly):
  Raoult's Law | NRTL | UNIQUAC | Peng-Robinson | Soave-Redlich-Kwong | Lee-Kesler-Plöcker"""


# ── Component 1: deterministic rule-based selection ────────────────────────────

class PackageSelector:
    """
    Selects a property package using ThermoRetriever hard rules.
    No LLM calls.  Returns a ranked candidate list (best first).
    """

    def __init__(self, retriever: Optional[Retriever] = None) -> None:
        self._retriever = retriever or Retriever()

    def select(
        self,
        graph:       FlowsheetGraph,
        description: str = "",
        exclude:     set[str] | None = None,
    ) -> list[str]:
        """Return ranked candidate packages; ambiguous if len > 1."""
        feed_T = _feed_temperature(graph)
        feed_P = _feed_pressure(graph)
        return self._retriever.select_package(
            graph.compounds, description, feed_P, feed_T, exclude)

    def is_unambiguous(self, candidates: list[str]) -> bool:
        return len(candidates) == 1


# ── Component 2: BIP injection ─────────────────────────────────────────────────

class BIPInjector:
    """
    Pure corpus lookup.  Injects binary interaction parameters into a graph.
    Returns (graph_with_bips, missing_pairs).
    No LLM calls.
    """

    def __init__(self, retriever: Optional[Retriever] = None) -> None:
        self._retriever = retriever or Retriever()

    def inject(
        self,
        graph: FlowsheetGraph,
        T_K:   Optional[float] = None,
    ) -> tuple[FlowsheetGraph, list[tuple[str, str]]]:
        """
        Inject BIPs for graph.property_package.
        Returns (patched_graph, missing_pairs).
        missing_pairs is empty iff full coverage exists.
        """
        pkg = graph.property_package
        if pkg not in ("NRTL", "UNIQUAC"):
            return graph, []

        bips, missing = self._retriever.query_bips(graph.compounds, pkg, T_K)
        if missing:
            return graph, missing

        g = graph.copy()
        g.binary_parameters = bips
        return g, []

    def has_coverage(
        self,
        graph: FlowsheetGraph,
        T_K:   Optional[float] = None,
    ) -> bool:
        _, missing = self.inject(graph, T_K)
        return len(missing) == 0


# ── Component 3: LLM fallback for ambiguous mixtures ──────────────────────────

class ThermoLLMFallback:
    """
    Resolves ambiguous package selection via a single focused LLM call.
    Only called when PackageSelector returns len(candidates) > 1.
    """

    def __init__(self, model: str = DEFAULT_MODEL,
                 retriever: Optional[Retriever] = None) -> None:
        self._model     = model
        self._retriever = retriever or Retriever()

    def select(
        self,
        graph:       FlowsheetGraph,
        description: str,
        candidates:  list[str],
        max_retries: int = 2,
    ) -> Optional[str]:
        """
        Ask LLM to choose the best package from candidates.
        Returns the chosen package name, or None on failure.
        """
        context = self._retriever.thermo_context(graph.compounds)
        prompt  = (
            f"Process: {description or 'not specified'}\n"
            f"Compounds: {', '.join(graph.compounds)}\n"
            f"Compound analysis:\n{context}\n"
            f"Candidate packages (best first): {', '.join(candidates)}\n\n"
            "Select the single best package."
        )
        last_error = ""
        for attempt in range(max_retries):
            raw = chat(
                prompt + (f"\n\nPrevious error: {last_error}" if last_error else ""),
                system=_SELECTION_SYSTEM,
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


# ── Helpers ────────────────────────────────────────────────────────────────────

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
