"""
Agent D — Thermo Mapper.

Input : FlowsheetGraph + Retriever
Output: FlowsheetGraph with property_package and binary_parameters set

Delegates to three single-responsibility components (agents/stage3/thermo_components.py):
  PackageSelector   — deterministic selection (Stage 1, free)
  ThermoLLMFallback — LLM confirmation when ambiguous (Stage 2, only if needed)
  BIPInjector       — pure corpus lookup for binary parameters

LLMs must not invent packages; the allowed list is injected explicitly.
"""
from __future__ import annotations

import os
from typing import Optional

from ir.graph import FlowsheetGraph
from agents.llm import DEFAULT_MODEL
from agents.stage3.thermo_components import (
    PackageSelector, BIPInjector, ThermoLLMFallback,
    _feed_temperature,
)
from rag.retriever import Retriever, ThermoCoverageGuard


class ThermoMapper:
    def __init__(self, model: str = DEFAULT_MODEL, retriever: Optional[Retriever] = None):
        ret = retriever or Retriever()
        self._ret       = ret
        self._selector  = PackageSelector(ret)
        self._injector  = BIPInjector(ret)
        self._llm       = ThermoLLMFallback(model, ret)

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
        g      = graph.copy()
        exclude = exclude or set()

        # CHANGE 1: the selector raises ThermoCoverageGuard when the coverage guard
        # is enabled and a polar/azeotropic system has no NRTL/UNIQUAC coverage.
        # Re-raise so the Stage-3 handler terminates the run (PARAM_MISSING → human)
        # rather than falling through to Raoult's Law (ideal).
        try:
            candidates = self._selector.select(g, description, exclude)
        except ThermoCoverageGuard as exc:
            print(f"[PARAM_MISSING] coverage-guard escalation: {exc}", flush=True)
            raise

        # CHANGE 5: config-gated tiebreak. THERMO_TIEBREAK=deterministic skips the LLM
        # entirely and takes candidates[0] (candidate GENERATION is untouched). Default
        # 'llm' is byte-identical to prior behaviour.
        _tiebreak = os.environ.get("THERMO_TIEBREAK", "llm").strip().lower()
        if self._selector.is_unambiguous(candidates):
            pkg = candidates[0]
        elif _tiebreak == "deterministic":
            pkg = candidates[0]
        else:
            pkg = self._llm.select(g, description, candidates, max_retries) \
                  or candidates[0]

        g.property_package = pkg

        # ── Option B: BIP-coverage provenance marker (no control-flow change) ──
        # When the selector saw a polar/azeotropic system with NO activity-model
        # coverage, the chosen package is a substitute served without BIPs. Record
        # MISSING on the graph so downstream — and the methodology figure — can show
        # Configuration writing a MISSING-tagged value, WITHOUT asserting an
        # escalation the control flow does not perform: the run still proceeds.
        # Inert to execution (to_dwsim ignores metadata); survives graph.copy().
        if getattr(self._ret.thermo, "last_coverage_status", None) == "MISSING":
            g.metadata["thermo_coverage"] = "MISSING"
            g.metadata["thermo_coverage_missing_pairs"] = list(
                getattr(self._ret.thermo, "last_coverage_missing_pairs", []))
            print(
                f"[THERMO] BIP coverage MISSING for "
                f"{g.metadata['thermo_coverage_missing_pairs']} — proceeding with "
                f"'{pkg}' (substitute, no BIPs); tagged MISSING, not escalated",
                flush=True)

        if pkg in ("NRTL", "UNIQUAC"):
            # T_K=None: skip feed-temperature guard so cold-feed cases (e.g. 25°C
            # feed into a 78°C flash) don't block BIP injection when the BIP fit
            # range covers only the operating temperature, not the feed temperature.
            g, missing = self._injector.inject(g, T_K=None)
            if missing:
                g.binary_parameters = []

        return g
