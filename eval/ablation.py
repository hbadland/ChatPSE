"""
Ablation hooks for systematic experiment design.

Each AblationMode disables one architectural component so its contribution
to the full-system accuracy can be isolated.

Ablation conditions:
  no_rag      — ThermoMapper/RepairAgent cannot query BIP corpus;
                  package is selected without BIP coverage check
  no_repair   — Stage 4 repair loop runs 0 iterations (execute once, then stop)
  no_classifier — ErrorClassifier skips Stage 2 LLM; only deterministic signals
  reduced_agents — Stage 1 collapses to single combined extraction call
  full        — baseline (all components enabled)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from agents.orchestrator_v2 import OrchestratorV2
from agents.llm import DEFAULT_MODEL


class AblationMode(str, Enum):
    FULL             = "full"
    NO_RAG           = "no_rag"
    NO_REPAIR        = "no_repair"
    NO_CLASSIFIER    = "no_classifier"
    REDUCED_AGENTS   = "reduced_agents"


@dataclass
class AblationConfig:
    mode:           AblationMode = AblationMode.FULL
    model:          str          = DEFAULT_MODEL
    max_iterations: int          = 6

    def build_orchestrator(self) -> OrchestratorV2:
        orch = OrchestratorV2(model=self.model, max_iterations=self.max_iterations)

        if self.mode == AblationMode.NO_RAG:
            # Replace retriever BIP lookup with empty stub
            from rag.retriever import Retriever, BIPRetriever
            orch._retriever = _NoRAGRetriever()
            orch._thermo._retriever = orch._retriever
            orch._params._retriever = orch._retriever
            orch._repair._retriever = orch._retriever

        elif self.mode == AblationMode.NO_REPAIR:
            # Zero repair iterations — run executor once, return result
            orch._max_iter = 1

        elif self.mode == AblationMode.NO_CLASSIFIER:
            # Disable LLM stage of ErrorClassifier
            orch._classifier._llm_classify = lambda *a, **kw: []

        elif self.mode == AblationMode.REDUCED_AGENTS:
            # Collapse Stage 1 to a single combined call via CombinedExtractor
            combined = _CombinedExtractor(model=self.model)
            orch._unit_ext   = combined
            orch._stream_ext = combined

        return orch


# ── Stub implementations ───────────────────────────────────────────────────────

class _NoRAGRetriever:
    """Retriever that returns no BIPs and selects packages without corpus check."""

    def query_bips(self, compounds, model, T_K=None):
        return [], list(zip(compounds, compounds[1:]))  # all pairs missing

    def select_package(self, compounds, description="", pressure_pa=101325.0,
                       temperature_k=300.0, exclude=None):
        from rag.retriever import ThermoRetriever
        tr = ThermoRetriever()
        return tr.select(compounds, description, pressure_pa, temperature_k,
                         exclude, bip_retriever=None)

    def unit_context(self, unit_type: str) -> str:
        from rag.retriever import UnitSpecRetriever
        return UnitSpecRetriever().context_for_prompt(unit_type)

    def thermo_context(self, compounds) -> str:
        from rag.retriever import ThermoRetriever
        return ThermoRetriever().context_for_prompt(compounds)

    class bip:
        @staticmethod
        def has_full_coverage(*a, **kw): return False

    class units:
        @staticmethod
        def defaults(unit_type):
            from rag.retriever import UnitSpecRetriever
            return UnitSpecRetriever().defaults(unit_type)


class _CombinedExtractor:
    """
    Reduced-agents stub: single LLM call returns both units and streams.
    Used to ablate the benefit of separate UnitExtractor + StreamExtractor.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self._model = model
        self._sem_units = None
        self._sem_topo  = None

    def extract(self, description, compounds, unit_tags=None, unit_roles=None,
                max_retries=3):
        if unit_tags is None:
            # First call: extract both units and streams together
            from agents.stage1.unit_extractor import UnitExtractor
            from agents.stage1.stream_extractor import StreamExtractor
            ue = UnitExtractor(model=self._model)
            self._sem_units = ue.extract(description, compounds, max_retries)
            se = StreamExtractor(model=self._model)
            self._sem_topo = se.extract(
                description, compounds,
                unit_tags  = [u.tag  for u in self._sem_units.units],
                unit_roles = {u.tag: u.role for u in self._sem_units.units},
                max_retries = max_retries,
            )
            return self._sem_units
        else:
            # Second call: return cached topology
            return self._sem_topo
