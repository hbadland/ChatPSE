"""
Ablation configurations for the CCS benchmark.

Five ablation conditions for the NeurIPS paper:

  full_ccs      — complete system (default): beam search + physics + rule store + coupling
  no_physics    — disable thermo estimation (bubble_point_K returns None)
  no_rule_store — empty RAG retriever (no BIP/package/unit-spec lookups)
  greedy        — beam_width=1 (single-candidate repair, no beam search)
  no_coupling   — ParameterCouplingMap returns no boosts (uncoupled sequential repair)

Each mode is a context manager that monkey-patches the relevant module-level
functions / class attributes for the duration of the benchmark run, then
restores originals on exit.  Core system files are NOT modified.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterator


ABLATION_MODES = [
    "full_ccs",
    "no_physics",
    "no_rule_store",
    "greedy",
    "no_coupling",
]


@dataclass(frozen=True)
class AblationConfig:
    mode:          str
    beam_width:    int   = 3      # overridden for "greedy"
    disable_physics: bool = False
    disable_rules:  bool  = False
    disable_coupling: bool = False

    def label(self) -> str:
        return self.mode

    def __str__(self) -> str:
        flags = []
        if self.disable_physics:  flags.append("no_physics")
        if self.disable_rules:    flags.append("no_rules")
        if self.disable_coupling: flags.append("no_coupling")
        if self.beam_width == 1:  flags.append("greedy")
        return f"AblationConfig({self.mode}: {', '.join(flags) or 'full'})"


CONFIGS: dict[str, AblationConfig] = {
    "full_ccs":    AblationConfig("full_ccs",    beam_width=3,  disable_physics=False, disable_rules=False,  disable_coupling=False),
    "no_physics":  AblationConfig("no_physics",  beam_width=3,  disable_physics=True,  disable_rules=False,  disable_coupling=False),
    "no_rule_store": AblationConfig("no_rule_store", beam_width=3, disable_physics=False, disable_rules=True, disable_coupling=False),
    "greedy":      AblationConfig("greedy",      beam_width=1,  disable_physics=False, disable_rules=False,  disable_coupling=False),
    "no_coupling": AblationConfig("no_coupling", beam_width=3,  disable_physics=False, disable_rules=False,  disable_coupling=True),
}


# ── Null Retriever ─────────────────────────────────────────────────────────────

class NullRetriever:
    """Drop-in replacement for rag.Retriever that returns empty results."""

    def query_bips(self, compounds: list[str], model: str = "NRTL",
                   T_K: float | None = None) -> list:
        return []

    def select_package(self, compounds: list[str],
                       description: str = "",
                       exclude: list[str] | None = None) -> list[str]:
        return ["Peng-Robinson"]   # always default EOS; never activity-coeff

    def unit_context(self, unit_type: str) -> str:
        return ""

    def has_full_coverage(self, compounds: list[str], model: str,
                          T_K: float | None = None) -> bool:
        return False

    def context_for_prompt(self, compounds: list[str]) -> str:
        return ""


# ── Context managers ───────────────────────────────────────────────────────────

@contextlib.contextmanager
def _patch(obj, attr: str, replacement):
    original = getattr(obj, attr, None)
    setattr(obj, attr, replacement)
    try:
        yield
    finally:
        if original is not None:
            setattr(obj, attr, original)
        else:
            try:
                delattr(obj, attr)
            except AttributeError:
                pass


@contextlib.contextmanager
def apply_ablation(config: AblationConfig) -> Iterator[None]:
    """
    Context manager that applies all patches for an AblationConfig.

    Usage:
        with apply_ablation(CONFIGS["greedy"]):
            result = orchestrator.run(description)
    """
    patches = []

    # ── No physics: bubble_point_K → None ─────────────────────────────────────
    if config.disable_physics:
        try:
            import ir.thermo_estimation as _te
            original_bp = _te.bubble_point_K

            def _null_bubble_point(*args, **kwargs):
                return None

            patches.append((_te, "bubble_point_K", original_bp))
            _te.bubble_point_K = _null_bubble_point

            # Also patch in repair_agent where it's imported at module level
            try:
                import agents.stage4.repair_agent as _ra
                if hasattr(_ra, "bubble_point_K"):
                    patches.append((_ra, "bubble_point_K", _ra.bubble_point_K))
                    _ra.bubble_point_K = _null_bubble_point
            except ImportError:
                pass

            try:
                import agents.stage4.beam_search as _bs
                if hasattr(_bs, "bubble_point_K"):
                    patches.append((_bs, "bubble_point_K", _bs.bubble_point_K))
                    _bs.bubble_point_K = _null_bubble_point
            except ImportError:
                pass
        except ImportError:
            pass

    # ── No rule store: replace Retriever with NullRetriever ───────────────────
    if config.disable_rules:
        try:
            import rag.retriever as _rag
            original_cls = _rag.Retriever
            patches.append((_rag, "Retriever", original_cls))
            _rag.Retriever = NullRetriever   # type: ignore[assignment]

            # Patch the cached singleton if instantiated
            try:
                import agents.stage3.thermo_mapper as _tm
                if hasattr(_tm, "_retriever"):
                    patches.append((_tm, "_retriever", _tm._retriever))
                    _tm._retriever = NullRetriever()
            except ImportError:
                pass
        except ImportError:
            pass

    # ── Greedy: beam_width = 1 ─────────────────────────────────────────────────
    if config.beam_width == 1:
        try:
            import agents.stage4.repair_agent as _ra
            if hasattr(_ra, "RepairAgent"):
                orig_init = _ra.RepairAgent.__init__

                def _greedy_init(self, *args, beam_width=1, **kwargs):
                    orig_init(self, *args, beam_width=1, **kwargs)

                patches.append((_ra.RepairAgent, "__init__", orig_init))
                _ra.RepairAgent.__init__ = _greedy_init
        except ImportError:
            pass

    # ── No coupling: ParameterCouplingMap.coupled_targets → [] ────────────────
    if config.disable_coupling:
        try:
            import ir.coupling as _coup
            if hasattr(_coup, "ParameterCouplingMap"):
                orig_ct = _coup.ParameterCouplingMap.coupled_targets

                def _no_coupling(self, *args, **kwargs):
                    return []

                patches.append((_coup.ParameterCouplingMap, "coupled_targets", orig_ct))
                _coup.ParameterCouplingMap.coupled_targets = _no_coupling
        except ImportError:
            pass

    try:
        yield
    finally:
        # Restore all patches in reverse order
        for obj, attr, original in reversed(patches):
            setattr(obj, attr, original)


def make_orchestrator(
    config:         AblationConfig,
    model:          str = "qwen3:14b",
    max_iterations: int = 6,
):
    """
    Create an OrchestratorV2 configured for the given ablation mode.
    The RepairAgent beam_width is set directly at construction.
    """
    from agents.orchestrator_v2 import OrchestratorV2
    from rag.retriever import Retriever

    retriever = NullRetriever() if config.disable_rules else Retriever()

    orch = OrchestratorV2(model=model, max_iterations=max_iterations)

    # Override RepairAgent beam width
    if hasattr(orch, "_repair"):
        orch._repair._beam_width = config.beam_width   # type: ignore[attr-defined]

    # Override retriever in all sub-agents
    if config.disable_rules:
        for attr in ("_retriever", "_thermo", "_params"):
            sub = getattr(orch, attr, None)
            if sub is not None and hasattr(sub, "_retriever"):
                sub._retriever = retriever         # type: ignore[attr-defined]
        if hasattr(orch, "_repair") and hasattr(orch._repair, "_retriever"):
            orch._repair._retriever = retriever    # type: ignore[attr-defined]

    return orch, config
