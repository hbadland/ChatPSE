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
    "completeness",   # full_ccs + completeness-verification loop around extraction
]


@dataclass(frozen=True)
class AblationConfig:
    mode:          str
    beam_width:    int   = 3      # overridden for "greedy"
    disable_physics: bool = False
    disable_rules:  bool  = False
    disable_coupling: bool = False
    completeness_loop: bool = False   # extract-verify-augment loop (default OFF)

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
    # full system + the completeness-verification loop around unit extraction.
    # Run this vs "full_ccs" to isolate the loop's effect on capture ratio.
    "completeness": AblationConfig("completeness", beam_width=3, disable_physics=False, disable_rules=False, disable_coupling=False, completeness_loop=True),
}


# ── Null Retriever ─────────────────────────────────────────────────────────────

class _NullUnitSpecs:
    """Stub for Retriever.units (UnitSpecRetriever interface)."""
    def defaults(self, unit_type: str) -> dict:
        return {}
    def context_for_prompt(self, unit_type: str) -> str:
        return ""
    def get(self, unit_type: str):
        return None


class NullRetriever:
    """
    Drop-in replacement for rag.Retriever that returns empty/safe results.

    Implements the full Retriever interface so agents don't crash when
    the no_rule_store ablation replaces the real retriever with this stub.
    """

    def __init__(self) -> None:
        # ParamMapper accesses .units.defaults() directly
        self.units = _NullUnitSpecs()

    # query_bips must return (list[dict], list[tuple]) — callers unpack as
    # `bips, missing = retriever.query_bips(...)`.
    def query_bips(self, compounds: list[str], model: str = "NRTL",
                   T_K: float | None = None) -> tuple[list, list]:
        return [], []

    # Signature must match Retriever.select_package (5 positional args after self)
    def select_package(self, compounds: list[str],
                       description: str = "",
                       pressure_pa: float = 101_325.0,
                       temperature_k: float = 300.0,
                       exclude: set | None = None) -> list[str]:
        return ["Peng-Robinson"]   # always default EOS; never activity-coeff

    def unit_context(self, unit_type: str) -> str:
        return ""

    def thermo_context(self, compounds: list[str]) -> str:
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
    import sys as _sys
    patches = []

    print(f"[ABLATION] ENTER mode={config.mode} "
          f"disable_physics={config.disable_physics} "
          f"disable_rules={config.disable_rules} "
          f"disable_coupling={config.disable_coupling} "
          f"beam_width={config.beam_width}", flush=True)

    # ── No physics: bubble_point_K → None ─────────────────────────────────────
    # `from ir.thermo_estimation import bubble_point_K` creates a module-level
    # name binding in every importer.  Patching only the source module leaves all
    # cached references (param_mapper, consistency, etc.) pointing to the real
    # function.  Use a sys.modules sweep to replace every live reference.
    if config.disable_physics:
        try:
            import ir.thermo_estimation as _te
            original_bp = _te.bubble_point_K

            def _null_bubble_point(*args, **kwargs):
                return None

            patched_mods: list[str] = []
            # Patch source first, then every module that imported it by name
            for _mod in [_te] + [m for m in _sys.modules.values()
                                  if m is not None and m is not _te
                                  and getattr(m, "bubble_point_K", None) is original_bp]:
                patches.append((_mod, "bubble_point_K", original_bp))
                _mod.bubble_point_K = _null_bubble_point
                patched_mods.append(getattr(_mod, "__name__", repr(_mod)))

            print(f"[ABLATION]   no_physics: patched {len(patched_mods)} modules: "
                  f"{patched_mods}", flush=True)

            # ── Verify the patch is live ──────────────────────────────────────
            # Call through each critical module's name binding; None = patch ok.
            def _verify_bp_mod(mod_name: str) -> str:
                mod = _sys.modules.get(mod_name)
                if mod is None:
                    return f"{mod_name}=NOT_LOADED"
                fn  = getattr(mod, "bubble_point_K", None)
                if fn is None:
                    return f"{mod_name}=NO_ATTR"
                try:
                    result = fn(["Water"], 101325.0)
                    status = "None=OK" if result is None else f"PATCH_FAILED({result})"
                except Exception as _exc:
                    status = f"ERROR({_exc})"
                return f"{mod_name}={status}"

            verifications = [
                _verify_bp_mod("ir.thermo_estimation"),
                _verify_bp_mod("agents.stage3.param_mapper"),
                _verify_bp_mod("ir.consistency"),
                _verify_bp_mod("agents.stage4.beam_search"),
            ]
            print(f"[ABLATION]   no_physics VERIFY: {verifications}", flush=True)
        except ImportError as _e:
            print(f"[ABLATION]   no_physics: import failed: {_e}", flush=True)

    # ── No rule store: replace Retriever with NullRetriever ───────────────────
    # The orchestrator sub-agents were already given NullRetriever instances by
    # make_orchestrator before apply_ablation is entered.  This class-level patch
    # prevents any NEW Retriever() calls inside orch.run() from getting real data.
    if config.disable_rules:
        try:
            import rag.retriever as _rag
            original_cls = _rag.Retriever
            patches.append((_rag, "Retriever", original_cls))
            _rag.Retriever = NullRetriever   # type: ignore[assignment]

            # ── Verify ────────────────────────────────────────────────────────
            test_inst = _rag.Retriever()
            print(f"[ABLATION]   no_rule_store VERIFY: "
                  f"rag.Retriever() type={type(test_inst).__name__} "
                  f"(NullRetriever=OK)", flush=True)
        except ImportError as _e:
            print(f"[ABLATION]   no_rule_store: import failed: {_e}", flush=True)

    # ── Greedy: beam_width=1 set directly on orch._repair in make_orchestrator ─
    if config.beam_width == 1:
        print(f"[ABLATION]   greedy: beam_width=1 applied by make_orchestrator "
              f"(not via apply_ablation)", flush=True)

    # ── No coupling: ParameterCouplingMap.get_coupled_boosts → {} ────────────
    # Class-level patch is sufficient: module-level singletons like beam_search._coupling
    # resolve methods via class lookup when no instance attribute overrides them.
    if config.disable_coupling:
        try:
            import ir.coupling as _coup
            orig_boosts = _coup.ParameterCouplingMap.get_coupled_boosts

            def _no_coupling_boosts(self, *args, **kwargs) -> dict:
                return {}

            patches.append((_coup.ParameterCouplingMap, "get_coupled_boosts", orig_boosts))
            _coup.ParameterCouplingMap.get_coupled_boosts = _no_coupling_boosts

            # ── Verify ────────────────────────────────────────────────────────
            # Call on the beam_search singleton — should return {} unconditionally.
            try:
                import agents.stage4.beam_search as _bs
                _coupling_inst = getattr(_bs, "_coupling", None)
                if _coupling_inst is not None:
                    probe = _coupling_inst.get_coupled_boosts(
                        None, "HT-01", "T_out", set())
                    ok = probe == {}
                    print(f"[ABLATION]   no_coupling VERIFY: "
                          f"beam_search._coupling.get_coupled_boosts(…) = {probe} "
                          f"({'{}=OK' if ok else 'PATCH_FAILED'})", flush=True)
                else:
                    print(f"[ABLATION]   no_coupling VERIFY: "
                          f"beam_search._coupling not found", flush=True)
            except Exception as _ve:
                print(f"[ABLATION]   no_coupling VERIFY error: {_ve}", flush=True)
        except (ImportError, AttributeError) as _e:
            print(f"[ABLATION]   no_coupling: patch failed: {_e}", flush=True)

    # ── Completeness loop: flip the module-level toggle read by _topology_node ─
    if getattr(config, "completeness_loop", False):
        try:
            import agents.stage1.completeness as _comp
            patches.append((_comp, "LOOP_ENABLED", _comp.LOOP_ENABLED))
            _comp.LOOP_ENABLED = True
            print(f"[ABLATION]   completeness: LOOP_ENABLED=True "
                  f"(extract-verify-augment loop active)", flush=True)
        except ImportError as _e:
            print(f"[ABLATION]   completeness: import failed: {_e}", flush=True)

    try:
        yield
    finally:
        # Restore all patches in reverse order
        for obj, attr, original in reversed(patches):
            setattr(obj, attr, original)
        print(f"[ABLATION] EXIT mode={config.mode} "
              f"(restored {len(patches)} patches)", flush=True)


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
    from agents.rule_store import FailureRuleStore
    from rag.retriever import Retriever

    retriever = NullRetriever() if config.disable_rules else Retriever()

    orch = OrchestratorV2(model=model, max_iterations=max_iterations)

    # Override RepairAgent beam width
    if hasattr(orch, "_repair"):
        orch._repair._beam_width = config.beam_width   # type: ignore[attr-defined]

    # Override retriever in all sub-agents
    if config.disable_rules:
        # Agents that store _retriever directly
        for attr in ("_retriever", "_params"):
            sub = getattr(orch, attr, None)
            if sub is not None and hasattr(sub, "_retriever"):
                sub._retriever = retriever         # type: ignore[attr-defined]
        if hasattr(orch, "_repair") and hasattr(orch._repair, "_retriever"):
            orch._repair._retriever = retriever    # type: ignore[attr-defined]

        # ThermoMapper stores the retriever only in its internal components
        # (_selector, _injector, _llm) — not as self._retriever — so patch
        # each component directly.
        _thermo = getattr(orch, "_thermo", None)
        if _thermo is not None:
            for _comp_attr in ("_selector", "_injector", "_llm"):
                _comp = getattr(_thermo, _comp_attr, None)
                if _comp is not None and hasattr(_comp, "_retriever"):
                    _comp._retriever = retriever   # type: ignore[attr-defined]

        # Replace the loaded rule store with an empty one so synthesized rules
        # from prior cases have no effect in the no_rule_store ablation.
        orch._rule_store = FailureRuleStore()

    return orch, config
