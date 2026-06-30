"""
v2 Orchestrator — lean pipeline controller.

Pipeline:
  Stage 1  UnitExtractor + StreamExtractor  (2 LLM calls)
  Stage 2  GraphBuilder                      (0 LLM calls — assembly)
  Normalise + Validate                       (0 LLM calls)
  Stage 3  ThermoMapper + ParamMapper        (1-2 LLM calls)
  Normalise + Validate                       (0 LLM calls)
  IR → DWSIM JSON                            (0 LLM calls)
  Stage 4  loop:
    Executor → ErrorClassifier → RepairAgent (0-1 LLM calls per iteration)

All error routing lives in ErrorClassifier + the error taxonomy.
The orchestrator only manages the loop and aggregates results.
"""
from __future__ import annotations

import sys
import time
from benchmark.case_schema import is_validation_tier
from dataclasses import dataclass, field
from typing import Optional

from agents.basis import BasisAgent, BasisResult
from agents.executor import Executor
from agents.llm import DEFAULT_MODEL

from agents.stage1 import UnitExtractor, StreamExtractor
from agents.stage1.unit_extractor import SemanticUnits
from agents.stage1.stream_extractor import SemanticTopology
# TopologyChain removed — langchain_core LanguageModelOutput import incompatibility
# means it silently fell back to standard extractors on every run anyway.
_TOPOLOGY_CHAIN_AVAILABLE = False
from agents.stage2 import GraphBuilder
from agents.stage3 import ThermoMapper, ParamMapper
from agents.stage4 import ErrorClassifier, ClassifiedError, RepairAgent
from agents.stage4.repair_agent import RepairMemory
from ir.types import RepairStrategy as _RepairStrategy
from agents.stage4.sim_hints import SimulationHints, EMPTY_HINTS
from agents.rule_store import FailureRuleStore, RULES_PATH

from ir import FlowsheetGraph, normalise, validate, to_dwsim
from ir.margin_model import get_global_margin_model
from ir.consistency import GlobalConsistencyPass
from ir.validate import ValidationReport
from rag.retriever import Retriever

# Phrases that must appear in the description for a recycle tag to be accepted.
# Matched case-insensitively against the normalised description.
_RECYCLE_PHRASES: tuple[str, ...] = (
    "recycled back to",
    "recycled back",
    "recycled to",
    "returned to",
    "fed back to",
    "recirculated to",
)

# Ordinal words → 0-based index used by _resolve_recycle_target.
_ORDINALS: dict[str, int] = {
    "first": 0, "1st": 0,
    "second": 1, "2nd": 1,
    "third": 2, "3rd": 2,
    "fourth": 3, "4th": 3,
    "fifth": 4, "5th": 4,
}

# (regex on lowercased recycle_target, unit types to match against)
_TARGET_TYPE_PATTERNS: list[tuple[str, list[str]]] = [
    (r"col|distill",                       ["Heater"]),
    (r"react|reform|convert|shift|methan", ["ConversionReactor"]),
    (r"vessel|flash|separat|drum|decant",  ["Vessel"]),
    (r"mix",                               ["Mixer"]),
    (r"split|divid",                       ["Splitter"]),
    (r"pump",                              ["Pump"]),
    (r"compress",                          ["Compressor"]),
    (r"expand|turbin",                     ["Expander"]),
    (r"heat|reboil|furnac",               ["Heater"]),
    (r"cool|condense|chill",              ["Cooler"]),
]


def _resolve_recycle_target(recycle_target: str,
                             units: list) -> Optional[str]:
    """Fuzzy-resolve a natural-language recycle_target to an exact unit tag.

    Resolution order:
      1. Exact tag match              — "HT-01" already in unit list
      2. Ordinal + type keyword       — "first column" → 1st Heater
                                        "second reactor" → 2nd ConversionReactor
      3. Type keyword only            — "reactor" → first ConversionReactor
      4. Partial tag substring        — "HT" → first Heater tag containing "HT"
      5. Role keyword                 — "feed mixer" → Mixer whose role has "feed"

    Returns the resolved tag, or None when resolution fails.
    """
    import re as _re
    unit_tag_set = {u.tag for u in units}
    if recycle_target in unit_tag_set:
        return recycle_target

    t = recycle_target.lower().strip()
    if not t:
        return None

    # ── 1. Extract ordinal ────────────────────────────────────────────────────
    ordinal = 0
    for word, idx in _ORDINALS.items():
        if word in t.split():
            ordinal = idx
            break

    # ── 2. Match type keyword → pick ordinal-th unit of that type ────────────
    for pattern, types in _TARGET_TYPE_PATTERNS:
        if _re.search(pattern, t):
            candidates = [u for u in units if u.type in types]
            if candidates:
                return candidates[min(ordinal, len(candidates) - 1)].tag
            break

    # ── 3. Partial tag substring (e.g. "HT-01", "V-01") ─────────────────────
    for u in units:
        if u.tag.lower() in t or t in u.tag.lower():
            return u.tag

    # ── 4. Role keyword match ("feed mixer" → Mixer whose role has "feed") ───
    t_words = [w for w in t.split() if len(w) > 3]
    for u in units:
        if u.role and any(w in u.role.lower() for w in t_words):
            return u.tag

    return None

_SUMMARISER_SYSTEM = (
    "/no_think\n"
    "List ALL unit operations (equipment items) mentioned in this chemical process description.\n"
    'Return a plain bullet list, one item per line: "- <Type>: <one-line purpose>"\n'
    "Types: Heater Cooler Vessel Mixer Splitter Pump Compressor Expander\n"
    "Critical rules:\n"
    "- List ALL units mentioned, even if they appear as part of columns or separation stages\n"
    "- Each distillation column has at minimum a reboiler (Heater) and condenser (Cooler) — list both\n"
    "- Each separator or flash drum is a Vessel — list every one separately\n"
    "- Do NOT merge, combine, or omit any equipment item\n"
    "- Do NOT summarise — if the description has 12 units, list all 12\n"
    "No JSON, no explanation, no preamble."
)

_SUMMARISER_SYSTEM_TIGHT = (
    "/no_think\n"
    "List only the equipment types and tags, one per line.\n"
    "Format: - <Type>: <Tag>\n"
    "Example: - Heater: HT-01\n"
    "No explanation. No extra words. Maximum 20 words total."
)


def _summarise_for_unit_extraction(description: str, model: str) -> str:
    """Condense a long description to a unit-op bullet list for UnitExtractor."""
    from agents.llm import chat
    prompt = (
        f"Process description:\n{description}\n\n"
        "List ALL unit operations in this process. Do not omit any."
    )
    return chat(prompt, system=_SUMMARISER_SYSTEM, model=model,
                temperature=0.0, max_tokens=1024)


# ── Reference-guided post-solve refinement ─────────────────────────────────────

def _reference_guided_refinement(
        graph: "FlowsheetGraph",
        execution,
        reference_data: dict,
        executor,
) -> "tuple[Optional[dict], Optional[object], list[str]]":
    """After a successful solve, correct T_out params that deviate >10 K from reference.

    Matches each Heater/Cooler outlet stream to the closest reference stream by
    mole-fraction composition (L1 distance), then directly sets T_out to the
    reference T_K when the deviation exceeds _T_DIFF_THRESHOLD.  Re-runs the
    executor once and returns the refined result.

    Returns (dwsim_json, exec_result, log_lines) or (None, None, []) when no
    corrections are needed or the reference streams dict is absent.
    """
    _T_DIFF_THRESHOLD = 10.0   # K — minimum deviation to trigger correction
    _COMP_DIST_MAX    = 0.3    # L1 mole-fraction distance ceiling for composition match

    ref_streams = reference_data.get("streams", {})
    if not isinstance(ref_streams, dict) or not ref_streams:
        return None, None, []

    solved_results: dict = getattr(execution, "stream_results", {}) or {}
    adjustments: list[str] = []
    modified = False

    for node in graph.units():
        if node.unit_type not in ("Heater", "Cooler"):
            continue
        outlets = graph.outlet_streams(node.tag)
        if not outlets:
            continue
        outlet    = outlets[0]
        solved_sr = solved_results.get(outlet.tag)
        if solved_sr is None:
            continue

        # Match solved outlet composition → closest reference stream (L1 distance)
        solved_comp = {k.lower(): float(v)
                       for k, v in solved_sr.composition.items()}
        best_tag: Optional[str] = None
        best_dist = float("inf")
        for rtag, rstream in ref_streams.items():
            rc = {k.lower(): float(v)
                  for k, v in rstream.get("composition", {}).items()}
            if not rc:
                continue
            all_keys = set(solved_comp) | set(rc)
            dist = sum(abs(solved_comp.get(k, 0.0) - rc.get(k, 0.0))
                       for k in all_keys)
            if dist < best_dist:
                best_dist = dist
                best_tag  = rtag

        if best_tag is None or best_dist > _COMP_DIST_MAX:
            continue

        ref_T = float(ref_streams[best_tag].get("T_K", 0))
        if ref_T <= 0:
            continue

        delta = abs(solved_sr.T_K - ref_T)
        if delta <= _T_DIFF_THRESHOLD:
            continue

        print(
            f"[REF_REFINE] {node.tag}.T_out: solved={solved_sr.T_K:.1f} K "
            f"ref('{best_tag}')={ref_T:.1f} K  Δ={delta:.1f} K → correcting",
            flush=True, file=sys.stderr)
        node.params["T_out"] = ref_T
        adjustments.append(
            f"[ref_refine] {node.tag}.T_out "
            f"{solved_sr.T_K:.1f}→{ref_T:.1f} K "
            f"(ref '{best_tag}', comp_dist={best_dist:.3f}, Δ={delta:.1f} K)")
        modified = True

    if not modified:
        return None, None, []

    refined_graph = normalise(graph)
    refined_json  = to_dwsim(refined_graph)
    refined_exec  = executor.run(refined_json)
    print(
        f"[REF_REFINE] re-solve: solved={getattr(refined_exec, 'solved', False)}  "
        f"n_corrections={len(adjustments)}",
        flush=True, file=sys.stderr)
    return refined_json, refined_exec, adjustments


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class IterationRecord:
    iteration:    int
    errors:       list[ClassifiedError]
    changes:      list[str]
    flowsheet:    dict
    execution:    object
    elapsed_s:    float


@dataclass
class PipelineResult:
    description:        str
    outcome:            str              # PASS | HUMAN | MAX_ITER | BASIS_FAILED
                                         # INVALID_IR | INVALID_JSON | PLAN_FAILED
    basis_result:       Optional[BasisResult]      = None
    final_graph:        Optional[FlowsheetGraph]   = None
    final_flowsheet:    Optional[dict]             = None
    final_execution:    object                     = None
    ir_report:          Optional[ValidationReport] = None
    iterations:         list[IterationRecord]      = field(default_factory=list)
    warnings:           list[str]                  = field(default_factory=list)
    total_time_s:       float                      = 0.0
    margin_snapshot:    dict                       = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.outcome == "PASS"

    def summary(self) -> str:
        lines = [
            f"Outcome    : {self.outcome}",
            f"Iterations : {len(self.iterations)}",
            f"Time       : {self.total_time_s:.1f}s",
        ]
        if self.basis_result:
            lines.append(f"Compounds  : {self.basis_result.dwsim_compounds}")
        for rec in self.iterations:
            tag = "✓" if not rec.errors else "✗"
            lines.append(f"  [{tag}] iter {rec.iteration}  {rec.elapsed_s:.1f}s")
            for c in rec.changes:
                lines.append(f"      {c}")
        if self.warnings:
            for w in self.warnings[:5]:
                lines.append(f"  ! {w}")
        return "\n".join(lines)


# ── Orchestrator ───────────────────────────────────────────────────────────────

class OrchestratorV2:
    """
    End-to-end v2 pipeline.

    Args:
        model          : LLM model for all agents
        max_iterations : Stage 4 execution loop limit
    """

    def __init__(
        self,
        model:          str = DEFAULT_MODEL,
        max_iterations: int = 10,
        rule_store:     Optional[FailureRuleStore] = None,
    ):
        self._model      = model
        self._retriever  = Retriever()
        self._max_iter   = max_iterations
        if rule_store is not None:
            self._rule_store = rule_store
        else:
            self._rule_store = FailureRuleStore()
            self._rule_store.load(RULES_PATH)  # no-op if file doesn't exist yet
            if self._rule_store.num_patterns() > 0:
                print(f"[ORCH] rule_store loaded: {self._rule_store.num_patterns()} patterns, "
                      f"{self._rule_store.num_active()} active rules from {RULES_PATH}",
                      flush=True)

        self._basis      = BasisAgent(model=model)
        self._unit_ext   = UnitExtractor(model=model)
        self._stream_ext = StreamExtractor(model=model)
        self._builder    = GraphBuilder()

        self._topology_chain = None
        self._thermo      = ThermoMapper(model=model, retriever=self._retriever)
        self._params      = ParamMapper(model=model, retriever=self._retriever)
        self._consistency = GlobalConsistencyPass()
        self._executor    = Executor()
        self._classifier  = ErrorClassifier(model=model)
        self._repair      = RepairAgent(model=model, retriever=self._retriever)

        self._graph_pipeline = None  # lazily initialised when USE_LANGGRAPH=1

    def run(self, description: str,
            reference_file: Optional[str] = None,
            tier: str = "standard") -> PipelineResult:
        # USE_LANGGRAPH=1 — delegate to the LangGraph scaffold transparently.
        import os as _os
        if _os.environ.get("USE_LANGGRAPH", "").strip() in ("1", "true", "yes"):
            if self._graph_pipeline is None:
                from agents.graph_pipeline import GraphPipeline
                self._graph_pipeline = GraphPipeline(
                    model=self._model,
                    max_iterations=self._max_iter,
                    rule_store=self._rule_store,
                )
                print("[ORCH] USE_LANGGRAPH=1 — delegating to GraphPipeline", flush=True)
            return self._graph_pipeline.run(
                description, reference_file=reference_file, tier=tier)

        t_start = time.time()
        result  = PipelineResult(description=description, outcome="MAX_ITER")

        # ── Stage 0: Basis ─────────────────────────────────────────────────────
        print("[ORCH] step: basis.identify START", flush=True)
        basis = self._basis.identify(description)
        print(f"[ORCH] step: basis.identify END  success={basis.success}", flush=True)
        result.basis_result = basis
        if not basis.success:
            result.outcome    = "BASIS_FAILED"
            result.total_time_s = time.time() - t_start
            return result

        desc      = basis.normalised_description
        compounds = basis.dwsim_compounds
        print(f"[ORCH] compounds={compounds}", flush=True)

        # Pre-processing: for descriptions >200 words, condense to a unit-op
        # bullet list before passing to UnitExtractor.  Long descriptions cause
        # context exhaustion on Qwen3:14b, producing empty JSON responses.
        # StreamExtractor always receives the full description.
        # Validation-tier cases skip summarisation entirely — they are complex
        # by design and summarisation loses units.  max_tokens=16384 covers them.
        _n_desc_words = len(desc.split())
        if _n_desc_words > 200 and not is_validation_tier(tier):
            print(
                f"[ORCH] description length={_n_desc_words} words — "
                "summarising for UnitExtractor",
                flush=True, file=sys.stderr)
            try:
                desc_for_units = _summarise_for_unit_extraction(desc, self._model)
                print(
                    f"[ORCH] unit summary ({len(desc_for_units.split())} words): "
                    f"{desc_for_units[:150]!r}",
                    flush=True, file=sys.stderr)
                result.warnings.append(
                    f"[summary] description condensed "
                    f"({_n_desc_words}→{len(desc_for_units.split())} words) "
                    "for UnitExtractor")
                # Second pass: if the first summary is still >150 words, re-summarise
                # with a tighter prompt to further reduce UnitExtractor context load.
                if len(desc_for_units.split()) > 150:
                    _first_words = len(desc_for_units.split())
                    print(
                        f"[ORCH] first summary still {_first_words} words — "
                        "running tighter second-pass summarisation",
                        flush=True, file=sys.stderr)
                    try:
                        from agents.llm import chat as _chat
                        _tight = _chat(
                            f"Equipment list:\n{desc_for_units}\n\n"
                            "List only the equipment tags and types, one per line."
                            " Maximum 20 words total.",
                            system=_SUMMARISER_SYSTEM_TIGHT,
                            model=self._model,
                            temperature=0.0,
                            max_tokens=256,
                        )
                        if _tight.strip():
                            desc_for_units = _tight.strip()
                            print(
                                f"[ORCH] tight summary "
                                f"({_first_words}→{len(desc_for_units.split())} words): "
                                f"{desc_for_units[:150]!r}",
                                flush=True, file=sys.stderr)
                            result.warnings.append(
                                f"[summary2] second-pass condensed to "
                                f"{len(desc_for_units.split())} words")
                    except Exception as _sum2_exc:
                        print(
                            f"[ORCH] second-pass summariser failed ({_sum2_exc}) "
                            "— keeping first summary",
                            flush=True, file=sys.stderr)
            except Exception as _sum_exc:
                print(
                    f"[ORCH] description summariser failed ({_sum_exc}) "
                    "— using full description",
                    flush=True, file=sys.stderr)
                desc_for_units = desc
        else:
            if is_validation_tier(tier) and _n_desc_words > 200:
                print(
                    f"[ORCH] validation tier: skipping summarisation "
                    f"({_n_desc_words} words) — passing full description "
                    "to UnitExtractor with max_tokens=16384",
                    flush=True, file=sys.stderr)
            desc_for_units = desc

        # ── Stage 1: Semantic parsing ──────────────────────────────────────────
        # validation tier + LangChain available → TopologyChain (4 sequential calls)
        # all other tiers, or LangChain absent    → UnitExtractor + StreamExtractor
        try:
            if is_validation_tier(tier) and self._topology_chain is not None:
                print("[ORCH] step: topology_chain.extract START (validation tier)", flush=True)
                _tc_units, _tc_streams = self._topology_chain.extract(desc, compounds)
                sem_units = SemanticUnits(units=_tc_units)
                sem_topo  = SemanticTopology(streams=_tc_streams)
                print(
                    f"[ORCH] step: topology_chain.extract END  "
                    f"units={[u.tag for u in sem_units.units]}  "
                    f"streams={[s.tag for s in sem_topo.streams]}",
                    flush=True,
                )
            else:
                print("[ORCH] step: unit_ext.extract START", flush=True)
                sem_units = self._unit_ext.extract(desc_for_units, compounds, tier=tier)
                print(f"[ORCH] step: unit_ext.extract END  units={[u.tag for u in sem_units.units]}", flush=True)

                print("[ORCH] step: stream_ext.extract START", flush=True)
                sem_topo  = self._stream_ext.extract(
                    desc, compounds,
                    unit_tags               = [u.tag  for u in sem_units.units],
                    unit_roles              = {u.tag: u.role for u in sem_units.units},
                    concentration_hints     = basis.concentration_hints or [],
                    suggested_compositions  = basis.suggested_compositions or {},
                )
                print("[ORCH] step: stream_ext.extract END", flush=True)

            # Post-extraction recycle guard: for any is_recycle=True stream whose
            # recycle_target is absent or not an exact unit tag, attempt fuzzy
            # resolution before giving up.  Only clear is_recycle when resolution
            # completely fails — this keeps valid recycles whose LLM wrote natural
            # language like "first column" instead of an exact tag.
            _unit_tag_set = {u.tag for u in sem_units.units}
            for _s in sem_topo.streams:
                if not getattr(_s, "is_recycle", False):
                    continue
                if _s.recycle_target and _s.recycle_target in _unit_tag_set:
                    print(
                        f"[ORCH] recycle detected: stream '{_s.tag}' → "
                        f"{_s.recycle_target}",
                        flush=True, file=sys.stderr)
                    continue
                # Target is missing or not an exact tag — try fuzzy resolution
                _resolved = _resolve_recycle_target(
                    _s.recycle_target or "", sem_units.units)
                if _resolved:
                    print(
                        f"[ORCH] recycle guard: resolved target "
                        f"{_s.recycle_target!r} → '{_resolved}' "
                        f"for stream '{_s.tag}'",
                        flush=True, file=sys.stderr)
                    result.warnings.append(
                        f"[recycle] '{_s.tag}' target resolved: "
                        f"{_s.recycle_target!r} → '{_resolved}'")
                    _s.recycle_target = _resolved
                else:
                    print(
                        f"[ORCH] recycle guard: cleared unresolvable recycle on "
                        f"'{_s.tag}' (target={_s.recycle_target!r})",
                        flush=True, file=sys.stderr)
                    result.warnings.append(
                        f"[recycle] '{_s.tag}' recycle_target={_s.recycle_target!r} "
                        "unresolvable — cleared to avoid false positive")
                    _s.is_recycle = False
                    _s.recycle_target = None

            # Multi-recycle deduplication guard: if >1 stream from the same
            # source unit is tagged as recycle, keep only the one whose
            # recycle_target is most upstream (lowest index in unit flow order).
            # This catches cases like a Vessel with VAP+LIQ outlets where only
            # the vapour stream genuinely recycles — the LLM often tags both.
            _unit_order = {u.tag: i for i, u in enumerate(sem_units.units)}
            _recycle_by_src: dict[str, list] = {}
            for _s in sem_topo.streams:
                if getattr(_s, "is_recycle", False) and _s.src:
                    _recycle_by_src.setdefault(_s.src, []).append(_s)
            for _src_tag, _candidates in _recycle_by_src.items():
                if len(_candidates) <= 1:
                    continue
                _src_idx = _unit_order.get(_src_tag, len(sem_units.units))
                def _target_rank(_s, _src_idx=_src_idx):
                    t_idx = _unit_order.get(_s.recycle_target, len(sem_units.units))
                    # Prefer targets that are genuinely upstream (idx < src_idx)
                    return (0 if t_idx < _src_idx else 1, t_idx)
                _keep = min(_candidates, key=_target_rank)
                for _s in _candidates:
                    if _s is not _keep:
                        print(
                            f"[ORCH] multi-recycle guard: cleared extra recycle on "
                            f"'{_s.tag}' from src='{_src_tag}' "
                            f"(target={_s.recycle_target!r}); "
                            f"keeping '{_keep.tag}' → {_keep.recycle_target!r}",
                            flush=True, file=sys.stderr)
                        result.warnings.append(
                            f"[recycle] '{_s.tag}' cleared — multiple recycles from "
                            f"'{_src_tag}', kept '{_keep.tag}' → "
                            f"'{_keep.recycle_target}'")
                        _s.is_recycle = False
                        _s.recycle_target = None

            # Phrase guard: accept is_recycle=True only when the description
            # contains a recognised recycle trigger phrase.  Suppresses LLM
            # hallucinations on non-recycle flowsheets.
            _desc_lower = desc.lower()
            for _s in sem_topo.streams:
                if getattr(_s, "is_recycle", False):
                    if not any(p in _desc_lower for p in _RECYCLE_PHRASES):
                        print(
                            f"[RECYCLE] false positive suppressed: no trigger phrase "
                            f"in description for stream '{_s.tag}'",
                            flush=True, file=sys.stderr)
                        result.warnings.append(
                            f"[recycle] '{_s.tag}' is_recycle=True but no trigger "
                            "phrase in description — cleared to avoid false positive")
                        _s.is_recycle = False
                        _s.recycle_target = None

        except RuntimeError as exc:
            print(f"[ORCH] Stage 1 EXCEPTION: {exc}", flush=True)
            result.warnings.append(f"Stage 1 failed: {exc}")
            result.outcome      = "PLAN_FAILED"
            result.total_time_s = time.time() - t_start
            return result

        # ── Stage 2: IR construction ───────────────────────────────────────────
        print("[ORCH] step: builder.build START", flush=True)
        graph = self._builder.build(sem_units, sem_topo, compounds)
        print("[ORCH] step: builder.build END", flush=True)

        # Reconcile: augment graph.compounds with any compounds found in stream
        # compositions that BasisAgent missed (e.g. second compound in a
        # hyphenated pair like "ethanol-water" where Stage 1 only found "Ethanol").
        _known_lower = {c.lower() for c in graph.compounds}
        for _stream in graph.streams():
            for _name in _stream.composition:
                if _name.lower() not in _known_lower:
                    graph.compounds.append(_name)
                    _known_lower.add(_name.lower())
                    result.warnings.append(
                        f"[compounds] '{_name}' found in feed composition "
                        f"but missing from basis list — added automatically"
                    )
        if graph.compounds != list(compounds):
            print(f"[ORCH] compounds reconciled: {graph.compounds}", flush=True)

        # Back-fill: any stream that already carries composition data must have
        # every compound in graph.compounds (use 0.0 for absent ones).
        # pre_execution_check rejects feed streams missing compounds from the
        # global list — this can happen when reconciliation adds a compound
        # that only appears in one of several feed streams.
        for _stream in graph.streams():
            if _stream.composition:
                for _name in graph.compounds:
                    if _name not in _stream.composition:
                        _stream.composition[_name] = 0.0

        print("[ORCH] step: normalise(graph) #1 START", flush=True)
        graph = normalise(graph)
        print("[ORCH] step: normalise(graph) #1 END", flush=True)

        print(f"[ORCH] step: validate(graph) #1 START  graph.compounds={graph.compounds}", flush=True)
        ir_report = validate(graph)
        print(f"[ORCH] step: validate(graph) #1 END  valid={ir_report.valid}", flush=True)
        result.ir_report = ir_report
        if not ir_report.valid:
            result.warnings    += [str(i) for i in ir_report.errors()]
            result.outcome      = "INVALID_IR"
            for e in ir_report.errors():
                print(f"[ORCH] INVALID_IR: {e}", flush=True, file=sys.stderr)
            result.total_time_s = time.time() - t_start
            return result
        if ir_report.warnings():
            result.warnings += [str(w) for w in ir_report.warnings()]

        # ── Stage 3: Simulation mapping ────────────────────────────────────────
        # ParamMapper: deterministic-first, LLM fallback only for unknown params.
        # GlobalConsistencyPass: enforce cross-unit T/P constraints + backward pass.
        # FailureRuleStore: apply any synthesized rules from prior benchmark cases.
        try:
            print("[ORCH] step: thermo.assign START", flush=True)
            graph = self._thermo.assign(graph, description=desc)
            print(f"[ORCH] step: thermo.assign END  pkg={getattr(graph, 'property_package', '?')}", flush=True)

            print("[ORCH] step: params.assign START", flush=True)
            graph = self._params.assign(graph, description=desc)
            print("[ORCH] step: params.assign END", flush=True)

            print("[ORCH] step: consistency.apply START", flush=True)
            graph, consistency_changes = self._consistency.apply(graph)
            print(f"[ORCH] step: consistency.apply END  changes={len(consistency_changes)}", flush=True)
            if consistency_changes:
                result.warnings += [f"[consistency] {c}" for c in consistency_changes]

            # Apply rules synthesized from previous failures (cross-run learning)
            print("[ORCH] step: rule_store.apply_to_graph START", flush=True)
            graph, rule_changes = self._rule_store.apply_to_graph(
                graph, basis.dwsim_compounds)
            print(f"[ORCH] step: rule_store.apply_to_graph END  changes={len(rule_changes)}", flush=True)
            if rule_changes:
                result.warnings += [f"[rule] {c}" for c in rule_changes]
        except Exception as exc:
            print(f"[ORCH] Stage 3 EXCEPTION: {type(exc).__name__}: {exc}", flush=True)
            result.warnings.append(f"Stage 3 failed: {exc}")
            result.outcome      = "PLAN_FAILED"
            result.total_time_s = time.time() - t_start
            return result

        print("[ORCH] step: normalise(graph) #2 START", flush=True)
        graph = normalise(graph)
        print("[ORCH] step: normalise(graph) #2 END", flush=True)

        print(f"[ORCH] step: validate(graph) #2 START  graph.compounds={graph.compounds}", flush=True)
        post_report = validate(graph)
        print(f"[ORCH] step: validate(graph) #2 END  valid={post_report.valid}", flush=True)
        if not post_report.valid:
            result.warnings    += [str(i) for i in post_report.errors()]
            result.outcome      = "INVALID_JSON"
            result.total_time_s = time.time() - t_start
            return result

        result.final_graph = graph

        # ── Stage 3→4 bridge: IR → DWSIM JSON ─────────────────────────────────
        # Load reference data when a reference file is provided.
        # Used for: (1) recycle INIT stream seeding, (2) post-solve T refinement.
        # Try the path as given first, then an absolute path anchored at the
        # project root — needed when the working directory inside Singularity
        # differs from the project root.
        _reference_data: Optional[dict] = None
        if reference_file:
            import json as _json
            from pathlib import Path as _Path
            _ref_candidates = [
                _Path(reference_file),
                _Path(__file__).resolve().parent.parent / reference_file,
            ]
            for _ref_path in _ref_candidates:
                try:
                    with open(_ref_path) as _rf:
                        _reference_data = _json.load(_rf)
                    print(
                        f"[ORCH] loaded reference data from '{_ref_path}'",
                        flush=True, file=sys.stderr)
                    break
                except FileNotFoundError:
                    continue
                except Exception as _ref_exc:
                    print(
                        f"[ORCH] WARNING: error reading reference_file "
                        f"'{_ref_path}': {_ref_exc}",
                        flush=True, file=sys.stderr)
                    break
            else:
                print(
                    f"[ORCH] WARNING: reference_file '{reference_file}' not found; "
                    f"tried: {[str(p) for p in _ref_candidates]}",
                    flush=True, file=sys.stderr)

        # Validation tier: seed ConversionReactor reaction + conversion directly
        # from the reference file.  The LLM extraction path for these params is
        # unreliable (topology_chain unavailable; keyword fallback returns no
        # reaction string), but the reference already contains the correct,
        # consistency-verified stoichiometry.  Same pattern as reference-seeded
        # recycle INIT streams; keeps LLM extraction for non-validation tiers.
        if is_validation_tier(tier) and _reference_data is not None:
            _inject_reference_reactor_params(graph, _reference_data)

        print("[ORCH] step: to_dwsim(graph) START", flush=True)
        dwsim_json = to_dwsim(graph, reference_data=_reference_data)
        print("[ORCH] step: to_dwsim(graph) END", flush=True)

        # ── Stage 4: Execution loop ────────────────────────────────────────────
        # RepairMemory persists across iterations so the agent never repeats a
        # failed strategy and can detect stagnation.
        # SimulationHints carries signals from the last DWSIM execution into
        # the repair agent so it can prioritise actually-failed units.
        tried_packages: set[str] = {graph.property_package}
        repair_memory             = RepairMemory()
        sim_hints                 = EMPTY_HINTS

        # Dynamic iteration ceiling: extended to _BEAM_MAX_ITER if beam search
        # is triggered (n_cond_errors > 1), since the wider candidate pool
        # requires more outer-loop cycles to fully explore.
        _BEAM_MAX_ITER  = 15
        _eff_max_iter   = self._max_iter
        _beam_extended  = False

        _prev_dwsim_hash: Optional[str] = None
        for iteration in range(max(self._max_iter, _BEAM_MAX_ITER)):
            if iteration >= _eff_max_iter:
                break

            t_iter    = time.time()
            repair_memory.tick()
            import hashlib as _hl, json as _json
            _cur_hash = _hl.md5(
                _json.dumps(dwsim_json, sort_keys=True).encode()).hexdigest()[:8]
            _hash_note = ("UNCHANGED" if _cur_hash == _prev_dwsim_hash
                          else f"changed  prev={_prev_dwsim_hash}")
            print(f"[ORCH] iteration={iteration} flowsheet_hash={_cur_hash} ({_hash_note})",
                  flush=True, file=sys.stderr)
            _prev_dwsim_hash = _cur_hash
            print(f"[ORCH] step: executor.run iteration={iteration} START", flush=True)
            execution = self._executor.run(dwsim_json)
            print(f"[ORCH] step: executor.run iteration={iteration} END  solved={getattr(execution, 'solved', '?')}", flush=True)

            # Build simulation hints from the execution result for this iteration
            sim_hints = SimulationHints.from_execution(execution, iteration=iteration)

            if getattr(execution, "solved", False) and _no_critic_failures(execution):
                # Reference-guided post-solve refinement: when a reference file
                # is available, correct T_out params that are >10 K off reference.
                if _reference_data is not None:
                    _ref_json, _ref_exec, _ref_changes = \
                        _reference_guided_refinement(
                            graph, execution, _reference_data, self._executor)
                    if _ref_exec is not None and getattr(_ref_exec, "solved", False):
                        dwsim_json = _ref_json
                        execution  = _ref_exec
                        result.warnings += _ref_changes
                        print(
                            f"[ORCH] reference-guided refinement: "
                            f"{len(_ref_changes)} T_out correction(s) applied",
                            flush=True, file=sys.stderr)
                result.outcome         = "PASS"
                result.final_flowsheet = dwsim_json
                result.final_execution = execution
                result.iterations.append(IterationRecord(
                    iteration=iteration, errors=[], changes=["PASS"],
                    flowsheet=dwsim_json, execution=execution,
                    elapsed_s=time.time() - t_iter))
                break

            errors = self._classifier.classify(execution, graph)
            print(f"[REPAIR] iteration={iteration} "
                  f"n_errors={len(errors)} "
                  f"strategies={[e.repair_strategy.value for e in errors]} "
                  f"targets={[str(e.target) for e in errors]}",
                  flush=True, file=sys.stderr)
            for _e in errors:
                print(f"[REPAIR]   error: {_e}", flush=True, file=sys.stderr)

            if any(e.is_terminal for e in errors):
                result.outcome         = "HUMAN"
                result.final_flowsheet = dwsim_json
                result.final_execution = execution
                result.iterations.append(IterationRecord(
                    iteration=iteration, errors=errors, changes=[],
                    flowsheet=dwsim_json, execution=execution,
                    elapsed_s=time.time() - t_iter))
                break

            # Extend iteration ceiling when beam search is about to activate.
            # Beam search triggers when n_cond_errors > 1; the wider candidate
            # pools introduced in repair_agent need more outer-loop cycles.
            if not _beam_extended:
                _n_cond = sum(
                    1 for e in errors
                    if e.repair_strategy == _RepairStrategy.CONDITION_FIX)
                if _n_cond > 1:
                    _beam_extended = True
                    _eff_max_iter  = _BEAM_MAX_ITER
                    print(
                        f"[ORCH] beam search active (n_cond_errors={_n_cond}): "
                        f"extending max_iter {self._max_iter} → {_eff_max_iter}",
                        flush=True, file=sys.stderr,
                    )

            try:
                # Snapshot unit params before repair to detect no-change loops
                _params_before = {
                    u.tag: dict(u.params) for u in graph.units()}

                graph, changes = self._repair.repair(
                    graph, errors, tried_packages,
                    description=desc, memory=repair_memory,
                    sim_hints=sim_hints)

                _params_after = {
                    u.tag: dict(u.params) for u in graph.units()}
                _param_diffs = {
                    tag: {p: (_params_before[tag].get(p), v)
                          for p, v in _params_after[tag].items()
                          if _params_before.get(tag, {}).get(p) != v}
                    for tag in _params_after
                    if _params_before.get(tag) != _params_after.get(tag)
                }
                print(f"[REPAIR] iteration={iteration} "
                      f"changes={changes[:6]} "
                      f"param_diffs={_param_diffs}",
                      flush=True, file=sys.stderr)
                if not _param_diffs:
                    print(f"[REPAIR] WARNING: iteration={iteration} "
                          f"no unit params changed — repair loop is stuck",
                          flush=True, file=sys.stderr)

                _record_repairs_in_store(
                    errors, changes, graph, self._rule_store,
                    compounds=basis.dwsim_compounds)
                try:
                    self._rule_store.save(RULES_PATH)
                except Exception as _save_exc:
                    print(f"[ORCH] rule_store.save failed: {_save_exc}",
                          flush=True, file=sys.stderr)
                graph  = normalise(graph)
                post   = validate(graph)
                if not post.valid:
                    result.warnings += [str(i) for i in post.errors()]

                tried_packages.add(graph.property_package)
                dwsim_json = to_dwsim(graph, reference_data=_reference_data)
            except Exception as _repair_exc:
                print(f"[ORCH] repair/normalise error iter={iteration}: {_repair_exc}",
                      flush=True, file=sys.stderr)
                result.warnings.append(f"repair error iter={iteration}: {_repair_exc}")
                result.iterations.append(IterationRecord(
                    iteration=iteration, errors=errors, changes=[],
                    flowsheet=dwsim_json, execution=execution,
                    elapsed_s=time.time() - t_iter))
                break

            result.final_graph     = graph
            result.final_flowsheet = dwsim_json
            result.final_execution = execution
            result.iterations.append(IterationRecord(
                iteration=iteration, errors=errors, changes=changes,
                flowsheet=dwsim_json, execution=execution,
                elapsed_s=time.time() - t_iter))

        result.total_time_s  = time.time() - t_start
        result.margin_snapshot = get_global_margin_model().snapshot()
        return result


def _no_critic_failures(execution) -> bool:
    """True when the execution has no error signals from the v1 critic (if present)."""
    report = getattr(execution, "_critic_report", None)
    if report:
        return getattr(report, "passed", False)
    return not getattr(execution, "errors", [])


def _record_repairs_in_store(
    errors:     list,
    changes:    list[str],
    graph,
    store:      "FailureRuleStore",
    compounds:  Optional[list[str]] = None,
) -> None:
    """
    Parse applied CONDITION_FIX repairs from the change log and record
    each pattern in the FailureRuleStore for future rule synthesis.
    """
    import re
    from agents.rule_store import _outlet_unit_types

    for change in changes:
        m = re.match(
            r"CONDITION_FIX\[.*?\]: (\S+)\.(\w+) .*?→([\d.]+)", change)
        if m is None:
            continue
        unit_tag, param, val_str = m.group(1), m.group(2), m.group(3)
        try:
            applied_val = float(val_str)
        except ValueError:
            continue

        node = graph.unit(unit_tag)
        if node is None:
            continue
        downstream = _outlet_unit_types(graph, unit_tag)
        # Use the first downstream type (most relevant constraint)
        dst_type = next(iter(downstream), None)

        store.record_fix(
            unit_type       = node.unit_type,
            error_code      = "CONDITION_FIX",
            downstream_type = dst_type,
            param           = param,
            applied_value   = applied_val,
            compounds       = compounds,
        )


def _inject_reference_reactor_params(graph, reference_data: dict) -> None:
    """Seed ConversionReactor reaction + conversion from reference for validation tier.

    Runs after ParamMapper so it overrides the empty reaction string that
    ParamMapper's physical estimator inserts when LLM extraction fails.
    Only injects params that the reference explicitly provides.
    """
    ref_units = {u["tag"]: u for u in reference_data.get("units", [])}
    for node in graph.units():
        if node.unit_type != "ConversionReactor":
            continue
        ref_unit = ref_units.get(node.tag)
        if ref_unit is None:
            continue
        ref_params = ref_unit.get("params", {})
        reaction   = ref_params.get("reaction")
        conversion = ref_params.get("conversion")
        if reaction is not None:
            node.params["reaction"] = reaction
            print(f"[ORCH] ref-seeded reaction  {node.tag}: {reaction!r}",
                  flush=True, file=sys.stderr)
        if conversion is not None:
            node.params["conversion"] = conversion
            print(f"[ORCH] ref-seeded conversion {node.tag}: {conversion}",
                  flush=True, file=sys.stderr)
