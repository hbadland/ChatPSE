"""
agents/graph_pipeline.py — LangGraph scaffold for the v2 flowsheet pipeline.

Phase 1: wraps existing agent calls in a StateGraph that reproduces
         orchestrator_v2.py results identically.  No new logic.

Phase 1 constraint: _topology_node always uses UnitExtractor + StreamExtractor.
TopologyChain (the 4-call LangChain path) is deliberately excluded here so the
checkpoint run is a clean apples-to-apples comparison against v2.  It will be
wired in during Phase 4 once the scaffold is proven equivalent.

Enable via: USE_LANGGRAPH=1 (checked in orchestrator_v2.py)
"""
from __future__ import annotations

import hashlib
import json
import operator
import sys
import time
import traceback
from typing import Annotated, Any, Optional

# ── LangGraph availability ────────────────────────────────────────────────────
try:
    from langgraph.graph import StateGraph, END
    from typing import TypedDict
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    from typing import TypedDict  # still available in stdlib

# ── Agent imports ─────────────────────────────────────────────────────────────
from agents.basis import BasisAgent
from agents.executor import Executor
from agents.llm import DEFAULT_MODEL
from agents.stage1 import UnitExtractor, StreamExtractor
from agents.stage1.unit_extractor import SemanticUnits
from agents.stage1.stream_extractor import SemanticTopology
# TopologyChain is NOT used in Phase 1 — imported only so GraphPipeline.__init__
# can log its availability, matching OrchestratorV2's startup message.
try:
    from agents.stage1.topology_chain import TopologyChain as _TopologyChain
    _TC_AVAILABLE = True
except ImportError:
    _TopologyChain = None  # type: ignore[assignment,misc]
    _TC_AVAILABLE = False
from agents.stage2 import GraphBuilder
from agents.stage3 import ThermoMapper, ParamMapper
from agents.stage4 import ErrorClassifier, RepairAgent
from agents.stage4.repair_agent import RepairMemory
from agents.stage4.sim_hints import SimulationHints, EMPTY_HINTS
from agents.rule_store import FailureRuleStore, RULES_PATH
from ir import FlowsheetGraph, normalise, validate, to_dwsim
from ir.consistency import GlobalConsistencyPass
from ir.margin_model import get_global_margin_model
from ir.types import RepairStrategy as _RepairStrategy
from rag.retriever import Retriever

# Pull shared helpers and types from orchestrator_v2 so we don't duplicate them.
from agents.orchestrator_v2 import (
    _RECYCLE_PHRASES,
    _SUMMARISER_SYSTEM_TIGHT,
    _no_critic_failures,
    _record_repairs_in_store,
    _reference_guided_refinement,
    _resolve_recycle_target,
    _summarise_for_unit_extraction,
    IterationRecord,
    PipelineResult,
)


# ── Pipeline state ────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    # ── Inputs (set once by run()) ────────────────────────────────────────────
    description:    str
    tier:           str
    reference_file: Optional[str]
    max_iterations: int
    t_start:        float

    # ── Stage 0 ───────────────────────────────────────────────────────────────
    basis_result:   Optional[Any]
    norm_desc:      str
    compounds:      list[str]

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    sem_units:      Optional[Any]   # SemanticUnits
    sem_topo:       Optional[Any]   # SemanticTopology

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    ir_graph:       Optional[Any]   # FlowsheetGraph
    ir_report:      Optional[Any]   # ValidationReport

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    dwsim_json:     Optional[dict]
    reference_data: Optional[dict]
    tried_packages: list[str]

    # ── Stage 4 loop ─────────────────────────────────────────────────────────
    iteration:      int
    eff_max_iter:   int
    beam_extended:  bool
    repair_memory:  Optional[Any]   # RepairMemory
    sim_hints:      Optional[Any]   # SimulationHints
    execution:      Optional[Any]   # ExecutionResult
    errors:         list[Any]       # list[ClassifiedError]
    prev_hash:      Optional[str]

    # ── Accumulating across nodes (operator.add reducer) ─────────────────────
    warnings:       Annotated[list[str], operator.add]
    iterations_log: Annotated[list[Any], operator.add]

    # ── Routing / output ─────────────────────────────────────────────────────
    outcome:        str


# ── Routing functions (module-level, stateless) ───────────────────────────────

def _route_basis(state: PipelineState) -> str:
    return END if state["outcome"] == "BASIS_FAILED" else "topology"


def _route_stage1(state: PipelineState) -> str:
    return END if state["outcome"] == "PLAN_FAILED" else "build"


def _route_validate(state: PipelineState) -> str:
    return END if state["outcome"] == "INVALID_IR" else "thermo"


def _route_thermo(state: PipelineState) -> str:
    return END if state["outcome"] in ("PLAN_FAILED", "INVALID_JSON") else "execute"


def _route_execute(state: PipelineState) -> str:
    return END if state["outcome"] in ("PASS", "HUMAN", "MAX_ITER") else "repair"


# ── Pipeline class ────────────────────────────────────────────────────────────

class GraphPipeline:
    """
    LangGraph-based flowsheet pipeline.  Identical results to OrchestratorV2;
    only the control flow is expressed as a StateGraph.
    """

    def __init__(
        self,
        model:          str = DEFAULT_MODEL,
        max_iterations: int = 10,
        rule_store:     Optional[FailureRuleStore] = None,
    ):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "langgraph is not installed — pip install langgraph")

        self._model    = model
        self._max_iter = max_iterations

        retriever = Retriever()

        self._basis       = BasisAgent(model=model)
        self._unit_ext    = UnitExtractor(model=model)
        self._stream_ext  = StreamExtractor(model=model)
        self._builder     = GraphBuilder()

        # Log TopologyChain availability (matches OrchestratorV2 startup messages)
        # but do NOT instantiate it — Phase 1 uses UnitExtractor + StreamExtractor only.
        if not _TC_AVAILABLE:
            print("[GP] LangChain not installed (TopologyChain excluded from Phase 1 anyway)",
                  flush=True)

        self._thermo      = ThermoMapper(model=model, retriever=retriever)
        self._params      = ParamMapper(model=model, retriever=retriever)
        self._consistency = GlobalConsistencyPass()
        self._executor    = Executor()
        self._classifier  = ErrorClassifier(model=model)
        self._repair      = RepairAgent(model=model, retriever=retriever)

        if rule_store is not None:
            self._rule_store = rule_store
        else:
            self._rule_store = FailureRuleStore()
            self._rule_store.load(RULES_PATH)
            if self._rule_store.num_patterns() > 0:
                print(f"[GP] rule_store loaded: {self._rule_store.num_patterns()} patterns",
                      flush=True)

        self._app = self._build_graph()

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self):
        g = StateGraph(PipelineState)

        g.add_node("basis",    self._basis_node)
        g.add_node("topology", self._topology_node)
        g.add_node("build",    self._build_node)
        g.add_node("validate", self._validate_node)
        g.add_node("thermo",   self._thermo_node)
        g.add_node("execute",  self._execute_node)
        g.add_node("repair",   self._repair_node)

        g.set_entry_point("basis")

        g.add_conditional_edges("basis",    _route_basis,    {"topology": "topology", END: END})
        g.add_conditional_edges("topology", _route_stage1,   {"build": "build",       END: END})
        g.add_edge("build", "validate")
        g.add_conditional_edges("validate", _route_validate, {"thermo": "thermo",     END: END})
        g.add_conditional_edges("thermo",   _route_thermo,   {"execute": "execute",   END: END})
        g.add_conditional_edges("execute",  _route_execute,  {"repair": "repair",     END: END})
        g.add_edge("repair", "execute")

        return g.compile()

    # ── Public interface ──────────────────────────────────────────────────────

    def run(
        self,
        description:    str,
        reference_file: Optional[str] = None,
        tier:           str = "standard",
    ) -> PipelineResult:
        initial: PipelineState = {
            "description":    description,
            "tier":           tier,
            "reference_file": reference_file,
            "max_iterations": self._max_iter,
            "t_start":        time.time(),
            "basis_result":   None,
            "norm_desc":      "",
            "compounds":      [],
            "sem_units":      None,
            "sem_topo":       None,
            "ir_graph":       None,
            "ir_report":      None,
            "dwsim_json":     None,
            "reference_data": None,
            "tried_packages": [],
            "iteration":      0,
            "eff_max_iter":   self._max_iter,
            "beam_extended":  False,
            "repair_memory":  None,
            "sim_hints":      None,
            "execution":      None,
            "errors":         [],
            "prev_hash":      None,
            "warnings":       [],
            "iterations_log": [],
            "outcome":        "MAX_ITER",
        }
        final = self._app.invoke(initial)
        return self._to_result(final)

    def _to_result(self, state: PipelineState) -> PipelineResult:
        result = PipelineResult(
            description  = state["description"],
            outcome      = state["outcome"],
            basis_result = state.get("basis_result"),
            ir_report    = state.get("ir_report"),
            warnings     = list(state.get("warnings", [])),
            total_time_s = time.time() - state["t_start"],
        )
        result.iterations      = list(state.get("iterations_log", []))
        result.final_graph     = state.get("ir_graph")
        result.final_flowsheet = state.get("dwsim_json")
        result.final_execution = state.get("execution")
        result.margin_snapshot = get_global_margin_model().snapshot()
        return result

    # ── Node implementations ──────────────────────────────────────────────────

    def _basis_node(self, state: PipelineState) -> dict:
        print("[GP] step: basis.identify START", flush=True)
        basis = self._basis.identify(state["description"])
        print(f"[GP] step: basis.identify END  success={basis.success}", flush=True)
        if not basis.success:
            return {"basis_result": basis, "outcome": "BASIS_FAILED"}
        print(f"[GP] compounds={basis.dwsim_compounds}", flush=True)
        return {
            "basis_result": basis,
            "norm_desc":    basis.normalised_description,
            "compounds":    list(basis.dwsim_compounds),
        }

    def _topology_node(self, state: PipelineState) -> dict:
        desc      = state["norm_desc"]
        compounds = state["compounds"]
        tier      = state["tier"]
        basis     = state["basis_result"]
        new_warns: list[str] = []

        # ── Optional summarisation (mirrors orchestrator_v2 exactly) ─────────
        _n_words = len(desc.split())
        if _n_words > 200 and tier != "validation":
            print(f"[GP] description length={_n_words} words — summarising for UnitExtractor",
                  flush=True, file=sys.stderr)
            try:
                desc_for_units = _summarise_for_unit_extraction(desc, self._model)
                print(f"[GP] unit summary ({len(desc_for_units.split())} words): "
                      f"{desc_for_units[:150]!r}",
                      flush=True, file=sys.stderr)
                new_warns.append(
                    f"[summary] description condensed "
                    f"({_n_words}→{len(desc_for_units.split())} words) for UnitExtractor")
                if len(desc_for_units.split()) > 150:
                    _first_words = len(desc_for_units.split())
                    print(f"[GP] first summary still {_first_words} words — "
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
                            new_warns.append(
                                f"[summary2] second-pass condensed to "
                                f"{len(desc_for_units.split())} words")
                    except Exception as _sum2_exc:
                        print(f"[GP] second-pass summariser failed ({_sum2_exc})",
                              flush=True, file=sys.stderr)
            except Exception as _sum_exc:
                print(f"[GP] summariser failed ({_sum_exc}) — using full description",
                      flush=True, file=sys.stderr)
                desc_for_units = desc
        else:
            if tier == "validation" and _n_words > 200:
                print(f"[GP] validation tier: skipping summarisation ({_n_words} words)",
                      flush=True, file=sys.stderr)
            desc_for_units = desc

        # ── Stage 1 extraction (Phase 1: always UnitExtractor + StreamExtractor) ─
        # TopologyChain is excluded from Phase 1 to keep the checkpoint a clean
        # reproduction of v2.  It will be wired in during Phase 4.
        try:
            print("[GP] step: unit_ext.extract START", flush=True)
            sem_units = self._unit_ext.extract(desc_for_units, compounds, tier=tier)
            print(f"[GP] step: unit_ext.extract END  "
                  f"units={[u.tag for u in sem_units.units]}", flush=True)

            print("[GP] step: stream_ext.extract START", flush=True)
            sem_topo = self._stream_ext.extract(
                desc, compounds,
                unit_tags              = [u.tag  for u in sem_units.units],
                unit_roles             = {u.tag: u.role for u in sem_units.units},
                concentration_hints    = basis.concentration_hints or [],
                suggested_compositions = basis.suggested_compositions or {},
            )
            print("[GP] step: stream_ext.extract END", flush=True)

            # ── Recycle guard 1: target-validity + fuzzy resolution ───────────
            _unit_tag_set = {u.tag for u in sem_units.units}
            for _s in sem_topo.streams:
                if not getattr(_s, "is_recycle", False):
                    continue
                if _s.recycle_target and _s.recycle_target in _unit_tag_set:
                    print(f"[GP] recycle detected: stream '{_s.tag}' → {_s.recycle_target}",
                          flush=True, file=sys.stderr)
                    continue
                _resolved = _resolve_recycle_target(
                    _s.recycle_target or "", sem_units.units)
                if _resolved:
                    print(f"[GP] recycle guard: resolved target "
                          f"{_s.recycle_target!r} → '{_resolved}' for stream '{_s.tag}'",
                          flush=True, file=sys.stderr)
                    new_warns.append(
                        f"[recycle] '{_s.tag}' target resolved: "
                        f"{_s.recycle_target!r} → '{_resolved}'")
                    _s.recycle_target = _resolved
                else:
                    print(f"[GP] recycle guard: cleared unresolvable recycle on "
                          f"'{_s.tag}' (target={_s.recycle_target!r})",
                          flush=True, file=sys.stderr)
                    new_warns.append(
                        f"[recycle] '{_s.tag}' recycle_target={_s.recycle_target!r} "
                        "unresolvable — cleared to avoid false positive")
                    _s.is_recycle     = False
                    _s.recycle_target = None

            # ── Recycle guard 2: multi-recycle deduplication ─────────────────
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
                    return (0 if t_idx < _src_idx else 1, t_idx)
                _keep = min(_candidates, key=_target_rank)
                for _s in _candidates:
                    if _s is not _keep:
                        print(f"[GP] multi-recycle guard: cleared extra recycle on "
                              f"'{_s.tag}' from src='{_src_tag}'; "
                              f"keeping '{_keep.tag}' → {_keep.recycle_target!r}",
                              flush=True, file=sys.stderr)
                        new_warns.append(
                            f"[recycle] '{_s.tag}' cleared — multiple recycles from "
                            f"'{_src_tag}', kept '{_keep.tag}' → '{_keep.recycle_target}'")
                        _s.is_recycle     = False
                        _s.recycle_target = None

            # ── Recycle guard 3: phrase guard ─────────────────────────────────
            _desc_lower = desc.lower()
            for _s in sem_topo.streams:
                if getattr(_s, "is_recycle", False):
                    if not any(p in _desc_lower for p in _RECYCLE_PHRASES):
                        print(f"[GP] phrase guard: suppressed false-positive recycle "
                              f"on stream '{_s.tag}'",
                              flush=True, file=sys.stderr)
                        new_warns.append(
                            f"[recycle] '{_s.tag}' is_recycle=True but no trigger "
                            "phrase in description — cleared to avoid false positive")
                        _s.is_recycle     = False
                        _s.recycle_target = None

        except RuntimeError as exc:
            print(f"[GP] Stage 1 EXCEPTION: {exc}", flush=True)
            return {"outcome": "PLAN_FAILED", "warnings": [f"Stage 1 failed: {exc}"]}

        return {
            "sem_units": sem_units,
            "sem_topo":  sem_topo,
            "warnings":  new_warns,
        }

    def _build_node(self, state: PipelineState) -> dict:
        sem_units = state["sem_units"]
        sem_topo  = state["sem_topo"]
        compounds = state["compounds"]
        new_warns: list[str] = []

        print("[GP] step: builder.build START", flush=True)
        graph = self._builder.build(sem_units, sem_topo, compounds)
        print("[GP] step: builder.build END", flush=True)

        # Compound reconciliation: augment graph.compounds with any found in
        # stream compositions that BasisAgent missed.
        _known_lower = {c.lower() for c in graph.compounds}
        for _stream in graph.streams():
            for _name in _stream.composition:
                if _name.lower() not in _known_lower:
                    graph.compounds.append(_name)
                    _known_lower.add(_name.lower())
                    new_warns.append(
                        f"[compounds] '{_name}' found in feed composition "
                        f"but missing from basis list — added automatically")
        if graph.compounds != list(compounds):
            print(f"[GP] compounds reconciled: {graph.compounds}", flush=True)

        # Back-fill: every stream with composition data must have all compounds.
        for _stream in graph.streams():
            if _stream.composition:
                for _name in graph.compounds:
                    if _name not in _stream.composition:
                        _stream.composition[_name] = 0.0

        print("[GP] step: normalise(graph) #1 START", flush=True)
        graph = normalise(graph)
        print("[GP] step: normalise(graph) #1 END", flush=True)

        return {"ir_graph": graph, "warnings": new_warns}

    def _validate_node(self, state: PipelineState) -> dict:
        graph = state["ir_graph"]
        print(f"[GP] step: validate(graph) #1 START  "
              f"graph.compounds={graph.compounds}", flush=True)
        ir_report = validate(graph)
        print(f"[GP] step: validate(graph) #1 END  valid={ir_report.valid}", flush=True)

        if not ir_report.valid:
            new_warns = [str(e) for e in ir_report.errors()]
            for e in ir_report.errors():
                print(f"[GP] INVALID_IR: {e}", flush=True, file=sys.stderr)
            return {"ir_report": ir_report, "outcome": "INVALID_IR", "warnings": new_warns}

        new_warns = [str(w) for w in ir_report.warnings()]
        return {"ir_report": ir_report, "warnings": new_warns}

    def _thermo_node(self, state: PipelineState) -> dict:
        graph     = state["ir_graph"]
        desc      = state["norm_desc"]
        compounds = state["compounds"]
        new_warns: list[str] = []

        try:
            print("[GP] step: thermo.assign START", flush=True)
            graph = self._thermo.assign(graph, description=desc)
            print(f"[GP] step: thermo.assign END  "
                  f"pkg={getattr(graph, 'property_package', '?')}", flush=True)

            print("[GP] step: params.assign START", flush=True)
            graph = self._params.assign(graph, description=desc)
            print("[GP] step: params.assign END", flush=True)

            print("[GP] step: consistency.apply START", flush=True)
            graph, consistency_changes = self._consistency.apply(graph)
            print(f"[GP] step: consistency.apply END  "
                  f"changes={len(consistency_changes)}", flush=True)
            new_warns += [f"[consistency] {c}" for c in consistency_changes]

            print("[GP] step: rule_store.apply_to_graph START", flush=True)
            graph, rule_changes = self._rule_store.apply_to_graph(graph, compounds)
            print(f"[GP] step: rule_store.apply_to_graph END  "
                  f"changes={len(rule_changes)}", flush=True)
            new_warns += [f"[rule] {c}" for c in rule_changes]

        except Exception as exc:
            print(f"[GP] Stage 3 EXCEPTION: {type(exc).__name__}: {exc}", flush=True)
            return {"outcome": "PLAN_FAILED", "warnings": [f"Stage 3 failed: {exc}"]}

        print("[GP] step: normalise(graph) #2 START", flush=True)
        graph = normalise(graph)
        print("[GP] step: normalise(graph) #2 END", flush=True)

        print(f"[GP] step: validate(graph) #2 START  "
              f"graph.compounds={graph.compounds}", flush=True)
        post_report = validate(graph)
        print(f"[GP] step: validate(graph) #2 END  valid={post_report.valid}", flush=True)
        if not post_report.valid:
            new_warns += [str(e) for e in post_report.errors()]
            return {"ir_graph": graph, "outcome": "INVALID_JSON", "warnings": new_warns}

        # Load reference data (Stage 3→4 bridge)
        reference_data: Optional[dict] = None
        reference_file = state["reference_file"]
        if reference_file:
            import json as _json
            from pathlib import Path as _Path
            _candidates = [
                _Path(reference_file),
                _Path(__file__).resolve().parent.parent / reference_file,
            ]
            for _p in _candidates:
                try:
                    with open(_p) as _rf:
                        reference_data = _json.load(_rf)
                    print(f"[GP] loaded reference data from '{_p}'",
                          flush=True, file=sys.stderr)
                    break
                except FileNotFoundError:
                    continue
                except Exception as _ref_exc:
                    print(f"[GP] WARNING: error reading reference_file: {_ref_exc}",
                          flush=True, file=sys.stderr)
                    break
            else:
                print(f"[GP] WARNING: reference_file '{reference_file}' not found",
                      flush=True, file=sys.stderr)

        print("[GP] step: to_dwsim(graph) START", flush=True)
        dwsim_json = to_dwsim(graph, reference_data=reference_data)
        print("[GP] step: to_dwsim(graph) END", flush=True)

        return {
            "ir_graph":       graph,
            "dwsim_json":     dwsim_json,
            "reference_data": reference_data,
            "tried_packages": [graph.property_package],
            "warnings":       new_warns,
        }

    def _execute_node(self, state: PipelineState) -> dict:
        iteration    = state["iteration"]
        eff_max_iter = state["eff_max_iter"]

        # Iteration ceiling check (mirrors the for-loop break in orchestrator_v2)
        if iteration >= eff_max_iter:
            return {"outcome": "MAX_ITER"}

        # RepairMemory: init on first call, carry across iterations
        repair_memory: RepairMemory = state["repair_memory"] or RepairMemory()
        repair_memory.tick()

        # Hash tracking for loop-detection logging
        _cur_hash = hashlib.md5(
            json.dumps(state["dwsim_json"], sort_keys=True).encode()
        ).hexdigest()[:8]
        _prev = state["prev_hash"]
        _note = "UNCHANGED" if _cur_hash == _prev else f"changed  prev={_prev}"
        print(f"[GP] iteration={iteration} flowsheet_hash={_cur_hash} ({_note})",
              flush=True, file=sys.stderr)

        print(f"[GP] step: executor.run iteration={iteration} START", flush=True)
        execution = self._executor.run(state["dwsim_json"])
        print(f"[GP] step: executor.run iteration={iteration} END  "
              f"solved={getattr(execution, 'solved', '?')}", flush=True)

        sim_hints = SimulationHints.from_execution(execution, iteration=iteration)

        # ── Solved path ───────────────────────────────────────────────────────
        if getattr(execution, "solved", False) and _no_critic_failures(execution):
            dwsim_json = state["dwsim_json"]
            ref_data   = state["reference_data"]
            new_warns: list[str] = []
            if ref_data is not None:
                _ref_json, _ref_exec, _ref_changes = _reference_guided_refinement(
                    state["ir_graph"], execution, ref_data, self._executor)
                if _ref_exec is not None and getattr(_ref_exec, "solved", False):
                    dwsim_json = _ref_json
                    execution  = _ref_exec
                    new_warns  = list(_ref_changes)
                    print(f"[GP] reference-guided refinement: "
                          f"{len(_ref_changes)} correction(s)",
                          flush=True, file=sys.stderr)

            iter_record = IterationRecord(
                iteration=iteration, errors=[], changes=["PASS"],
                flowsheet=dwsim_json, execution=execution, elapsed_s=0.0)
            return {
                "outcome":        "PASS",
                "execution":      execution,
                "dwsim_json":     dwsim_json,
                "repair_memory":  repair_memory,
                "sim_hints":      sim_hints,
                "prev_hash":      _cur_hash,
                "warnings":       new_warns,
                "iterations_log": [iter_record],
            }

        # ── Classify errors ───────────────────────────────────────────────────
        errors = self._classifier.classify(execution, state["ir_graph"])
        print(f"[REPAIR] iteration={iteration} "
              f"n_errors={len(errors)} "
              f"strategies={[e.repair_strategy.value for e in errors]} "
              f"targets={[str(e.target) for e in errors]}",
              flush=True, file=sys.stderr)
        for _e in errors:
            print(f"[REPAIR]   error: {_e}", flush=True, file=sys.stderr)

        # ── Terminal path ─────────────────────────────────────────────────────
        if any(e.is_terminal for e in errors):
            iter_record = IterationRecord(
                iteration=iteration, errors=errors, changes=[],
                flowsheet=state["dwsim_json"], execution=execution, elapsed_s=0.0)
            return {
                "outcome":        "HUMAN",
                "execution":      execution,
                "errors":         errors,
                "repair_memory":  repair_memory,
                "sim_hints":      sim_hints,
                "prev_hash":      _cur_hash,
                "iterations_log": [iter_record],
            }

        # ── Beam search extension ─────────────────────────────────────────────
        beam_extended   = state["beam_extended"]
        eff_max_iter_new = eff_max_iter
        _BEAM_MAX_ITER  = 15
        if not beam_extended:
            n_cond = sum(
                1 for e in errors
                if e.repair_strategy == _RepairStrategy.CONDITION_FIX)
            if n_cond > 1:
                beam_extended    = True
                eff_max_iter_new = _BEAM_MAX_ITER
                print(f"[GP] beam search active (n_cond_errors={n_cond}): "
                      f"extending max_iter {eff_max_iter} → {_BEAM_MAX_ITER}",
                      flush=True, file=sys.stderr)

        # Continue to repair_node
        return {
            "execution":     execution,
            "errors":        errors,
            "repair_memory": repair_memory,
            "sim_hints":     sim_hints,
            "prev_hash":     _cur_hash,
            "eff_max_iter":  eff_max_iter_new,
            "beam_extended": beam_extended,
            "outcome":       "CONTINUE",
        }

    def _repair_node(self, state: PipelineState) -> dict:
        iteration     = state["iteration"]
        graph         = state["ir_graph"]
        errors        = state["errors"]
        tried_pkgs    = list(state["tried_packages"])
        repair_memory = state["repair_memory"]
        sim_hints     = state["sim_hints"] or EMPTY_HINTS
        execution     = state["execution"]
        ref_data      = state["reference_data"]
        new_warns: list[str] = []

        try:
            _params_before = {u.tag: dict(u.params) for u in graph.units()}

            graph, changes = self._repair.repair(
                graph, errors, tried_pkgs,
                description=state["norm_desc"],
                memory=repair_memory,
                sim_hints=sim_hints,
            )

            _params_after = {u.tag: dict(u.params) for u in graph.units()}
            _param_diffs  = {
                tag: {p: (_params_before[tag].get(p), v)
                      for p, v in _params_after[tag].items()
                      if _params_before.get(tag, {}).get(p) != v}
                for tag in _params_after
                if _params_before.get(tag) != _params_after.get(tag)
            }
            print(f"[REPAIR] iteration={iteration} "
                  f"changes={changes[:6]} param_diffs={_param_diffs}",
                  flush=True, file=sys.stderr)
            if not _param_diffs:
                print(f"[REPAIR] WARNING: iteration={iteration} "
                      "no unit params changed — repair loop is stuck",
                      flush=True, file=sys.stderr)

            _record_repairs_in_store(
                errors, changes, graph, self._rule_store,
                compounds=state["compounds"])
            try:
                self._rule_store.save(RULES_PATH)
            except Exception as _save_exc:
                print(f"[GP] rule_store.save failed: {_save_exc}",
                      flush=True, file=sys.stderr)

            graph = normalise(graph)
            post  = validate(graph)
            if not post.valid:
                new_warns += [str(e) for e in post.errors()]

            if graph.property_package not in tried_pkgs:
                tried_pkgs = tried_pkgs + [graph.property_package]

            dwsim_json = to_dwsim(graph, reference_data=ref_data)

            iter_record = IterationRecord(
                iteration=iteration, errors=errors, changes=changes,
                flowsheet=dwsim_json, execution=execution, elapsed_s=0.0)

            return {
                "ir_graph":       graph,
                "dwsim_json":     dwsim_json,
                "tried_packages": tried_pkgs,
                "iteration":      iteration + 1,
                "warnings":       new_warns,
                "iterations_log": [iter_record],
                "outcome":        "CONTINUE",
            }

        except Exception as _repair_exc:
            print(f"[GP] repair/normalise error iter={iteration}: {_repair_exc}",
                  flush=True, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            iter_record = IterationRecord(
                iteration=iteration, errors=errors, changes=[],
                flowsheet=state["dwsim_json"], execution=execution, elapsed_s=0.0)
            # Force MAX_ITER to exit the loop on the next execute_node call by
            # setting iteration to eff_max_iter — mirrors the orchestrator's break.
            return {
                "iteration":      state["eff_max_iter"],
                "warnings":       [f"repair error iter={iteration}: {_repair_exc}"],
                "iterations_log": [iter_record],
                "outcome":        "CONTINUE",
            }
