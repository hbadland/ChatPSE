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

import time
from dataclasses import dataclass, field
from typing import Optional

from agents.basis import BasisAgent, BasisResult
from agents.executor import Executor
from agents.llm import DEFAULT_MODEL

from agents.stage1 import UnitExtractor, StreamExtractor
from agents.stage2 import GraphBuilder
from agents.stage3 import ThermoMapper, ParamMapper
from agents.stage4 import ErrorClassifier, ClassifiedError, RepairAgent

from ir import FlowsheetGraph, normalise, validate, to_dwsim
from ir.validate import ValidationReport
from rag.retriever import Retriever


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
        max_iterations: int = 6,
    ):
        self._model      = model
        self._retriever  = Retriever()
        self._max_iter   = max_iterations

        self._basis      = BasisAgent(model=model)
        self._unit_ext   = UnitExtractor(model=model)
        self._stream_ext = StreamExtractor(model=model)
        self._builder    = GraphBuilder()
        self._thermo     = ThermoMapper(model=model, retriever=self._retriever)
        self._params     = ParamMapper(model=model, retriever=self._retriever)
        self._executor   = Executor()
        self._classifier = ErrorClassifier(model=model)
        self._repair     = RepairAgent(model=model, retriever=self._retriever)

    def run(self, description: str) -> PipelineResult:
        t_start = time.time()
        result  = PipelineResult(description=description, outcome="MAX_ITER")

        # ── Stage 0: Basis ─────────────────────────────────────────────────────
        basis = self._basis.identify(description)
        result.basis_result = basis
        if not basis.success:
            result.outcome    = "BASIS_FAILED"
            result.total_time_s = time.time() - t_start
            return result

        desc      = basis.normalised_description
        compounds = basis.dwsim_compounds

        # ── Stage 1: Semantic parsing ──────────────────────────────────────────
        try:
            sem_units = self._unit_ext.extract(desc, compounds)
            sem_topo  = self._stream_ext.extract(
                desc, compounds,
                unit_tags  = [u.tag  for u in sem_units.units],
                unit_roles = {u.tag: u.role for u in sem_units.units},
            )
        except RuntimeError as exc:
            result.warnings.append(f"Stage 1 failed: {exc}")
            result.outcome    = "PLAN_FAILED"
            result.total_time_s = time.time() - t_start
            return result

        # ── Stage 2: IR construction ───────────────────────────────────────────
        graph = self._builder.build(sem_units, sem_topo, compounds)
        graph = normalise(graph)
        ir_report = validate(graph)
        result.ir_report = ir_report
        if not ir_report.valid:
            result.warnings += [str(i) for i in ir_report.errors()]
            result.outcome    = "INVALID_IR"
            result.total_time_s = time.time() - t_start
            return result
        if ir_report.warnings():
            result.warnings += [str(w) for w in ir_report.warnings()]

        # ── Stage 3: Simulation mapping ────────────────────────────────────────
        try:
            graph = self._thermo.assign(graph, description=desc)
            graph = self._params.assign(graph, description=desc)
        except Exception as exc:
            result.warnings.append(f"Stage 3 failed: {exc}")
            result.outcome    = "PLAN_FAILED"
            result.total_time_s = time.time() - t_start
            return result

        graph = normalise(graph)
        post_report = validate(graph)
        if not post_report.valid:
            result.warnings += [str(i) for i in post_report.errors()]
            result.outcome    = "INVALID_JSON"
            result.total_time_s = time.time() - t_start
            return result

        result.final_graph = graph

        # ── Stage 3→4 bridge: IR → DWSIM JSON ─────────────────────────────────
        dwsim_json = to_dwsim(graph)

        # ── Stage 4: Execution loop ────────────────────────────────────────────
        tried_packages: set[str] = {graph.property_package}

        for iteration in range(self._max_iter):
            t_iter    = time.time()
            execution = self._executor.run(dwsim_json)

            if getattr(execution, "solved", False) and _no_critic_failures(execution):
                result.outcome          = "PASS"
                result.final_flowsheet  = dwsim_json
                result.final_execution  = execution
                result.iterations.append(IterationRecord(
                    iteration=iteration, errors=[], changes=["PASS"],
                    flowsheet=dwsim_json, execution=execution,
                    elapsed_s=time.time() - t_iter))
                break

            errors = self._classifier.classify(execution, graph)

            if any(e.is_terminal() for e in errors):
                result.outcome         = "HUMAN"
                result.final_flowsheet = dwsim_json
                result.final_execution = execution
                result.iterations.append(IterationRecord(
                    iteration=iteration, errors=errors, changes=[],
                    flowsheet=dwsim_json, execution=execution,
                    elapsed_s=time.time() - t_iter))
                break

            graph, changes = self._repair.repair(
                graph, errors, tried_packages, description=desc)
            graph  = normalise(graph)
            post   = validate(graph)
            if not post.valid:
                result.warnings += [str(i) for i in post.errors()]

            tried_packages.add(graph.property_package)
            dwsim_json = to_dwsim(graph)

            result.final_graph     = graph
            result.final_flowsheet = dwsim_json
            result.final_execution = execution
            result.iterations.append(IterationRecord(
                iteration=iteration, errors=errors, changes=changes,
                flowsheet=dwsim_json, execution=execution,
                elapsed_s=time.time() - t_iter))

        result.total_time_s = time.time() - t_start
        return result


def _no_critic_failures(execution) -> bool:
    """True when the execution has no error signals from the v1 critic (if present)."""
    report = getattr(execution, "_critic_report", None)
    if report:
        return getattr(report, "passed", False)
    return not getattr(execution, "errors", [])
