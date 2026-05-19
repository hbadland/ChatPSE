"""
Candidate generation and selection (Item 7, extended for publication).

Generates N independent semantic parses of the same description, builds N
FlowsheetGraphs, validates each, then selects the best candidate by the
multi-factor CandidateScore (ir/scoring.py).

Scoring at selection time (Stage 1–2):
  valid_ir, valid_json, unit_appropriateness, separation_feasibility,
  repair_economy, param_completeness, phase_consistency, excess_units_penalty

Thermo_consistency is updated by the orchestrator after ThermoMapper (Stage 3).
Converged is updated after DWSIM execution.

N=3 is the standard configuration for publication experiments.
N=1 is used in the REDUCED_AGENTS ablation (equivalent to single-pass).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ir.graph import FlowsheetGraph, make_node, EdgeIR
from ir.normalise import normalise
from ir.validate import validate, ValidationReport
from ir.scoring import CandidateScore, score_candidate, compute_margin
from agents.stage1.unit_extractor import UnitExtractor, SemanticUnits
from agents.stage1.stream_extractor import StreamExtractor, SemanticTopology
from agents.llm import DEFAULT_MODEL


@dataclass
class Candidate:
    graph:          FlowsheetGraph
    sem_units:      SemanticUnits
    sem_topo:       SemanticTopology
    report:         ValidationReport = field(default_factory=ValidationReport)
    candidate_score: Optional[CandidateScore] = None
    repair_count:   int  = 0

    # Backward-compatible accessors used by some tests/orchestrator
    @property
    def issue_count(self) -> int:
        return len(self.report.issues)

    @property
    def valid_ir(self) -> bool:
        return self.candidate_score.valid_ir >= 1.0 if self.candidate_score else True

    @property
    def valid_json(self) -> bool:
        return self.candidate_score.valid_json >= 1.0 if self.candidate_score else True

    def score(self) -> float:
        """Scalar score for sorting. Wraps CandidateScore.total."""
        return self.candidate_score.total if self.candidate_score else 0.0

    def score_tuple(self) -> tuple:
        """Legacy tuple for backward compatibility."""
        return (self.valid_ir, self.valid_json, -self.repair_count, -self.issue_count)


class CandidateSelector:
    """
    Generates N parse candidates and returns the best by CandidateScore.

    n=1 — single-pass mode (ablation: REDUCED_AGENTS)
    n=3 — standard for publication experiments
    """

    def __init__(self, model: str = DEFAULT_MODEL, n: int = 3) -> None:
        self._model  = model
        self._n      = n
        self._unit_extractor   = UnitExtractor(model)
        self._stream_extractor = StreamExtractor(model)

    def select(
        self,
        description: str,
        compounds:   list[str],
        max_retries: int = 3,
    ) -> Candidate:
        """
        Generate up to self._n candidates, score each, return the best.
        The winning candidate's margin field is set to the gap over second-best.
        """
        candidates: list[Candidate] = []

        for i in range(self._n):
            try:
                cand = self._generate_one(
                    description, compounds, max_retries, candidate_idx=i)
                candidates.append(cand)
            except RuntimeError:
                continue

        if not candidates:
            raise RuntimeError(
                f"CandidateSelector: all {self._n} generation attempts failed")

        # Compute margins and sort
        scores = [c.candidate_score for c in candidates if c.candidate_score]
        compute_margin(scores)
        candidates.sort(key=lambda c: c.score(), reverse=True)
        return candidates[0]

    def select_top_k(
        self,
        description: str,
        compounds:   list[str],
        k:           int = 3,
        max_retries: int = 3,
    ) -> list[Candidate]:
        """Return top-K candidates for multi-candidate repair (Item 6)."""
        candidates: list[Candidate] = []
        for i in range(max(self._n, k)):
            try:
                cand = self._generate_one(
                    description, compounds, max_retries, candidate_idx=i)
                candidates.append(cand)
            except RuntimeError:
                continue
        candidates.sort(key=lambda c: c.score(), reverse=True)
        return candidates[:k]

    def _generate_one(
        self,
        description:   str,
        compounds:     list[str],
        max_retries:   int,
        candidate_idx: int = 0,
    ) -> Candidate:
        sem_units = self._unit_extractor.extract(description, compounds, max_retries)
        sem_topo  = self._stream_extractor.extract(
            description, compounds,
            unit_tags  = [u.tag  for u in sem_units.units],
            unit_roles = {u.tag: u.role for u in sem_units.units},
            max_retries = max_retries,
        )

        graph, repair_count = _assemble_graph(sem_units, sem_topo, compounds)
        report  = validate(graph)
        cscore  = score_candidate(
            graph, report, repair_count, description, candidate_idx)

        return Candidate(
            graph           = graph,
            sem_units       = sem_units,
            sem_topo        = sem_topo,
            report          = report,
            candidate_score = cscore,
            repair_count    = repair_count,
        )


# ── Graph assembly helper ──────────────────────────────────────────────────────

def _assemble_graph(
    units:     SemanticUnits,
    topology:  SemanticTopology,
    compounds: list[str],
) -> tuple[FlowsheetGraph, int]:
    """
    Build a FlowsheetGraph from Stage 1 output; returns (graph, n_insertions).
    n_insertions is the number of Mixer/Splitter nodes added by the normaliser.
    """
    from ir.graph import SUPPORTED_UNIT_TYPES

    graph = FlowsheetGraph()
    graph.compounds = list(compounds)

    for sem_unit in units.units:
        if sem_unit.type not in SUPPORTED_UNIT_TYPES:
            continue
        node = make_node(sem_unit.type, sem_unit.tag, {},
                         metadata={"role": sem_unit.role})
        graph.add_unit(node, strict=False)

    for sem_stream in topology.streams:
        edge = EdgeIR(
            tag         = sem_stream.tag,
            T           = sem_stream.T,
            P           = sem_stream.P,
            flow        = sem_stream.flow,
            composition = dict(sem_stream.composition),
            metadata    = {"is_feed": sem_stream.is_feed},
        )
        graph.add_stream(edge, sem_stream.src, sem_stream.dst,
                         enforce_phase=False)

    before = len(graph.unit_tags())
    graph  = normalise(graph)
    after  = len(graph.unit_tags())

    return graph, max(0, after - before)
