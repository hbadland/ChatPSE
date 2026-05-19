"""
Multi-candidate repair loop (Item 6).

Maintains a beam of top-K candidates through the Stage 4 repair loop.
At each iteration, all K candidates are repaired and re-validated; the
bottom candidates are pruned by CandidateScore, and the loop continues
with the survivors.

Beam search over the repair space reduces the chance of getting stuck
in a local minimum (e.g., a topology that cannot be fixed by THERMO_SWITCH
but a parallel candidate can be fixed by BIP injection).

Typical configuration for publication:
  k=3, max_iterations=6, prune_to=2 (keep top 2 after each round)

Single-candidate mode (k=1) is equivalent to the original RepairAgent loop
and is used as the REDUCED_REPAIR ablation baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ir.graph import FlowsheetGraph
from ir.validate import validate
from ir.scoring import score_candidate, update_thermo, CandidateScore
from ir.types import SimError, RepairStrategy
from agents.stage4.repair_agent import RepairAgent
from agents.candidate_selector import Candidate
from rag.retriever import Retriever


@dataclass
class BeamState:
    """One candidate in the repair beam."""
    candidate:      Candidate
    tried_packages: set[str]  = field(default_factory=set)
    change_log:     list[str] = field(default_factory=list)
    iteration:      int       = 0
    converged:      bool      = False

    def score(self) -> float:
        cs = self.candidate.candidate_score
        return cs.total if cs else 0.0


@dataclass
class MultiRepairResult:
    """Output of MultiCandidateRepairLoop.run()."""
    best:         BeamState
    all_states:   list[BeamState]
    n_iterations: int
    pruned:       int   = 0   # total candidates pruned across all iterations

    @property
    def graph(self) -> FlowsheetGraph:
        return self.best.candidate.graph

    @property
    def change_log(self) -> list[str]:
        return self.best.change_log


class MultiCandidateRepairLoop:
    """
    Beam-search repair loop.

    Parameters
    ----------
    k               : beam width (number of candidates to maintain)
    max_iterations  : maximum repair iterations across the whole beam
    prune_to        : beam width after each pruning step (default: k - 1)
    model           : LLM model for LLMRepair
    retriever       : RAG retriever
    """

    def __init__(
        self,
        k:              int = 3,
        max_iterations: int = 6,
        prune_to:       Optional[int] = None,
        model:          str = "",
        retriever:      Optional[Retriever] = None,
    ) -> None:
        from agents.llm import DEFAULT_MODEL
        self._k         = k
        self._max_iter  = max_iterations
        self._prune_to  = prune_to if prune_to is not None else max(1, k - 1)
        self._repair    = RepairAgent(
            model     = model or DEFAULT_MODEL,
            retriever = retriever or Retriever(),
        )
        self._retriever = retriever or Retriever()

    def run(
        self,
        candidates:  list[Candidate],
        description: str = "",
    ) -> MultiRepairResult:
        """
        Run the multi-candidate repair loop.

        Parameters
        ----------
        candidates : initial beam (from CandidateSelector.select_top_k())
        description: process description, passed to LLMRepair
        """
        beam = [BeamState(candidate=c) for c in candidates[:self._k]]
        total_pruned = 0
        n_iter       = 0

        for iteration in range(self._max_iter):
            n_iter = iteration + 1
            new_beam: list[BeamState] = []

            for state in beam:
                # Get errors from current validation report
                errors = state.candidate.report.sim_errors()
                if not errors:
                    state.converged = True
                    new_beam.append(state)
                    continue

                # Apply repairs
                g, changes = self._repair.repair(
                    state.candidate.graph,
                    errors,
                    tried_packages = state.tried_packages,
                    description    = description,
                )
                state.change_log.extend(changes)
                state.iteration = iteration + 1

                # Track tried packages
                if g.property_package != state.candidate.graph.property_package:
                    state.tried_packages.add(
                        state.candidate.graph.property_package)

                # Re-validate and re-score
                report = validate(g)
                score  = score_candidate(
                    g, report, state.candidate.repair_count,
                    description, state.candidate.candidate_score.candidate_idx
                    if state.candidate.candidate_score else 0)
                update_thermo(score, g)

                state.candidate.graph           = g
                state.candidate.report          = report
                state.candidate.candidate_score = score
                new_beam.append(state)

            # Prune beam: keep top prune_to by score, always keep at least 1
            new_beam.sort(key=lambda s: s.score(), reverse=True)
            if len(new_beam) > self._prune_to:
                pruned = len(new_beam) - self._prune_to
                total_pruned += pruned
                new_beam = new_beam[:self._prune_to]

            beam = new_beam

            # Early exit: top candidate fully valid
            if beam and not beam[0].candidate.report.errors():
                break

        beam.sort(key=lambda s: s.score(), reverse=True)
        return MultiRepairResult(
            best        = beam[0],
            all_states  = beam,
            n_iterations = n_iter,
            pruned      = total_pruned,
        )
