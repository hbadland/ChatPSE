"""
Candidate generation and selection (Item 7).

Generates N independent semantic parses of the same description, builds N
FlowsheetGraphs, validates each, then selects the best candidate by a
lexicographic score:

  (valid_ir, valid_json, -repair_count, -issue_count)

highest first (True > False, fewer repairs/issues wins).

The N-best approach costs N×(Stage 1 LLM tokens) but yields a much more
reliable graph entering Stage 3, particularly on small models that
occasionally misidentify unit types or miss a stream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ir.graph import FlowsheetGraph, make_node, EdgeIR
from ir.normalise import normalise
from ir.validate import validate
from agents.stage1.unit_extractor import UnitExtractor, SemanticUnits
from agents.stage1.stream_extractor import StreamExtractor, SemanticTopology
from agents.llm import DEFAULT_MODEL


@dataclass
class Candidate:
    graph:        FlowsheetGraph
    sem_units:    SemanticUnits
    sem_topo:     SemanticTopology
    issue_count:  int = 0
    repair_count: int = 0    # normaliser insertions
    valid_ir:     bool = True
    valid_json:   bool = True

    def score(self) -> tuple:
        return (self.valid_ir, self.valid_json, -self.repair_count, -self.issue_count)


class CandidateSelector:
    """
    Generates N parse candidates and returns the best one.

    n=1 — identical to single-pass (default in ablation mode)
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
        Generate up to self._n candidates and return the best by score.
        Falls back to the first valid candidate if ranking produces a tie.
        """
        candidates: list[Candidate] = []

        for i in range(self._n):
            try:
                cand = self._generate_one(description, compounds, max_retries)
                candidates.append(cand)
            except RuntimeError:
                continue

        if not candidates:
            raise RuntimeError(
                f"CandidateSelector: all {self._n} generation attempts failed")

        candidates.sort(key=lambda c: c.score(), reverse=True)
        return candidates[0]

    def _generate_one(
        self,
        description: str,
        compounds:   list[str],
        max_retries: int,
    ) -> Candidate:
        sem_units = self._unit_extractor.extract(description, compounds, max_retries)
        sem_topo  = self._stream_extractor.extract(
            description, compounds,
            unit_tags  = [u.tag  for u in sem_units.units],
            unit_roles = {u.tag: u.role for u in sem_units.units},
            max_retries = max_retries,
        )

        graph, repair_count = _assemble_graph(sem_units, sem_topo, compounds)
        report = validate(graph)

        return Candidate(
            graph        = graph,
            sem_units    = sem_units,
            sem_topo     = sem_topo,
            issue_count  = len(report.issues),
            repair_count = repair_count,
            valid_ir     = all(
                i.level != "SCHEMA" for i in report.issues),
            valid_json   = True,
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
        unit_type = sem_unit.type
        if unit_type not in SUPPORTED_UNIT_TYPES:
            continue
        node = make_node(unit_type, sem_unit.tag, {},
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
        try:
            graph.add_stream(edge, sem_stream.src, sem_stream.dst,
                             enforce_phase=False)
        except ValueError:
            graph.add_stream(edge, sem_stream.src, sem_stream.dst,
                             enforce_phase=False)

    before = len(graph.unit_tags())
    graph  = normalise(graph)
    after  = len(graph.unit_tags())

    return graph, max(0, after - before)
