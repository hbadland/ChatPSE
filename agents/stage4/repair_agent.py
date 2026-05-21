"""
Search-based RepairAgent (v4).

Architecture — four layers:

  RepairMemory       — tracks per-target what was tried, when, and whether it worked.
                       Now includes credit assignment (Item 9): tracks how much each
                       fix reduced the error count for source-quality scoring.

  PhysicsCandidates  — generates candidates directly from violated constraints (Item 2):
                       directional fixes based on sim_hints, downstream unit type,
                       and bubble point relationships.  LLM is a fallback only.

  MarginModel        — provides learned margins (Item 4) for physical offsets.
                       Falls back to hard-coded defaults when data is scarce.

  CandidateRanker    — extended score: validation errors (primary) + warnings
                       (secondary) + magnitude penalty + credit-weighted adaptive
                       penalty (tiebreaker prefers historically successful sources).

Safety guarantees:
  - LLM output is clamped to the unit-specific physical range before use
  - Values present in tried_vals are skipped by all generators
  - Only the ONE flagged param is written back (no OVERCORRECTION_RISK)
  - HUMAN errors are logged and returned unchanged
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from ir.graph import FlowsheetGraph, HeaterNode, CoolerNode, SeparatorNode, PumpNode
from ir.thermo_estimation import bubble_point_K
from ir.margin_model import get_global_margin_model
from ir.repair import DeterministicRepair
from ir.validate import validate
from ir.types import RepairStrategy, SimError, ErrorType
from agents.llm import chat, DEFAULT_MODEL, retry_temperature
from agents.stage4.sim_hints import SimulationHints, EMPTY_HINTS
from rag.retriever import Retriever

_det_repair = DeterministicRepair()

# Minimum spacing between temperature candidates (K) and pressure candidates (ratio)
_MIN_T_SPACING = 10.0
_MIN_P_RATIO   = 1.25

# RepairMemory limits
_MAX_HISTORY_PER_TAG = 20   # max records per target tag (oldest trimmed)
_CREDIT_WINDOW       = 10   # recent records used for credit_score()
_OSC_ROUND_DP        = 1    # decimal places for oscillation value comparison


# ── Structured repair action ───────────────────────────────────────────────────

@dataclass
class RepairAction:
    """A proposed single-parameter fix, fully typed."""
    param:     str        # e.g. "T_out", "P_out"
    new_value: Any        # the proposed value
    source:    str        # "physics" | "deterministic" | "heuristic" | "llm"
    rationale: str


# ── Repair candidate ───────────────────────────────────────────────────────────

@dataclass
class RepairCandidate:
    graph:            FlowsheetGraph
    action:           RepairAction
    n_errors:         int   = 0
    n_warnings:       int   = 0
    magnitude:        float = 0.0
    adaptive_penalty: float = 0.0

    @property
    def score(self) -> float:
        return (self.n_errors * 100
                + self.n_warnings
                + self.magnitude * 0.5
                + self.adaptive_penalty)


# ── Iteration memory with credit tracking ─────────────────────────────────────

@dataclass
class _AttemptRecord:
    iteration:       int
    strategy:        str
    param:           str
    value:           Any
    source:          str
    n_errors_before: int   # IR errors before this attempt (Item 9)
    n_errors_after:  int

    @property
    def error_reduction(self) -> int:
        return max(0, self.n_errors_before - self.n_errors_after)

    @property
    def credit(self) -> float:
        """Fractional improvement: 0.0 = no change, 1.0 = fixed all errors."""
        if self.n_errors_before == 0:
            return 0.0
        return self.error_reduction / self.n_errors_before


class RepairMemory:
    """
    Tracks repair attempts across iterations.  One instance lives in the
    orchestrator and is passed to each repair() call.
    """

    def __init__(self) -> None:
        self._log: dict[str, list[_AttemptRecord]] = defaultdict(list)
        self._iteration: int = 0

    def tick(self) -> None:
        self._iteration += 1

    def record(
        self,
        target_tag:      str,
        strategy:        str,
        param:           str,
        value:           Any,
        source:          str,
        n_errors_after:  int,
        n_errors_before: int = 0,
    ) -> None:
        self._log[target_tag].append(_AttemptRecord(
            iteration       = self._iteration,
            strategy        = strategy,
            param           = param,
            value           = value,
            source          = source,
            n_errors_before = n_errors_before,
            n_errors_after  = n_errors_after,
        ))
        if len(self._log[target_tag]) > _MAX_HISTORY_PER_TAG:
            self._log[target_tag] = self._log[target_tag][-_MAX_HISTORY_PER_TAG:]

    def tried_values(self, target_tag: str, param: str) -> list[Any]:
        return [a.value for a in self._log[target_tag] if a.param == param]

    def tried_strategies(self, target_tag: str) -> set[str]:
        return {a.strategy for a in self._log[target_tag]}

    def recent_error_counts(self, target_tag: str, last_n: int = 3) -> list[int]:
        return [a.n_errors_after for a in self._log[target_tag][-last_n:]]

    def is_stagnating(self, target_tag: str) -> bool:
        counts = self.recent_error_counts(target_tag)
        if len(counts) < 2:
            return False
        return all(counts[i] >= counts[i - 1] for i in range(1, len(counts)))

    def uncertainty_score(self, target_tag: str, param: str) -> float:
        """Range-based uncertainty: 2 × (explored_range / |mean|)."""
        vals = [a.value for a in self._log[target_tag]
                if a.param == param and isinstance(a.value, (int, float))]
        if len(vals) < 2:
            return 0.0
        val_range = max(vals) - min(vals)
        mean      = sum(vals) / len(vals)
        return min(1.0, 2.0 * val_range / max(abs(mean), 1.0))

    def source_successes(self, target_tag: str, param: str) -> dict[str, list[bool]]:
        """Return {source: [was_improvement, ...]} for all attempts on (tag, param)."""
        result: dict[str, list[bool]] = defaultdict(list)
        records = [a for a in self._log[target_tag] if a.param == param]
        for i, rec in enumerate(records):
            if i == 0:
                continue
            improved = records[i].n_errors_after < records[i - 1].n_errors_after
            result[rec.source].append(improved)
        return result

    def credit_score(self, target_tag: str, param: str) -> float:
        """
        Mean credit over the most recent _CREDIT_WINDOW attempts on (tag, param).
        Uses a sliding window so stale trajectories don't skew the score.
        """
        recs = [a for a in self._log[target_tag] if a.param == param]
        if not recs:
            return 0.0
        recent = recs[-_CREDIT_WINDOW:]
        return sum(r.credit for r in recent) / len(recent)

    def best_value(self, target_tag: str, param: str) -> Optional[Any]:
        """Return the value that achieved the best (lowest) error count."""
        recs = [a for a in self._log[target_tag] if a.param == param]
        if not recs:
            return None
        return min(recs, key=lambda r: r.n_errors_after).value

    def detect_oscillation(
        self, target_tag: str, param: str
    ) -> tuple[bool, Optional[float]]:
        """
        Issue 1 / Issue 6 — detect ping-pong between two values.

        Returns (is_oscillating, oscillation_center).

        Oscillation criterion: the last ≥4 values alternate between exactly
        two distinct rounded values.
        """
        vals = [v for v in self.tried_values(target_tag, param)
                if isinstance(v, (int, float))]
        if len(vals) < 4:
            return False, None

        recent  = vals[-4:]
        rounded = [round(v, _OSC_ROUND_DP) for v in recent]
        unique  = set(rounded)

        if len(unique) == 2:
            is_alt = all(rounded[i] != rounded[i + 1] for i in range(len(rounded) - 1))
            if is_alt:
                v1, v2 = sorted(unique)
                return True, (v1 + v2) / 2.0

        return False, None

    def record_trajectory_outcome(
        self,
        trajectory:   list[tuple[str, str, Any]],   # [(tag, param, value), ...]
        final_errors: int,
        init_errors:  int = 0,
        discount:     float = 0.7,
    ) -> None:
        """
        Issue 6 — back-propagate trajectory credit.

        When a full trajectory yields final_errors < init_errors, each fix
        in the trajectory receives proportional credit discounted by distance
        from the end (last fix = full credit, earlier fixes = discount^k).

        Credit is stored by updating the matching AttemptRecord's
        n_errors_before field so credit() reflects the full trajectory gain.
        This avoids a separate data structure.
        """
        if final_errors >= init_errors or not trajectory:
            return

        n_steps = len(trajectory)
        total_reduction = init_errors - final_errors

        for i, (tag, param, value) in enumerate(reversed(trajectory)):
            # Discount increases with distance from the last fix
            step_credit = total_reduction * (discount ** i)

            # Find the most recent record for this (tag, param, value)
            recs = [a for a in self._log[tag]
                    if a.param == param and a.value == value]
            if recs:
                last = recs[-1]
                # Boost n_errors_before to reflect the full trajectory gain
                # (credit property = n_errors_before - n_errors_after / n_errors_before)
                last.n_errors_before = max(
                    last.n_errors_before,
                    last.n_errors_after + int(step_credit + 0.5),
                )

    def summary(self, target_tag: str) -> str:
        attempts = self._log.get(target_tag, [])
        if not attempts:
            return "(no prior attempts)"
        lines = [
            f"  iter={a.iteration} {a.param}={a.value} ({a.source})"
            f" → {a.n_errors_after} errors (credit={a.credit:.2f})"
            for a in attempts[-5:]
        ]
        return "\n".join(lines)


# ── LLM system prompt ──────────────────────────────────────────────────────────

_LLM_SYSTEM = """\
Fix ONE failing unit parameter. Return ONLY: {"param": "<name>", "value": <number>}
No prose, no markdown. Temperatures in Kelvin (K). Pressures in Pascals (Pa).
Change ONLY the parameter specified — do not return other parameters.

━━━ EXAMPLE 1 ━━━
Error: Heater T_out=293 K is below feed T=298 K
Unit: HT-01 (Heater), feed T=298.15 K
Constraint: T_out must be > 298.15 K
Bubble point: 354 K → target 362–374 K
Tried values: none
{"param": "T_out", "value": 369.15}

━━━ EXAMPLE 2 ━━━
Error: Pump P_out=50000 Pa is below feed P=101325 Pa
Unit: PM-01 (Pump), feed P=101325 Pa
Constraint: P_out must be > 101325 Pa
Tried values: [80000]  — do not repeat
{"param": "P_out", "value": 1013250.0}

━━━ EXAMPLE 3 ━━━
Error: Cooler T_out=430 K is above feed T=423 K
Unit: CL-01 (Cooler), feed T=423.15 K
Constraint: T_out must be < 423.15 K
Tried values: [410.0]  — do not repeat
{"param": "T_out", "value": 323.15}"""


# ── Main repair agent ──────────────────────────────────────────────────────────

class RepairAgent:
    """
    Search-based repair: generates diverse candidates, validates against IR,
    selects the best by score, and records the outcome in RepairMemory.

    disable_llm=True: LLM candidate always returns None (benchmark / deterministic mode).
    """

    def __init__(
        self,
        model:        str  = DEFAULT_MODEL,
        retriever:    Optional[Retriever] = None,
        beam_width:   int  = 3,
        beam_depth:   int  = 2,
        disable_llm:  bool = False,
    ) -> None:
        self._model      = model
        self._retriever  = retriever or Retriever()
        self._beam_width = beam_width
        self._beam_depth = beam_depth
        self._disable_llm = disable_llm

    def repair(
        self,
        graph:          FlowsheetGraph,
        errors:         list[SimError],
        tried_packages: set[str] | None        = None,
        description:    str                    = "",
        memory:         Optional[RepairMemory] = None,
        sim_hints:      SimulationHints        = EMPTY_HINTS,
    ) -> tuple[FlowsheetGraph, list[str]]:
        g              = graph.copy()
        changes:list[str] = []
        tried_packages = tried_packages or set()
        memory         = memory or RepairMemory()

        cond_errors  = [e for e in errors
                        if e.repair_strategy == RepairStrategy.CONDITION_FIX]
        other_errors = [e for e in errors
                        if e.repair_strategy != RepairStrategy.CONDITION_FIX]

        # ── Non-CONDITION_FIX errors: always deterministic ─────────────────────
        for error in other_errors:
            if error.repair_strategy == RepairStrategy.HUMAN:
                changes.append(f"HUMAN: {error.target} — {error.evidence[:80]}")
                continue
            try:
                g, msgs = _det_repair.apply(
                    g, error, self._retriever, tried_packages)
                if error.repair_strategy == RepairStrategy.THERMO_SWITCH:
                    tried_packages.add(g.property_package)
            except ValueError as exc:
                msgs = [f"REPAIR_ERROR: {exc}"]
            changes.extend(msgs)

        # ── CONDITION_FIX errors: beam search (multi) vs single-step (one) ─────
        if len(cond_errors) > 1:
            from agents.stage4.beam_search import BeamRepairSearch
            searcher = BeamRepairSearch(
                width = self._beam_width,
                depth = self._beam_depth,
            )
            g, beam_changes = searcher.search(
                g, cond_errors,
                memory    = memory,
                sim_hints = sim_hints,
                llm_agent = self,
            )
            changes.extend(beam_changes)

        elif len(cond_errors) == 1:
            g, msgs = self._search_condition_fix(
                g, cond_errors[0], description, memory, sim_hints)
            changes.extend(msgs)

        return g, changes

    # ── Search-based condition fix ────────────────────────────────────────────

    def _search_condition_fix(
        self,
        graph:       FlowsheetGraph,
        error:       SimError,
        description: str,
        memory:      RepairMemory,
        sim_hints:   SimulationHints = EMPTY_HINTS,
    ) -> tuple[FlowsheetGraph, list[str]]:
        from ir.types import TargetKind

        if error.target.kind == TargetKind.STREAM:
            conv_error = SimError(
                error_type      = error.error_type,
                target          = error.target,
                evidence        = error.evidence,
                repair_strategy = RepairStrategy.UNIT_CONVERSION,
                severity        = error.severity,
            )
            return _det_repair.fix_unit_conversions(graph, conv_error)

        node = graph.unit(error.target.tag)
        if node is None:
            return graph, [f"CONDITION_FIX: unit {error.target.tag} not found"]

        param_name = _infer_param(error)
        if param_name is None:
            return graph, [f"CONDITION_FIX: cannot infer param from {error.evidence}"]

        feed_T, feed_P = _inlet_conditions(graph, node.tag)
        bp             = bubble_point_K(graph.compounds, feed_P or 101_325.0)
        tried_vals     = memory.tried_values(error.target.tag, param_name)
        ref_val        = node.params.get(param_name)
        uncertainty    = memory.uncertainty_score(error.target.tag, param_name)

        n_errors_before = len(validate(graph).errors())

        candidates = _deterministic_candidates(
            graph, node, param_name, feed_T, feed_P, bp,
            tried_vals, error, uncertainty=uncertainty, sim_hints=sim_hints,
            memory=memory)

        llm_cand = self._llm_candidate(
            graph, node, param_name, error, feed_T, feed_P, bp, tried_vals)
        if llm_cand is not None:
            candidates.append(llm_cand)

        if not candidates:
            return graph, [
                f"CONDITION_FIX: no viable candidates for {node.tag}.{param_name}"]

        _score_candidates(
            candidates,
            ref_value       = float(ref_val) if ref_val is not None else None,
            memory          = memory,
            target_tag      = error.target.tag,
            unit_type       = node.unit_type,
            sim_hints       = sim_hints,
            target_tag_sim  = error.target.tag,
        )
        best = min(candidates, key=lambda c: c.score)

        post_report = validate(best.graph)
        memory.record(
            target_tag      = error.target.tag,
            strategy        = "CONDITION_FIX",
            param           = best.action.param,
            value           = best.action.new_value,
            source          = best.action.source,
            n_errors_after  = len(post_report.errors()),
            n_errors_before = n_errors_before,
        )

        # Record margin only when fully repaired — prevents noisy learning
        if len(post_report.errors()) == 0 and bp is not None:
            _record_successful_margin(graph, node, param_name,
                                      best.action.new_value, bp)

        old_val = node.params.get(param_name, "?")
        return best.graph, [
            f"CONDITION_FIX[{best.action.source}]: {node.tag}.{param_name} "
            f"{old_val}→{best.action.new_value} "
            f"({best.action.rationale}, score={best.score:.1f}, "
            f"credit={best.action.source})"
        ]

    # ── LLM candidate (with output clamping) ──────────────────────────────────

    def _llm_candidate(
        self,
        graph:      FlowsheetGraph,
        node,
        param_name: str,
        error:      SimError,
        feed_T:     Optional[float],
        feed_P:     Optional[float],
        bp:         Optional[float],
        tried_vals: list,
    ) -> Optional[RepairCandidate]:
        if self._disable_llm:
            return None

        param_unit, constraint = _param_constraint(
            node.unit_type, param_name, feed_T, feed_P)
        role = node.metadata.get("role", "")

        prompt = (
            f"Error: {error.evidence}\n"
            f"Unit: {node.tag} ({node.unit_type}) — {role}\n"
            f"Feed: T={feed_T} K, P={feed_P} Pa\n"
            f"Constraint: {param_name} [{param_unit}] {constraint}\n"
        )
        if bp is not None:
            prompt += f"Bubble point: {bp} K\n"
        if tried_vals:
            prompt += f"Tried values (do NOT repeat): {tried_vals}\n"
        prompt += f'\nReturn: {{"param": "{param_name}", "value": <number>}}'

        for attempt in range(2):
            raw = chat(
                prompt,
                system      = _LLM_SYSTEM,
                model       = self._model,
                temperature = retry_temperature(attempt),
                max_tokens  = 64,
            )
            try:
                data  = _parse_json(raw)
                value = float(data["value"])

                value = _clamp_value(node.unit_type, param_name, value, feed_T, feed_P)
                if value is None:
                    continue

                errs = _validate_single(node.unit_type, param_name, value)
                if errs or value in [float(v) for v in tried_vals
                                     if isinstance(v, (int, float))]:
                    continue

                g2 = graph.copy()
                n  = g2.unit(node.tag)
                n.params[param_name] = value
                return RepairCandidate(
                    graph  = g2,
                    action = RepairAction(
                        param     = param_name,
                        new_value = value,
                        source    = "llm",
                        rationale = "LLM structured fix",
                    ),
                )
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                continue
        return None


# ── Physics-derived candidate generation (Item 2) ─────────────────────────────

def _physics_candidates(
    graph:      FlowsheetGraph,
    node,
    param_name: str,
    feed_T:     Optional[float],
    feed_P:     Optional[float],
    bp:         Optional[float],
    tried_vals: list,
    error:      SimError,
    sim_hints:  SimulationHints = EMPTY_HINTS,
) -> list[tuple[float, str, str]]:
    """
    Generate candidates derived directly from violated physical constraints.

    Returns list of (value, rationale, source="physics") raw tuples.

    Rules:
      Heater feeding Vessel : T_out = bp + learned_margin (flash feasibility)
      Cooler feeding Pump   : T_out = bp - safety_margin  (liquid feed required)
      Pump cavitation risk  : P_out = feed_P * safety_factor
      Directional hint      : if sim_hints says increase/decrease, generate
                              candidates biased in that direction
    """
    from agents.rule_store import classify_compounds

    margin_model = get_global_margin_model()
    compound_classes = classify_compounds(graph.compounds)
    raw: list[tuple[float, str, str]] = []

    # ── Heater → Vessel: flash feasibility target ──────────────────────────────
    if param_name == "T_out" and isinstance(node, HeaterNode):
        # Check downstream type for context
        downstream_types = _direct_downstream_types(graph, node.tag)
        has_vessel = "Vessel" in downstream_types

        if has_vessel and bp is not None:
            margin = margin_model.get_margin(
                "Heater", "Vessel", compound_classes, "T_out", default=20.0)
            raw.append((
                round(bp + margin, 2),
                f"physics/flash: bp={bp:.1f}+{margin:.1f} K",
                "physics",
            ))
            # Also a tighter and a wider candidate bracketing the learned margin
            raw.append((
                round(bp + max(margin - 5.0, 5.0), 2),
                f"physics/flash: bp={bp:.1f}+{max(margin-5,5):.1f} K [tighter]",
                "physics",
            ))
            raw.append((
                round(bp + margin + 10.0, 2),
                f"physics/flash: bp={bp:.1f}+{margin+10:.1f} K [wider]",
                "physics",
            ))

    # ── Cooler → Pump: liquid feed required ───────────────────────────────────
    if param_name == "T_out" and isinstance(node, CoolerNode):
        downstream_types = _direct_downstream_types(graph, node.tag)
        has_pump = "Pump" in downstream_types

        if has_pump and bp is not None:
            margin = margin_model.get_margin(
                "Cooler", "Pump", compound_classes, "T_out", default=15.0)
            raw.append((
                round(max(bp - margin, 273.15), 2),
                f"physics/pump_liquid: bp={bp:.1f}-{margin:.1f} K",
                "physics",
            ))

    # ── Simulation directional hint biasing ───────────────────────────────────
    direction = sim_hints.directional_hint(node.tag, param_name)
    ft = feed_T or 298.15
    fp = feed_P or 101_325.0

    if direction == "increase" and param_name == "T_out":
        if bp is not None:
            raw.append((round(bp + 15.0, 2), "physics/sim_hint: increase T to above BP", "physics"))
        raw.append((round(ft + 40.0, 2), "physics/sim_hint: increase T from feed", "physics"))

    elif direction == "decrease" and param_name == "T_out":
        raw.append((round(max(ft - 30.0, 273.15), 2), "physics/sim_hint: decrease T", "physics"))

    elif direction == "increase" and param_name == "P_out":
        raw.append((round(fp * 4.0, 0), "physics/sim_hint: increase P", "physics"))

    elif direction == "decrease" and param_name == "P_out":
        raw.append((round(max(fp / 2.0, 101_325.0), 0), "physics/sim_hint: decrease P", "physics"))

    return raw


# ── Deterministic candidate generation (diversity-enforced) ───────────────────

def _deterministic_candidates(
    graph:       FlowsheetGraph,
    node,
    param_name:  str,
    feed_T:      Optional[float],
    feed_P:      Optional[float],
    bp:          Optional[float],
    tried_vals:  list,
    error:       SimError,
    uncertainty: float = 0.0,
    sim_hints:   SimulationHints = EMPTY_HINTS,
    memory:      Optional["RepairMemory"] = None,   # Issue 1: oscillation detection
) -> list[RepairCandidate]:
    """
    Generate diverse repair candidates in physics + low/medium/high tiers.

    Physics-derived candidates (Item 2) are generated first with higher
    priority; deterministic heuristics follow.  Diversity constraints
    (spacing enforcement) are applied across all sources together.

    Issue 1 — oscillation escape: when RepairMemory detects ping-pong between
    two values, escape candidates are injected at the oscillation midpoint and
    far outside the oscillation range to break the cycle.
    """
    raw_candidates: list[tuple[float, str, str]] = []
    tried_set = {round(float(v), 2) for v in tried_vals
                 if isinstance(v, (int, float))}
    high_uncertainty = uncertainty > 0.5

    # ── Oscillation escape (Issue 1) ──────────────────────────────────────────
    if memory is not None:
        is_osc, osc_center = memory.detect_oscillation(node.tag, param_name)
        if is_osc and osc_center is not None:
            all_tried = [v for v in tried_vals if isinstance(v, (int, float))]
            spread    = max(all_tried) - min(all_tried) if len(all_tried) >= 2 else 30.0
            escape    = max(spread * 1.5, 30.0)  # at least 30 K / 30% spread

            if param_name == "T_out":
                raw_candidates += [
                    (round(osc_center, 2),
                     f"escape/oscillation-center={osc_center:.1f}", "heuristic"),
                    (round(osc_center + escape, 2),
                     f"escape/+{escape:.0f} K outside osc-range", "heuristic"),
                    (round(max(osc_center - escape, 273.15), 2),
                     f"escape/-{escape:.0f} K outside osc-range", "heuristic"),
                ]
            elif param_name == "P_out":
                raw_candidates += [
                    (round(osc_center, 0),
                     f"escape/oscillation-center={osc_center:.0f}", "heuristic"),
                    (round(osc_center * (1.0 + escape / 100.0), 0),
                     f"escape/+{escape:.0f}% outside osc-range", "heuristic"),
                    (round(max(osc_center * (1.0 - escape / 100.0), 101_325.0), 0),
                     f"escape/-{escape:.0f}% outside osc-range", "heuristic"),
                ]

    # ── Physics-derived candidates first ──────────────────────────────────────
    raw_candidates += _physics_candidates(
        graph, node, param_name, feed_T, feed_P, bp,
        tried_vals, error, sim_hints)

    # ── Rule-based heuristic candidates ───────────────────────────────────────
    if param_name == "T_out":
        ft = feed_T or 298.15

        if isinstance(node, HeaterNode):
            if bp is not None:
                raw_candidates += [
                    (bp + 10.0, f"bp({bp:.1f} K)+10 [low]",  "deterministic"),
                    (bp + 20.0, f"bp({bp:.1f} K)+20 [med]",  "deterministic"),
                    (bp + 35.0, f"bp({bp:.1f} K)+35 [high]", "deterministic"),
                ]
                if high_uncertainty:
                    raw_candidates += [
                        (bp + 55.0, "bp+55 [wide/uncertain]", "heuristic"),
                        (bp + 80.0, "bp+80 [wide/uncertain]", "heuristic"),
                    ]
            raw_candidates += [
                (ft + 25.0, "feed T+25 [low]",  "heuristic"),
                (ft + 50.0, "feed T+50 [med]",  "heuristic"),
                (ft + 80.0, "feed T+80 [high]", "heuristic"),
            ]
            if high_uncertainty:
                raw_candidates.append((ft + 120.0, "feed T+120 [wide]", "heuristic"))

        elif isinstance(node, CoolerNode):
            raw_candidates += [
                (max(ft - 20.0, 273.15), "feed T-20 [low]",  "heuristic"),
                (max(ft - 50.0, 273.15), "feed T-50 [med]",  "heuristic"),
                (max(ft - 80.0, 273.15), "feed T-80 [high]", "heuristic"),
            ]
            if high_uncertainty:
                raw_candidates.append((max(ft - 120.0, 273.15), "feed T-120 [wide]", "heuristic"))

    elif param_name == "P_out":
        fp = feed_P or 101_325.0

        if node.unit_type in ("Pump", "Compressor"):
            raw_candidates += [
                (fp * 3.0,  "feed P×3 [low]",   "deterministic"),
                (fp * 5.0,  "feed P×5 [med]",   "deterministic"),
                (fp * 10.0, "feed P×10 [high]",  "deterministic"),
            ]
            if high_uncertainty:
                raw_candidates.append((fp * 20.0, "feed P×20 [wide]", "heuristic"))
        elif node.unit_type == "Expander":
            raw_candidates += [
                (max(fp / 2.0, 101_325.0), "feed P/2 [low]",  "deterministic"),
                (max(fp / 3.0, 101_325.0), "feed P/3 [med]",  "deterministic"),
                (max(fp / 5.0, 101_325.0), "feed P/5 [high]", "deterministic"),
            ]

    # ── Build candidates, enforce diversity ───────────────────────────────────
    candidates: list[RepairCandidate] = []
    accepted_vals: list[float] = []

    for raw_val, rationale, source in raw_candidates:
        value = round(raw_val, 2) if param_name == "T_out" else round(raw_val, 0)

        if value in tried_set:
            continue
        if _validate_single(node.unit_type, param_name, value):
            continue

        if param_name == "T_out":
            if any(abs(value - v) < _MIN_T_SPACING for v in accepted_vals):
                continue
        elif param_name == "P_out":
            if any(abs(value / max(v, 1.0) - 1.0) < (_MIN_P_RATIO - 1.0)
                   for v in accepted_vals):
                continue

        g2 = graph.copy()
        n  = g2.unit(node.tag)
        n.params[param_name] = value
        candidates.append(RepairCandidate(
            graph  = g2,
            action = RepairAction(
                param     = param_name,
                new_value = value,
                source    = source,
                rationale = rationale,
            ),
        ))
        accepted_vals.append(value)

    return candidates


def _score_candidates(
    candidates:     list[RepairCandidate],
    ref_value:      Optional[float]       = None,
    memory:         Optional[RepairMemory] = None,
    target_tag:     Optional[str]         = None,
    unit_type:      Optional[str]         = None,
    sim_hints:      Optional[SimulationHints] = None,
    target_tag_sim: Optional[str]         = None,
) -> None:
    """
    In-place: run IR validation on each candidate, set n_errors, n_warnings,
    magnitude, and adaptive_penalty.

    adaptive_penalty combines:
      - Per-source historical success rate (from RepairMemory source_successes)
      - Credit-weighted adjustment (Item 9): sources with high credit get a bonus
      - Simulation hint priority boost
    """
    source_rates:   dict[str, float] = {}
    source_credits: dict[str, float] = {}

    if memory is not None and target_tag is not None and candidates:
        param = candidates[0].action.param
        successes = memory.source_successes(target_tag, param)
        for src, outcomes in successes.items():
            if outcomes:
                source_rates[src] = sum(outcomes) / len(outcomes)
        source_credits["physics"]      = memory.credit_score(target_tag, param)
        source_credits["deterministic"]= memory.credit_score(target_tag, param)

    for cand in candidates:
        report = validate(cand.graph)
        cand.n_errors   = len(report.errors())
        cand.n_warnings = len(report.warnings())

        if ref_value is not None and abs(ref_value) > 1e-9:
            cand.magnitude = min(
                abs(cand.action.new_value - ref_value) / abs(ref_value), 10.0)
        else:
            cand.magnitude = 0.0

        # Adaptive penalty: low-success sources penalised, high-credit sources rewarded
        src_rate = source_rates.get(cand.action.source)
        if src_rate is not None:
            cand.adaptive_penalty = (1.0 - src_rate) * 5.0
        else:
            cand.adaptive_penalty = 0.0

        # Credit bonus: physics candidates that historically worked get a boost
        credit = source_credits.get(cand.action.source, 0.0)
        cand.adaptive_penalty -= credit * 2.0   # negative = better score

        if sim_hints is not None and target_tag_sim is not None:
            cand.adaptive_penalty += sim_hints.priority_boost(target_tag_sim)


# ── Margin recording ───────────────────────────────────────────────────────────

def _record_successful_margin(
    graph:     FlowsheetGraph,
    node:      object,
    param:     str,
    new_value: float,
    bp:        float,
) -> None:
    """Record the successful margin in the global MarginModel (Item 4)."""
    from agents.rule_store import classify_compounds
    margin_model     = get_global_margin_model()
    compound_classes = classify_compounds(graph.compounds)
    downstream_types = _direct_downstream_types(graph, node.tag)
    dst_type = next(iter(downstream_types), None)

    if param == "T_out" and bp > 0:
        margin = new_value - bp
        if margin > 0:
            margin_model.record(
                node.unit_type, dst_type, compound_classes, margin, param="T_out")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _direct_downstream_types(graph: FlowsheetGraph, unit_tag: str) -> set[str]:
    types: set[str] = set()
    for stream in graph.outlet_streams(unit_tag):
        dst_tag = graph.stream_dest(stream.tag)
        if dst_tag:
            dst = graph.unit(dst_tag)
            if dst:
                types.add(dst.unit_type)
    return types


def _clamp_value(
    unit_type: str,
    param:     str,
    value:     float,
    feed_T:    Optional[float],
    feed_P:    Optional[float],
) -> Optional[float]:
    if param == "T_out":
        if not (50.0 < value < 2000.0):
            return None
        if unit_type == "Heater" and feed_T is not None and value <= feed_T:
            return round(feed_T + 10.0, 2)
        if unit_type == "Cooler" and feed_T is not None and value >= feed_T:
            return round(feed_T - 10.0, 2)
        return value

    if param == "P_out":
        if not (100.0 < value < 1e8):
            return None
        if unit_type in ("Pump", "Compressor") and feed_P is not None and value <= feed_P:
            return round(feed_P * 2.0, 0)
        if unit_type == "Expander" and feed_P is not None and value >= feed_P:
            return round(feed_P / 2.0, 0)
        return value

    return value


def _infer_param(error: SimError) -> Optional[str]:
    ev = error.evidence.lower()
    if "t_out" in ev or "temperature" in ev or "heater" in ev or "cooler" in ev:
        return "T_out"
    if "p_out" in ev or "pressure" in ev or "pump" in ev or "compressor" in ev:
        return "P_out"
    attr = getattr(error.target, "attribute", None)
    if attr in ("T_out", "P_out"):
        return attr
    if error.error_type == ErrorType.INVALID_UNIT_CONFIG:
        if re.search(r'T_out\s*=', error.evidence):
            return "T_out"
        if re.search(r'P_out\s*=', error.evidence):
            return "P_out"
    return None


def _inlet_conditions(
    graph: FlowsheetGraph,
    unit_tag: str,
) -> tuple[Optional[float], Optional[float]]:
    for stream in graph.inlet_streams(unit_tag):
        if stream.T is not None:
            return stream.T, stream.P
    return None, None


def _param_constraint(
    unit_type: str, param: str,
    feed_T: Optional[float], feed_P: Optional[float],
) -> tuple[str, str]:
    if param == "T_out":
        if unit_type == "Heater" and feed_T:
            return "K", f"must be > {feed_T:.2f} K (feed T)"
        if unit_type == "Cooler" and feed_T:
            return "K", f"must be < {feed_T:.2f} K (feed T)"
        return "K", "must be in range 100–1500 K"
    if param == "P_out":
        if unit_type in ("Pump", "Compressor") and feed_P:
            return "Pa", f"must be > {feed_P:.0f} Pa (feed P)"
        if unit_type == "Expander" and feed_P:
            return "Pa", f"must be < {feed_P:.0f} Pa (feed P)"
        return "Pa", "must be in range 1000–1e8 Pa"
    return "?", "must be physically reasonable"


def _validate_single(unit_type: str, param: str, value: float) -> list[str]:
    errors: list[str] = []
    if param == "T_out" and not (50.0 < value < 2000.0):
        errors.append(f"T_out={value} K out of range")
    if param == "P_out" and not (100.0 < value < 1e8):
        errors.append(f"P_out={value} Pa out of range")
    return errors


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$",       "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group() if m else text)
