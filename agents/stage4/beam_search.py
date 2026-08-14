"""
Multi-step beam search for CONDITION_FIX repair (v3).

v3 additions on top of v2:
  Issue 1  — CoupledSettler.settle() runs after each consistency pass so
             P→BP→T coupled parameters are corrected deterministically before
             the next beam step, preventing inter-iteration oscillation.
  Issue 5  — _diverse_beam_prune() replaces plain score-sorted slicing.
             Beam states too similar in parameter space are penalised so the
             beam explores meaningfully different regions.
  Issue 5  — explicit trajectory tracking in BeamState enables multi-step
             credit back-propagation at the end of search.
  Issue 6  — memory.record_trajectory_outcome() is called on the winning
             trajectory so earlier fixes that enabled later fixes receive
             discounted credit.

Algorithm:
  beam = [(graph, changes=[], trajectory=[], fixed={}, score=initial_ir_errors)]
  for step in range(depth):
      for each beam_state:
          pick next unfixed error (coupling + uncertainty + sim-hint biased)
          generate physics + deterministic + LLM candidates
          apply GlobalConsistencyPass → CoupledSettler.settle()
          validate each candidate (state-cached)
          append to next_beam
      prune next_beam → diverse top-width states
      update explore/exploit scheduler
  post-process: coordinate_descent on best state
  back-propagate credit for winning trajectory

Candidate import note:
  _deterministic_candidates, _score_candidates, _infer_param, _inlet_conditions
  are imported from repair_agent at module level — safe because repair_agent.py
  only imports BeamRepairSearch lazily, breaking the circular dependency.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from ir.graph import FlowsheetGraph
from ir.consistency import GlobalConsistencyPass
from ir.thermo_estimation import bubble_point_K
from ir.validate import validate
from ir.types import SimError, RepairStrategy
from ir.coupling import ParameterCouplingMap, CoupledSettler
from ir.state_cache import StateCache
from ir.local_optimiser import coordinate_descent

from agents.stage4.repair_agent import (
    _deterministic_candidates,
    _record_successful_margin,
    _score_candidates,
    _infer_param,
    _inlet_conditions,
    RepairMemory,
    RepairCandidate,
)
from agents.stage4.sim_hints import SimulationHints, EMPTY_HINTS
from agents.stage4.explore_exploit import ExploreExploitScheduler

logger       = logging.getLogger(__name__)
_consistency = GlobalConsistencyPass()
_coupling    = ParameterCouplingMap()
_settler     = CoupledSettler()

# Diversity thresholds for beam pruning (normalised units — 1.0 = just-diverse-enough)
_T_DIV_THRESH = 15.0   # K
_P_DIV_THRESH = 0.25   # fractional


# ── Beam state ─────────────────────────────────────────────────────────────────

@dataclass
class BeamState:
    graph:            FlowsheetGraph
    changes:          list[str]                   = field(default_factory=list)
    trajectory:       list[tuple[str, str, Any]]  = field(default_factory=list)
    fixed_tags:       set[str]                    = field(default_factory=set)
    last_fixed_tag:   Optional[str]               = None
    last_fixed_param: Optional[str]               = None
    ir_errors:        int                         = 0
    ir_warnings:      int                         = 0
    sim_penalty:      float                       = 0.0

    @property
    def score(self) -> float:
        return self.ir_errors * 100 + self.ir_warnings + self.sim_penalty


# ── Beam search ────────────────────────────────────────────────────────────────

class BeamRepairSearch:
    """
    Diversity-pruned multi-step beam search with coupled settling and
    trajectory credit back-propagation.
    """

    def __init__(
        self,
        width:         int  = 3,
        depth:         int  = 2,
        run_local_opt: bool = True,
    ) -> None:
        self.width         = width
        self.depth         = depth
        self.run_local_opt = run_local_opt

    def search(
        self,
        graph:     FlowsheetGraph,
        errors:    list[SimError],
        memory:    Optional[RepairMemory] = None,
        sim_hints: SimulationHints        = EMPTY_HINTS,
        llm_agent: Optional[Any]          = None,
    ) -> tuple[FlowsheetGraph, list[str]]:
        cond_errors = [
            e for e in errors
            if e.repair_strategy == RepairStrategy.CONDITION_FIX
            and _infer_param(e) is not None
        ]
        if not cond_errors:
            return graph, []

        import sys as _sys
        _bp_probe = bubble_point_K(graph.compounds, 101_325.0)
        print(f"[BEAM] search start: compounds={graph.compounds[:3]} "
              f"bp_probe={_bp_probe} n_cond_errors={len(cond_errors)} "
              f"beam_width={self.width}",
              flush=True, file=_sys.stderr)

        # Ablation activation counters — persisted into ABLATION_STATS_LOG
        _ablation_bp_calls:        int       = 1  # probe call above
        _ablation_phase_cands:     int       = 0
        _ablation_coupling_queries: int      = 0
        _ablation_nonempty_boosts: list[dict] = []

        memory    = memory or RepairMemory()
        cache     = StateCache()
        scheduler = ExploreExploitScheduler(
            exploration_phase=max(1, self.depth - 1))

        init_report = cache.cached_validate(graph)
        beam = [BeamState(
            graph       = graph,
            ir_errors   = len(init_report.errors()),
            ir_warnings = len(init_report.warnings()),
        )]

        for step in range(self.depth):
            eff_width   = scheduler.effective_beam_width(self.width)
            next_states: list[BeamState] = []

            for state in beam:
                unfixed = [e for e in cond_errors
                           if e.target.tag not in state.fixed_tags]
                if not unfixed:
                    next_states.append(state)
                    continue

                # Coupling-aware target selection
                unfixed_tags = {e.target.tag for e in unfixed}
                coupling_boosts: dict[str, float] = {}
                if state.last_fixed_tag and state.last_fixed_param:
                    _ablation_coupling_queries += 1
                    coupling_boosts = _coupling.get_coupled_boosts(
                        state.graph,
                        state.last_fixed_tag,
                        state.last_fixed_param,
                        unfixed_tags,
                    )
                    if coupling_boosts:
                        _ablation_nonempty_boosts.append({
                            "source_tag":   state.last_fixed_tag,
                            "source_param": state.last_fixed_param,
                            "boosted":      dict(coupling_boosts),
                        })
                        import sys as _sys
                        print(f"[BEAM] coupling_boosts={coupling_boosts} "
                              f"(from {state.last_fixed_tag}.{state.last_fixed_param})",
                              flush=True, file=_sys.stderr)

                target_error = _pick_target(
                    unfixed, memory, sim_hints, coupling_boosts)
                param_name = _infer_param(target_error)
                if param_name is None:
                    next_states.append(state)
                    continue

                node = state.graph.unit(target_error.target.tag)
                if node is None:
                    next_states.append(state)
                    continue

                feed_T, feed_P = _inlet_conditions(state.graph, node.tag)
                bp             = bubble_point_K(
                    state.graph.compounds, feed_P or 101_325.0)
                _ablation_bp_calls += 1
                tried_vals     = memory.tried_values(
                    target_error.target.tag, param_name)
                uncertainty    = memory.uncertainty_score(
                    target_error.target.tag, param_name)

                # Exploration phase → force wide-range candidates
                effective_uncertainty = (
                    max(uncertainty, 0.6) if scheduler.use_wide_range()
                    else uncertainty
                )

                candidates = _deterministic_candidates(
                    state.graph, node, param_name,
                    feed_T, feed_P, bp, tried_vals, target_error,
                    uncertainty = effective_uncertainty,
                    sim_hints   = sim_hints,
                    memory      = memory,
                )
                _ablation_phase_cands += sum(
                    1 for c in candidates if c.action.source == "physics")

                if llm_agent is not None:
                    llm_c = llm_agent._llm_candidate(
                        state.graph, node, param_name, target_error,
                        feed_T, feed_P, bp, tried_vals)
                    if llm_c is not None:
                        candidates.append(llm_c)

                if not candidates:
                    import sys as _sys
                    print(f"[BEAM] step={step} {node.tag}.{param_name}: "
                          f"no candidates (tried={len(tried_vals)}) — "
                          f"passing state through unchanged",
                          flush=True, file=_sys.stderr)
                    next_states.append(state)
                    continue

                ref_val = node.params.get(param_name)
                _score_candidates(
                    candidates,
                    ref_value      = float(ref_val) if ref_val is not None else None,
                    memory         = memory,
                    target_tag     = target_error.target.tag,
                    sim_hints      = sim_hints,
                    target_tag_sim = target_error.target.tag,
                )

                old_val = node.params.get(param_name, "?")
                for cand in candidates:
                    # Consistency pass → coupled settling → cache-validated
                    cand_g, cons_changes = _consistency.apply(cand.graph)
                    cand_g, settle_changes = _settler.settle(
                        cand_g, target_error.target.tag, param_name)
                    report = cache.cached_validate(cand_g)
                    n_errs = len(report.errors())
                    n_warn = len(report.warnings())
                    # Carry RepairCandidate physics score into BeamState.sim_penalty
                    # so the beam can differentiate candidates even when all have
                    # ir_errors=0 (IR validation passes for all parameter changes).
                    # Without this, BeamState.score=0 for every candidate and
                    # min() picks arbitrarily — discarding all the reasoning done
                    # by _score_candidates (magnitude penalty, source credit, etc).
                    sp     = (sim_hints.convergence_penalty()
                              + sim_hints.priority_boost(target_error.target.tag)
                              + cand.magnitude * 0.5
                              + cand.adaptive_penalty)

                    fix_msg = (
                        f"[beam/step{step}/{cand.action.source}] "
                        f"{node.tag}.{param_name} "
                        f"{old_val}→{cand.action.new_value} "
                        f"({cand.action.rationale})"
                    )

                    new_traj = state.trajectory + [
                        (target_error.target.tag, param_name, cand.action.new_value)
                    ]

                    next_states.append(BeamState(
                        graph            = cand_g,
                        changes          = state.changes + [fix_msg]
                                          + cons_changes + settle_changes,
                        trajectory       = new_traj,
                        fixed_tags       = state.fixed_tags | {target_error.target.tag},
                        last_fixed_tag   = target_error.target.tag,
                        last_fixed_param = param_name,
                        ir_errors        = n_errs,
                        ir_warnings      = n_warn,
                        sim_penalty      = sp,
                    ))

            if not next_states:
                break

            # Diversity-aware pruning (Issue 5)
            best = min(next_states, key=lambda s: s.score)
            beam = _diverse_beam_prune(next_states, eff_width)

            # Update adaptive scheduler with best current error count
            scheduler.update(best.ir_errors)

            if best.ir_errors == 0 and best.ir_warnings == 0:
                break

        best = min(beam, key=lambda s: s.score)

        # ── Trajectory credit back-propagation (Issue 6) ──────────────────────
        if best.trajectory:
            init_errs = len(init_report.errors())
            memory.record_trajectory_outcome(
                best.trajectory, best.ir_errors, init_errs)
            # Record every proposed value from the winning trajectory so that
            # tried_values() filters them on the next search() call.  Without
            # this, beam search re-proposes the same candidates every iteration
            # because RepairMemory only receives records from the single-error
            # path (_search_condition_fix), never from multi-error beam search.
            for _tag, _param, _val in best.trajectory:
                memory.record(
                    target_tag      = _tag,
                    strategy        = "CONDITION_FIX",
                    param           = _param,
                    value           = _val,
                    source          = "beam",
                    n_errors_after  = best.ir_errors,
                    n_errors_before = init_errs,
                )

        # ── Ablation stats log (persisted into per-run JSON) ──────────────────
        _ablation_phase_selected = any(
            isinstance(c, str) and "[physics]" in c for c in best.changes)
        _abl_log = "ABLATION_STATS_LOG:" + json.dumps({
            "n_bp_calls":              _ablation_bp_calls,
            "n_phase_candidates":      _ablation_phase_cands,
            "phase_candidate_selected": _ablation_phase_selected,
            "n_coupling_queries":      _ablation_coupling_queries,
            "nonempty_boosts":         _ablation_nonempty_boosts,
        })

        # ── Local optimisation post-processing (Item 3) ───────────────────────
        after_beam_errs = best.ir_errors
        if self.run_local_opt and (best.ir_errors > 0 or best.ir_warnings > 0):
            opt_graph, opt_changes = coordinate_descent(best.graph)
            opt_report = validate(opt_graph)
            opt_errs   = len(opt_report.errors())
            if (opt_errs * 100 + len(opt_report.warnings())
                    < best.ir_errors * 100 + best.ir_warnings):
                if opt_errs == 0 and best.trajectory:
                    _record_trajectory_margins(opt_graph, best.trajectory)
                stage_log = (
                    f'REPAIR_STAGE_LOG:{{"after_beam":{after_beam_errs},'
                    f'"after_local_opt":{opt_errs}}}'
                )
                return opt_graph, best.changes + opt_changes + [stage_log, _abl_log]

        stage_log = (
            f'REPAIR_STAGE_LOG:{{"after_beam":{after_beam_errs},'
            f'"after_local_opt":{after_beam_errs}}}'
        )
        if best.ir_errors == 0 and best.trajectory:
            _record_trajectory_margins(best.graph, best.trajectory)
        return best.graph, best.changes + [stage_log, _abl_log]


# ── Target selection ───────────────────────────────────────────────────────────

def _pick_target(
    unfixed:         list[SimError],
    memory:          RepairMemory,
    sim_hints:       SimulationHints,
    coupling_boosts: dict[str, float],
) -> SimError:
    if len(unfixed) == 1:
        return unfixed[0]

    def _key(error: SimError) -> tuple:
        tag   = error.target.tag
        param = _infer_param(error) or ""

        coupling   = -coupling_boosts.get(tag, 0.0)
        sim_pri    = 0 if sim_hints.unit_failed(tag) else 1
        n_tried    = len(memory.tried_values(tag, param))
        uncertainty = _local_uncertainty(memory, tag, param)
        credit     = memory.credit_score(tag, param)

        return (coupling, sim_pri, -n_tried, -uncertainty, -credit)

    return min(unfixed, key=_key)


def _local_uncertainty(memory: RepairMemory, tag: str, param: str) -> float:
    vals = [v for v in memory.tried_values(tag, param) if isinstance(v, (int, float))]
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    var  = sum((v - mean) ** 2 for v in vals) / len(vals)
    return min(1.0, var / (mean ** 2 + 1e-9))


# ── Diversity-aware beam pruning (Issue 5) ────────────────────────────────────

def _diverse_beam_prune(
    states: list[BeamState],
    width:  int,
) -> list[BeamState]:
    """
    Greedy diverse selection: always include the global best, then add
    candidates that are sufficiently diverse in parameter space.
    Falls back to score order when diversity requirement cannot be met.
    Logs diversity vs score-fallback counts and average pairwise distance.
    """
    if len(states) <= width:
        return sorted(states, key=lambda s: s.score)

    sorted_states = sorted(states, key=lambda s: s.score)
    selected: list[BeamState] = [sorted_states[0]]
    remaining = sorted_states[1:]

    accepted_diversity = 0
    accepted_fallback  = 0

    for cand in remaining:
        if len(selected) >= width:
            break
        if _min_param_distance(cand, selected) >= 1.0:
            selected.append(cand)
            accepted_diversity += 1

    # Fill remaining slots with best by score (no diversity requirement)
    if len(selected) < width:
        for cand in remaining:
            if cand not in selected:
                selected.append(cand)
                accepted_fallback += 1
                if len(selected) >= width:
                    break

    avg_dist = _avg_pairwise_distance(selected)
    logger.debug(
        "DIVERSITY_PRUNE: width=%d | best=1 | diversity=%d | fallback=%d | "
        "avg_pairwise_dist=%.2f",
        width, accepted_diversity, accepted_fallback, avg_dist,
    )

    return selected


def _min_param_distance(state: BeamState, others: list[BeamState]) -> float:
    """
    Minimum normalised parameter distance from `state` to any of `others`.
    Returns ≥ 1.0 when the state is sufficiently diverse from all others.

    Normalisation:
      T_out differences divided by _T_DIV_THRESH (15 K)
      P_out fractional differences divided by _P_DIV_THRESH (0.25)
    """
    def _extract(s: BeamState) -> dict[str, float]:
        result: dict[str, float] = {}
        for unit in s.graph.units():
            for pname in ("T_out", "P_out"):
                val = unit.params.get(pname)
                if val is not None and isinstance(val, (int, float)):
                    result[f"{unit.tag}.{pname}"] = float(val)
        return result

    my = _extract(state)
    min_dist = float("inf")

    for other in others:
        their = _extract(other)
        shared = set(my) & set(their)
        if not shared:
            min_dist = min(min_dist, 1.0)
            continue

        total = 0.0
        for key in shared:
            va, vb = my[key], their[key]
            if ".T_out" in key:
                total += abs(va - vb) / _T_DIV_THRESH
            else:
                ref = max(abs(va), abs(vb), 1.0)
                total += abs(va - vb) / ref / _P_DIV_THRESH
        min_dist = min(min_dist, total / len(shared))

    return min_dist if min_dist != float("inf") else 1.0


def _avg_pairwise_distance(states: list[BeamState]) -> float:
    """Average normalised L1 distance across all unique pairs in `states`."""
    if len(states) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            d = _min_param_distance(states[i], [states[j]])
            total += d
            count += 1
    return total / count if count > 0 else 0.0


def _record_trajectory_margins(
    graph:      FlowsheetGraph,
    trajectory: list[tuple[str, str, Any]],
) -> None:
    """Record learned T_out margins for all fixes in a zero-error trajectory."""
    for tag, param, value in trajectory:
        if param != "T_out":
            continue
        node = graph.unit(tag)
        if node is None:
            continue
        _, feed_P = _inlet_conditions(graph, tag)
        bp = bubble_point_K(graph.compounds, feed_P or 101_325.0)
        if bp is None:
            continue
        _record_successful_margin(graph, node, param, float(value), bp)
