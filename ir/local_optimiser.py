"""
Local continuous optimiser — adaptive coordinate descent (Item 3 — Issue 3 hardened).

Runs after beam search has produced its best candidate graph.
Performs gradient-free coordinate descent over T_out and P_out parameters.

Issue-3 mitigation — adaptive step sizes:
  When a full sweep of all parameters produces no improvement, the step sizes
  are doubled (up to MAX_EXPANSIONS times).  This allows the optimiser to
  escape shallow local minima without multi-start restarts, while keeping the
  deterministic, constraint-respecting structure of coordinate descent.

  Step sequence: [5K, 10K, 20K, 40K] × [1×, 2×, 4×]
  ≡ effective range up to 160 K, which covers typical bubble-point margin errors.

Objective (lower = better):
    score = n_errors × 100 + n_warnings + magnitude_penalty × 0.5
"""
from __future__ import annotations

import logging

from ir.graph import (
    FlowsheetGraph,
    HeaterNode, CoolerNode,
    PumpNode, CompressorNode, ExpanderNode,
    SeparatorNode, Source,
)
from ir.validate import validate
from ir.constraint_solver import ConstraintSolver, Constraint, ConstraintPriority
from ir.thermo_estimation import bubble_point_K

logger  = logging.getLogger(__name__)
_solver = ConstraintSolver()

# Base step sizes — multiplied by expansion factor when stuck
_T_BASE_STEPS:  list[float] = [5.0, 10.0, 20.0, 40.0]    # K
_P_BASE_RATIOS: list[float] = [0.10, 0.25, 0.50]          # fractional

_MAX_OUTER_ITER  = 12   # outer sweeps per expansion level
_MAX_EXPANSIONS  = 2    # step-size doublings before giving up (0→2×→4×)


def coordinate_descent(
    graph:       FlowsheetGraph,
    max_iter:    int = _MAX_OUTER_ITER,
) -> tuple[FlowsheetGraph, list[str]]:
    """
    Adaptive coordinate descent over T_out / P_out parameters.

    Returns (best_graph, change_log).  The input graph is never mutated.

    Expansion schedule:
      Level 0: base steps.
      Level 1: 2× base steps (triggered when level 0 produces no improvement).
      Level 2: 4× base steps (triggered when level 1 produces no improvement).
    """
    g       = graph.copy()
    changes: list[str] = []
    report  = validate(g)
    score   = _obj(report, 0.0)

    if score == 0.0:
        return g, changes

    initial_score  = score
    accepted_total = 0

    for expansion in range(_MAX_EXPANSIONS + 1):
        multiplier   = 2.0 ** expansion
        t_steps      = [s * multiplier for s in _T_BASE_STEPS]
        p_ratios     = [r * multiplier for r in _P_BASE_RATIOS]
        improved_any = False

        if expansion > 0:
            logger.debug(
                "LOCAL_OPT: no improvement at level %d — expanding steps to ×%.0f "
                "(max T=%.0f K, max P=%.0f%%)",
                expansion - 1, multiplier,
                max(t_steps), max(p_ratios) * 100,
            )

        for _outer in range(max_iter):
            improved_pass = False

            for node in list(g.units()):
                tag = node.tag
                if isinstance(node, (HeaterNode, CoolerNode)):
                    g, score, changes, step_ok = _try_temperature(
                        g, tag, score, changes, t_steps)
                elif isinstance(node, (PumpNode, CompressorNode, ExpanderNode)):
                    g, score, changes, step_ok = _try_pressure(
                        g, tag, score, changes, p_ratios)
                else:
                    continue

                if step_ok:
                    accepted_total += 1
                    improved_pass   = True
                    improved_any    = True
                    node = g.unit(tag)   # re-fetch after graph copy

            if not improved_pass:
                break
            if score == 0.0:
                logger.debug(
                    "LOCAL_OPT: zero errors reached (%d accepted moves total)",
                    accepted_total,
                )
                if accepted_total > 0:
                    changes = changes + [
                        f'LOCAL_OPT_LOG:{{"triggered":true,"level":{expansion},'
                        f'"improvement":{initial_score - score:.1f},'
                        f'"accepted":{accepted_total}}}'
                    ]
                return g, changes

        if improved_any or score == 0.0:
            break   # improvements found at this level — don't expand further

    logger.debug(
        "LOCAL_OPT complete: %d accepted moves, final score=%.1f",
        accepted_total, score,
    )
    if accepted_total > 0:
        changes = changes + [
            f'LOCAL_OPT_LOG:{{"triggered":true,"level":{expansion},'
            f'"improvement":{initial_score - score:.1f},'
            f'"accepted":{accepted_total}}}'
        ]
    return g, changes


# ── Parameter-specific optimisation ──────────────────────────────────────────

def _try_temperature(
    graph:   FlowsheetGraph,
    tag:     str,
    score:   float,
    changes: list[str],
    steps:   list[float],
) -> tuple[FlowsheetGraph, float, list[str], bool]:
    node = graph.unit(tag)
    if node is None:
        return graph, score, changes, False

    current = node.params.get("T_out")
    if current is None:
        return graph, score, changes, False
    current = float(current)

    constraints = _build_temp_constraints(graph, node)
    rejected_constraint = 0

    for step in steps:
        for delta in (+step, -step):
            candidate = round(current + delta, 2)
            if not (50.0 < candidate < 2000.0):
                continue

            result = _solver.resolve("T_out", candidate, constraints)
            new_val = round(result.resolved_value, 2)
            if abs(new_val - candidate) > 2.0:
                rejected_constraint += 1
                continue  # constraint would require a large forced adjustment

            g2 = graph.copy()
            # Sanctioned physical correction: override + honest retag (no silent overwrite).
            g2.unit(tag).correct_param("T_out", new_val, Source.COMPUTED)

            rep2 = validate(g2)
            mag  = abs(new_val - current) / max(abs(current), 1.0)
            sc2  = _obj(rep2, mag)

            if sc2 < score:
                changes = changes + [
                    f"LOCAL_OPT: {tag}.T_out {current:.1f}→{new_val:.1f} K "
                    f"(Δ={delta:+.1f} K, score {score:.1f}→{sc2:.1f})"
                ]
                if rejected_constraint > 0:
                    logger.debug(
                        "LOCAL_OPT %s.T_out: accepted Δ%+.1f K "
                        "(%d constraint-rejected moves before this)",
                        tag, delta, rejected_constraint,
                    )
                return g2, sc2, changes, True

    if rejected_constraint > 0:
        logger.debug(
            "LOCAL_OPT %s.T_out: %d constraint rejections, no improvement found",
            tag, rejected_constraint,
        )
    return graph, score, changes, False


def _try_pressure(
    graph:   FlowsheetGraph,
    tag:     str,
    score:   float,
    changes: list[str],
    ratios:  list[float],
) -> tuple[FlowsheetGraph, float, list[str], bool]:
    node = graph.unit(tag)
    if node is None:
        return graph, score, changes, False

    current = node.params.get("P_out")
    if current is None:
        return graph, score, changes, False
    current = float(current)

    for ratio in ratios:
        for delta_r in (+ratio, -ratio):
            candidate = round(current * (1.0 + delta_r), 0)
            if not (100.0 < candidate < 1e8):
                continue

            g2 = graph.copy()
            # Sanctioned physical correction: override + honest retag (no silent overwrite).
            g2.unit(tag).correct_param("P_out", candidate, Source.COMPUTED)

            rep2 = validate(g2)
            mag  = abs(candidate - current) / max(current, 1.0)
            sc2  = _obj(rep2, mag)

            if sc2 < score:
                changes = changes + [
                    f"LOCAL_OPT: {tag}.P_out {current:.0f}→{candidate:.0f} Pa "
                    f"(Δ={delta_r:+.0%}, score {score:.1f}→{sc2:.1f})"
                ]
                logger.debug(
                    "LOCAL_OPT %s.P_out: accepted Δ%+.0%% (score %.1f→%.1f)",
                    tag, delta_r, score, sc2,
                )
                return g2, sc2, changes, True

    return graph, score, changes, False


# ── Objective ─────────────────────────────────────────────────────────────────

def _obj(report: object, magnitude: float) -> float:
    return len(report.errors()) * 100 + len(report.warnings()) + magnitude * 0.5


# ── Constraint builder ────────────────────────────────────────────────────────

def _build_temp_constraints(
    graph: FlowsheetGraph,
    node:  object,
) -> list[Constraint]:
    constraints: list[Constraint] = []

    inlets = graph.inlet_streams(node.tag)
    in_T = next((s.T for s in inlets if s.T is not None), None)
    in_P = next((s.P for s in inlets if s.P is not None), 101_325.0)

    if isinstance(node, HeaterNode):
        if in_T is not None:
            constraints.append(Constraint(
                param    = "T_out",
                priority = ConstraintPriority.UNIT_CONSTRAINT,
                source   = "Heater: T_out > feed T",
                min_val  = in_T + 1.0,
            ))
        bp = bubble_point_K(graph.compounds, in_P or 101_325.0)
        for outlet in graph.outlet_streams(node.tag):
            dst_tag = graph.stream_dest(outlet.tag)
            if dst_tag:
                dst = graph.unit(dst_tag)
                if isinstance(dst, SeparatorNode) and bp is not None:
                    constraints.append(Constraint(
                        param    = "T_out",
                        priority = ConstraintPriority.PHYSICAL_FEASIBILITY,
                        source   = f"Vessel flash: T > BP={bp:.1f}+5",
                        min_val  = bp + 5.0,
                    ))

    elif isinstance(node, CoolerNode):
        if in_T is not None:
            constraints.append(Constraint(
                param    = "T_out",
                priority = ConstraintPriority.UNIT_CONSTRAINT,
                source   = "Cooler: T_out < feed T",
                max_val  = in_T - 1.0,
            ))
        constraints.append(Constraint(
            param    = "T_out",
            priority = ConstraintPriority.PHYSICAL_FEASIBILITY,
            source   = "absolute lower bound",
            min_val  = 150.0,
        ))

    return constraints
