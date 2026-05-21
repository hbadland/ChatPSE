"""
Priority-based constraint conflict resolution.

When multiple constraints apply to the same parameter (e.g. T_out), they
may conflict: the description parser may extract 320 K while physical
feasibility demands T ≥ 366 K for flash separation.

Resolution algorithm:
  1. Collect all constraints on the parameter
  2. Build feasibility window: max(all lower bounds), min(all upper bounds)
  3. If window is empty (conflict), resolve by priority:
     - Keep highest-priority bounds; relax lower-priority ones until feasible
  4. Apply minimal adjustment: clamp current value into the feasibility window

Priority levels (lower int = higher priority):
  PHYSICAL_FEASIBILITY  1  — physics laws, thermodynamic limits
  UNIT_CONSTRAINT       2  — per-unit process constraints (heater must heat)
  HEURISTIC_DEFAULT     3  — estimator / description-parser values
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class ConstraintPriority(IntEnum):
    PHYSICAL_FEASIBILITY = 1   # highest — can never be relaxed
    UNIT_CONSTRAINT      = 2
    HEURISTIC_DEFAULT    = 3   # lowest — relaxed first in conflict


@dataclass
class Constraint:
    """Single bound on a parameter."""
    param:    str
    priority: ConstraintPriority
    source:   str                   # human-readable origin for logging
    min_val:  Optional[float] = None
    max_val:  Optional[float] = None

    def __post_init__(self) -> None:
        if self.min_val is not None and self.max_val is not None:
            if self.min_val > self.max_val:
                raise ValueError(
                    f"Constraint {self.source!r}: min_val={self.min_val} > max_val={self.max_val}")


@dataclass
class ResolutionResult:
    resolved_value: float
    conflict:       bool              = False
    dropped_sources: list[str]        = field(default_factory=list)
    reason:         str               = ""


class ConstraintSolver:
    """
    Resolves conflicts between multiple constraints on the same parameter.

    Usage:
        solver = ConstraintSolver()
        result = solver.resolve("T_out", current_value=320.0, constraints=[...])
        if result.conflict:
            log(f"Conflict: {result.reason} — dropped {result.dropped_sources}")
        node.params["T_out"] = result.resolved_value
    """

    def resolve(
        self,
        param:       str,
        current:     float,
        constraints: list[Constraint],
    ) -> ResolutionResult:
        if not constraints:
            return ResolutionResult(resolved_value=current)

        # ── Step 1: Find feasibility window across all constraints ─────────────
        lo = -math.inf
        hi =  math.inf
        for c in constraints:
            if c.min_val is not None:
                lo = max(lo, c.min_val)
            if c.max_val is not None:
                hi = min(hi, c.max_val)

        # ── Step 2: If feasible, apply minimal adjustment ──────────────────────
        if lo <= hi:
            if current < lo:
                return ResolutionResult(
                    resolved_value = lo,
                    conflict       = False,
                    reason         = f"clamped up to min={lo:.2f}",
                )
            if current > hi:
                return ResolutionResult(
                    resolved_value = hi,
                    conflict       = False,
                    reason         = f"clamped down to max={hi:.2f}",
                )
            return ResolutionResult(resolved_value=current)

        # ── Step 3: Conflict — resolve by priority ─────────────────────────────
        # Sort descending by priority (highest priority = smallest IntEnum value)
        by_priority = sorted(constraints, key=lambda c: c.priority)
        dropped: list[str] = []

        lo_res = -math.inf
        hi_res =  math.inf
        last_priority = None

        for c in by_priority:
            # Try adding this constraint's bound
            new_lo = max(lo_res, c.min_val) if c.min_val is not None else lo_res
            new_hi = min(hi_res, c.max_val) if c.max_val is not None else hi_res

            if new_lo <= new_hi:
                lo_res, hi_res = new_lo, new_hi
                last_priority  = c.priority
            else:
                # This constraint causes conflict — drop it
                dropped.append(c.source)

        if lo_res > hi_res:
            # Still infeasible after priority ordering — use highest-priority bounds only
            phys = [c for c in by_priority
                    if c.priority == ConstraintPriority.PHYSICAL_FEASIBILITY]
            if phys:
                lo_res = max((c.min_val or -math.inf) for c in phys)
                hi_res = min((c.max_val or  math.inf) for c in phys)

        # Minimal adjustment from current
        resolved = current
        if current < lo_res:
            resolved = lo_res
        elif current > hi_res:
            resolved = hi_res

        return ResolutionResult(
            resolved_value  = resolved,
            conflict        = True,
            dropped_sources = dropped,
            reason          = (
                f"conflict resolved by priority; dropped "
                f"{dropped} to achieve window [{lo_res:.2f}, {hi_res:.2f}]"
            ),
        )
