"""
Real-case hook for the benchmarking pipeline.

Real cases are actual industrial or lab process descriptions collected from
literature or domain experts. They are structurally identical to BenchmarkCase
but are NEVER used for optimisation, tuning, or prompt development.

Usage:
  from eval.real_cases import REAL_CASES, load_real_cases
  results, metrics = run_benchmark(orchestrator, cases=REAL_CASES)

Adding a real case:
  Append a RealCase entry to REAL_CASES below.
  Do NOT add expected_units or expected_pkg unless you have a verified
  simulation result — leave them empty to avoid biasing evaluation.

Design rules:
  - No real case may share a description with any benchmark case.
  - Real cases are never split into dev/holdout — all are treated as test-only.
  - The source field documents provenance (textbook, paper, internal report).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from eval.benchmark_cases import BenchmarkCase


@dataclass
class RealCase(BenchmarkCase):
    """
    A real industrial/lab case. Inherits all BenchmarkCase fields.

    Extra fields:
      source      : provenance string (textbook, DOI, internal)
      verified    : True if a reference DWSIM simulation exists
      ref_pkg     : property package used in reference simulation (if known)
    """
    source:   str = ""
    verified: bool = False
    ref_pkg:  str = ""

    # Override tier to always be "real"
    def __post_init__(self) -> None:
        object.__setattr__(self, "tier", "real")


# ── Real case registry ─────────────────────────────────────────────────────────
# Add entries here. Keep descriptions verbatim from the source.

REAL_CASES: list[RealCase] = [

    RealCase(
        case_id       = "REAL01",
        description   = (
            "Distil a binary mixture of ethanol and water at 1 atm. "
            "Feed is equimolar at 80°C. Recover ethanol at 92 mol% purity "
            "in the distillate."
        ),
        compounds     = ["Ethanol", "Water"],
        tier          = "real",
        expected_pkg  = "NRTL",
        expected_units = ["DistillationColumn"],
        notes         = "Classic binary distillation from Perry's Handbook, 8th ed., §13.",
        source        = "Perry's Chemical Engineers' Handbook, 8th ed., §13",
        verified      = False,
        ref_pkg       = "NRTL",
    ),

    RealCase(
        case_id       = "REAL02",
        description   = (
            "Natural gas dehydration using triethylene glycol (TEG). "
            "Wet gas at 70 bar and 35°C contacts TEG in an absorber. "
            "Lean TEG is regenerated in a stripper at 1.2 bar."
        ),
        compounds     = ["Methane", "Ethane", "Water", "Triethylene glycol"],
        tier          = "real",
        expected_pkg  = "Peng-Robinson",
        expected_units = ["AbsorptionColumn", "DistillationColumn", "Pump", "Heater"],
        notes         = "TEG dehydration reference case; Soave-Redlich-Kwong also acceptable.",
        source        = "Gas Conditioning and Processing, Vol. 2, Campbell Petroleum Series",
        verified      = False,
        ref_pkg       = "Peng-Robinson",
    ),

    RealCase(
        case_id       = "REAL03",
        description   = (
            "Refrigeration cycle using propane as refrigerant. "
            "Compress saturated vapour from 1 bar to 10 bar, "
            "condense at 10 bar, expand through a valve, and evaporate at 1 bar."
        ),
        compounds     = ["Propane"],
        tier          = "real",
        expected_pkg  = "Peng-Robinson",
        expected_units = ["Compressor", "Cooler", "Expander", "Heater"],
        notes         = "Simple single-stage propane refrigeration loop.",
        source        = "Smith, J.M. et al., Introduction to Chemical Engineering Thermodynamics, 8th ed.",
        verified      = False,
        ref_pkg       = "Peng-Robinson",
    ),

]


# ── Accessor ───────────────────────────────────────────────────────────────────

def load_real_cases(verified_only: bool = False) -> list[RealCase]:
    """Return all real cases, optionally filtered to verified ones only."""
    if verified_only:
        return [c for c in REAL_CASES if c.verified]
    return list(REAL_CASES)


def real_case_by_id(case_id: str) -> Optional[RealCase]:
    for c in REAL_CASES:
        if c.case_id == case_id:
            return c
    return None
