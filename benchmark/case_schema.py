"""
Case schema: load, validate and query benchmark cases from JSON files.

Cases are stored in benchmark/cases/<tier>.json, one file per tier.
Each case is a dict that matches the BenchmarkCaseSpec dataclass.

Usage:
    from benchmark.case_schema import load_all, load_tier, BenchmarkCaseSpec

    all_cases = load_all()
    hard_cases = load_tier("hard")
    case = load_by_id("HARD_01")
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

_CASES_DIR = os.path.join(os.path.dirname(__file__), "cases")

TIERS = [
    "sanity", "easy", "medium", "hard", "perturbation", "generalisation",
    # Capability-demo cases (P1/F1/C1/...): independently-built references, each
    # with an independent validation route (NIST / closed-form / published VLE).
    "capability",
    # Extended hard-benchmark tiers — each maps to a separate file via _TIER_FILENAMES
    "multi_unit", "missing_bip", "ambiguous", "adversarial",
    # Ground-truth validation tiers — real DWSIM/FOSSEE flowsheets, split by the
    # CORRECTED material-unit count into complexity bins (replaces the flat
    # "validation" tier).  Default "<tier>.json" filename pattern applies.
    "val_3_5", "val_6_9", "val_10_14", "val_15plus",
]

# The validation family — real-sourced reference cases that need the large
# extraction token budget, the no-summarisation path, and reference-MAPE scoring.
# Behaviour keys on this predicate, NOT a literal tier string, so all complexity
# bins (and the legacy "validation" string) are treated identically.
def is_validation_tier(tier: str) -> bool:
    return tier == "validation" or tier.startswith("val_")

# Tiers whose filename differs from the default "<tier>.json" pattern.
_TIER_FILENAMES: dict[str, str] = {
    "multi_unit": "hard_benchmark_mu.json",
    "missing_bip": "hard_benchmark_mb.json",
    "ambiguous": "hard_benchmark_amb.json",
    "adversarial": "hard_benchmark_adv.json",
}


@dataclass
class PhysicsCheck:
    check_type:  str
    params:      dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"PhysicsCheck({self.check_type}, {self.params})"


@dataclass
class ReferenceStructure:
    n_units:    int
    unit_types: list[str]
    connections: list[list[str]]


@dataclass
class ExpectedBehaviour:
    property_package_class: Optional[str]  # "activity_coefficient" | "eos" | "ideal"
    n_units_min:             int = 1
    n_units_max:             int = 10
    convergence_expected:    bool = True
    physics_checks:          list[PhysicsCheck] = field(default_factory=list)


@dataclass
class BenchmarkCaseSpec:
    id:                  str
    name:                str
    tier:                str
    difficulty:          str    # easy | medium | hard
    coupling_level:      str    # low | medium | high
    perturbation:        str    # none | mild | severe
    domain:              str    # hydrocarbon | polar | azeotrope | mixed
    description:         str
    compounds:           list[str]
    expected:            ExpectedBehaviour
    reference_structure: Optional[ReferenceStructure] = None
    # Perturbation-specific
    base_id:             Optional[str] = None
    perturbation_type:   Optional[str] = None
    perturbation_magnitude: Optional[float] = None
    perturbation_direction: Optional[str] = None
    recovery_expected:   bool = True
    # Generalisation-specific
    transfer_from:       Optional[str] = None
    structural_analog:   Optional[str] = None
    notes:               str = ""
    # Validation-specific — path to ground-truth reference flowsheet JSON
    reference_file:      Optional[str] = None
    # Validation-specific — short NL prompt fed to the pipeline (distinct from
    # the long documentation description).  Falls back to description if absent.
    prompt:              Optional[str] = None


def _parse_physics_check(raw: dict) -> PhysicsCheck:
    t = raw.get("type", raw.get("check_type", ""))
    params = {k: v for k, v in raw.items() if k not in ("type", "check_type")}
    return PhysicsCheck(check_type=t, params=params)


def _parse_expected(raw: dict) -> ExpectedBehaviour:
    checks = [_parse_physics_check(c) for c in raw.get("physics_checks", [])]
    return ExpectedBehaviour(
        property_package_class = raw.get("property_package_class"),
        n_units_min            = raw.get("n_units_min", 1),
        n_units_max            = raw.get("n_units_max", 10),
        convergence_expected   = raw.get("convergence_expected", True),
        physics_checks         = checks,
    )


def _parse_reference(raw: Optional[dict]) -> Optional[ReferenceStructure]:
    if raw is None:
        return None
    return ReferenceStructure(
        n_units     = raw.get("n_units", 0),
        unit_types  = raw.get("unit_types", []),
        connections = raw.get("connections", []),
    )


def _parse_case(raw: dict) -> BenchmarkCaseSpec:
    return BenchmarkCaseSpec(
        id                      = raw["id"],
        name                    = raw.get("name", raw["id"]),
        tier                    = raw["tier"],
        difficulty              = raw.get("difficulty", "medium"),
        coupling_level          = raw.get("coupling_level", "medium"),
        perturbation            = raw.get("perturbation", "none"),
        domain                  = raw.get("domain", "mixed"),
        description             = raw["description"],
        compounds               = raw["compounds"],
        expected                = _parse_expected(raw.get("expected", {})),
        reference_structure     = _parse_reference(raw.get("reference_structure")),
        base_id                 = raw.get("base_id"),
        perturbation_type       = raw.get("perturbation_type"),
        perturbation_magnitude  = raw.get("perturbation_magnitude"),
        perturbation_direction  = raw.get("perturbation_direction"),
        recovery_expected       = raw.get("recovery_expected", True),
        transfer_from           = raw.get("transfer_from"),
        structural_analog       = raw.get("structural_analog"),
        notes                   = raw.get("notes", ""),
        reference_file          = raw.get("reference_file"),
        prompt                  = raw.get("prompt"),
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def load_tier(tier: str) -> list[BenchmarkCaseSpec]:
    """Load all cases from a single tier JSON file."""
    filename = _TIER_FILENAMES.get(tier, f"{tier}.json")
    path = os.path.join(_CASES_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No case file for tier '{tier}' at {path}")
    with open(path, encoding="utf-8") as f:
        raw_cases = json.load(f)
    return [_parse_case(c) for c in raw_cases]


def load_from_files(paths: list[str]) -> list[BenchmarkCaseSpec]:
    """Load cases from arbitrary JSON file paths, bypassing the tier system."""
    cases: list[BenchmarkCaseSpec] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for raw in json.load(f):
                cases.append(_parse_case(raw))
    return cases


def load_all(
    tiers: list[str] | None = None,
    difficulty: str | None = None,
    domain: str | None = None,
    perturbation: str | None = None,
) -> list[BenchmarkCaseSpec]:
    """
    Load cases across all (or specified) tiers, with optional filters.

    Parameters
    ----------
    tiers       : subset of TIERS to load; None = all
    difficulty  : filter by difficulty label
    domain      : filter by domain label
    perturbation: filter by perturbation label
    """
    selected_tiers = tiers or TIERS
    cases: list[BenchmarkCaseSpec] = []
    for tier in selected_tiers:
        filename = _TIER_FILENAMES.get(tier, f"{tier}.json")
        path = os.path.join(_CASES_DIR, filename)
        if os.path.exists(path):
            cases.extend(load_tier(tier))
    if difficulty:
        cases = [c for c in cases if c.difficulty == difficulty]
    if domain:
        cases = [c for c in cases if c.domain == domain]
    if perturbation:
        cases = [c for c in cases if c.perturbation == perturbation]
    return cases


def load_by_id(case_id: str) -> BenchmarkCaseSpec:
    """Return a single case by ID, searching all tiers."""
    for tier in TIERS:
        filename = _TIER_FILENAMES.get(tier, f"{tier}.json")
        path = os.path.join(_CASES_DIR, filename)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for raw in json.load(f):
                if raw.get("id") == case_id:
                    return _parse_case(raw)
    raise KeyError(f"Case '{case_id}' not found in any tier")


def case_index() -> dict[str, BenchmarkCaseSpec]:
    """Return all cases as {id: spec} dict."""
    return {c.id: c for c in load_all()}


def summary() -> dict:
    """Return a count summary for reporting."""
    counts: dict[str, int] = {}
    total = 0
    for tier in TIERS:
        filename = _TIER_FILENAMES.get(tier, f"{tier}.json")
        path = os.path.join(_CASES_DIR, filename)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                n = len(json.load(f))
            counts[tier] = n
            total += n
    counts["total"] = total
    return counts
