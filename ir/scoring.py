"""
Multi-factor candidate scoring for flowsheet ranking (Item 1).

CandidateScore encodes 8 interpretable components combined into a scalar
via a fixed weighted sum.  Weights are justified by ablation analysis
(see ARCHITECTURE.md §Ablation Study Design).

Component weights (sum = 1.00):
  valid_ir               0.35  — no schema/graph critical errors
  unit_appropriateness   0.15  — units match process description keywords
  param_completeness     0.12  — required params present across units
  thermo_consistency     0.10  — package/BIP alignment with compound system
  repair_economy         0.10  — low normaliser insertion count
  valid_json             0.08  — IR serialises without exception
  separation_feasibility 0.07  — Vessel has appropriate upstream conditioning
  phase_consistency      0.03  — labelled stream phases are coherent

Penalties (subtracted from weighted sum, floor at 0):
  excess_units_penalty       0.05 per unit above expected count for the unit set
  unphysical_param_penalty   0.05 per PHYSICS-level CRITICAL issue

The score is computed in two passes:
  Pass 1 (Stage 1–2): all components except thermo_consistency (no package yet)
  Pass 2 (Stage 3+):  thermo_consistency updated once ThermoMapper has run
  Pass 3 (optional):  converged component updated after DWSIM execution
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Optional

from ir.graph import (
    FlowsheetGraph, SeparatorNode, HeaterNode, CoolerNode,
    PumpNode, CompressorNode, ExpanderNode,
)
from ir.types import ErrorSeverity


# ── Unit-keyword matching ──────────────────────────────────────────────────────

_UNIT_KEYWORDS: dict[str, set[str]] = {
    "Heater":     {"heat", "warm", "preheat", "vaporis", "vaporiz", "raise temp",
                   "hot", "boil", "reboil", "furnace"},
    "Cooler":     {"cool", "chill", "condense", "condenser", "refrigerat", "cold",
                   "quench", "freeze"},
    "Vessel":     {"flash", "separat", "vessel", "drum", "distil", "phase",
                   "liquid-vapour", "vapor-liquid", "vle", "strip", "absorb"},
    "Mixer":      {"mix", "blend", "combin", "merge", "join"},
    "Splitter":   {"split", "divid", "recycle", "bypass", "purge"},
    "Pump":       {"pump", "pressuri", "pressurize"},
    "Compressor": {"compress", "compressor", "pressuri"},
    "Expander":   {"expand", "expander", "turbine", "let down", "throttle"},
}

# Package–compound-class compatibility
# (polar = ALCOHOLS | KETONES | ESTERS | ETHERS | POLAR_OTHER | WATER)
_PKG_POLAR_COMPAT = {
    "Raoult's Law":          0.3,   # forbidden for polar/azeotropic systems
    "NRTL":                  1.0,
    "UNIQUAC":               1.0,
    "Peng-Robinson":         0.5,   # usable but suboptimal for polar
    "Soave-Redlich-Kwong":   0.4,
    "Lee-Kesler-Plöcker":    0.2,
}
_PKG_NONPOLAR_COMPAT = {
    "Raoult's Law":          0.7,
    "NRTL":                  0.6,
    "UNIQUAC":               0.6,
    "Peng-Robinson":         1.0,
    "Soave-Redlich-Kwong":   0.9,
    "Lee-Kesler-Plöcker":    0.8,
}
_PKG_GAS_COMPAT = {
    "Raoult's Law":          0.2,
    "NRTL":                  0.4,
    "UNIQUAC":               0.4,
    "Peng-Robinson":         1.0,
    "Soave-Redlich-Kwong":   0.9,
    "Lee-Kesler-Plöcker":    1.0,
}


# ── Score dataclass ────────────────────────────────────────────────────────────

@dataclass
class CandidateScore:
    """
    Scalar score with interpretable components.
    All components are in [0, 1]; total is weighted sum minus penalties, clipped to [0, 1].
    """
    # Validation
    valid_ir:               float = 0.0
    valid_json:             float = 0.0

    # Process quality
    unit_appropriateness:   float = 0.5   # 0.5 = neutral (unknown)
    separation_feasibility: float = 1.0   # 1.0 = N/A (no Vessel)
    thermo_consistency:     float = 0.5   # 0.5 until ThermoMapper runs

    # Construction cost
    repair_economy:         float = 1.0   # 1.0 = no normaliser insertions
    param_completeness:     float = 0.5   # 0.5 = no units have required params

    # Phase
    phase_consistency:      float = 1.0   # 1.0 = no labelled streams or all consistent

    # Penalties
    excess_units_penalty:      float = 0.0
    unphysical_param_penalty:  float = 0.0

    # Post-execution (optional)
    converged:              float = 0.0
    n_repair_iterations:    int   = 0

    # Context
    candidate_idx:  int   = 0
    n_units:        int   = 0
    n_streams:      int   = 0
    margin:         float = 0.0   # gap to next-best candidate score

    _WEIGHTS: ClassVar[dict[str, float]] = {
        "valid_ir":               0.35,
        "unit_appropriateness":   0.15,
        "param_completeness":     0.12,
        "thermo_consistency":     0.10,
        "repair_economy":         0.10,
        "valid_json":             0.08,
        "separation_feasibility": 0.07,
        "phase_consistency":      0.03,
    }

    @property
    def total(self) -> float:
        raw = sum(w * getattr(self, k) for k, w in self._WEIGHTS.items())
        penalties = self.excess_units_penalty + self.unphysical_param_penalty
        return max(0.0, min(1.0, raw - penalties))

    def breakdown(self) -> dict[str, float]:
        """Return weighted contribution of each component (for reporting)."""
        d = {
            k: round(self._WEIGHTS[k] * getattr(self, k), 4)
            for k in self._WEIGHTS
        }
        d["excess_units_penalty"]    = round(-self.excess_units_penalty,   4)
        d["unphysical_param_penalty"] = round(-self.unphysical_param_penalty, 4)
        d["TOTAL"] = round(self.total, 4)
        return d

    def __lt__(self, other: "CandidateScore") -> bool:
        return self.total < other.total


# ── Main scoring function ──────────────────────────────────────────────────────

def score_candidate(
    graph:        FlowsheetGraph,
    report,                          # ValidationReport
    repair_count: int = 0,
    description:  str = "",
    candidate_idx: int = 0,
) -> CandidateScore:
    """
    Compute CandidateScore from a graph and its ValidationReport.
    Call update_thermo() after ThermoMapper sets property_package.
    """
    from ir.validate import ValidationReport
    assert isinstance(report, ValidationReport)

    score = CandidateScore(candidate_idx=candidate_idx)
    score.n_units   = len(graph.units())
    score.n_streams = len(graph.streams())

    # ── valid_ir ──────────────────────────────────────────────────────────────
    critical_schema_graph = [
        i for i in report.issues
        if i.level in ("SCHEMA", "GRAPH")
        and i.error.severity == ErrorSeverity.CRITICAL
    ]
    score.valid_ir = 1.0 if not critical_schema_graph else 0.0

    # ── valid_json ────────────────────────────────────────────────────────────
    try:
        from ir.to_dwsim import to_dwsim
        to_dwsim(graph)
        score.valid_json = 1.0
    except Exception:
        score.valid_json = 0.0

    # ── unit_appropriateness ──────────────────────────────────────────────────
    score.unit_appropriateness = _unit_appropriateness(graph, description)

    # ── separation_feasibility ────────────────────────────────────────────────
    score.separation_feasibility = _separation_feasibility(graph)

    # ── repair_economy ────────────────────────────────────────────────────────
    score.repair_economy = max(0.0, 1.0 - repair_count / 4.0)

    # ── param_completeness ────────────────────────────────────────────────────
    score.param_completeness = _param_completeness(graph)

    # ── phase_consistency ─────────────────────────────────────────────────────
    score.phase_consistency = _phase_consistency(graph, report)

    # ── excess_units_penalty ──────────────────────────────────────────────────
    score.excess_units_penalty = _excess_units_penalty(graph, description)

    # ── unphysical_param_penalty ──────────────────────────────────────────────
    physics_critical = [
        i for i in report.issues
        if i.level == "PHYSICS"
        and i.error.severity == ErrorSeverity.CRITICAL
    ]
    score.unphysical_param_penalty = min(0.30, len(physics_critical) * 0.05)

    # thermo_consistency remains 0.5 until update_thermo() is called
    return score


def update_thermo(score: CandidateScore, graph: FlowsheetGraph) -> CandidateScore:
    """
    Update thermo_consistency once property_package is set by ThermoMapper.
    Returns the same score object (mutated in-place) for chaining.
    """
    score.thermo_consistency = _thermo_consistency(graph)
    return score


def update_convergence(
    score:             CandidateScore,
    converged:         bool,
    n_repair_iters:    int = 0,
) -> CandidateScore:
    """Update score after DWSIM execution."""
    score.converged         = 1.0 if converged else 0.0
    score.n_repair_iterations = n_repair_iters
    return score


def compute_margin(scores: list[CandidateScore]) -> list[CandidateScore]:
    """
    Compute pairwise margin between top-1 and top-2.
    Mutates the top candidate's margin field.
    """
    if len(scores) < 2:
        if scores:
            scores[0].margin = 1.0   # only candidate — maximum confidence
        return scores
    sorted_scores = sorted(scores, key=lambda s: s.total, reverse=True)
    sorted_scores[0].margin = sorted_scores[0].total - sorted_scores[1].total
    return sorted_scores


# ── Component implementations ──────────────────────────────────────────────────

def _unit_appropriateness(graph: FlowsheetGraph, description: str) -> float:
    units = graph.units()
    if not units:
        return 0.0
    if not description:
        return 0.5   # neutral when description absent

    desc_lower = description.lower()
    n_appropriate = 0
    for node in units:
        kw_set = _UNIT_KEYWORDS.get(node.unit_type, set())
        if any(kw in desc_lower for kw in kw_set):
            n_appropriate += 1

    return n_appropriate / len(units)


def _separation_feasibility(graph: FlowsheetGraph) -> float:
    vessels = [n for n in graph.units() if isinstance(n, SeparatorNode)]
    if not vessels:
        return 1.0   # N/A — no Vessel present

    scores = []
    for vessel in vessels:
        # Walk upstream to find nearest conditioning unit
        inlets = graph.inlet_streams(vessel.tag)
        upstream_unit_types: set[str] = set()
        for s in inlets:
            src_tag = graph.stream_source(s.tag)
            if src_tag:
                upstream_node = graph.unit(src_tag)
                if upstream_node:
                    upstream_unit_types.add(upstream_node.unit_type)

        # Has T-conditioning upstream → good
        has_heater  = "Heater" in upstream_unit_types
        has_cooler  = "Cooler" in upstream_unit_types
        has_feed_t  = any(s.T is not None and s.T > 273.15 + 50
                         for s in inlets)   # T meaningfully above ambient

        vessel_score = 0.3   # baseline
        if has_heater:
            vessel_score = 1.0
        elif has_cooler:
            vessel_score = 0.8   # partial condensation is valid
        elif has_feed_t:
            vessel_score = 0.7   # some T info available
        scores.append(vessel_score)

    return sum(scores) / len(scores)


def _thermo_consistency(graph: FlowsheetGraph) -> float:
    pkg = graph.property_package
    if not pkg or not graph.compounds:
        return 0.5

    # Classify compound system
    from rag.retriever import ThermoRetriever
    tr = ThermoRetriever()
    classes = tr._classify(graph.compounds)
    has_azeo = tr._has_azeotrope(graph.compounds)

    is_polar = bool(classes & {"ALCOHOLS", "KETONES", "ESTERS", "ETHERS",
                                "POLAR_OTHER", "WATER"})
    is_light_gas = "LIGHT_GASES" in classes and not is_polar

    if has_azeo or is_polar:
        return _PKG_POLAR_COMPAT.get(pkg, 0.5)
    if is_light_gas:
        return _PKG_GAS_COMPAT.get(pkg, 0.5)
    return _PKG_NONPOLAR_COMPAT.get(pkg, 0.5)


def _repair_economy(repair_count: int) -> float:
    return max(0.0, 1.0 - repair_count / 4.0)


def _param_completeness(graph: FlowsheetGraph) -> float:
    total_required = 0
    total_present  = 0
    for node in graph.units():
        req = node.__class__.REQUIRED_PARAMS
        for p in req:
            total_required += 1
            if p in node.params:
                total_present += 1
    if total_required == 0:
        return 1.0   # no required params for this unit set
    return total_present / total_required


def _phase_consistency(graph: FlowsheetGraph, report) -> float:
    labelled = [s for s in graph.streams() if s.phase not in ("mixed", "any")]
    if not labelled:
        return 1.0   # no phase labels — cannot evaluate, treat as consistent

    from ir.types import ErrorType
    phase_issues = {
        i.error.target.tag
        for i in report.issues
        if i.error.error_type in (ErrorType.PHASE_MISMATCH, ErrorType.INVALID_TOPOLOGY)
        and i.level == "PHYSICS"
    }
    n_inconsistent = sum(1 for s in labelled if s.tag in phase_issues)
    return 1.0 - n_inconsistent / len(labelled)


def _excess_units_penalty(graph: FlowsheetGraph, description: str) -> float:
    """
    Penalise units that have no keyword justification in the description.
    0.05 penalty per unjustified unit, capped at 0.20.
    """
    if not description:
        return 0.0
    desc_lower = description.lower()
    unjustified = 0
    for node in graph.units():
        kw_set = _UNIT_KEYWORDS.get(node.unit_type, set())
        if kw_set and not any(kw in desc_lower for kw in kw_set):
            unjustified += 1
    return min(0.20, unjustified * 0.05)
