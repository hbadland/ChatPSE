"""
Flowsheet confidence scoring (Item 5).

FlowsheetConfidence is computed from:
  - Validation success (did the IR pass schema/graph/physics?)
  - Candidate score and ranking margin (was the best candidate clearly better?)
  - Repair iteration count (how many loops were needed?)
  - BIP coverage (are all thermodynamic parameters available?)
  - Known limitations (flags for user awareness)

Confidence levels:
  HIGH      ≥ 0.85  — production-ready; all checks pass, low repair cost
  MEDIUM    0.65–0.85 — usable; minor issues or moderate repair cost
  LOW       0.40–0.65 — use with caution; significant issues or many repairs
  VERY_LOW  < 0.40  — unreliable; failed validation or max repairs exhausted

Exposed to the caller via PipelineResult.confidence (orchestrator_v2.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ir.validate import ValidationReport
from ir.scoring import CandidateScore


@dataclass
class FlowsheetConfidence:
    """Per-flowsheet confidence summary."""

    # Scalar confidence in [0, 1]
    score: float

    # Component breakdown
    validation_component:   float = 0.0   # 1.0 = fully valid IR
    repair_component:       float = 0.0   # 1.0 = zero repairs
    margin_component:       float = 0.0   # 1.0 = clear winner (margin ≥ 0.2)
    bip_component:          float = 0.0   # 1.0 = full BIP coverage
    convergence_component:  float = 0.0   # 1.0 = DWSIM solved (if known)

    # Flags (for human-readable limitations list)
    has_missing_bips:        bool = False
    has_unspecified_params:  bool = False
    has_phase_uncertainty:   bool = False
    has_unresolved_issues:   bool = False
    has_max_repairs:         bool = False

    # Limitations text
    limitations: list[str] = field(default_factory=list)

    @property
    def level(self) -> str:
        if self.score >= 0.85:
            return "HIGH"
        if self.score >= 0.65:
            return "MEDIUM"
        if self.score >= 0.40:
            return "LOW"
        return "VERY_LOW"

    def as_dict(self) -> dict:
        return {
            "score":                 round(self.score, 4),
            "level":                 self.level,
            "validation_component":  round(self.validation_component, 4),
            "repair_component":      round(self.repair_component, 4),
            "margin_component":      round(self.margin_component, 4),
            "bip_component":         round(self.bip_component, 4),
            "convergence_component": round(self.convergence_component, 4),
            "has_missing_bips":      self.has_missing_bips,
            "has_unspecified_params": self.has_unspecified_params,
            "has_phase_uncertainty": self.has_phase_uncertainty,
            "has_unresolved_issues": self.has_unresolved_issues,
            "limitations":           self.limitations,
        }

    def __str__(self) -> str:
        lim_str = "; ".join(self.limitations) if self.limitations else "none"
        return (f"FlowsheetConfidence({self.level}, score={self.score:.3f}, "
                f"limitations=[{lim_str}])")


# Component weights
_W_VALIDATION   = 0.35
_W_REPAIR       = 0.20
_W_MARGIN       = 0.15
_W_BIP          = 0.15
_W_CONVERGENCE  = 0.15


def compute_confidence(
    report:          Optional[ValidationReport] = None,
    candidate_score: Optional[CandidateScore]   = None,
    repair_iters:    int  = 0,
    max_iters:       int  = 6,
    has_bips:        bool = True,
    converged:       Optional[bool] = None,
    n_missing_params: int = 0,
) -> FlowsheetConfidence:
    """
    Compute FlowsheetConfidence from pipeline state.

    Parameters
    ----------
    report          : final ValidationReport after repair
    candidate_score : CandidateScore of the winning candidate
    repair_iters    : number of Stage 4 repair loop iterations used
    max_iters       : maximum allowed iterations (OrchestratorV2._max_iter)
    has_bips        : True if BIPs are fully covered for this package
    converged       : True/False if DWSIM ran; None if not yet executed
    n_missing_params: count of required params still unset after Stage 3
    """
    conf = FlowsheetConfidence(score=0.0)
    limitations: list[str] = []

    # ── Validation component ───────────────────────────────────────────────────
    if report is not None:
        n_critical = sum(
            1 for i in report.issues
            if i.error.severity.value == "CRITICAL")
        n_warnings = len(report.warnings())
        conf.has_unresolved_issues = n_critical > 0
        if n_critical == 0:
            conf.validation_component = 1.0 - min(0.3, n_warnings * 0.05)
        else:
            conf.validation_component = max(0.0, 0.5 - n_critical * 0.15)
        if n_critical > 0:
            limitations.append(f"{n_critical} unresolved validation error(s)")
        if n_warnings > 0:
            limitations.append(f"{n_warnings} validation warning(s)")
    else:
        conf.validation_component = 0.5   # unknown

    # ── Repair component ───────────────────────────────────────────────────────
    conf.has_max_repairs = (repair_iters >= max_iters and max_iters > 0)
    if max_iters > 0:
        conf.repair_component = max(0.0, 1.0 - repair_iters / max_iters)
    else:
        conf.repair_component = 1.0
    if conf.has_max_repairs:
        limitations.append("repair loop exhausted (may not be optimal)")

    # ── Margin component ───────────────────────────────────────────────────────
    if candidate_score is not None:
        margin = getattr(candidate_score, "margin", 0.0)
        conf.margin_component = min(1.0, margin / 0.20)   # full confidence at margin≥0.20
        if margin < 0.05:
            limitations.append("candidate selection was ambiguous (low margin)")
    else:
        conf.margin_component = 0.5

    # ── BIP component ─────────────────────────────────────────────────────────
    conf.has_missing_bips = not has_bips
    conf.bip_component    = 1.0 if has_bips else 0.2
    if not has_bips:
        limitations.append("binary interaction parameters unavailable — "
                           "VLE accuracy reduced")

    # ── Convergence component ─────────────────────────────────────────────────
    if converged is True:
        conf.convergence_component = 1.0
    elif converged is False:
        conf.convergence_component = 0.0
        limitations.append("DWSIM did not converge")
    else:
        conf.convergence_component = 0.5   # not yet executed

    # ── Additional flags ──────────────────────────────────────────────────────
    if n_missing_params > 0:
        conf.has_unspecified_params = True
        limitations.append(
            f"{n_missing_params} required parameter(s) unset — "
            "using defaults may give inaccurate results")

    if candidate_score is not None and candidate_score.phase_consistency < 0.8:
        conf.has_phase_uncertainty = True
        limitations.append("stream phase labels are partially inconsistent")

    conf.limitations = limitations

    # ── Weighted total ─────────────────────────────────────────────────────────
    conf.score = (
        _W_VALIDATION  * conf.validation_component
        + _W_REPAIR    * conf.repair_component
        + _W_MARGIN    * conf.margin_component
        + _W_BIP       * conf.bip_component
        + _W_CONVERGENCE * conf.convergence_component
    )

    return conf
