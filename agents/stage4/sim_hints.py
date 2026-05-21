"""
Simulation outcome signals for repair scoring and target prioritisation (Item 5 upgrade).

SimulationHints is extracted from an ExecutionResult after DWSIM runs.
It is passed to RepairAgent so the repair loop can:
  • prioritise units that actually failed in simulation (not just IR-invalid)
  • penalise candidates that are unlikely to converge (e.g., same thermo failure)
  • skip re-fixing units that already converged correctly
  • bias candidate generation direction (increase/decrease T or P)
  • detect phase mismatches and severity of failure per unit

New in v2 (Item 5 — Simulation Signal Backpropagation):
  UnitSignal  — per-unit directional hint + severity + phase mismatch type
  directional_hint(tag, param) — "increase" | "decrease" | None
  severity(tag) — 0.0–1.0 failure severity
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

# Match unit tags in error strings: 'H-01', "V-01", [P-01], or bare H01
_TAG_RE = re.compile(r"""['\"\[]?([A-Z][A-Z0-9]*-\d+)['\"\]]?""", re.ASCII)

# Direction-extracting patterns  (match against lower-cased error string)
_INCREASE_T_PATTERNS = [
    "below bubble", "below dew", "flash failed", "no vapour",
    "two-phase", "two phase", "under-heated", "cold feed", "subcooled",
]
_DECREASE_T_PATTERNS = [
    "above dew", "superheated", "no liquid", "overheated",
    "temperature too high", "vapour fraction",
]
_INCREASE_P_PATTERNS = [
    "cavitation", "pump inlet", "below vapour pressure", "suction pressure",
    "too low pressure",
]
_DECREASE_P_PATTERNS = [
    "pressure too high", "supercritical", "above critical",
]


@dataclass
class UnitSignal:
    """
    Directional and severity signal extracted from one unit's simulation error.
    """
    tag:           str
    direction_T:   Optional[str]  = None   # "increase" | "decrease" | None
    direction_P:   Optional[str]  = None   # "increase" | "decrease" | None
    severity:      float          = 0.5    # 0.0 (minor) – 1.0 (fatal)
    phase_mismatch: Optional[str] = None   # "liquid_needed" | "vapour_needed" | None
    raw_message:   str            = ""


@dataclass
class SimulationHints:
    """
    Lightweight summary of a DWSIM execution result.

    Fields:
        converged      — True when the solver finished without fatal error
        unit_errors    — map unit_tag → error message for units that failed
        stream_zero    — set of stream tags where flow rate was ~0 (no product)
        iteration      — which repair loop iteration produced this execution
        unit_signals   — per-unit directional / severity signals (v2)
    """
    converged:    bool                      = False
    unit_errors:  dict[str, str]            = field(default_factory=dict)
    stream_zero:  set[str]                  = field(default_factory=set)
    iteration:    int                       = 0
    unit_signals: dict[str, UnitSignal]     = field(default_factory=dict)

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_execution(
        cls,
        execution: Any,
        iteration: int = 0,
    ) -> "SimulationHints":
        """
        Build SimulationHints from whatever ExecutionResult the executor returns.
        Tolerant of missing attributes so it works across executor versions.
        """
        hints = cls(
            converged = bool(getattr(execution, "solved", False)),
            iteration = iteration,
        )

        diag_unit = None
        diag = getattr(execution, "diagnostics", None) or {}
        if isinstance(diag, dict):
            diag_unit = diag.get("unit")

        all_errors = list(getattr(execution, "errors", []) or [])
        all_errors += list(getattr(execution, "solver_errors", []) or [])

        for err in all_errors:
            tag = (getattr(err, "unit_tag", None)
                   or getattr(err, "tag", None)
                   or getattr(err, "target", None))
            if tag is None:
                m = _TAG_RE.search(str(err))
                if m:
                    tag = m.group(1)
            if tag is None and diag_unit:
                tag = diag_unit
            if tag:
                err_str = str(err)
                hints.unit_errors[str(tag)] = err_str
                hints.unit_signals[str(tag)] = _parse_unit_signal(str(tag), err_str)

        for stream in getattr(execution, "streams", []) or []:
            tag  = getattr(stream, "tag", None)
            flow = getattr(stream, "flow", None) or getattr(stream, "molar_flow", None)
            if tag and flow is not None and float(flow) < 1e-12:
                hints.stream_zero.add(str(tag))

        return hints

    # ── Query helpers ──────────────────────────────────────────────────────────

    def failed_units(self) -> set[str]:
        return set(self.unit_errors.keys())

    def unit_failed(self, tag: str) -> bool:
        return tag in self.unit_errors

    def directional_hint(self, tag: str, param: str) -> Optional[str]:
        """
        Return "increase" or "decrease" for (tag, param), or None if unknown.
        Used by candidate generation to bias direction.
        """
        sig = self.unit_signals.get(tag)
        if sig is None:
            return None
        if param == "T_out":
            return sig.direction_T
        if param == "P_out":
            return sig.direction_P
        return None

    def unit_severity(self, tag: str) -> float:
        """Failure severity 0–1 for the given unit (0.0 if not failed)."""
        sig = self.unit_signals.get(tag)
        return sig.severity if sig else 0.0

    def phase_mismatch(self, tag: str) -> Optional[str]:
        sig = self.unit_signals.get(tag)
        return sig.phase_mismatch if sig else None

    def convergence_penalty(self) -> float:
        """
        Additional score penalty for non-converged states.
        Higher penalty at later iterations.
        """
        if self.converged:
            return 0.0
        return 30.0 + self.iteration * 5.0

    def priority_boost(self, unit_tag: str) -> float:
        """
        Score boost for candidates targeting simulation-failed units.
        Negative = better (lower score = better candidate).
        """
        if unit_tag not in self.unit_errors:
            return 0.0
        severity = self.unit_severity(unit_tag)
        return -20.0 * (1.0 + severity)   # severe failures boosted more

    def no_hints(self) -> bool:
        return not self.unit_errors and not self.stream_zero


EMPTY_HINTS = SimulationHints(converged=True)


# ── Signal extraction ─────────────────────────────────────────────────────────

def _tokens_match(
    tokens:   set[str],
    patterns: list[str],
    full_str: str,
) -> bool:
    """
    Match patterns against an error string.
    Single-word patterns: check token membership (handles articles).
    Multi-word patterns: check all words present in any order.
    """
    for pattern in patterns:
        words = pattern.split()
        if len(words) == 1:
            if words[0] in tokens:
                return True
        else:
            if all(w in tokens for w in words):
                return True
        if pattern in full_str:
            return True
    return False


def _parse_unit_signal(tag: str, message: str) -> UnitSignal:
    """
    Extract directional and severity signals from a unit error message.
    Heuristic pattern matching — purely deterministic.
    """
    lower = message.lower()

    # Direction for temperature — use token-based matching to tolerate articles
    tokens = set(lower.split())
    dir_T: Optional[str] = None
    if _tokens_match(tokens, _INCREASE_T_PATTERNS, lower):
        dir_T = "increase"
    elif _tokens_match(tokens, _DECREASE_T_PATTERNS, lower):
        dir_T = "decrease"

    # Direction for pressure
    dir_P: Optional[str] = None
    if _tokens_match(tokens, _INCREASE_P_PATTERNS, lower):
        dir_P = "increase"
    elif _tokens_match(tokens, _DECREASE_P_PATTERNS, lower):
        dir_P = "decrease"

    # Phase mismatch
    phase_mm: Optional[str] = None
    if "liquid" in lower and ("required" in lower or "needed" in lower or "inlet" in lower):
        phase_mm = "liquid_needed"
    elif "vapour" in lower and ("required" in lower or "needed" in lower or "inlet" in lower):
        phase_mm = "vapour_needed"
    elif "gas" in lower and ("required" in lower or "needed" in lower):
        phase_mm = "vapour_needed"

    # Severity heuristic: fatal keywords → 1.0, solver keywords → 0.7, others → 0.5
    if any(kw in lower for kw in ("fatal", "infeasible", "singular", "diverged")):
        severity = 1.0
    elif any(kw in lower for kw in ("failed", "solver", "no solution", "did not converge")):
        severity = 0.7
    else:
        severity = 0.5

    return UnitSignal(
        tag           = tag,
        direction_T   = dir_T,
        direction_P   = dir_P,
        severity      = severity,
        phase_mismatch= phase_mm,
        raw_message   = message[:200],
    )
