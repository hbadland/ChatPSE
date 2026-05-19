"""
Shared enums and typed error structures used across IR, validation, and agents.

Keeping enums here (not in agents/) means the IR layer stays self-contained
and agents import upward from ir/, not laterally between stage packages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Union


# ── Error taxonomy ─────────────────────────────────────────────────────────────

class ErrorType(str, Enum):
    MISSING_PARAM       = "MISSING_PARAM"        # BIPs absent, required param not set
    INVALID_TOPOLOGY    = "INVALID_TOPOLOGY"      # connectivity / port violation
    CONVERGENCE_FAILURE = "CONVERGENCE_FAILURE"   # solver diverged
    INVALID_UNIT_CONFIG = "INVALID_UNIT_CONFIG"   # T_out below bubble point, etc.
    UNPHYSICAL_VALUES   = "UNPHYSICAL_VALUES"     # T in °C instead of K, etc.
    PHASE_MISMATCH      = "PHASE_MISMATCH"        # vapour into Pump, liquid into Compressor
    MASS_BALANCE        = "MASS_BALANCE"          # inlet ≠ outlet flow
    INFEASIBLE          = "INFEASIBLE"            # thermodynamically impossible


class RepairStrategy(str, Enum):
    PARAM_INJECT    = "PARAM_INJECT"    # inject BIPs from corpus
    TOPOLOGY_FIX    = "TOPOLOGY_FIX"    # re-run graph normaliser
    THERMO_SWITCH   = "THERMO_SWITCH"   # change property package via RAG
    CONDITION_FIX   = "CONDITION_FIX"   # correct numerical parameter (LLM fallback)
    UNIT_CONVERSION = "UNIT_CONVERSION" # deterministic °C→K, bar→Pa
    DEFAULT_FILL    = "DEFAULT_FILL"    # fill missing params with spec defaults
    PORT_REPAIR     = "PORT_REPAIR"     # swap/reassign port numbers
    HUMAN           = "HUMAN"           # escalate — cannot repair automatically


class ErrorSeverity(str, Enum):
    CRITICAL = "CRITICAL"  # blocks simulation
    WARNING  = "WARNING"   # may cause wrong results


# ── Error target ───────────────────────────────────────────────────────────────

class TargetKind(str, Enum):
    UNIT   = "unit"
    STREAM = "stream"
    GLOBAL = "global"


@dataclass(frozen=True)
class ErrorTarget:
    kind: TargetKind
    tag:  str               # unit/stream tag, or "global"
    field: Optional[str] = None  # specific param/attribute, e.g. "T_out"

    @classmethod
    def unit(cls, tag: str, field: Optional[str] = None) -> "ErrorTarget":
        return cls(TargetKind.UNIT, tag, field)

    @classmethod
    def stream(cls, tag: str, field: Optional[str] = None) -> "ErrorTarget":
        return cls(TargetKind.STREAM, tag, field)

    @classmethod
    def global_(cls) -> "ErrorTarget":
        return cls(TargetKind.GLOBAL, "global")

    def __str__(self) -> str:
        s = f"{self.kind.value}:{self.tag}"
        if self.field:
            s += f".{self.field}"
        return s


# ── Typed simulation error ─────────────────────────────────────────────────────

@dataclass
class SimError:
    """
    A fully typed simulation error.  No free-form strings in routing logic —
    all branching is on enum values.
    """
    error_type:      ErrorType
    target:          ErrorTarget
    evidence:        str               # human-readable value / message
    repair_strategy: RepairStrategy
    severity:        ErrorSeverity = ErrorSeverity.CRITICAL
    metadata:        dict          = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.repair_strategy == RepairStrategy.HUMAN

    @property
    def is_deterministic(self) -> bool:
        """True when the repair requires no LLM call."""
        return self.repair_strategy in (
            RepairStrategy.PARAM_INJECT,
            RepairStrategy.TOPOLOGY_FIX,
            RepairStrategy.THERMO_SWITCH,
            RepairStrategy.UNIT_CONVERSION,
            RepairStrategy.DEFAULT_FILL,
            RepairStrategy.PORT_REPAIR,
        )

    def __str__(self) -> str:
        return (f"[{self.severity.value}/{self.error_type.value}] "
                f"{self.target} → {self.repair_strategy.value}: {self.evidence[:80]}")
