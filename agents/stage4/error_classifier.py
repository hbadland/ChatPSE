"""
Agent F — Error Classifier.

Input : ExecutionResult + FlowsheetGraph
Output: list[SimError] — typed errors with repair strategies

Stage 1 (deterministic): maps raw DWSIM signals to error taxonomy.
Stage 2 (LLM, only for ambiguous cases): short targeted diagnosis prompt.

ClassifiedError is retained as a backward-compatible alias for SimError;
new code should work with SimError directly.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from agents.llm import chat, DEFAULT_MODEL, retry_temperature
from context import FAILURE_TAXONOMY
from ir.types import (
    ErrorType, RepairStrategy, ErrorSeverity,
    ErrorTarget, TargetKind, SimError,
)

# Keep old name as alias — orchestrator_v2 and tests import ClassifiedError
ClassifiedError = SimError

_SYSTEM = f"""\
Classify a DWSIM simulation failure and determine the repair strategy.
Return ONLY a JSON object — no explanation, no markdown.

Schema:
{{
  "errors": [
    {{
      "error_type": "<MISSING_PARAM|INVALID_TOPOLOGY|CONVERGENCE_FAILURE|INVALID_UNIT_CONFIG|UNPHYSICAL_VALUES|INFEASIBLE>",
      "location": "<unit or stream tag, or 'global'>",
      "evidence": "<specific values or message>",
      "repair_strategy": "<PARAM_INJECT|TOPOLOGY_FIX|THERMO_SWITCH|CONDITION_FIX|UNIT_CONVERSION|DEFAULT_FILL|PORT_REPAIR|HUMAN>",
      "severity": "<CRITICAL|WARNING>"
    }}
  ]
}}

Taxonomy reference:
{FAILURE_TAXONOMY[:1500]}"""

# Deterministic signal → (ErrorType, RepairStrategy) mapping
_SIGNAL_MAP: dict[str, tuple[ErrorType, RepairStrategy]] = {
    "SOLVER_FAIL":       (ErrorType.CONVERGENCE_FAILURE, RepairStrategy.TOPOLOGY_FIX),
    "NUMERIC_FAIL":      (ErrorType.CONVERGENCE_FAILURE, RepairStrategy.THERMO_SWITCH),
    "MASS_BALANCE":      (ErrorType.MASS_BALANCE,        RepairStrategy.TOPOLOGY_FIX),
    "UNPHYSICAL_T":      (ErrorType.UNPHYSICAL_VALUES,   RepairStrategy.UNIT_CONVERSION),
    "UNPHYSICAL_P":      (ErrorType.UNPHYSICAL_VALUES,   RepairStrategy.UNIT_CONVERSION),
    "ENERGY_UNPHYSICAL": (ErrorType.UNPHYSICAL_VALUES,   RepairStrategy.CONDITION_FIX),
    "ZERO_OUTLET":       (ErrorType.INVALID_UNIT_CONFIG, RepairStrategy.CONDITION_FIX),
    "NO_SEPARATION":     (ErrorType.MISSING_PARAM,       RepairStrategy.PARAM_INJECT),
    "PARAM_MISSING":     (ErrorType.MISSING_PARAM,       RepairStrategy.PARAM_INJECT),
    "WRONG_PHASE_DIR":   (ErrorType.INVALID_TOPOLOGY,    RepairStrategy.TOPOLOGY_FIX),
    "INFEASIBLE":        (ErrorType.INFEASIBLE,          RepairStrategy.HUMAN),
}


class ErrorClassifier:
    def __init__(self, model: str = DEFAULT_MODEL):
        self._model = model

    def classify(self, execution, graph) -> list[SimError]:
        errors = _deterministic_classify(execution)

        if not errors:
            return []
        if any(e.is_terminal for e in errors):
            return errors
        if _all_unambiguous(errors):
            return errors

        llm_errors = self._llm_classify(execution, graph)
        if llm_errors:
            return llm_errors
        return errors

    def _llm_classify(self, execution, graph) -> list[SimError]:
        summary = _execution_summary(execution)
        prompt  = (
            f"DWSIM execution summary:\n{summary}\n\n"
            "Classify each failure and assign the correct repair strategy."
        )
        for attempt in range(2):
            raw = chat(
                prompt,
                system=_SYSTEM,
                model=self._model,
                temperature=retry_temperature(attempt),
                max_tokens=1024,
            )
            try:
                data   = _parse_json(raw)
                errors = [_parse_sim_error(e) for e in data.get("errors", [])]
                if errors:
                    return errors
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
        return []


def _deterministic_classify(execution) -> list[SimError]:
    errors: list[SimError] = []

    critic_report = getattr(execution, "_critic_report", None)
    if critic_report:
        for sig in getattr(critic_report, "signals", []):
            mapping = _SIGNAL_MAP.get(sig.code)
            if mapping:
                etype, strategy = mapping
                loc = getattr(sig, "location", "global")
                errors.append(SimError(
                    error_type      = etype,
                    target          = _infer_target(loc),
                    evidence        = getattr(sig, "evidence", ""),
                    repair_strategy = strategy,
                    severity        = _parse_severity(getattr(sig, "severity", "CRITICAL")),
                ))
        return errors

    # Fallback: infer from ExecutionResult fields directly
    if not getattr(execution, "solved", True):
        solver_errs = getattr(execution, "solver_errors", [])
        errors.append(SimError(
            error_type      = ErrorType.CONVERGENCE_FAILURE,
            target          = ErrorTarget.global_(),
            evidence        = "; ".join(str(e) for e in solver_errs[:3]),
            repair_strategy = RepairStrategy.TOPOLOGY_FIX,
            severity        = ErrorSeverity.CRITICAL,
        ))

    for err in getattr(execution, "errors", []):
        err_str = str(err).upper()
        if "PARAM" in err_str or "BIP" in err_str or "BINARY" in err_str:
            errors.append(SimError(
                error_type      = ErrorType.MISSING_PARAM,
                target          = ErrorTarget.global_(),
                evidence        = str(err),
                repair_strategy = RepairStrategy.PARAM_INJECT,
            ))
        elif "UNPHYSICAL" in err_str or "NAN" in err_str or "INF" in err_str:
            errors.append(SimError(
                error_type      = ErrorType.UNPHYSICAL_VALUES,
                target          = ErrorTarget.global_(),
                evidence        = str(err),
                repair_strategy = RepairStrategy.CONDITION_FIX,
            ))

    return errors


def _all_unambiguous(errors: list[SimError]) -> bool:
    ambiguous = {ErrorType.CONVERGENCE_FAILURE}
    return not any(e.error_type in ambiguous for e in errors)


def _infer_target(location: str) -> ErrorTarget:
    if not location or location == "global":
        return ErrorTarget.global_()
    if location.startswith("stream:"):
        return ErrorTarget.stream(location[7:])
    return ErrorTarget.unit(location)


def _parse_severity(s: str) -> ErrorSeverity:
    try:
        return ErrorSeverity(s.upper())
    except ValueError:
        return ErrorSeverity.CRITICAL


def _parse_sim_error(e: dict) -> SimError:
    etype    = _safe_enum(ErrorType,      e.get("error_type",      "CONVERGENCE_FAILURE"))
    strategy = _safe_enum(RepairStrategy, e.get("repair_strategy", "TOPOLOGY_FIX"))
    sev      = _parse_severity(e.get("severity", "CRITICAL"))
    loc      = e.get("location", "global")
    return SimError(
        error_type      = etype,
        target          = _infer_target(loc),
        evidence        = e.get("evidence", ""),
        repair_strategy = strategy,
        severity        = sev,
    )


def _safe_enum(cls, value: str):
    try:
        return cls(value)
    except ValueError:
        return list(cls)[0]


def _execution_summary(execution) -> str:
    lines = [f"solved={getattr(execution,'solved','?')}"]
    for err in list(getattr(execution, "errors", []))[:5]:
        lines.append(f"  error: {err}")
    for err in list(getattr(execution, "solver_errors", []))[:5]:
        lines.append(f"  solver_error: {err}")
    streams = getattr(execution, "streams", {})
    for tag, result in list(streams.items())[:4]:
        lines.append(f"  stream {tag}: {result}")
    return "\n".join(lines)


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    return json.loads(text)
