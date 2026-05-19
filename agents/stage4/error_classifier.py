"""
Agent F — Error Classifier.

Input : ExecutionResult + FlowsheetGraph
Output: list[ClassifiedError] — structured error taxonomy with repair strategies

Stage 1 (deterministic): maps raw DWSIM signals to error taxonomy.
Stage 2 (LLM, only for ambiguous cases): short targeted diagnosis prompt.

Error taxonomy:
  MISSING_PARAM       → PARAM_INJECT     (BIPs absent; fill from corpus)
  INVALID_TOPOLOGY    → TOPOLOGY_FIX     (graph normaliser re-applied)
  CONVERGENCE_FAILURE → THERMO_SWITCH    (change property package via RAG)
  INVALID_UNIT_CONFIG → CONDITION_FIX    (ReviseAgent rewrites specific params)
  UNPHYSICAL_VALUES   → CONDITION_FIX    (unit conversion / out-of-range values)
  INFEASIBLE          → HUMAN            (cannot be recovered automatically)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from agents.llm import chat, DEFAULT_MODEL, retry_temperature
from context import FAILURE_TAXONOMY

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
      "repair_strategy": "<PARAM_INJECT|TOPOLOGY_FIX|THERMO_SWITCH|CONDITION_FIX|HUMAN>",
      "severity": "<CRITICAL|WARNING>"
    }}
  ]
}}

Taxonomy reference:
{FAILURE_TAXONOMY[:1500]}"""

# Deterministic signal → (error_type, repair_strategy) mapping
_SIGNAL_MAP: dict[str, tuple[str, str]] = {
    "SOLVER_FAIL":       ("CONVERGENCE_FAILURE", "TOPOLOGY_FIX"),
    "NUMERIC_FAIL":      ("CONVERGENCE_FAILURE", "THERMO_SWITCH"),
    "MASS_BALANCE":      ("INVALID_TOPOLOGY",    "TOPOLOGY_FIX"),
    "UNPHYSICAL_T":      ("UNPHYSICAL_VALUES",   "CONDITION_FIX"),
    "UNPHYSICAL_P":      ("UNPHYSICAL_VALUES",   "CONDITION_FIX"),
    "ENERGY_UNPHYSICAL": ("UNPHYSICAL_VALUES",   "CONDITION_FIX"),
    "ZERO_OUTLET":       ("INVALID_UNIT_CONFIG", "CONDITION_FIX"),
    "NO_SEPARATION":     ("MISSING_PARAM",       "PARAM_INJECT"),
    "PARAM_MISSING":     ("MISSING_PARAM",       "PARAM_INJECT"),
    "WRONG_PHASE_DIR":   ("INVALID_TOPOLOGY",    "TOPOLOGY_FIX"),
    "INFEASIBLE":        ("INFEASIBLE",           "HUMAN"),
}


@dataclass
class ClassifiedError:
    error_type:      str
    location:        str
    evidence:        str
    repair_strategy: str
    severity:        str  = "CRITICAL"
    metadata:        dict = field(default_factory=dict)

    def is_terminal(self) -> bool:
        return self.repair_strategy == "HUMAN"


class ErrorClassifier:
    def __init__(self, model: str = DEFAULT_MODEL):
        self._model = model

    def classify(self, execution, graph) -> list[ClassifiedError]:
        """
        execution: ExecutionResult (from agents/executor.py)
        graph:     FlowsheetGraph
        """
        # Stage 1 — deterministic from CriticAgent signals
        errors = _deterministic_classify(execution)

        if not errors:
            return []

        # Don't call LLM if already terminal or if Stage 1 is unambiguous
        if any(e.is_terminal() for e in errors):
            return errors
        if _all_unambiguous(errors):
            return errors

        # Stage 2 — LLM for ambiguous multi-signal cases
        llm_errors = self._llm_classify(execution, graph)
        if llm_errors:
            return llm_errors
        return errors

    def _llm_classify(self, execution, graph) -> list[ClassifiedError]:
        summary = _execution_summary(execution)
        prompt  = (
            f"DWSIM execution summary:\n{summary}\n\n"
            "Classify each failure and assign the correct repair strategy."
        )
        raw = ""
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
                errors = [_parse_error(e) for e in data.get("errors", [])]
                if errors:
                    return errors
            except (json.JSONDecodeError, KeyError):
                pass
        return []


def _deterministic_classify(execution) -> list[ClassifiedError]:
    errors: list[ClassifiedError] = []

    # Use CriticReport signals if available (v1 critic already ran)
    critic_report = getattr(execution, "_critic_report", None)
    if critic_report:
        for sig in getattr(critic_report, "signals", []):
            mapping = _SIGNAL_MAP.get(sig.code)
            if mapping:
                error_type, strategy = mapping
                errors.append(ClassifiedError(
                    error_type      = error_type,
                    location        = sig.location,
                    evidence        = sig.evidence,
                    repair_strategy = strategy,
                    severity        = sig.severity,
                ))
        return errors

    # Fallback: infer from ExecutionResult fields directly
    if not getattr(execution, "solved", True):
        solver_errs = getattr(execution, "solver_errors", [])
        errors.append(ClassifiedError(
            error_type      = "CONVERGENCE_FAILURE",
            location        = "global",
            evidence        = "; ".join(str(e) for e in solver_errs[:3]),
            repair_strategy = "TOPOLOGY_FIX",
            severity        = "CRITICAL",
        ))

    for err in getattr(execution, "errors", []):
        err_str = str(err).upper()
        if "PARAM" in err_str or "BIP" in err_str or "BINARY" in err_str:
            errors.append(ClassifiedError(
                error_type="MISSING_PARAM", location="global",
                evidence=str(err), repair_strategy="PARAM_INJECT"))
        elif "UNPHYSICAL" in err_str or "NAN" in err_str or "INF" in err_str:
            errors.append(ClassifiedError(
                error_type="UNPHYSICAL_VALUES", location="global",
                evidence=str(err), repair_strategy="CONDITION_FIX"))

    return errors


def _all_unambiguous(errors: list[ClassifiedError]) -> bool:
    """True when every error has a clear deterministic repair strategy."""
    ambiguous_types = {"CONVERGENCE_FAILURE"}
    return not any(e.error_type in ambiguous_types for e in errors)


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


def _parse_error(e: dict) -> ClassifiedError:
    return ClassifiedError(
        error_type      = e.get("error_type", "CONVERGENCE_FAILURE"),
        location        = e.get("location", "global"),
        evidence        = e.get("evidence", ""),
        repair_strategy = e.get("repair_strategy", "TOPOLOGY_FIX"),
        severity        = e.get("severity", "CRITICAL"),
    )


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    return json.loads(text)
