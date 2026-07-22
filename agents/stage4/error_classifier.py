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

_TAXONOMY_COMPACT = """\
SOLVER_FAIL        → TOPOLOGY_FIX    : solver did not converge; simplify topology
NUMERIC_FAIL       → THERMO_SWITCH   : NaN/Inf in stream; check T/P units and BIPs
MASS_BALANCE       → TOPOLOGY_FIX   : outlet flow ≠ feed flow; check connections
UNPHYSICAL_T       → UNIT_CONVERSION: T < 100 K or > 2000 K; convert °C→K
UNPHYSICAL_P       → UNIT_CONVERSION: P < 100 Pa or > 1e8 Pa; convert bar/atm→Pa
ENERGY_UNPHYSICAL  → CONDITION_FIX  : Heater outlet < feed T, or Cooler outlet > feed T
ZERO_OUTLET        → CONDITION_FIX  : terminal stream flow = 0; check flash conditions
NO_SEPARATION      → THERMO_SWITCH  : outlet ≈ feed; NRTL/UNIQUAC missing BIPs
WRONG_PHASE_DIR    → TOPOLOGY_FIX   : heavy in vapour, light in liquid; swap src_port
COMP_SUM           → TOPOLOGY_FIX   : mole fractions do not sum to 1.0
PARAM_MISSING      → PARAM_INJECT   : NRTL/UNIQUAC with no binary parameters
INFEASIBLE         → HUMAN          : same failure after 3+ iterations"""

_SYSTEM = f"""\
Classify a DWSIM simulation failure and assign the repair strategy.
Return ONLY a JSON object — no explanation, no markdown.

Schema:
{{
  "errors": [
    {{
      "error_type": "<MISSING_PARAM|INVALID_TOPOLOGY|CONVERGENCE_FAILURE|INVALID_UNIT_CONFIG|UNPHYSICAL_VALUES|INFEASIBLE>",
      "location": "<unit or stream tag, or 'global'>",
      "evidence": "<specific values or message from the execution summary>",
      "repair_strategy": "<PARAM_INJECT|TOPOLOGY_FIX|THERMO_SWITCH|CONDITION_FIX|UNIT_CONVERSION|DEFAULT_FILL|PORT_REPAIR|HUMAN>",
      "severity": "<CRITICAL|WARNING>"
    }}
  ]
}}

━━━ EXAMPLE ━━━
Execution: solved=False, error: "NRTL BIP missing for Ethanol/Water", stream FEED: T=298 P=101325
Output:
{{"errors": [{{"error_type": "MISSING_PARAM", "location": "global",
  "evidence": "NRTL BIP missing for Ethanol/Water",
  "repair_strategy": "PARAM_INJECT", "severity": "CRITICAL"}}]}}

Routing reference (signal → strategy):
{_TAXONOMY_COMPACT}"""

# Deterministic signal → (ErrorType, RepairStrategy) mapping
# Unit types with adjustable operating conditions → the parameter beam search repairs
_CONDITION_FIX_PARAMS: dict[str, str] = {
    "Heater":     "T_out",
    "Cooler":     "T_out",
    "Pump":       "P_out",
    "Compressor": "P_out",
    "Expander":   "P_out",
}

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

        # Reactor mis-specification takes precedence: a ConversionReactor with no
        # reaction string converts nothing, so DWSIM stalls.  Surface this as a
        # distinct error TARGETING the reactor's reaction — otherwise the loop
        # below blames a downstream cooler's T_out and wastes every iteration on a
        # parameter that cannot be the cause.
        if not getattr(execution, "solved", True):
            _rxn_errors = _reactor_missing_reaction(graph)
            if _rxn_errors:
                import sys as _sys
                print(f"[CLASSIFIER] ConversionReactor missing reaction → "
                      f"{[str(e.target) for e in _rxn_errors]} "
                      f"(not a downstream-condition problem)",
                      flush=True, file=_sys.stderr)
                return _rxn_errors

        # When the only classification is a generic TOPOLOGY_FIX (solved=False with
        # no specific signal), swap it for CONDITION_FIX errors targeting each unit
        # that has adjustable operating conditions (T_out / P_out).
        #
        # TOPOLOGY_FIX → DeterministicRepair.fix_topology() → normalise(graph) is a
        # no-op on already-validated IR.  The repair loop produces identical changes
        # every iteration and DWSIM receives the same flowsheet each time.  Beam
        # search only runs on CONDITION_FIX errors, so without this replacement it
        # never executes at all.
        if (not getattr(execution, "solved", True)
                and all(e.repair_strategy == RepairStrategy.TOPOLOGY_FIX
                        for e in errors)):
            cond_errors = _condition_fix_from_graph(graph)
            if cond_errors:
                print(f"[CLASSIFIER] generic TOPOLOGY_FIX → "
                      f"{len(cond_errors)} CONDITION_FIX errors "
                      f"({[str(e.target) for e in cond_errors]})",
                      flush=True, file=__import__('sys').stderr)
                errors = cond_errors

        if _all_unambiguous(errors):
            final = errors
        else:
            llm_errors = self._llm_classify(execution, graph)
            final = llm_errors if llm_errors else errors

        # When NRTL or UNIQUAC fails to converge and no THERMO_SWITCH is already
        # present, inject one. Topology fixes alone will never unstick a VLE solver
        # that can't converge for this compound pair.
        if not getattr(execution, "solved", True):
            pkg = getattr(graph, "property_package", "")
            if (pkg in ("NRTL", "UNIQUAC")
                    and not any(e.repair_strategy == RepairStrategy.THERMO_SWITCH
                                for e in final)):
                final = list(final) + [SimError(
                    error_type      = ErrorType.CONVERGENCE_FAILURE,
                    target          = ErrorTarget.global_(),
                    evidence        = (f"{pkg} VLE solver did not converge — "
                                       "switching property package"),
                    repair_strategy = RepairStrategy.THERMO_SWITCH,
                    severity        = ErrorSeverity.CRITICAL,
                )]

        return final

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


def _reactor_missing_reaction(graph) -> list[SimError]:
    """A ConversionReactor with an empty/missing reaction string cannot convert
    anything.  Return one CRITICAL INVALID_UNIT_CONFIG error per such reactor,
    targeting <tag>.reaction, so the repair loop addresses the reactor rather than
    a downstream unit's operating conditions."""
    out: list[SimError] = []
    if graph is None:
        return out
    for node in graph.units():
        if getattr(node, "unit_type", "") != "ConversionReactor":
            continue
        rxn = str(getattr(node, "params", {}).get("reaction", "") or "").strip()
        if not rxn:
            out.append(SimError(
                error_type      = ErrorType.INVALID_UNIT_CONFIG,
                target          = ErrorTarget.unit(node.tag, "reaction"),
                evidence        = (f"ConversionReactor {node.tag} has no reaction "
                                   "definition (reaction string is empty); DWSIM "
                                   "performs no conversion and cannot solve"),
                repair_strategy = RepairStrategy.HUMAN,
                severity        = ErrorSeverity.CRITICAL,
            ))
    return out


_DEGENERATE_TOKENS = ("near-degenerate split", "not separable by a shortcut")


def _degenerate_split_error(execution) -> Optional[SimError]:
    """A shortcut column with relative volatility ≈ 1 cannot separate its keys —
    no reflux, condition, or topology change fixes it (the FUG equations divide by
    ~0). The DWSIM wrapper flags it explicitly. Classify it as INFEASIBLE → HUMAN
    (terminal) so the repair loop stops on iteration 0 with the physical cause
    named, instead of re-normalising to no effect until MAX_ITER."""
    texts: list[str] = []
    cr = getattr(execution, "_critic_report", None)
    if cr:
        texts += [str(getattr(s, "evidence", "")) for s in getattr(cr, "signals", [])]
    texts += [str(e) for e in getattr(execution, "solver_errors", [])]
    texts += [str(e) for e in getattr(execution, "errors", [])]
    for t in texts:
        if any(tok in t.lower() for tok in _DEGENERATE_TOKENS):
            tag = t.split(":", 1)[0].strip()
            target = (_infer_target(tag)
                      if tag and " " not in tag and 0 < len(tag) <= 16
                      else ErrorTarget.global_())
            return SimError(
                error_type      = ErrorType.INFEASIBLE,
                target          = target,
                evidence        = t,      # already names the physical cause
                repair_strategy = RepairStrategy.HUMAN,
                severity        = ErrorSeverity.CRITICAL,
            )
    return None


def _deterministic_classify(execution) -> list[SimError]:
    errors: list[SimError] = []

    # A near-degenerate column split (relative volatility ≈ 1) is thermodynamically
    # infeasible: no repair separates the keys. Terminal → HUMAN, ahead of any
    # other signal (which would otherwise route it to a no-op TOPOLOGY_FIX).
    _deg = _degenerate_split_error(execution)
    if _deg is not None:
        return [_deg]

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


def _condition_fix_from_graph(graph) -> list[SimError]:
    """
    Generate CONDITION_FIX errors for every unit that has an adjustable
    operating condition (T_out or P_out).

    Called when DWSIM fails with a generic convergence error (no specific signal).
    The target.field is set so _infer_param() in beam_search / repair_agent
    resolves the parameter name without needing to parse the evidence string.
    """
    errors: list[SimError] = []
    for node in graph.units():
        param = _CONDITION_FIX_PARAMS.get(node.unit_type)
        if param is None:
            continue
        current = node.params.get(param, "unset")
        errors.append(SimError(
            error_type      = ErrorType.INVALID_UNIT_CONFIG,
            target          = ErrorTarget.unit(node.tag, param),
            evidence        = (f"{node.tag} ({node.unit_type}) {param}={current} "
                               f"— DWSIM convergence failure; {param} likely wrong"),
            repair_strategy = RepairStrategy.CONDITION_FIX,
            severity        = ErrorSeverity.CRITICAL,
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
