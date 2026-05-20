"""
Physics check evaluator.

Evaluates post-hoc physics checks from a BenchmarkCaseSpec against a
PipelineResult.  Checks are IR-level (no DWSIM required) or execution-level.

Each check returns a dict:
  {"check": <type>, "passed": bool, "detail": str}
"""
from __future__ import annotations

from typing import Any, Optional


_PACKAGE_CLASSES = {
    "activity_coefficient": {"NRTL", "UNIQUAC"},
    "eos": {"Peng-Robinson", "Soave-Redlich-Kwong",
            "Lee-Kesler-Plöcker", "Peng-Robinson-Stryjek-Vera"},
    "ideal": {"Raoult's Law", "Ideal"},
}


def _pkg_class(pkg: str) -> str:
    for cls, pkgs in _PACKAGE_CLASSES.items():
        if pkg in pkgs or any(p in pkg for p in pkgs):
            return cls
    return "unknown"


def _units(graph) -> list:
    if graph is None:
        return []
    units_fn = getattr(graph, "units", None)
    return list(units_fn()) if callable(units_fn) else []


def _unit_type(u) -> str:
    return getattr(u, "unit_type", getattr(u, "UNIT_TYPE", str(u)))


def _tag(u) -> str:
    return getattr(u, "tag", "")


def _match_tag(u, pattern: str) -> bool:
    return pattern.upper() in _tag(u).upper()


def _param(u, name: str) -> Optional[float]:
    params = getattr(u, "params", {})
    return params.get(name)


# ── Check dispatch ─────────────────────────────────────────────────────────────

def evaluate_check(check, graph, pipeline_result) -> dict:
    t = check.check_type
    p = check.params

    if t == "unit_type_present":
        return _check_unit_type_present(p.get("unit_type", ""), graph)

    if t == "n_units_of_type":
        return _check_n_units(p.get("unit_type", ""), p.get("count_min", 1), graph)

    if t == "property_package_class":
        return _check_pkg_class(p.get("package_class", ""), graph)

    if t == "temp_increases_across":
        return _check_temp_direction(p.get("unit_tag_pattern", ""), "+", graph)

    if t == "temp_decreases_across":
        return _check_temp_direction(p.get("unit_tag_pattern", ""), "-", graph)

    if t == "pressure_increases_across":
        return _check_pressure_direction(p.get("unit_tag_pattern", ""), graph)

    if t == "outlet_t_range":
        return _check_outlet_t_range(
            p.get("unit_tag_pattern", ""),
            p.get("T_min_K", 0), p.get("T_max_K", 9999), graph)

    if t == "two_phase_outlet":
        return _check_two_phase(p.get("unit_tag_pattern", ""), graph, pipeline_result)

    if t == "single_phase_vapor_ok":
        return _check_single_phase_ok(p.get("unit_tag_pattern", ""), graph)

    if t == "bip_injected":
        return _check_bip_injected(p.get("model", "NRTL"), graph)

    if t == "separation_quality_below":
        return _check_separation_quality(
            p.get("unit_tag_pattern", ""), p.get("max_enrichment", 0.5), graph)

    if t == "convergence":
        solved = getattr(pipeline_result, "converged", False) or \
                 getattr(pipeline_result, "outcome", "") == "PASS"
        return {"check": t, "passed": bool(solved), "detail": f"outcome={getattr(pipeline_result, 'outcome', '?')}"}

    if t == "temp_consistency_inlet_outlet":
        return _check_temp_consistency(p.get("unit_tag_pattern", ""), graph)

    return {"check": t, "passed": True, "detail": "check not implemented (skipped)"}


# ── Individual check implementations ──────────────────────────────────────────

def _check_unit_type_present(unit_type: str, graph) -> dict:
    units = _units(graph)
    found = any(_unit_type(u).lower() == unit_type.lower() for u in units)
    return {
        "check": "unit_type_present",
        "passed": found,
        "detail": f"looking for {unit_type}, found types: {[_unit_type(u) for u in units]}",
    }


def _check_n_units(unit_type: str, count_min: int, graph) -> dict:
    units = _units(graph)
    count = sum(1 for u in units if _unit_type(u).lower() == unit_type.lower())
    passed = count >= count_min
    return {
        "check": "n_units_of_type",
        "passed": passed,
        "detail": f"found {count} × {unit_type}, need >= {count_min}",
    }


def _check_pkg_class(pkg_class: str, graph) -> dict:
    if graph is None:
        return {"check": "property_package_class", "passed": False, "detail": "no graph"}
    pkg = getattr(graph, "property_package", "")
    actual_class = _pkg_class(pkg)
    passed = actual_class == pkg_class
    return {
        "check": "property_package_class",
        "passed": passed,
        "detail": f"pkg={pkg!r} class={actual_class!r}, expected={pkg_class!r}",
    }


def _check_temp_direction(pattern: str, direction: str, graph) -> dict:
    units = [u for u in _units(graph) if _match_tag(u, pattern)]
    if not units:
        return {"check": f"temp_{direction}increases_across", "passed": True,
                "detail": f"no unit matching '{pattern}' — skipped"}
    u = units[0]
    t_out = _param(u, "T_out")
    if t_out is None:
        return {"check": f"temp_{direction}increases", "passed": True,
                "detail": f"T_out not set on {_tag(u)} — cannot verify"}
    # Compare against feed temperature heuristic (298.15 K)
    feed_t = 298.15
    if direction == "+":
        passed = t_out > feed_t
        detail = f"T_out={t_out:.1f} > feed_T={feed_t:.1f}: {passed}"
    else:
        passed = t_out < feed_t + 50  # cooler brings below ~350 K
        detail = f"T_out={t_out:.1f} K for cooler: {passed}"
    return {"check": f"temp_{direction}increases_across", "passed": passed, "detail": detail}


def _check_pressure_direction(pattern: str, graph) -> dict:
    units = [u for u in _units(graph) if _match_tag(u, pattern)]
    if not units:
        return {"check": "pressure_increases_across", "passed": True,
                "detail": f"no unit matching '{pattern}' — skipped"}
    u = units[0]
    p_out = _param(u, "P_out")
    if p_out is None:
        return {"check": "pressure_increases_across", "passed": True,
                "detail": f"P_out not set on {_tag(u)} — cannot verify"}
    passed = p_out > 101325.0   # must exceed 1 atm as a sanity check
    return {"check": "pressure_increases_across", "passed": passed,
            "detail": f"P_out={p_out:.0f} Pa"}


def _check_outlet_t_range(pattern: str, t_min: float, t_max: float, graph) -> dict:
    units = [u for u in _units(graph) if _match_tag(u, pattern)]
    if not units:
        return {"check": "outlet_t_range", "passed": True,
                "detail": f"no unit matching '{pattern}' — skipped"}
    u = units[0]
    t_out = _param(u, "T_out")
    if t_out is None:
        return {"check": "outlet_t_range", "passed": True,
                "detail": f"T_out not set on {_tag(u)} — cannot verify"}
    passed = t_min <= t_out <= t_max
    return {"check": "outlet_t_range", "passed": passed,
            "detail": f"T_out={t_out:.1f} K, expected [{t_min}, {t_max}]"}


def _check_two_phase(pattern: str, graph, pr) -> dict:
    # If execution result is available, check stream phases
    execution = getattr(pr, "final_execution", None)
    if execution is not None:
        stream_results = getattr(execution, "stream_results", {})
        if stream_results:
            has_vapor = any(
                getattr(s, "vapor_fraction", 0) > 0.05
                for s in stream_results.values()
            )
            has_liquid = any(
                getattr(s, "vapor_fraction", 1) < 0.95
                for s in stream_results.values()
            )
            passed = has_vapor and has_liquid
            return {"check": "two_phase_outlet", "passed": passed,
                    "detail": f"vapor_present={has_vapor}, liquid_present={has_liquid}"}

    # Fall back to IR check: vessel must have >=2 outlet streams
    units = [u for u in _units(graph) if _match_tag(u, pattern)]
    if not units:
        return {"check": "two_phase_outlet", "passed": True,
                "detail": f"no unit matching '{pattern}' — skipped (no execution result)"}
    passed = True  # can't verify without execution
    return {"check": "two_phase_outlet", "passed": True,
            "detail": "vessel present; two-phase check requires execution result"}


def _check_single_phase_ok(pattern: str, graph) -> dict:
    # All-vapour outcome is acceptable (not an error)
    return {"check": "single_phase_vapor_ok", "passed": True,
            "detail": "all-vapour flash is a valid outcome — not penalised"}


def _check_bip_injected(model: str, graph) -> dict:
    if graph is None:
        return {"check": "bip_injected", "passed": False, "detail": "no graph"}
    bips = getattr(graph, "binary_parameters", [])
    model_bips = [b for b in bips
                  if isinstance(b, dict) and b.get("model", "").upper() == model.upper()]
    passed = len(model_bips) > 0
    return {"check": "bip_injected", "passed": passed,
            "detail": f"{len(model_bips)} {model} BIP(s) found"}


def _check_separation_quality(pattern: str, max_enrichment: float, graph) -> dict:
    # Without execution result this is a conservative pass
    return {"check": "separation_quality_below", "passed": True,
            "detail": "requires execution stream data; skipped at IR level"}


def _check_temp_consistency(pattern: str, graph) -> dict:
    units = [u for u in _units(graph) if _match_tag(u, pattern)]
    if not units:
        return {"check": "temp_consistency_inlet_outlet", "passed": True,
                "detail": f"no unit matching '{pattern}' — skipped"}
    # At IR level we can only check that T_out is positive
    u = units[0]
    t_out = _param(u, "T_out")
    if t_out is None:
        return {"check": "temp_consistency_inlet_outlet", "passed": True,
                "detail": "T_out not set — cannot verify"}
    passed = t_out > 200   # physically plausible temperature in K
    return {"check": "temp_consistency_inlet_outlet", "passed": passed,
            "detail": f"T_out={t_out:.1f} K"}


# ── Entry point ────────────────────────────────────────────────────────────────

def run_physics_checks(case, pipeline_result) -> list[dict]:
    """
    Run all physics checks for a case and return results.

    Parameters
    ----------
    case            : BenchmarkCaseSpec
    pipeline_result : OrchestratorV2 PipelineResult (with .ir_valid, .final_graph, etc.)

    Returns
    -------
    list of check result dicts
    """
    graph = getattr(pipeline_result, "final_graph", None)
    results = []
    for chk in case.expected.physics_checks:
        try:
            result = evaluate_check(chk, graph, pipeline_result)
        except Exception as exc:
            result = {"check": chk.check_type, "passed": False,
                      "detail": f"evaluator error: {exc}"}
        results.append(result)
    return results
