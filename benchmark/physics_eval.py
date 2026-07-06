"""
Physics check evaluator — post-hoc benchmark verification.

Each check returns:
  {
    "check":    <check_type str>,
    "passed":   bool,
    "severity": "CRITICAL" | "WARNING" | "INFO",
    "detail":   str,
    "source":   "execution" | "IR" | "pipeline" | "none",
  }

severity semantics
  CRITICAL — counts against the critical_physics_pass_rate; the simulation
             result is physically wrong or untrustworthy.
  WARNING  — outside expected range but may be valid; informational for the
             reviewer, does not count against the critical pass rate.
  INFO     — always passes; structural/presence checks or skipped checks.

source semantics
  execution — reads from pipeline_result.final_execution.stream_results
              (actual DWSIM output — thermodynamic correctness verified).
  IR        — reads from pipeline_result.final_graph (LLM-generated params —
              verifies LLM intent, not simulated outcome).
  pipeline  — reads pipeline_result.outcome string directly.
  none      — check was skipped (unit not found, or no data available).

For each check, the source and false-positive/negative exposure are documented
in the implementation comments below.

Audit summary (Task 1 findings, implemented as Task 2/3 upgrades)
─────────────────────────────────────────────────────────────────────
check                      | old source | new source     | FP risk | FN risk
unit_type_present          | IR         | IR             | low     | low
n_units_of_type            | IR         | IR             | low     | low
property_package_class     | IR         | IR             | low     | low
temp_increases_across      | IR (fixed) | exec→IR        | medium  | low
temp_decreases_across      | IR (fixed) | exec→IR        | medium  | low
pressure_increases_across  | IR (fixed) | exec→IR        | medium  | low
outlet_t_range             | IR         | exec→IR        | medium  | medium *
two_phase_outlet           | exec (bug) | exec (fixed)   | HIGH†   | medium
single_phase_vapor_ok      | —          | —              | trivial pass
bip_injected               | IR         | IR             | low     | low
separation_quality_below   | —          | exec→skip      | trivial pass (was)
temp_consistency_inlet_out | IR         | exec→IR        | low     | low
convergence                | pipeline   | pipeline       | low     | low
mass_balance               | NEW        | execution      | low     | low
flash_vapor_fraction       | NEW        | execution      | low     | medium ‡
energy_balance_heater      | NEW        | execution      | n/a     | low

* outlet_t_range FN: consistency pass may override T_out after description
  processing, so IR T_out and the DWSIM outlet T can diverge by 20–40 K.
  The exec path eliminates this.

† two_phase_outlet old bug: checked VF across ALL streams, not just the
  vessel's outlet streams. A multi-unit flowsheet where any upstream stream
  has VF in (0.05, 0.95) would pass even if the vessel itself produced no
  separation.  Fixed: now uses graph.outlet_streams(vessel_tag) to restrict
  to the vessel's own outputs.

‡ flash_vapor_fraction FN: bubble-point operation where the vapour outlet
  carries < 0.1% of the feed is below the flow threshold and marked as
  a non-genuine flash.  This is intentional — a DWSIM result at exactly
  the bubble point is thermodynamically marginal and the outcome is
  sensitive to small T/P perturbations.
"""
from __future__ import annotations

from typing import Optional


# ── Severity constants ─────────────────────────────────────────────────────────
# Stored as plain strings in result dicts so JSON serialisation is trivial.

class CheckSeverity:
    CRITICAL = "CRITICAL"
    WARNING  = "WARNING"
    INFO     = "INFO"


_PACKAGE_CLASSES = {
    "activity_coefficient": {"NRTL", "UNIQUAC"},
    "eos": {"Peng-Robinson", "Soave-Redlich-Kwong",
            "Lee-Kesler-Plöcker", "Peng-Robinson-Stryjek-Vera"},
    "ideal": {"Raoult's Law", "Ideal"},
}

_VF_VAPOR_MIN  = 0.05    # VF above this = vapour-phase outlet
_VF_LIQUID_MIN = 0.05    # VF below (1 - this) = liquid-phase outlet
_FLOW_FRAC_MIN = 0.001   # outlet must carry > 0.1% of feed to be "real"
_MB_TOL        = 0.01    # 1% mass balance tolerance

# Reference-match thresholds (stricter than comparison.py archive tolerances)
_REF_TOL_T_K   = 5.0    # K absolute  — used for reference_match_T check
_REF_TOL_P_REL = 0.05   # fractional  — used for reference_match_P check
_REF_TOL_VF    = 0.05   # absolute    — used for reference_match_vf check (WARNING)

# Minimum matched streams before a reference-MAPE is trustworthy.  Below this the
# MAPE is a misleading artifact (e.g. 1 perfectly-matched stream reads 0.0% and
# passes spuriously), so it is reported as "insufficient_match", NOT a number.
_MIN_MATCH_FOR_MAPE = 3


def _pkg_class(pkg: str) -> str:
    for cls, pkgs in _PACKAGE_CLASSES.items():
        if pkg in pkgs or any(p in pkg for p in pkgs):
            return cls
    return "unknown"


def _units(graph) -> list:
    if graph is None:
        return []
    fn = getattr(graph, "units", None)
    return list(fn()) if callable(fn) else []


def _unit_type(u) -> str:
    return getattr(u, "unit_type", getattr(u, "UNIT_TYPE", str(u)))


def _tag(u) -> str:
    return getattr(u, "tag", "")


def _match_tag(u, pattern: str) -> bool:
    return pattern.upper() in _tag(u).upper()


def _param(u, name: str) -> Optional[float]:
    return getattr(u, "params", {}).get(name)


# ── Execution-result accessors ─────────────────────────────────────────────────

def _stream_results(pr) -> dict:
    """Return {tag: StreamResult} from DWSIM execution, or empty dict."""
    execution = getattr(pr, "final_execution", None)
    if execution is None:
        return {}
    return getattr(execution, "stream_results", {}) or {}


def _inlet_stream_tags(unit_tag: str, graph) -> list[str]:
    fn = getattr(graph, "inlet_streams", None)
    return [e.tag for e in fn(unit_tag)] if callable(fn) else []


def _outlet_stream_tags(unit_tag: str, graph) -> list[str]:
    fn = getattr(graph, "outlet_streams", None)
    return [e.tag for e in fn(unit_tag)] if callable(fn) else []


def _feed_tags(graph) -> list[str]:
    """Streams with no upstream unit (global feed streams)."""
    fn_streams = getattr(graph, "streams",       None)
    fn_src     = getattr(graph, "stream_source",  None)
    if not callable(fn_streams) or not callable(fn_src):
        return []
    return [s.tag for s in fn_streams() if fn_src(s.tag) is None]


def _terminal_outlet_tags(graph) -> list[str]:
    """Streams with no downstream unit (final product streams)."""
    fn_streams = getattr(graph, "streams",     None)
    fn_dst     = getattr(graph, "stream_dest",  None)
    if not callable(fn_streams) or not callable(fn_dst):
        return []
    return [s.tag for s in fn_streams() if fn_dst(s.tag) is None]


def _get_T(sr) -> Optional[float]:
    return getattr(sr, "T_K", None)


def _get_P(sr) -> Optional[float]:
    return getattr(sr, "P_Pa", None)


def _get_flow(sr) -> Optional[float]:
    return getattr(sr, "flow_mol_s", None)


def _get_vf(sr) -> float:
    return getattr(sr, "vapor_fraction", 0.0)


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
        return _check_temp_direction(
            p.get("unit_tag_pattern", ""), "+", graph, pipeline_result)

    if t == "temp_decreases_across":
        return _check_temp_direction(
            p.get("unit_tag_pattern", ""), "-", graph, pipeline_result)

    if t == "pressure_increases_across":
        return _check_pressure_direction(
            p.get("unit_tag_pattern", ""), graph, pipeline_result)

    if t == "pressure_decreases_across":
        return _check_pressure_decreases(
            p.get("unit_tag_pattern", ""), graph, pipeline_result)

    if t == "outlet_t_range":
        return _check_outlet_t_range(
            p.get("unit_tag_pattern", ""),
            p.get("T_min_K", 0), p.get("T_max_K", 9999),
            graph, pipeline_result)

    if t == "two_phase_outlet":
        return _check_two_phase(p.get("unit_tag_pattern", ""), graph, pipeline_result)

    if t == "single_phase_vapor_ok":
        return _check_single_phase_ok()

    if t == "bip_injected":
        return _check_bip_injected(p.get("model", "NRTL"), graph)

    if t == "separation_quality_below":
        return _check_separation_quality(
            p.get("unit_tag_pattern", ""), p.get("max_enrichment", 0.5),
            graph, pipeline_result)

    if t == "convergence":
        solved = (getattr(pipeline_result, "converged", False)
                  or getattr(pipeline_result, "outcome", "") == "PASS")
        return {
            "check":    t,
            "passed":   bool(solved),
            "severity": CheckSeverity.CRITICAL,
            "detail":   f"outcome={getattr(pipeline_result, 'outcome', '?')}",
            "source":   "pipeline",
        }

    if t == "temp_consistency_inlet_outlet":
        return _check_temp_consistency(p.get("unit_tag_pattern", ""), graph, pipeline_result)

    # ── New simulation-grounded checks ─────────────────────────────────────────

    if t == "mass_balance":
        return _check_mass_balance(graph, pipeline_result)

    if t == "flash_vapor_fraction":
        return _check_flash_vf(p.get("unit_tag_pattern", ""), graph, pipeline_result)

    if t == "energy_balance_heater":
        return _check_energy_balance(
            p.get("unit_tag_pattern", ""), p.get("max_dT_K", 500.0),
            graph, pipeline_result)

    if t == "vle_bubble_point_spot_check":
        return _check_vle_bubble_point(
            p.get("unit_tag_pattern", ""), p.get("T_margin_K", 10.0),
            graph, pipeline_result)

    if t == "separation_achieved":
        return _check_separation_achieved(
            p.get("unit_tag_pattern", ""), p.get("min_relative_diff", 0.10),
            p.get("domain", ""),
            graph, pipeline_result)

    if t == "reference_match":
        return _check_reference_match_overall(
            p.get("reference_file", ""), pipeline_result)

    return {
        "check":    t,
        "passed":   True,
        "severity": CheckSeverity.INFO,
        "detail":   "check not implemented (skipped)",
        "source":   "none",
    }


# ── Individual check implementations ──────────────────────────────────────────

def _check_unit_type_present(unit_type: str, graph) -> dict:
    """
    Source: IR.  FP: low — unit type name mismatch would cause FP.
    FN: low — if graph has wrong unit type due to LLM error, check fails correctly.
    """
    units = _units(graph)
    found = any(_unit_type(u).lower() == unit_type.lower() for u in units)
    return {
        "check":    "unit_type_present",
        "passed":   found,
        "severity": CheckSeverity.WARNING,
        "detail":   f"looking for {unit_type!r}, found: {[_unit_type(u) for u in units]}",
        "source":   "IR",
    }


def _check_n_units(unit_type: str, count_min: int, graph) -> dict:
    """Source: IR."""
    units = _units(graph)
    count = sum(1 for u in units if _unit_type(u).lower() == unit_type.lower())
    return {
        "check":    "n_units_of_type",
        "passed":   count >= count_min,
        "severity": CheckSeverity.WARNING,
        "detail":   f"found {count} × {unit_type!r}, need >= {count_min}",
        "source":   "IR",
    }


def _check_pkg_class(pkg_class: str, graph) -> dict:
    """Source: IR."""
    if graph is None:
        return {
            "check": "property_package_class", "passed": False,
            "severity": CheckSeverity.WARNING, "detail": "no graph", "source": "IR",
        }
    pkg          = getattr(graph, "property_package", "")
    actual_class = _pkg_class(pkg)
    return {
        "check":    "property_package_class",
        "passed":   actual_class == pkg_class,
        "severity": CheckSeverity.WARNING,
        "detail":   f"pkg={pkg!r} → class={actual_class!r}, expected={pkg_class!r}",
        "source":   "IR",
    }


def _check_temp_direction(pattern: str, direction: str, graph, pr) -> dict:
    """
    Check temperature direction across a unit (Heater: +, Cooler: -).

    Primary source: DWSIM execution stream results.
      Reads inlet stream T_K and outlet stream T_K for the matched unit.
      Severity = CRITICAL (actual simulation behaviour verified).

    Fallback (no execution data): reads T_out from IR params and compares
      against the IR inlet stream T (or 298.15 K if IR inlet has no T).
      Severity = WARNING (LLM intent only, not verified against simulation).

    Old bug fixed: the previous implementation compared T_out against a
    hardcoded 298.15 K constant regardless of actual feed temperature.
    A heater on a cryogenic feed (-50°C → -20°C) would correctly raise
    temperature but fail the check because T_out < 298.15. The execution
    path compares against the actual inlet stream temperature.
    """
    check_name = "temp_increases_across" if direction == "+" else "temp_decreases_across"
    units = [u for u in _units(graph) if _match_tag(u, pattern)]
    if not units:
        return {
            "check": check_name, "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": f"no unit matching '{pattern}' — skipped", "source": "none",
        }

    u        = units[0]
    unit_tag = _tag(u)
    sr       = _stream_results(pr)

    if sr and graph is not None:
        in_Ts  = [_get_T(sr[t]) for t in _inlet_stream_tags(unit_tag, graph)
                  if t in sr and _get_T(sr[t]) is not None]
        out_Ts = [_get_T(sr[t]) for t in _outlet_stream_tags(unit_tag, graph)
                  if t in sr and _get_T(sr[t]) is not None]
        if in_Ts and out_Ts:
            t_in  = sum(in_Ts)  / len(in_Ts)
            t_out = sum(out_Ts) / len(out_Ts)
            passed = (t_out > t_in) if direction == "+" else (t_out < t_in)
            arrow  = "↑" if t_out > t_in else "↓"
            return {
                "check":    check_name,
                "passed":   passed,
                "severity": CheckSeverity.CRITICAL,
                "detail":   (f"execution: {unit_tag} T_in={t_in:.1f} K → "
                             f"T_out={t_out:.1f} K ({arrow}), "
                             f"expected {'↑' if direction == '+' else '↓'}"),
                "source":   "execution",
            }

    # Fallback: IR
    t_out = _param(u, "T_out")
    if t_out is None:
        return {
            "check": check_name, "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": f"T_out not set on {unit_tag} — cannot verify", "source": "IR",
        }
    feed_t = _ir_inlet_T(unit_tag, graph)
    passed = (t_out > feed_t) if direction == "+" else (t_out < feed_t)
    return {
        "check":    check_name,
        "passed":   passed,
        "severity": CheckSeverity.WARNING,
        "detail":   (f"IR: {unit_tag} T_out={t_out:.1f} K vs "
                     f"T_in={feed_t:.1f} K (IR/heuristic), "
                     f"expected {'↑' if direction == '+' else '↓'}"),
        "source":   "IR",
    }


def _check_pressure_direction(pattern: str, graph, pr) -> dict:
    """
    Check that outlet pressure exceeds inlet pressure (Pump / Compressor).

    Primary source: DWSIM execution stream results.
      Compares actual inlet and outlet stream P_Pa. Severity = CRITICAL.

    Fallback: IR P_out vs IR inlet stream P (or 101325 Pa if not set).
      Severity = WARNING.

    Old bug fixed: the previous implementation checked P_out > 101325.0
    (hardcoded 1 atm) regardless of inlet pressure. A pump boosting from
    200,000 Pa to 300,000 Pa passed correctly, but a pump in a vacuum
    system (inlet 50,000 Pa → outlet 80,000 Pa) would fail even though
    the pump is working correctly.
    """
    units = [u for u in _units(graph) if _match_tag(u, pattern)]
    if not units:
        return {
            "check": "pressure_increases_across", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": f"no unit matching '{pattern}' — skipped", "source": "none",
        }

    u        = units[0]
    unit_tag = _tag(u)
    sr       = _stream_results(pr)

    if sr and graph is not None:
        in_Ps  = [_get_P(sr[t]) for t in _inlet_stream_tags(unit_tag, graph)
                  if t in sr and _get_P(sr[t]) is not None]
        out_Ps = [_get_P(sr[t]) for t in _outlet_stream_tags(unit_tag, graph)
                  if t in sr and _get_P(sr[t]) is not None]
        if in_Ps and out_Ps:
            p_in  = sum(in_Ps)  / len(in_Ps)
            p_out = sum(out_Ps) / len(out_Ps)
            return {
                "check":    "pressure_increases_across",
                "passed":   p_out > p_in,
                "severity": CheckSeverity.CRITICAL,
                "detail":   (f"execution: {unit_tag} "
                             f"P_in={p_in:.0f} Pa → P_out={p_out:.0f} Pa"),
                "source":   "execution",
            }

    # Fallback: IR
    p_out = _param(u, "P_out")
    if p_out is None:
        return {
            "check": "pressure_increases_across", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": f"P_out not set on {unit_tag} — cannot verify", "source": "IR",
        }
    p_in = _ir_inlet_P(unit_tag, graph)
    return {
        "check":    "pressure_increases_across",
        "passed":   p_out > p_in,
        "severity": CheckSeverity.WARNING,
        "detail":   (f"IR: {unit_tag} P_out={p_out:.0f} Pa vs "
                     f"P_in={p_in:.0f} Pa (IR/heuristic)"),
        "source":   "IR",
    }


def _check_pressure_decreases(pattern: str, graph, pr) -> dict:
    """
    Check that outlet pressure is less than inlet pressure (Expander / valve).
    Mirror of _check_pressure_direction with the inequality reversed.
    """
    units = [u for u in _units(graph) if _match_tag(u, pattern)]
    if not units:
        return {
            "check": "pressure_decreases_across", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": f"no unit matching '{pattern}' — skipped", "source": "none",
        }

    u        = units[0]
    unit_tag = _tag(u)
    sr       = _stream_results(pr)

    if sr and graph is not None:
        in_Ps  = [_get_P(sr[t]) for t in _inlet_stream_tags(unit_tag, graph)
                  if t in sr and _get_P(sr[t]) is not None]
        out_Ps = [_get_P(sr[t]) for t in _outlet_stream_tags(unit_tag, graph)
                  if t in sr and _get_P(sr[t]) is not None]
        if in_Ps and out_Ps:
            p_in  = sum(in_Ps)  / len(in_Ps)
            p_out = sum(out_Ps) / len(out_Ps)
            return {
                "check":    "pressure_decreases_across",
                "passed":   p_out < p_in,
                "severity": CheckSeverity.CRITICAL,
                "detail":   (f"execution: {unit_tag} "
                             f"P_in={p_in:.0f} Pa → P_out={p_out:.0f} Pa"),
                "source":   "execution",
            }

    # Fallback: IR
    p_out = _param(u, "P_out")
    if p_out is None:
        return {
            "check": "pressure_decreases_across", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": f"P_out not set on {unit_tag} — cannot verify", "source": "IR",
        }
    p_in = _ir_inlet_P(unit_tag, graph)
    return {
        "check":    "pressure_decreases_across",
        "passed":   p_out < p_in,
        "severity": CheckSeverity.WARNING,
        "detail":   (f"IR: {unit_tag} P_out={p_out:.0f} Pa vs "
                     f"P_in={p_in:.0f} Pa (IR/heuristic)"),
        "source":   "IR",
    }


def _check_outlet_t_range(
        pattern: str, t_min: float, t_max: float, graph, pr) -> dict:
    """
    Check that the outlet temperature of a unit falls within [T_min_K, T_max_K].

    Primary source: DWSIM execution outlet stream T_K. Severity = WARNING.

    Fallback: IR T_out parameter. Severity = WARNING.

    Severity is WARNING (not CRITICAL) because the range is case-specified and
    may be approximate; a result just outside the range is not definitively wrong.

    Old issue: the consistency pass can overwrite T_out in the IR after param
    mapping, making the IR value diverge from both the description and DWSIM
    output. The execution path avoids this — it reads the actual stream temperature
    regardless of what the IR param says.
    """
    units = [u for u in _units(graph) if _match_tag(u, pattern)]
    if not units:
        return {
            "check": "outlet_t_range", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": f"no unit matching '{pattern}' — skipped", "source": "none",
        }

    u        = units[0]
    unit_tag = _tag(u)
    sr       = _stream_results(pr)

    if sr and graph is not None:
        out_tags = _outlet_stream_tags(unit_tag, graph)
        out_Ts   = [_get_T(sr[t]) for t in out_tags
                    if t in sr and _get_T(sr[t]) is not None]
        if out_Ts:
            t_out = sum(out_Ts) / len(out_Ts)
            return {
                "check":    "outlet_t_range",
                "passed":   t_min <= t_out <= t_max,
                "severity": CheckSeverity.WARNING,
                "detail":   (f"execution: {unit_tag} T_out={t_out:.1f} K, "
                             f"expected [{t_min}, {t_max}]"),
                "source":   "execution",
            }

    # Fallback: IR
    t_out = _param(u, "T_out")
    if t_out is None:
        return {
            "check": "outlet_t_range", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": f"T_out not set on {unit_tag} — cannot verify", "source": "IR",
        }
    return {
        "check":    "outlet_t_range",
        "passed":   t_min <= t_out <= t_max,
        "severity": CheckSeverity.WARNING,
        "detail":   f"IR: {unit_tag} T_out={t_out:.1f} K, expected [{t_min}, {t_max}]",
        "source":   "IR",
    }


def _check_two_phase(pattern: str, graph, pr) -> dict:
    """
    Check that a flash vessel genuinely produces two phases.

    Primary source: DWSIM execution — reads the vessel's OWN outlet streams
      (not all streams in the flowsheet). Passes if:
        - at least one outlet stream has VF > 0.05 (vapour present), AND
        - at least one outlet stream has VF < 0.95 (liquid present), AND
        - all outlet streams carry > 0.1% of feed flow (no zero-flow outlet).
      Severity = CRITICAL.

    Fallback (no execution data): vessel present in IR (structural check only).
      Severity = INFO (cannot verify thermodynamic outcome).

    Old bug fixed: the previous implementation checked VF across ALL streams in
    the flowsheet via stream_results.values(). On a two-unit Heater+Vessel
    flowsheet, the heater outlet stream (partial vapour) would satisfy
    has_vapor=True, causing the two_phase check to pass even if the vessel
    produced no separation. The fix restricts to the vessel's outlet streams.
    """
    sr = _stream_results(pr)

    if sr and graph is not None:
        matched = [u for u in _units(graph) if _match_tag(u, pattern)]
        if not matched:
            return {
                "check": "two_phase_outlet", "passed": True,
                "severity": CheckSeverity.INFO,
                "detail": f"no unit matching '{pattern}' — skipped", "source": "none",
            }

        u        = matched[0]
        unit_tag = _tag(u)
        out_tags = _outlet_stream_tags(unit_tag, graph)
        outlets  = [(t, sr[t]) for t in out_tags if t in sr]

        if outlets:
            in_tags   = _inlet_stream_tags(unit_tag, graph)
            feed_flow = sum(_get_flow(sr[t]) or 0.0 for t in in_tags if t in sr)
            min_flow  = feed_flow * _FLOW_FRAC_MIN if feed_flow > 0 else 0.0

            has_vapor  = any(_get_vf(s) > _VF_VAPOR_MIN           for _, s in outlets)
            has_liquid = any(_get_vf(s) < 1.0 - _VF_LIQUID_MIN    for _, s in outlets)
            both_nz    = all((_get_flow(s) or 0.0) > min_flow      for _, s in outlets)

            vf_summary = [(t, round(_get_vf(s), 3)) for t, s in outlets]
            return {
                "check":    "two_phase_outlet",
                "passed":   has_vapor and has_liquid and both_nz,
                "severity": CheckSeverity.CRITICAL,
                "detail":   (f"execution: {unit_tag} outlet VFs {vf_summary}, "
                             f"has_vapor={has_vapor}, has_liquid={has_liquid}, "
                             f"both_nonzero_flow={both_nz} "
                             f"(threshold={min_flow:.5f} mol/s)"),
                "source":   "execution",
            }

    # Fallback: IR structural check
    units = [u for u in _units(graph) if _match_tag(u, pattern)]
    if not units:
        return {
            "check": "two_phase_outlet", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": f"no unit matching '{pattern}' — skipped (no execution data)",
            "source": "none",
        }
    return {
        "check":    "two_phase_outlet",
        "passed":   True,
        "severity": CheckSeverity.INFO,
        "detail":   "vessel present in IR; two-phase cannot be verified without execution",
        "source":   "IR",
    }


def _check_single_phase_ok() -> dict:
    """
    All-vapour flash is a valid outcome — not penalised.
    Always INFO / always passes.
    """
    return {
        "check":    "single_phase_vapor_ok",
        "passed":   True,
        "severity": CheckSeverity.INFO,
        "detail":   "all-vapour flash is a valid outcome",
        "source":   "none",
    }


def _check_bip_injected(model: str, graph) -> dict:
    """
    Source: IR (graph.binary_parameters list).
    CRITICAL: NRTL/UNIQUAC without BIPs silently produces outlet ≈ feed.
    FP risk: low — BIPs are either in the list or not.
    FN risk: low.
    """
    if graph is None:
        return {
            "check": "bip_injected", "passed": False,
            "severity": CheckSeverity.CRITICAL, "detail": "no graph", "source": "IR",
        }
    bips = getattr(graph, "binary_parameters", [])
    model_bips = [b for b in bips
                  if isinstance(b, dict) and b.get("model", "").upper() == model.upper()]
    return {
        "check":    "bip_injected",
        "passed":   len(model_bips) > 0,
        "severity": CheckSeverity.CRITICAL,
        "detail":   f"{len(model_bips)} {model} BIP(s) found in IR",
        "source":   "IR",
    }


def _check_separation_quality(
        pattern: str, max_enrichment: float, graph, pr) -> dict:
    """
    Check that component enrichment across a unit is below a threshold.

    Source: execution stream compositions.
    Cannot be evaluated from IR alone — always skipped without execution data.
    Severity = WARNING.
    """
    sr = _stream_results(pr)
    if not sr or graph is None:
        return {
            "check": "separation_quality_below", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": "requires execution stream data — skipped", "source": "none",
        }
    units = [u for u in _units(graph) if _match_tag(u, pattern)]
    if not units:
        return {
            "check": "separation_quality_below", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": f"no unit matching '{pattern}' — skipped", "source": "none",
        }

    u        = units[0]
    unit_tag = _tag(u)
    in_tags  = _inlet_stream_tags(unit_tag, graph)
    out_tags = _outlet_stream_tags(unit_tag, graph)

    in_sr  = next((sr[t] for t in in_tags  if t in sr), None)
    out_sr = next((sr[t] for t in out_tags if t in sr), None)
    if in_sr is None or out_sr is None:
        return {
            "check": "separation_quality_below", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": "inlet or outlet stream absent from execution results — skipped",
            "source": "none",
        }

    in_comp  = getattr(in_sr,  "composition", {}) or {}
    out_comp = getattr(out_sr, "composition", {}) or {}
    enrichments = [
        out_comp.get(c, 0.0) / y_in
        for c, y_in in in_comp.items()
        if y_in > 0.01
    ]
    if not enrichments:
        return {
            "check": "separation_quality_below", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": "no significant feed components — skipped", "source": "none",
        }

    max_enrich = max(enrichments)
    return {
        "check":    "separation_quality_below",
        "passed":   max_enrich <= max_enrichment,
        "severity": CheckSeverity.WARNING,
        "detail":   f"{unit_tag} max enrichment factor={max_enrich:.3f}, limit={max_enrichment}",
        "source":   "execution",
    }


def _check_temp_consistency(pattern: str, graph, pr) -> dict:
    """
    Check that outlet temperature is physically plausible (> 200 K).

    Primary source: execution outlet stream T_K. Severity = CRITICAL.
    Fallback: IR T_out. Severity = CRITICAL (temperature < 200 K is always wrong).
    """
    units = [u for u in _units(graph) if _match_tag(u, pattern)]
    if not units:
        return {
            "check": "temp_consistency_inlet_outlet", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": f"no unit matching '{pattern}' — skipped", "source": "none",
        }

    u        = units[0]
    unit_tag = _tag(u)
    sr       = _stream_results(pr)

    if sr and graph is not None:
        out_Ts = [_get_T(sr[t]) for t in _outlet_stream_tags(unit_tag, graph)
                  if t in sr and _get_T(sr[t]) is not None]
        if out_Ts:
            t_min = min(out_Ts)
            return {
                "check":    "temp_consistency_inlet_outlet",
                "passed":   t_min > 200,
                "severity": CheckSeverity.CRITICAL,
                "detail":   f"execution: {unit_tag} min outlet T={t_min:.1f} K (must > 200 K)",
                "source":   "execution",
            }

    t_out = _param(u, "T_out")
    if t_out is None:
        return {
            "check": "temp_consistency_inlet_outlet", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": "T_out not set — cannot verify", "source": "IR",
        }
    return {
        "check":    "temp_consistency_inlet_outlet",
        "passed":   t_out > 200,
        "severity": CheckSeverity.CRITICAL,
        "detail":   f"IR: {unit_tag} T_out={t_out:.1f} K (must > 200 K)",
        "source":   "IR",
    }


# ── New simulation-grounded checks ────────────────────────────────────────────

def _check_mass_balance(graph, pr) -> dict:
    """
    Verify total molar flow is conserved: Σ feed_flow ≈ Σ terminal_outlet_flow.

    Source: execution stream flows (mol/s). CRITICAL.
    Tolerance: 1% relative error.
    Cannot be checked from IR alone — skipped without execution data.

    FP risk: low — a DWSIM result with mass balance error indicates a genuine
    solver problem (disconnected stream, unconverged unit).
    FN risk: low — a 1% relative tolerance is tight enough to catch all real errors
    while allowing for floating-point rounding.
    """
    sr = _stream_results(pr)
    if not sr or graph is None:
        return {
            "check": "mass_balance", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": "no execution data — skipped", "source": "none",
        }

    feed_tags     = _feed_tags(graph)
    terminal_tags = _terminal_outlet_tags(graph)

    feed_flows   = [_get_flow(sr[t]) for t in feed_tags     if t in sr and _get_flow(sr[t]) is not None]
    outlet_flows = [_get_flow(sr[t]) for t in terminal_tags if t in sr and _get_flow(sr[t]) is not None]

    if not feed_flows or not outlet_flows:
        return {
            "check": "mass_balance", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": (f"insufficient stream data: "
                       f"feed_tags={feed_tags}, terminal_tags={terminal_tags}"),
            "source": "none",
        }

    total_in  = sum(feed_flows)
    total_out = sum(outlet_flows)

    if total_in <= 0:
        return {
            "check":    "mass_balance",
            "passed":   False,
            "severity": CheckSeverity.CRITICAL,
            "detail":   f"total inlet flow = {total_in:.6f} mol/s (zero or negative)",
            "source":   "execution",
        }

    rel_err = abs(total_in - total_out) / total_in
    return {
        "check":    "mass_balance",
        "passed":   rel_err <= _MB_TOL,
        "severity": CheckSeverity.CRITICAL,
        "detail":   (f"feed={total_in:.4f} mol/s, outlets={total_out:.4f} mol/s, "
                     f"rel_error={rel_err:.3%} "
                     f"({'within' if rel_err <= _MB_TOL else 'exceeds'} 1% tolerance)"),
        "source":   "execution",
    }


def _check_flash_vf(pattern: str, graph, pr) -> dict:
    """
    Verify a flash vessel is genuinely in the two-phase region.

    Complements two_phase_outlet: instead of checking stream VF values,
    this check uses molar flow fractions to detect near-degenerate cases.

    Passes if:
      - at least one outlet stream (VF > 0.05) carries > 0.1% of feed flow
        (a real vapour phase, not a trace), AND
      - at least one outlet stream (VF < 0.95) carries > 0.1% of feed flow.

    Source: execution. CRITICAL.

    Use case: a flash at the bubble point may have VF > 0.05 (technically
    two-phase) but the vapour outlet flow is 0.08% of feed. DWSIM will
    converge, but the result is thermodynamically marginal — small errors in
    T or P flip it to single-phase. This check flags that case explicitly.
    """
    sr = _stream_results(pr)
    if not sr or graph is None:
        return {
            "check": "flash_vapor_fraction", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": "no execution data — skipped", "source": "none",
        }

    units = [u for u in _units(graph) if _match_tag(u, pattern)]
    if not units:
        return {
            "check": "flash_vapor_fraction", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": f"no unit matching '{pattern}' — skipped", "source": "none",
        }

    u        = units[0]
    unit_tag = _tag(u)
    in_tags  = _inlet_stream_tags(unit_tag, graph)
    out_tags = _outlet_stream_tags(unit_tag, graph)

    feed_flow = sum(_get_flow(sr[t]) or 0.0 for t in in_tags if t in sr)
    outlets   = [(t, sr[t]) for t in out_tags if t in sr]

    if not outlets or feed_flow <= 0:
        return {
            "check": "flash_vapor_fraction", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": f"insufficient stream data for {unit_tag}", "source": "none",
        }

    min_flow = feed_flow * _FLOW_FRAC_MIN

    vapour_outlets = [(t, s) for t, s in outlets if _get_vf(s) > _VF_VAPOR_MIN]
    liquid_outlets = [(t, s) for t, s in outlets if _get_vf(s) < 1.0 - _VF_LIQUID_MIN]

    has_vapour_flow = any((_get_flow(s) or 0.0) > min_flow for _, s in vapour_outlets)
    has_liquid_flow = any((_get_flow(s) or 0.0) > min_flow for _, s in liquid_outlets)

    vf_summary = [(t, round(_get_vf(s), 3), round(_get_flow(s) or 0, 5))
                  for t, s in outlets]
    return {
        "check":    "flash_vapor_fraction",
        "passed":   has_vapour_flow and has_liquid_flow,
        "severity": CheckSeverity.CRITICAL,
        "detail":   (f"{unit_tag} outlets (tag, VF, flow_mol_s): {vf_summary}. "
                     f"feed={feed_flow:.4f} mol/s, min_threshold={min_flow:.6f} mol/s. "
                     f"has_vapour_flow={has_vapour_flow}, has_liquid_flow={has_liquid_flow}"),
        "source":   "execution",
    }


def _check_energy_balance(
        pattern: str, max_dT_K: float, graph, pr) -> dict:
    """
    Sanity-check: ΔT across a Heater must not exceed max_dT_K (default 500 K).

    A temperature rise larger than ~500 K from a single heater is physically
    implausible in a chemical process benchmark and almost always indicates a
    unit conversion error (T_out specified in °C instead of K, giving a
    spurious +273 K offset).

    Source: execution (inlet and outlet stream T_K). WARNING.
    Cannot be checked without execution data — skipped otherwise.

    This is NOT a full energy balance (no Cp calculation) — it is a bounding
    check. A result within 500 K is not guaranteed to be energy-balanced;
    it is only guaranteed to not be wildly wrong.
    """
    sr = _stream_results(pr)
    if not sr or graph is None:
        return {
            "check": "energy_balance_heater", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": "no execution data — skipped", "source": "none",
        }

    units = [u for u in _units(graph) if _match_tag(u, pattern)]
    if not units:
        return {
            "check": "energy_balance_heater", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": f"no unit matching '{pattern}' — skipped", "source": "none",
        }

    u        = units[0]
    unit_tag = _tag(u)
    in_Ts    = [_get_T(sr[t]) for t in _inlet_stream_tags(unit_tag, graph)
                if t in sr and _get_T(sr[t]) is not None]
    out_Ts   = [_get_T(sr[t]) for t in _outlet_stream_tags(unit_tag, graph)
                if t in sr and _get_T(sr[t]) is not None]

    if not in_Ts or not out_Ts:
        return {
            "check": "energy_balance_heater", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": f"no T data for {unit_tag} — skipped", "source": "none",
        }

    t_in  = sum(in_Ts)  / len(in_Ts)
    t_out = sum(out_Ts) / len(out_Ts)
    dT    = t_out - t_in
    passed = dT <= max_dT_K
    return {
        "check":    "energy_balance_heater",
        "passed":   passed,
        "severity": CheckSeverity.WARNING,
        "detail":   (f"execution: {unit_tag} ΔT={dT:.1f} K "
                     f"(T_in={t_in:.1f} K → T_out={t_out:.1f} K), "
                     f"limit={max_dT_K:.0f} K"
                     + ("" if passed else " — EXCEEDS: possible unit conversion error")),
        "source":   "execution",
    }


def _check_vle_bubble_point(pattern: str, T_margin_K: float, graph, pr) -> dict:
    """
    Cross-validate DWSIM vapour fraction against an independent bubble-point estimate.

    Algorithm
    ---------
    1. Find the vessel's inlet stream in DWSIM execution results → T_flash, P_flash.
    2. Call bubble_point_K(graph.compounds, P_flash) from ir/thermo_estimation —
       an independent Raoult's Law estimate that does NOT use DWSIM.
    3. Compute VF_overall = vapour_outlet_flow / total_inlet_flow from DWSIM.
    4. Flag WARNING when the two signals are qualitatively inconsistent:
         a. T_flash < T_bubble - T_margin_K  AND  VF_overall > 0.15
            (below bubble point yet DWSIM reports two-phase)
         b. T_flash > T_bubble + 30 K        AND  VF_overall < 0.05
            (well above bubble point yet DWSIM reports all-liquid)

    Severity: WARNING — the bubble-point estimate assumes equimolar feed via
    Raoult's Law and is approximate for non-ideal systems.  A mismatch is a
    red flag, not a definitive error.

    Source: execution (T_flash, P_flash, VF_overall from DWSIM stream results).

    Skipped when: no execution data, compounds not in Antoine database,
    or inlet stream absent from execution results.
    """
    sr = _stream_results(pr)
    if not sr or graph is None:
        return {
            "check": "vle_bubble_point_spot_check", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": "no execution data — skipped", "source": "none",
        }

    units = [u for u in _units(graph) if _match_tag(u, pattern)]
    if not units:
        return {
            "check": "vle_bubble_point_spot_check", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": f"no unit matching '{pattern}' — skipped", "source": "none",
        }

    u        = units[0]
    unit_tag = _tag(u)
    in_tags  = _inlet_stream_tags(unit_tag, graph)
    out_tags = _outlet_stream_tags(unit_tag, graph)

    # Get flash operating conditions from vessel inlet stream
    inlet_srs = [sr[t] for t in in_tags if t in sr]
    if not inlet_srs:
        return {
            "check": "vle_bubble_point_spot_check", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": f"inlet stream(s) {in_tags} absent from execution results — skipped",
            "source": "none",
        }

    T_flash = _get_T(inlet_srs[0])
    P_flash = _get_P(inlet_srs[0])
    if T_flash is None or P_flash is None:
        return {
            "check": "vle_bubble_point_spot_check", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": "inlet stream missing T or P — skipped", "source": "none",
        }

    # Independent bubble-point estimate
    compounds = list(getattr(graph, "compounds", []) or [])
    if not compounds:
        return {
            "check": "vle_bubble_point_spot_check", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": "no compounds on graph — skipped", "source": "none",
        }

    try:
        from ir.thermo_estimation import bubble_point_K
        T_bubble = bubble_point_K(compounds, P_flash)
    except Exception as exc:
        return {
            "check": "vle_bubble_point_spot_check", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": f"bubble_point_K raised {exc} — skipped", "source": "none",
        }

    if T_bubble is None:
        return {
            "check": "vle_bubble_point_spot_check", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": (f"bubble_point_K returned None for {compounds} — "
                       "compounds not in Antoine database, skipped"),
            "source": "none",
        }

    # Overall vapour fraction from flow balance
    outlet_srs     = [(t, sr[t]) for t in out_tags if t in sr]
    total_in_flow  = sum(_get_flow(s) or 0.0 for s in inlet_srs)
    vapour_flow    = sum(
        _get_flow(s) or 0.0 for _, s in outlet_srs if _get_vf(s) > _VF_VAPOR_MIN
    )
    VF_overall = vapour_flow / total_in_flow if total_in_flow > 0 else 0.0

    delta_T = T_flash - T_bubble
    note    = "(estimate: equimolar Raoult's Law — approximate for non-ideal systems)"

    # Case a: below bubble point but DWSIM reports two-phase
    if delta_T < -T_margin_K and VF_overall > 0.15:
        return {
            "check":    "vle_bubble_point_spot_check",
            "passed":   False,
            "severity": CheckSeverity.WARNING,
            "detail":   (
                f"{unit_tag}: T_flash={T_flash:.1f} K is {-delta_T:.1f} K BELOW "
                f"estimated T_bubble={T_bubble:.1f} K at P={P_flash:.0f} Pa, "
                f"yet DWSIM reports VF_overall={VF_overall:.3f} (expect ≈0 below bubble). "
                f"{note}"
            ),
            "source":   "execution",
        }

    # Case b: well above bubble point but DWSIM reports all-liquid
    if delta_T > 30.0 and VF_overall < 0.05:
        return {
            "check":    "vle_bubble_point_spot_check",
            "passed":   False,
            "severity": CheckSeverity.WARNING,
            "detail":   (
                f"{unit_tag}: T_flash={T_flash:.1f} K is {delta_T:.1f} K ABOVE "
                f"estimated T_bubble={T_bubble:.1f} K at P={P_flash:.0f} Pa, "
                f"yet DWSIM reports VF_overall={VF_overall:.3f} (expect >0 above bubble). "
                f"{note}"
            ),
            "source":   "execution",
        }

    return {
        "check":    "vle_bubble_point_spot_check",
        "passed":   True,
        "severity": CheckSeverity.WARNING,
        "detail":   (
            f"{unit_tag}: T_flash={T_flash:.1f} K, T_bubble_est={T_bubble:.1f} K "
            f"(ΔT={delta_T:+.1f} K), VF_overall={VF_overall:.3f} — consistent. {note}"
        ),
        "source":   "execution",
    }


_AZEOTROPE_THRESHOLD = 0.03   # applied instead of min_relative_diff for azeotrope+activity pkg
_ACTIVITY_PKGS       = {"NRTL", "UNIQUAC"}


def _check_separation_achieved(
        pattern: str, min_relative_diff: float, domain: str,
        graph, pr) -> dict:
    """
    Verify that flash-separator outlet streams have meaningfully different compositions.

    A flash where vapour and liquid outlets carry nearly identical compositions
    indicates one of three genuine failure modes:
      (a) NRTL/UNIQUAC missing BIPs → DWSIM treats mixture as ideal → outlet ≈ feed
          for both phases.
      (b) Flash conditions outside the two-phase region → trivial single-phase split
          miscategorised as two-phase by DWSIM's topology.
      (c) Solver converged to a trivial solution.

    These all pass DWSIM's solved=True check, pass mass balance, and produce
    two non-zero outlet streams — the ONLY way to detect them is composition analysis.

    Near-azeotrope guard
    --------------------
    When the check spec includes ``domain="azeotrope"`` AND the property package
    is NRTL or UNIQUAC, the effective threshold is lowered to
    ``_AZEOTROPE_THRESHOLD`` (3%) instead of the default 10%.

    Rationale: near an azeotrope, vapour and liquid compositions converge toward
    the azeotropic point (y_i → x_i). A correctly converged DWSIM result for an
    ethanol-water flash near the 89.4 mol% azeotrope may show only 3–8% relative
    composition difference — far below the 10% default threshold — even though the
    separation is physically correct.  The 3% floor still catches the BIP-missing
    failure mode (rel_diff ≈ 0.5–1%) while not penalising genuine near-azeotrope
    equilibrium.

    The ``domain`` parameter comes from the check spec (``p.get("domain", "")``),
    which maps to the ``BenchmarkCaseSpec.domain`` field.  Set it in the case JSON:
      ``{"type": "separation_achieved", "unit_tag_pattern": "VES",
         "min_relative_diff": 0.10, "domain": "azeotrope"}``

    Algorithm
    ---------
    1. Identify the vapour outlet (highest VF) and liquid outlet (lowest VF).
    2. For each compound at > 1% combined:
         rel_diff_i = |y_i - x_i| / max(y_i, x_i)
    3. max_rel_diff = max of rel_diff_i.
    4. FAIL (CRITICAL) if max_rel_diff < effective_threshold.

    Single-component systems are skipped (INFO — vessel separates phases, not species).

    Source: execution stream composition data. CRITICAL.
    """
    sr = _stream_results(pr)
    if not sr or graph is None:
        return {
            "check": "separation_achieved", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": "no execution data — skipped", "source": "none",
        }

    units = [u for u in _units(graph) if _match_tag(u, pattern)]
    if not units:
        return {
            "check": "separation_achieved", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": f"no unit matching '{pattern}' — skipped", "source": "none",
        }

    u        = units[0]
    unit_tag = _tag(u)
    out_tags = _outlet_stream_tags(unit_tag, graph)
    outlets  = [(t, sr[t]) for t in out_tags if t in sr]

    if len(outlets) < 2:
        return {
            "check": "separation_achieved", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": (f"{unit_tag} has {len(outlets)} outlet stream(s) in results "
                       "— need ≥2 for composition comparison"),
            "source": "none",
        }

    # Sort by VF: [0] = most liquid, [-1] = most vapour
    outlets_by_vf = sorted(outlets, key=lambda x: _get_vf(x[1]))
    liq_tag, liq_sr = outlets_by_vf[0]
    vap_tag, vap_sr = outlets_by_vf[-1]

    y_vap = getattr(vap_sr, "composition", {}) or {}
    x_liq = getattr(liq_sr, "composition", {}) or {}

    # Collect compounds present in either stream above 1% combined
    all_comps = set(y_vap) | set(x_liq)
    significant = [c for c in all_comps if y_vap.get(c, 0) + x_liq.get(c, 0) > 0.01]

    if not significant:
        return {
            "check": "separation_achieved", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": "no compounds above 1% threshold in outlet streams — skipped",
            "source": "none",
        }

    if len(significant) == 1:
        return {
            "check": "separation_achieved", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": (f"single-component system: compositions identical in both "
                       "phases by definition — not a separation failure"),
            "source": "execution",
        }

    # Near-azeotrope guard: lower threshold when the case is flagged as an
    # azeotrope domain AND an activity-coefficient package is in use.
    pkg = getattr(graph, "property_package", "") or ""
    is_azeotrope = domain.lower() == "azeotrope" and pkg in _ACTIVITY_PKGS
    effective_threshold = _AZEOTROPE_THRESHOLD if is_azeotrope else min_relative_diff
    guard_note = (
        f" [near-azeotrope guard active: pkg={pkg!r}, domain=azeotrope → "
        f"threshold={effective_threshold:.2f} instead of {min_relative_diff:.2f}]"
        if is_azeotrope else ""
    )

    # Compute relative differences
    rel_diffs = []
    for c in significant:
        y = y_vap.get(c, 0.0)
        x = x_liq.get(c, 0.0)
        denom = max(y, x)
        if denom > 0.005:
            rel_diffs.append((c, abs(y - x) / denom))

    if not rel_diffs:
        return {
            "check": "separation_achieved", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": "all outlet compositions below 0.5% — cannot evaluate",
            "source": "none",
        }

    # Most separated compound
    key_comp, max_rel_diff = max(rel_diffs, key=lambda x: x[1])

    # Most volatile component (highest enrichment in vapour)
    most_volatile = max(
        significant,
        key=lambda c: (y_vap.get(c, 0) / max(x_liq.get(c, 1e-9), 1e-9))
    )
    enrichment = y_vap.get(most_volatile, 0) / max(x_liq.get(most_volatile, 1e-9), 1e-9)

    passed = max_rel_diff >= effective_threshold
    return {
        "check":    "separation_achieved",
        "passed":   passed,
        "severity": CheckSeverity.CRITICAL,
        "detail":   (
            f"{unit_tag} [{vap_tag}(vap) vs {liq_tag}(liq)]: "
            f"max rel_diff={max_rel_diff:.3f} on '{key_comp}' "
            f"(y={y_vap.get(key_comp, 0):.3f}, x={x_liq.get(key_comp, 0):.3f}), "
            f"threshold={effective_threshold:.2f}.{guard_note} "
            f"Most volatile: '{most_volatile}' @ {enrichment:.2f}× enrichment in vapour."
            + ("" if passed else
               " FAIL: compositions nearly identical — no separation achieved.")
        ),
        "source":   "execution",
    }


# ── Reference-match checks ────────────────────────────────────────────────────
#
# Two entry points:
#   run_reference_comparison(case, pr)  — called by runner for all validation cases;
#                                         returns (list[dict], mape_T, mape_P, mape_vf)
#   evaluate_check() dispatch           — handles "reference_match" check type when
#                                         explicitly listed in physics_checks; returns
#                                         one combined CRITICAL check dict.
#
# Thresholds are stricter than the comparison.py archive tolerances (±5 K vs ±10 K,
# ±0.05 VF vs ±0.10) because the physics-check score should reflect tighter engineering
# accuracy, not the generous bounds used for research-reporting match_score.

def run_reference_comparison(
    case, pipeline_result
) -> "tuple[list[dict], float, float, float]":
    """
    Run ground-truth stream comparison for a validation case.

    Called from the runner after physics checks for any case with reference_file.
    Returns (checks, mape_T_pct, mape_P_pct, mean_vf_abs_err).

    Three checks are returned:
      reference_match_T   — CRITICAL: all matched streams within ±5 K
      reference_match_P   — CRITICAL: all matched streams within ±5% relative
      reference_match_vf  — WARNING:  all comparable streams within ±0.05 absolute
    """
    ref_file = getattr(case, "reference_file", None)
    if not ref_file:
        return [], 0.0, 0.0, 0.0
    return _do_reference_comparison(ref_file, pipeline_result)


def _check_reference_match_overall(
    reference_file: str, pipeline_result
) -> dict:
    """
    Single-check entry for evaluate_check() dispatch when 'reference_match' is
    explicitly listed in a case's physics_checks.  Returns one CRITICAL check
    whose detail summarises all three sub-checks.
    """
    checks, mape_T, mape_P, mape_vf = _do_reference_comparison(
        reference_file, pipeline_result)
    if not checks:
        return {
            "check": "reference_match", "passed": True,
            "severity": CheckSeverity.INFO,
            "detail": "reference comparison skipped (file missing or no execution data)",
            "source": "none",
        }
    crit_checks = [c for c in checks if c["severity"] == CheckSeverity.CRITICAL]
    active_crit = [c for c in crit_checks if c.get("source") != "none"]
    all_pass    = bool(active_crit) and all(c["passed"] for c in active_crit)
    return {
        "check":    "reference_match",
        "passed":   all_pass,
        "severity": CheckSeverity.CRITICAL,
        "detail":   (f"T_MAPE={mape_T:.2f}% P_MAPE={mape_P:.2f}% "
                     f"VF_MAE={mape_vf:.4f}. "
                     f"Sub-check pass: "
                     f"{sum(c['passed'] for c in checks)}/{len(checks)}"),
        "source":   "execution",
    }


def _do_reference_comparison(
    reference_file: str, pipeline_result
) -> "tuple[list[dict], float, float, float]":
    """
    Match system streams to reference streams by CONTENT (composition/T/P/vf, not
    tag) via benchmark.stream_matcher, then compute reference-MAPE over the
    CONFIDENT matched pairs only.  Also emits a 'reference_stream_matching' detail
    check carrying every matched pair (ref<->sys, confidence, dT/dP/dvf) plus the
    unmatched counts, so the matching is fully inspectable in the per-run JSON.
    """
    try:
        from benchmark.comparison import load_reference
        from benchmark.stream_matcher import match_streams
    except ImportError:
        return [], 0.0, 0.0, 0.0

    ref = load_reference(reference_file)
    if ref is None:
        return [], 0.0, 0.0, 0.0

    execution = getattr(pipeline_result, "final_execution", None)
    if execution is None:
        return _skipped_ref_checks("DWSIM did not execute"), 0.0, 0.0, 0.0
    sys_sr = getattr(execution, "stream_results", {}) or {}
    if not sys_sr:
        return _skipped_ref_checks("no stream results from DWSIM"), 0.0, 0.0, 0.0

    # System streams (P in Pa; carry is_feed for feed anchoring).
    sys_streams: dict = {}
    for tag, s in sys_sr.items():
        T = getattr(s, "T_K", None); P = getattr(s, "P_Pa", None)
        vf = getattr(s, "vapor_fraction", None)
        sys_streams[str(tag)] = {
            "T_K":            (float(T)  if T  is not None else None),
            "P_Pa":           (float(P)  if P  is not None else None),
            "vapor_fraction": (float(vf) if vf is not None else None),
            "composition":    dict(getattr(s, "composition", {}) or {}),
            "is_feed":        bool(getattr(s, "is_feed", False)),
        }

    ref_raw = ref.get("streams", {})
    if not ref_raw:
        return _skipped_ref_checks("reference file has no streams"), 0.0, 0.0, 0.0

    # Reference streams normalised to the same fields (P in Pa; accept P_Pa or P_bar).
    ref_streams: dict = {}
    for rt, rs in ref_raw.items():
        P_Pa = rs.get("P_Pa")
        if P_Pa is None and rs.get("P_bar") is not None:
            P_Pa = float(rs["P_bar"]) * 1e5
        ref_streams[str(rt)] = {
            "T_K":            rs.get("T_K"),
            "P_Pa":           P_Pa,
            "vapor_fraction": rs.get("vapor_fraction"),
            "composition":    rs.get("composition", {}) or {},
            "is_feed":        rs.get("is_feed"),
        }

    match = match_streams(sys_streams, ref_streams)
    pairs = match["pairs"]

    t_apes: list = []; p_apes: list = []; vf_errs: list = []
    t_fails: list = []; p_fails: list = []; vf_fails: list = []
    for pr_ in pairs:
        rt, st = pr_["ref_tag"], pr_["sys_tag"]
        ref_s, sys_s = ref_streams[rt], sys_streams[st]
        ref_T, sys_T = ref_s["T_K"], sys_s["T_K"]
        if ref_T and sys_T is not None and float(ref_T) > 1e-9:
            ae = abs(float(sys_T) - float(ref_T))
            t_apes.append(ae / float(ref_T) * 100.0)
            if ae > _REF_TOL_T_K:
                t_fails.append(f"{rt}->{st}: |dT|={ae:.1f}K "
                               f"(sys={float(sys_T):.1f} ref={float(ref_T):.1f})")
        ref_P, sys_P = ref_s["P_Pa"], sys_s["P_Pa"]
        if ref_P and sys_P is not None and float(ref_P) > 1e-9:
            rel = abs(float(sys_P) - float(ref_P)) / float(ref_P)
            p_apes.append(rel * 100.0)
            if rel > _REF_TOL_P_REL:
                p_fails.append(f"{rt}->{st}: |dP/P|={rel:.1%}")
        ref_vf, sys_vf = ref_s["vapor_fraction"], sys_s["vapor_fraction"]
        if ref_vf is not None and sys_vf is not None:
            ae = abs(float(sys_vf) - float(ref_vf))
            vf_errs.append(ae)
            if ae > _REF_TOL_VF:
                vf_fails.append(f"{rt}->{st}: |dvf|={ae:.3f}")

    n_matched = match["n_matched"]
    if n_matched < _MIN_MATCH_FOR_MAPE:
        # Too few matched streams to compute a trustworthy MAPE.  Emit the T/P/vf
        # checks as INFO/none (so they do NOT count as passing critical checks —
        # reference_match_pass will be False) and return None for every MAPE so
        # the caller reports "insufficient_match", never a bare 0.0.
        _reason = (f"insufficient_match: {n_matched} matched stream(s) < "
                   f"{_MIN_MATCH_FOR_MAPE} required — MAPE not computed")
        checks = [
            {"check": c, "passed": True, "severity": CheckSeverity.INFO,
             "source": "none", "detail": _reason}
            for c in ("reference_match_T", "reference_match_P", "reference_match_vf")
        ]
        checks.append(_matching_detail_check(match))
        return checks, None, None, None

    mape_T  = round(sum(t_apes)  / len(t_apes),  2) if t_apes  else 0.0
    mape_P  = round(sum(p_apes)  / len(p_apes),  2) if p_apes  else 0.0
    mape_vf = round(sum(vf_errs) / len(vf_errs), 4) if vf_errs else 0.0
    n_vf_cmp = len(vf_errs)

    checks = [
        {"check": "reference_match_T", "passed": not t_fails,
         "severity": CheckSeverity.CRITICAL,
         "detail": (f"{n_matched - len(t_fails)}/{n_matched} matched streams within "
                    f"+/-{_REF_TOL_T_K:.0f} K. MAPE={mape_T:.2f}%."
                    + (f" Failures: {'; '.join(t_fails)}" if t_fails else "")),
         "source": "execution"},
        {"check": "reference_match_P", "passed": not p_fails,
         "severity": CheckSeverity.CRITICAL,
         "detail": (f"{n_matched - len(p_fails)}/{n_matched} matched streams within "
                    f"+/-{_REF_TOL_P_REL:.0%}. MAPE={mape_P:.2f}%."
                    + (f" Failures: {'; '.join(p_fails)}" if p_fails else "")),
         "source": "execution"},
        {"check": "reference_match_vf", "passed": not vf_fails,
         "severity": CheckSeverity.WARNING,
         "detail": ((f"{n_vf_cmp - len(vf_fails)}/{n_vf_cmp} matched streams within "
                     f"+/-{_REF_TOL_VF}. MAE={mape_vf:.4f}.")
                    if n_vf_cmp > 0 else "no comparable vapour-fraction data")
                   + (f" Failures: {'; '.join(vf_fails)}" if vf_fails else ""),
         "source": "execution" if n_vf_cmp > 0 else "none"},
        _matching_detail_check(match),
    ]
    return checks, mape_T, mape_P, mape_vf


def _matching_detail_check(match: dict) -> dict:
    """INFO check carrying the full stream-matching detail for the per-run JSON."""
    return {
        "check":    "reference_stream_matching",
        "passed":   True,
        "severity": CheckSeverity.INFO,
        "source":   "execution",
        "detail":   (f"{match['n_matched']} matched (confidence >= "
                     f"{match['threshold']:.2f}); {match['n_system_unmatched']} system "
                     f"+ {match['n_reference_unmatched']} reference unmatched"),
        "n_matched":             match["n_matched"],
        "n_system_unmatched":    match["n_system_unmatched"],
        "n_reference_unmatched": match["n_reference_unmatched"],
        "matches":               match["pairs"],
        "system_unmatched":      match["system_unmatched"],
        "reference_unmatched":   match["reference_unmatched"],
    }



def _match_ref_stream(
    ref_tag: str,
    ref_s: dict,
    sys_streams: dict,
    used: set,
    best_comp_fn,
) -> "Optional[str]":
    """
    Find the system stream that best matches a reference stream tag.

    Strategy:
      1. Exact tag match
      2. Case-insensitive tag match
      3. Composition nearest-neighbour (delegates to comparison._best_composition_match)
    """
    if ref_tag in sys_streams and ref_tag not in used:
        return ref_tag
    for k in sys_streams:
        if k not in used and k.lower() == ref_tag.lower():
            return k
    try:
        best, _ = best_comp_fn(ref_s, sys_streams, used)
        return best
    except Exception:
        return None


def _skipped_ref_checks(reason: str = "no execution data") -> list:
    """Return three INFO-level skipped checks when comparison cannot proceed."""
    return [
        {
            "check":    name,
            "passed":   True,
            "severity": CheckSeverity.INFO,
            "detail":   f"reference comparison skipped: {reason}",
            "source":   "none",
        }
        for name in ("reference_match_T", "reference_match_P", "reference_match_vf")
    ]


# ── IR topology helpers ────────────────────────────────────────────────────────

def _ir_inlet_T(unit_tag: str, graph) -> float:
    """IR inlet stream T for a unit, or 298.15 K if not set."""
    if graph is None:
        return 298.15
    fn = getattr(graph, "stream", None)
    if not callable(fn):
        return 298.15
    Ts = [e.T for tag in _inlet_stream_tags(unit_tag, graph)
          for e in [fn(tag)] if e is not None and e.T is not None]
    return sum(Ts) / len(Ts) if Ts else 298.15


def _ir_inlet_P(unit_tag: str, graph) -> float:
    """IR inlet stream P for a unit, or 101325 Pa if not set."""
    if graph is None:
        return 101325.0
    fn = getattr(graph, "stream", None)
    if not callable(fn):
        return 101325.0
    Ps = [e.P for tag in _inlet_stream_tags(unit_tag, graph)
          for e in [fn(tag)] if e is not None and e.P is not None]
    return sum(Ps) / len(Ps) if Ps else 101325.0


# ── Entry point ────────────────────────────────────────────────────────────────

def run_physics_checks(case, pipeline_result) -> list[dict]:
    """
    Run all physics checks defined in case.expected.physics_checks.

    Parameters
    ----------
    case            : BenchmarkCaseSpec
    pipeline_result : OrchestratorV2 PipelineResult

    Returns
    -------
    List of check result dicts, each containing:
      check, passed, severity, detail, source
    """
    graph   = getattr(pipeline_result, "final_graph", None)
    results = []
    for chk in case.expected.physics_checks:
        try:
            result = evaluate_check(chk, graph, pipeline_result)
        except Exception as exc:
            result = {
                "check":    chk.check_type,
                "passed":   False,
                "severity": CheckSeverity.CRITICAL,
                "detail":   f"evaluator error: {exc}",
                "source":   "error",
            }
        results.append(result)
    return results
