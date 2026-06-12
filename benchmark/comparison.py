"""
Flowsheet comparison engine for validation cases.

Compares a system-generated flowsheet against a ground-truth reference
produced from an original DWSIM simulation file.

Tolerances (chosen to reflect real modelling uncertainty):
  Temperature  : ±10 K
  Pressure     : ±5 % relative
  Composition  : ±0.05 mole fraction (absolute)
  Vapour fraction: ±0.10 (absolute)
  Unit types   : exact match (count and set of types)
  Property pkg : exact match
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

# ── Tolerances ─────────────────────────────────────────────────────────────────

TOL_T_K        = 10.0    # K
TOL_P_REL      = 0.05    # fractional
TOL_FLOW_REL   = 0.20    # fractional (molar flow — looser than T/P)
TOL_COMP       = 0.05    # absolute mole fraction
TOL_VF         = 0.10    # absolute vapour fraction


# ── Result containers ──────────────────────────────────────────────────────────

@dataclass
class FieldResult:
    stream:       str
    field:        str
    system:       float
    reference:    float
    diff:         float
    passed:       bool
    tolerance:    str
    is_diagnostic: bool = False  # if True: informational only, excluded from score/overall_pass


@dataclass
class ComparisonResult:
    case_id:           str
    case_name:         str
    reference_file:    str
    overall_pass:      bool
    match_score:       float          # 0.0 – 1.0 (diagnostic fields excluded)
    pkg_match:         bool
    pkg_system:        str
    pkg_reference:     str
    unit_type_match:   bool           # exact sorted match (diagnostic only)
    unit_count_in_range: bool         # n_units_min <= len(sys) <= n_units_max
    unit_types_system:    list[str]
    unit_types_reference: list[str]
    field_results:     list[FieldResult] = field(default_factory=list)
    warnings:          list[str]        = field(default_factory=list)
    failure_modes:     dict             = field(default_factory=dict)  # structured failure categories
    mape_T_pct:        float           = 0.0   # mean |ΔT/T_ref| × 100 across matched streams


# ── Main comparison function ───────────────────────────────────────────────────

def compare_flowsheets(
    system_fs:      dict,
    reference_fs:   dict,
    reference_file: str = "",
    n_units_min:    Optional[int] = None,
    n_units_max:    Optional[int] = None,
) -> ComparisonResult:
    """
    Compare a system-generated flowsheet dict against a reference dict.

    Both dicts must follow the _flowsheet_dict / reference JSON schema:
    keys: case_id, compounds, property_package, units, connections, streams.
    """
    warnings: list[str] = []
    field_results: list[FieldResult] = []

    # ── Property package ──────────────────────────────────────────────────────
    pkg_sys = system_fs.get("property_package", "")
    pkg_ref = reference_fs.get("property_package", "")
    pkg_match = pkg_sys.strip().lower() == pkg_ref.strip().lower()
    if not pkg_match:
        warnings.append(
            f"Property package mismatch: system='{pkg_sys}' reference='{pkg_ref}'")

    # ── Unit types ────────────────────────────────────────────────────────────
    sys_types = sorted(u["type"] for u in system_fs.get("units", []))
    ref_types = sorted(u["type"] for u in reference_fs.get("units", []))
    unit_type_match = sys_types == ref_types
    if not unit_type_match:
        warnings.append(
            f"Unit type mismatch: system={sys_types} reference={ref_types}")

    # Unit count range check (uses caller-supplied bounds; falls back to exact
    # reference count when no bounds are provided for backward compatibility).
    _lo = n_units_min if n_units_min is not None else len(ref_types)
    _hi = n_units_max if n_units_max is not None else len(ref_types)
    unit_count_in_range = _lo <= len(sys_types) <= _hi
    if not unit_count_in_range:
        warnings.append(
            f"Unit count out of range: system={len(sys_types)} "
            f"expected=[{_lo}, {_hi}]")

    # ── Stream conditions ─────────────────────────────────────────────────────
    sys_streams  = system_fs.get("streams", {})
    ref_streams  = reference_fs.get("streams", {})
    compounds    = reference_fs.get("compounds", [])
    stream_roles = reference_fs.get("stream_roles", {})  # optional: {ref_tag: role_label}

    # matched_sys keeps track of which system stream tags have already been
    # claimed so composition fallback never matches the same stream twice.
    matched_sys: set[str] = set()

    for stag, ref_s in ref_streams.items():
        sys_tag  = None
        match_how = "name"

        # 1. Exact name match
        if stag in sys_streams:
            sys_tag = stag

        # 2. Case-insensitive name match
        if sys_tag is None:
            for k in sys_streams:
                if k not in matched_sys and k.lower() == stag.lower():
                    sys_tag   = k
                    match_how = "name (case-insensitive)"
                    break

        # 3. Composition nearest-neighbour fallback
        if sys_tag is None:
            best_tag, best_dist = _best_composition_match(
                ref_s, sys_streams, matched_sys)
            if best_tag is not None:
                sys_tag   = best_tag
                match_how = f"composition (L1={best_dist:.3f} → '{best_tag}')"
                role_str  = f" [{stream_roles[stag]}]" if stag in stream_roles else ""
                warnings.append(
                    f"Stream '{stag}'{role_str} matched by composition to "
                    f"system stream '{best_tag}' (L1={best_dist:.3f}) — "
                    f"names differ but conditions will be compared")

        if sys_tag is None:
            role_str = f" [{stream_roles[stag]}]" if stag in stream_roles else ""
            warnings.append(
                f"Stream '{stag}'{role_str} has no match in system output "
                f"(tried name and composition)")
            continue

        matched_sys.add(sys_tag)
        sys_s = sys_streams[sys_tag]

        # Temperature
        field_results.append(_check(
            stag, "T_K",
            sys_s.get("T_K", 0.0), ref_s.get("T_K", 0.0),
            abs_tol=TOL_T_K, tol_str=f"±{TOL_T_K} K",
        ))

        # Pressure (relative)
        field_results.append(_check_rel(
            stag, "P_bar",
            sys_s.get("P_bar", 0.0), ref_s.get("P_bar", 0.0),
            rel_tol=TOL_P_REL, tol_str=f"±{TOL_P_REL*100:.0f}% rel",
        ))

        # Molar flow (relative, diagnostic only) — the NL prompt never states
        # feed flow rate so the system must invent one; absolute flow comparison
        # is therefore meaningless for pass/fail.  Kept for information only.
        ref_flow = ref_s.get("flow_mol_s", 0.0)
        if ref_flow > 1e-9:
            field_results.append(_check_rel(
                stag, "flow_mol_s",
                sys_s.get("flow_mol_s", 0.0), ref_flow,
                rel_tol=TOL_FLOW_REL, tol_str=f"±{TOL_FLOW_REL*100:.0f}% rel",
                is_diagnostic=True,
            ))

        # Vapour fraction — skip if reference value is null (unknown/mixed phase)
        ref_vf = ref_s.get("vapor_fraction")
        sys_vf = sys_s.get("vapor_fraction")
        if ref_vf is not None and sys_vf is not None:
            field_results.append(_check(
                stag, "vapor_fraction",
                float(sys_vf), float(ref_vf),
                abs_tol=TOL_VF, tol_str=f"±{TOL_VF}",
            ))

        # Composition per compound
        sys_comp = sys_s.get("composition", {})
        ref_comp = ref_s.get("composition", {})
        for cmp in compounds:
            sys_x = sys_comp.get(cmp, 0.0)
            ref_x = ref_comp.get(cmp, 0.0)
            field_results.append(_check(
                stag, f"x({cmp})",
                sys_x, ref_x,
                abs_tol=TOL_COMP, tol_str=f"±{TOL_COMP}",
            ))

    # ── MAPE on temperature ────────────────────────────────────────────────────
    _t_apes = [
        abs(fr.diff / fr.reference) * 100.0
        for fr in field_results
        if fr.field == "T_K" and not fr.is_diagnostic and abs(fr.reference) > 1e-9
    ]
    mape_T_pct = round(sum(_t_apes) / len(_t_apes), 2) if _t_apes else 0.0

    # ── Score ─────────────────────────────────────────────────────────────────
    # Diagnostic fields (flow_mol_s) are excluded from both score and
    # overall_pass — the system has no way to know the feed flow rate from
    # the NL prompt, so absolute flow agreement is informational only.
    scored = [fr for fr in field_results if not fr.is_diagnostic]
    n_total  = len(scored) + 1 + 1   # +1 pkg, +1 unit count range
    n_passed = sum(1 for fr in scored if fr.passed)
    if pkg_match:
        n_passed += 1
    if unit_count_in_range:
        n_passed += 1

    score = n_passed / max(n_total, 1)

    # overall_pass: pkg correct + unit count in range + every scored field OK.
    overall_pass = pkg_match and unit_count_in_range and all(
        fr.passed for fr in scored)

    # ── Failure mode classification ────────────────────────────────────────────
    failure_modes: dict = {}
    if not pkg_match:
        failure_modes["pkg_mismatch"] = True
    if not unit_count_in_range:
        failure_modes["unit_count_out_of_range"] = True
    if any("has no match" in w for w in warnings):
        failure_modes["stream_not_found"] = True
    for fr in field_results:
        if fr.passed:
            continue
        if fr.is_diagnostic:
            if fr.field == "flow_mol_s":
                failure_modes["flow_fail"] = True
        elif fr.field == "T_K":
            failure_modes["temperature_fail"] = True
        elif fr.field == "P_bar":
            failure_modes["pressure_fail"] = True
        elif fr.field == "vapor_fraction":
            failure_modes["vapor_fraction_fail"] = True
        elif fr.field.startswith("x("):
            failure_modes["composition_fail"] = True

    return ComparisonResult(
        case_id              = system_fs.get("case_id", ""),
        case_name            = system_fs.get("case_name", ""),
        reference_file       = reference_file,
        overall_pass         = overall_pass,
        match_score          = round(score, 4),
        pkg_match            = pkg_match,
        pkg_system           = pkg_sys,
        pkg_reference        = pkg_ref,
        unit_type_match      = unit_type_match,
        unit_count_in_range  = unit_count_in_range,
        unit_types_system    = sys_types,
        unit_types_reference = ref_types,
        field_results        = field_results,
        warnings             = warnings,
        failure_modes        = failure_modes,
        mape_T_pct           = mape_T_pct,
    )


_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))


def load_reference(reference_file: str) -> Optional[dict]:
    """Load a reference flowsheet JSON. Returns None if file not found or unfilled.

    Paths may be absolute or relative to the repo root — resolved in that order.
    Files that still contain the '_instructions' key are treated as unfilled
    templates and skipped silently.
    """
    path = reference_file
    if not os.path.isabs(path):
        path = os.path.join(_REPO_ROOT, reference_file)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "_instructions" in data:
        return None  # template not yet filled in
    return data


# ── Plain-text diff report ─────────────────────────────────────────────────────

def comparison_report(cr: ComparisonResult) -> str:
    """Render a human-readable comparison report."""
    W    = 64
    bar  = "═" * W
    line = "─" * W
    tick = "✓"
    cross = "✗"

    overall = f"{'PASS' if cr.overall_pass else 'FAIL'}  (match score: {cr.match_score:.1%})"

    mape_str = f"{cr.mape_T_pct:.2f}%" if cr.mape_T_pct > 0.0 else "—"

    lines = [
        bar,
        f"COMPARISON REPORT — {cr.case_id}",
        cr.case_name,
        f"Reference: {cr.reference_file}",
        bar,
        f"Overall : {overall}",
        f"MAPE T  : {mape_str}  (mean |ΔT/T_ref|×100 across matched streams)",
        "",
        "TOPOLOGY",
        line,
        f"Property package : {cr.pkg_system:<20} ref={cr.pkg_reference:<20} "
        f"{tick if cr.pkg_match else cross}",
        f"Unit count       : sys={len(cr.unit_types_system)}  "
        f"ref={len(cr.unit_types_reference)}  "
        f"{tick if cr.unit_count_in_range else cross}",
        f"Unit types (sys) : {', '.join(cr.unit_types_system) or '—'}",
        f"Unit types (ref) : {', '.join(cr.unit_types_reference) or '—'} "
        f"{'(exact match)' if cr.unit_type_match else '(differs — diagnostic only)'}",
        "",
    ]

    if cr.warnings:
        lines.append("WARNINGS")
        lines.append(line)
        for w in cr.warnings:
            lines.append(f"  ! {w}")
        lines.append("")

    # ── Stream field table ─────────────────────────────────────────────────────
    lines.append(
        f"STREAM CONDITIONS  "
        f"(T ±{TOL_T_K}K | P ±{TOL_P_REL*100:.0f}% | "
        f"comp ±{TOL_COMP} | VF ±{TOL_VF} | "
        f"F ±{TOL_FLOW_REL*100:.0f}% (i)=diagnostic only)")
    lines.append(line)
    lines.append(
        f"{'Stream':<12}  {'Field':<18}  {'System':>9}  {'Reference':>9}  "
        f"{'Diff':>9}  {'':>2}")
    lines.append(line)

    current_stream = None
    for fr in cr.field_results:
        if fr.stream != current_stream:
            if current_stream is not None:
                lines.append("")
            current_stream = fr.stream
        if fr.is_diagnostic:
            status = "(i)"
        elif fr.passed:
            status = tick
        else:
            status = cross
        lines.append(
            f"{fr.stream:<12}  {fr.field:<18}  {fr.system:>9.4f}  "
            f"{fr.reference:>9.4f}  {fr.diff:>+9.4f}  {status}")

    lines += ["", bar]
    return "\n".join(lines) + "\n"


# ── Helpers ────────────────────────────────────────────────────────────────────

# Maximum L1 composition distance to accept a composition-based match.
# L1 ranges 0–2; 0.40 allows ~20% average error per compound in a 2-component
# system — generous enough to catch right-direction mismatches but tight enough
# to reject completely wrong streams.
_COMP_MATCH_MAX_L1 = 0.40


def _compose_distance(comp_a: dict, comp_b: dict) -> float:
    """L1 distance between two mole-fraction dicts over their union of keys."""
    keys = set(comp_a) | set(comp_b)
    return sum(abs(comp_a.get(k, 0.0) - comp_b.get(k, 0.0)) for k in keys)


def _best_composition_match(
    ref_s:       dict,
    sys_streams: dict,
    used_tags:   set,
) -> tuple:
    """
    Find the unmatched system stream with the smallest L1 composition distance
    to ref_s.  Returns (best_tag, best_dist) or (None, inf) if nothing is
    within _COMP_MATCH_MAX_L1.
    """
    ref_comp = ref_s.get("composition", {})
    if not ref_comp:
        return None, float("inf")

    best_tag  = None
    best_dist = float("inf")
    for stag, ss in sys_streams.items():
        if stag in used_tags:
            continue
        sys_comp = ss.get("composition", {})
        if not sys_comp:
            continue
        d = _compose_distance(ref_comp, sys_comp)
        if d < best_dist:
            best_dist = d
            best_tag  = stag

    if best_dist <= _COMP_MATCH_MAX_L1:
        return best_tag, best_dist
    return None, best_dist


def _check(
    stream: str, field: str,
    system: float, reference: float,
    abs_tol: float, tol_str: str,
    is_diagnostic: bool = False,
) -> FieldResult:
    diff   = system - reference
    passed = abs(diff) <= abs_tol
    return FieldResult(stream, field, system, reference, diff, passed, tol_str, is_diagnostic)


def _check_rel(
    stream: str, field: str,
    system: float, reference: float,
    rel_tol: float, tol_str: str,
    is_diagnostic: bool = False,
) -> FieldResult:
    diff = system - reference
    denom  = abs(reference) if abs(reference) > 1e-9 else 1.0
    passed = abs(diff / denom) <= rel_tol
    return FieldResult(stream, field, system, reference, diff, passed, tol_str, is_diagnostic)
