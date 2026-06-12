"""
Executor: translates a validated flowsheet JSON into DWSIM API calls,
runs the solver, and returns a structured ExecutionResult.

Pre-execution checks catch issues before touching DWSIM.
Post-execution checks verify physical plausibility of results.
All errors are structured for the Critic Agent to diagnose.
"""
from __future__ import annotations
import sys
import math
from dataclasses import dataclass, field

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── DWSIM property package name mapping ───────────────────────────────────────
# Our canonical schema names → exact DWSIM CreateAndAddPropertyPackage strings.
# Verify with `sim.AvailablePropertyPackages` if adding new packages.
_PP_MAP: dict[str, str] = {
    "Raoult's Law":         "Raoult's Law",
    "NRTL":                 "NRTL",
    "UNIQUAC":              "UNIQUAC",
    "Peng-Robinson":        "Peng-Robinson (PR)",
    "Soave-Redlich-Kwong":  "Soave-Redlich-Kwong (SRK)",
    "Lee-Kesler-Plöcker":   "Lee-Kesler-Plöcker",
}

# Parameters each unit type requires in the JSON
_REQUIRED_UNIT_PARAMS: dict[str, list[str]] = {
    "Heater":     ["T_out"],
    "Cooler":     ["T_out"],
    "Vessel":     [],
    "Mixer":      [],
    "Splitter":   ["split_fractions"],
    "Pump":       ["P_out"],
    "Compressor": ["P_out"],
    "Expander":   ["P_out"],
}

# Physical plausibility bounds
_T_MIN_K   = 100.0       # below this is almost certainly wrong
_T_MAX_K   = 2000.0
_P_MIN_PA  = 100.0       # near-vacuum — usually a sign of unit mismatch
_P_MAX_PA  = 1e8         # 1000 bar
_COMP_TOL  = 0.02        # composition sum must be within this of 1.0
_MB_TOL    = 0.01        # mass balance relative tolerance


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class StreamResult:
    tag:          str
    T_K:          float
    P_Pa:         float
    flow_mol_s:   float
    composition:    dict[str, float]
    vapor_fraction: float = 0.0
    is_feed:        bool  = False

    @property
    def T_C(self) -> float:
        return self.T_K - 273.15

    @property
    def P_bar(self) -> float:
        return self.P_Pa / 1e5

    def composition_sum(self) -> float:
        return sum(self.composition.values())


@dataclass
class ExecutionResult:
    solved:           bool
    stream_results:   dict[str, StreamResult] = field(default_factory=dict)
    errors:           list[str]               = field(default_factory=list)
    warnings:         list[str]               = field(default_factory=list)
    diagnostics:      dict                    = field(default_factory=dict)
    solver_errors:    list[str]               = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Solved: {self.solved}"]
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            lines.extend(f"  - {e}" for e in self.errors)
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            lines.extend(f"  - {w}" for w in self.warnings)
        for tag, s in self.stream_results.items():
            lines.append(
                f"  {tag}: T={s.T_C:.1f}°C  P={s.P_bar:.3f}bar  "
                f"flow={s.flow_mol_s:.4f} mol/s  "
                f"comp={s.composition}")
        return "\n".join(lines)


# ── Pre-execution checks ───────────────────────────────────────────────────────

def pre_execution_check(flowsheet: dict) -> list[str]:
    """
    Validate the flowsheet for executor-specific requirements.
    Returns a list of errors (empty = safe to proceed).
    These checks run before any DWSIM object is created.
    """
    errors = []
    compounds  = flowsheet.get("compounds", [])
    streams    = {s["tag"]: s for s in flowsheet.get("streams", [])}
    units      = {u["tag"]: u for u in flowsheet.get("units", [])}
    conns      = flowsheet.get("connections", [])

    # 1. Property package must be mappable to a DWSIM name
    pp = flowsheet.get("property_package", "")
    if pp not in _PP_MAP:
        errors.append(
            f"Property package '{pp}' has no DWSIM mapping. "
            f"Known packages: {list(_PP_MAP)}")
    for u in flowsheet.get("units", []):
        upp = u.get("property_package")
        if upp and upp not in _PP_MAP:
            errors.append(
                f"Unit '{u['tag']}' property_package '{upp}' "
                f"has no DWSIM mapping.")

    # 2. Identify feed streams (no incoming connections)
    has_incoming = {conn[1] for conn in conns if len(conn) >= 2}
    feed_tags = {tag for tag in streams if tag not in has_incoming}

    # 3. Feed streams must have T, P, flow, and composition
    for tag in feed_tags:
        s = streams[tag]
        for field_name in ("T", "P", "flow"):
            if field_name not in s or s[field_name] is None:
                errors.append(
                    f"Feed stream '{tag}' is missing required field '{field_name}'.")
        comp = s.get("composition", {})
        if not comp:
            errors.append(
                f"Feed stream '{tag}' has no composition specified.")
        else:
            missing = [c for c in compounds if c not in comp]
            if missing:
                errors.append(
                    f"Feed stream '{tag}' composition missing compounds: {missing}. "
                    "All compounds must have a mole fraction (use 0.0 for absent).")
            total = sum(comp.values())
            if abs(total - 1.0) > _COMP_TOL:
                errors.append(
                    f"Feed stream '{tag}' composition sums to {total:.4f}, not 1.0.")

    # 4. Required unit parameters
    for tag, u in units.items():
        utype = u.get("type", "")
        required = _REQUIRED_UNIT_PARAMS.get(utype, [])
        for param in required:
            if param not in u or u[param] is None:
                errors.append(
                    f"Unit '{tag}' ({utype}) is missing required "
                    f"parameter '{param}'.")
        # Heater/Cooler T_out must be physically reasonable
        if utype in ("Heater", "Cooler"):
            t_out = u.get("T_out")
            if t_out is not None:
                if t_out < _T_MIN_K or t_out > _T_MAX_K:
                    errors.append(
                        f"Unit '{tag}' T_out={t_out} K is outside physical "
                        f"range [{_T_MIN_K}, {_T_MAX_K}] K.")
        # Efficiency must be a fraction in (0, 1] — LLM sometimes outputs 75 not 0.75
        if utype in ("Pump", "Compressor", "Expander"):
            eff = u.get("efficiency")
            if eff is not None and not (0 < eff <= 1.0):
                errors.append(
                    f"Unit '{tag}' efficiency={eff} is invalid — "
                    "must be a fraction in (0, 1], e.g. 0.75 not 75.")

    # 5. No direct unit-to-unit connections (must route through streams)
    for conn in conns:
        if len(conn) < 2:
            continue
        src, dst = conn[0], conn[1]
        if src in units and dst in units:
            errors.append(
                f"Connection '{src}' → '{dst}' links two unit operations directly. "
                "All connections must route through a MaterialStream.")

    # 6. Every object has at least one connection
    all_tags   = set(streams) | set(units)
    conn_srcs  = {c[0] for c in conns if len(c) >= 2}
    conn_dsts  = {c[1] for c in conns if len(c) >= 2}
    connected  = conn_srcs | conn_dsts
    isolated   = all_tags - connected - feed_tags
    for tag in isolated:
        errors.append(
            f"Object '{tag}' has no connections — it will be ignored by the solver.")

    # 7. At least one feed stream exists
    if not feed_tags:
        errors.append(
            "No feed streams identified. At least one stream must have "
            "no incoming connections and carry feed conditions.")

    # 8. Splitter split_fractions must sum to 1.0
    for tag, u in units.items():
        if u.get("type") == "Splitter":
            fracs = u.get("split_fractions", {})
            if fracs:
                total = sum(fracs.values())
                if abs(total - 1.0) > _COMP_TOL:
                    errors.append(
                        f"Unit '{tag}' (Splitter) split_fractions sum to "
                        f"{total:.4f}, not 1.0.")

    return errors


# ── Executor ───────────────────────────────────────────────────────────────────

class Executor:
    """
    Translates a validated flowsheet JSON to DWSIM calls and returns results.
    Designed to feed directly into the Critic Agent.
    """

    def run(self, flowsheet: dict) -> ExecutionResult:
        """Full pipeline: pre-check → build → solve → post-check."""

        # Pre-execution checks
        pre_errors = pre_execution_check(flowsheet)
        if pre_errors:
            print(f"  [EXEC] pre_check FAILED: {pre_errors}", flush=True, file=sys.stderr)
            return ExecutionResult(
                solved=False,
                errors=pre_errors,
                diagnostics={"stage": "pre_execution", "flowsheet": flowsheet})

        # Build and solve
        try:
            result = self._build_and_solve(flowsheet)
        except Exception as e:
            return ExecutionResult(
                solved=False,
                errors=[f"Unexpected executor error: {type(e).__name__}: {e}"],
                diagnostics={"stage": "build_or_solve", "flowsheet": flowsheet})

        return result

    def _build_and_solve(self, flowsheet: dict) -> ExecutionResult:
        import sys as _sys
        import json as _json

        def _chk(label):
            print(f"  [EXEC] {label}", flush=True, file=_sys.stderr)

        # Dump the flowsheet JSON so we can see exactly what DWSIM receives.
        # This appears on stderr and does not affect benchmark stdout output.
        try:
            print(f"  [EXEC] flowsheet JSON:\n{_json.dumps(flowsheet, indent=2)}",
                  flush=True, file=_sys.stderr)
        except Exception:
            pass

        _chk("importing DWSIMFlowsheet")
        from dwsim.dwsim_wrapper import DWSIMFlowsheet

        compounds   = flowsheet["compounds"]
        streams_cfg = {s["tag"]: s for s in flowsheet["streams"]}
        units_cfg   = {u["tag"]: u for u in flowsheet["units"]}
        conns       = flowsheet["connections"]
        default_pp  = flowsheet["property_package"]

        # Identify feeds (streams with no incoming connection)
        has_incoming  = {c[1] for c in conns if len(c) >= 2}
        feed_tags     = {t for t in streams_cfg if t not in has_incoming}

        _chk("DWSIMFlowsheet()")
        sim = DWSIMFlowsheet()
        _chk("DWSIMFlowsheet() done")

        # ── Compounds ─────────────────────────────────────────────────────────
        _chk(f"add_compounds {compounds}")
        try:
            sim.add_compounds(compounds)
            added = sim._compounds
            missing = [c for c in compounds if c not in added]
            if missing:
                return ExecutionResult(
                    solved=False,
                    errors=[f"DWSIM could not add compounds: {missing}. "
                            "Check compound names match DWSIM's database exactly."],
                    diagnostics={"stage": "add_compounds"})
        except Exception as e:
            return ExecutionResult(
                solved=False,
                errors=[f"Failed to add compounds {compounds}: {e}"],
                diagnostics={"stage": "add_compounds"})

        # ── Property package ──────────────────────────────────────────────────
        _chk(f"add_compounds done; set_property_package {default_pp}")
        dwsim_pp = _PP_MAP[default_pp]
        try:
            sim.set_property_package(dwsim_pp)
        except Exception as e:
            return ExecutionResult(
                solved=False,
                errors=[f"Failed to set property package '{dwsim_pp}': {e}. "
                        "Verify the package name is supported by this DWSIM version."],
                diagnostics={"stage": "set_property_package"})

        # ── Disable AutoEstimate when NRTL/UNIQUAC has no pre-supplied BIPs ────
        # AutoEstimate produces spurious pseudo-separation that prevents
        # NO_SEPARATION / PARAM_MISSING from firing, bypassing CALIBRATION routing.
        # When BIPs ARE supplied (binary_parameters populated), set_nrtl_parameters
        # already disables it — this only fires for the no-BIP case.
        early_warnings: list[str] = []
        if default_pp in ("NRTL", "UNIQUAC") and not flowsheet.get("binary_parameters"):
            try:
                sim.disable_auto_estimate(dwsim_pp)
            except Exception as e:
                early_warnings.append(
                    f"Could not disable AutoEstimate for '{dwsim_pp}': {e}")

        # ── Binary interaction parameter injection ────────────────────────────
        for bp in flowsheet.get("binary_parameters", []):
            try:
                if bp["model"] == "NRTL":
                    sim.set_nrtl_parameters(
                        bp["compound_a"], bp["compound_b"],
                        bp["A12"], bp["A21"], bp["alpha12"],
                        bp.get("B12", 0.0), bp.get("B21", 0.0),
                        bp.get("source", ""),
                    )
                elif bp["model"] == "UNIQUAC":
                    sim.set_uniquac_parameters(
                        bp["compound_a"], bp["compound_b"],
                        bp["A12"], bp["A21"],
                        bp.get("B12", 0.0), bp.get("B21", 0.0),
                        bp.get("source", ""),
                    )
            except Exception as e:
                return ExecutionResult(
                    solved=False,
                    errors=[f"Failed to inject binary parameters for "
                            f"{bp.get('compound_a')}/{bp.get('compound_b')}: {e}"],
                    diagnostics={"stage": "binary_parameter_injection"})

        # ── Auto-layout positions ─────────────────────────────────────────────
        _chk("property_package done; add_streams/units")
        positions = _auto_layout(flowsheet)

        # ── Add streams ───────────────────────────────────────────────────────
        for tag, s_cfg in streams_cfg.items():
            x, y = positions.get(tag, (0, 0))
            try:
                sim.add_stream(tag, x=x, y=y)
            except Exception as e:
                return ExecutionResult(
                    solved=False,
                    errors=[f"Failed to add stream '{tag}': {e}"],
                    diagnostics={"stage": "add_streams"})

        # ── Add units ─────────────────────────────────────────────────────────
        for tag, u_cfg in units_cfg.items():
            x, y = positions.get(tag, (0, 0))
            try:
                sim.add_unit(tag, u_cfg["type"], x=x, y=y)
            except Exception as e:
                return ExecutionResult(
                    solved=False,
                    errors=[f"Failed to add unit '{tag}' ({u_cfg['type']}): {e}"],
                    diagnostics={"stage": "add_units"})

        # ── Apply per-unit property package overrides (if specified in flowsheet) ──
        for tag, u_cfg in units_cfg.items():
            unit_pp = u_cfg.get("property_package")
            if unit_pp and unit_pp != default_pp:
                dwsim_unit_pp = _PP_MAP[unit_pp]  # pre_execution_check already verified key
                try:
                    sim.set_unit_property_package(tag, dwsim_unit_pp)
                except Exception as e:
                    return ExecutionResult(
                        solved=False,
                        errors=[f"Failed to set property package '{unit_pp}' "
                                f"on unit '{tag}': {e}"],
                        diagnostics={"stage": "set_unit_property_packages",
                                     "unit": tag})

        # ── Connect objects ───────────────────────────────────────────────────
        _chk("streams/units added; connect")
        warnings: list[str] = early_warnings
        for conn in conns:
            src, dst   = conn[0], conn[1]
            src_port   = conn[2] if len(conn) > 2 else 0
            dst_port   = conn[3] if len(conn) > 3 else 0
            try:
                sim.connect(src, dst, src_port=src_port, dst_port=dst_port)
            except Exception as e:
                return ExecutionResult(
                    solved=False,
                    errors=[f"Failed to connect '{src}' → '{dst}': {e}"],
                    diagnostics={"stage": "connect_objects",
                                 "connection": [src, dst, src_port, dst_port]})

        # ── Add recycle convergence blocks (only when recycle edges are present) ─
        for rb in flowsheet.get("recycle_blocks", []):
            _chk(f"add_recycle_block {rb['tag']}")
            try:
                sim.add_recycle_block(
                    rb["tag"],
                    inlet_stream=rb["inlet_stream"],
                    outlet_stream=rb["outlet_stream"],
                )
            except Exception as e:
                return ExecutionResult(
                    solved=False,
                    errors=[f"Failed to add recycle block '{rb['tag']}': "
                            f"{type(e).__name__}: {e}"],
                    diagnostics={"stage": "add_recycle_blocks", "block": rb})

        # ── Set feed stream conditions ────────────────────────────────────────
        for tag in feed_tags:
            s_cfg = streams_cfg[tag]
            try:
                sim.set_stream(
                    tag,
                    T=s_cfg["T"],
                    P=s_cfg["P"],
                    flow=s_cfg["flow"],
                    composition=s_cfg["composition"],
                )
            except Exception as e:
                return ExecutionResult(
                    solved=False,
                    errors=[f"Failed to set conditions on feed stream '{tag}': {e}"],
                    diagnostics={"stage": "set_stream_conditions", "stream": tag})

        # ── Set unit conditions ───────────────────────────────────────────────
        for tag, u_cfg in units_cfg.items():
            errs = _set_unit_conditions(sim, u_cfg)
            if errs:
                return ExecutionResult(
                    solved=False, errors=errs,
                    diagnostics={"stage": "set_unit_conditions", "unit": tag})

        # ── Solve ─────────────────────────────────────────────────────────────
        import os as _os
        _timeout = int(_os.environ.get("DWSIM_SOLVER_TIMEOUT", "120"))
        _chk(f"connected; calling solve(timeout={_timeout}s)")
        raw = sim.solve(timeout=_timeout)
        _chk(f"solve() returned: solved={raw.get('solved')} errors={raw.get('errors')}")
        solver_errors = raw.get("errors", [])
        solved = raw.get("solved", False)

        # Extract stream results
        stream_results: dict[str, StreamResult] = {}
        for tag in streams_cfg:
            r = raw.get(tag)
            if r:
                stream_results[tag] = StreamResult(
                    tag=tag,
                    T_K=r["T_K"],
                    P_Pa=r["P_Pa"],
                    flow_mol_s=r["flow_mol_s"],
                    composition=r["composition"],
                    vapor_fraction=r.get("vapor_fraction", 0.0),
                    is_feed=(tag in feed_tags),
                )

        if not solved:
            return ExecutionResult(
                solved=False,
                stream_results=stream_results,
                errors=["DWSIM solver did not converge."] + list(solver_errors),
                solver_errors=list(solver_errors),
                diagnostics={
                    "stage": "solve",
                    "hint": _convergence_hint(flowsheet, stream_results),
                })

        # ── Post-execution checks ─────────────────────────────────────────────
        post_errors, post_warnings = _post_execution_check(
            stream_results, feed_tags, flowsheet)

        return ExecutionResult(
            solved=True,
            stream_results=stream_results,
            errors=post_errors,
            warnings=post_warnings,
            solver_errors=list(solver_errors),
            diagnostics={
                "stage": "complete",
                "feed_tags": list(feed_tags),
                "outlet_tags": [t for t in streams_cfg if t not in feed_tags],
            })


# ── Unit condition setters ────────────────────────────────────────────────────

def _set_unit_conditions(sim, u_cfg: dict) -> list[str]:
    """Set operating conditions on a unit op. Returns errors (empty = ok)."""
    tag   = u_cfg["tag"]
    utype = u_cfg["type"]
    try:
        if utype == "Heater":
            sim.set_heater(tag, T_out=u_cfg["T_out"],
                           dP=u_cfg.get("dP", 0.0))
        elif utype == "Cooler":
            sim.set_cooler(tag, T_out=u_cfg["T_out"],
                           dP=u_cfg.get("dP", 0.0))
        elif utype == "Vessel":
            sim.set_vessel(tag, dP=u_cfg.get("dP", 0.0))
        elif utype == "Mixer":
            _set_mixer(sim, tag, dP=u_cfg.get("dP", 0.0))
        elif utype == "Splitter":
            sim.set_splitter(tag, u_cfg["split_fractions"], dP=u_cfg.get("dP", 0.0))
        elif utype == "Pump":
            sim.set_pump(tag, u_cfg["P_out"], efficiency=u_cfg.get("efficiency", 0.75))
        elif utype == "Compressor":
            sim.set_compressor(tag, u_cfg["P_out"],
                               efficiency=u_cfg.get("efficiency", 0.75))
        elif utype == "Expander":
            sim.set_expander(tag, u_cfg["P_out"],
                             efficiency=u_cfg.get("efficiency", 0.75))
        elif utype == "ConversionReactor":
            sim.set_conversion_reactor(
                tag,
                temperature_K = u_cfg["temperature_K"],
                pressure_Pa   = u_cfg["pressure_Pa"],
                conversion    = u_cfg["conversion"],
                reaction      = u_cfg.get("reaction", ""),
            )
        else:
            return [f"Unit '{tag}' type '{utype}' condition-setting "
                    "not yet implemented in executor."]
    except Exception as e:
        return [f"Failed to set conditions on unit '{tag}' ({utype}): "
                f"{type(e).__name__}: {e}"]
    return []


def _set_mixer(sim, tag: str, dP: float = 0.0) -> None:
    """Set mixer pressure drop via PROP_MIX_0."""
    obj = sim._sim.GetFlowsheetSimulationObject(tag)
    obj.SetPropertyValue("PROP_MIX_0", float(dP))


# ── Post-execution physical checks ────────────────────────────────────────────

def _post_execution_check(
        results: dict[str, StreamResult],
        feed_tags: set[str],
        flowsheet: dict,
) -> tuple[list[str], list[str]]:
    errors:   list[str] = []
    warnings: list[str] = []

    # Terminal outlets: streams with no outgoing connections (not feeds, not intermediate)
    conn_srcs   = {c[0] for c in flowsheet.get("connections", []) if len(c) >= 2}
    outlet_tags = [t for t in results if t not in feed_tags and t not in conn_srcs]

    for tag, s in results.items():
        label = f"Stream '{tag}'"

        # Temperature bounds
        if s.T_K < _T_MIN_K:
            errors.append(
                f"{label}: T={s.T_K:.1f} K is below physical minimum "
                f"({_T_MIN_K} K). Likely a unit conversion error or solver failure.")
        if s.T_K > _T_MAX_K:
            errors.append(
                f"{label}: T={s.T_K:.1f} K is above physical maximum "
                f"({_T_MAX_K} K). Check heater outlet temperature units (must be K).")

        # Pressure bounds
        if s.P_Pa < _P_MIN_PA:
            errors.append(
                f"{label}: P={s.P_Pa:.1f} Pa is near vacuum. "
                "Check pressure units (must be Pa, not bar or kPa).")
        if s.P_Pa > _P_MAX_PA:
            errors.append(
                f"{label}: P={s.P_Pa:.2e} Pa exceeds 1000 bar. "
                "Check pressure units.")

        # Composition sum
        comp_sum = s.composition_sum()
        if abs(comp_sum - 1.0) > _COMP_TOL:
            errors.append(
                f"{label}: composition sums to {comp_sum:.4f}. "
                "Possible solver or phase-split error.")

        # Zero-flow outlet (not necessarily an error for all cases)
        if tag in outlet_tags and s.flow_mol_s == 0.0:
            warnings.append(
                f"Outlet stream '{tag}' has zero molar flow. "
                "Verify phase split conditions — this may be expected "
                "(e.g. no vapour at sub-bubble-point conditions).")

        # NaN or Inf check
        for field_name, val in [("T_K", s.T_K), ("P_Pa", s.P_Pa),
                                 ("flow_mol_s", s.flow_mol_s)]:
            if math.isnan(val) or math.isinf(val):
                errors.append(
                    f"{label}: {field_name} = {val}. "
                    "Solver returned non-finite value — convergence failure.")

    # Mass balance: feeds vs terminal outlets only (excludes intermediate streams).
    # Skip for recycle-containing flowsheets — the recycle init stream appears as
    # a feed (no src in connections), so feed_total would include recycle flow and
    # the check would produce a false positive.
    terminal_tags = set(outlet_tags)
    if not flowsheet.get("recycle_blocks"):
        mb_errors = _mass_balance_check(results, feed_tags, terminal_tags)
        errors.extend(mb_errors)

    return errors, warnings


def _mass_balance_check(
        results: dict[str, StreamResult],
        feed_tags: set[str],
        terminal_tags: set[str],
) -> list[str]:
    """Check total molar flow in ≈ total molar flow out (terminal outlets only)."""
    errors = []
    feeds   = [s for t, s in results.items() if t in feed_tags]
    outlets = [s for t, s in results.items() if t in terminal_tags]
    if not feeds or not outlets:
        return errors
    total_in  = sum(s.flow_mol_s for s in feeds)
    total_out = sum(s.flow_mol_s for s in outlets)
    if total_in > 0:
        rel_err = abs(total_in - total_out) / total_in
        if rel_err > _MB_TOL:
            errors.append(
                f"Mass balance error: feed total = {total_in:.4f} mol/s, "
                f"outlet total = {total_out:.4f} mol/s "
                f"(relative error {rel_err:.1%}). "
                "Check for disconnected streams or solver non-convergence.")
    return errors


def _convergence_hint(flowsheet: dict,
                      partial: dict[str, StreamResult]) -> str:
    """Return a diagnostic hint to help the Critic Agent diagnose solver failure."""
    hints = []
    pp = flowsheet.get("property_package", "")
    if pp in ("NRTL", "UNIQUAC"):
        hints.append(
            "NRTL/UNIQUAC requires binary interaction parameters in DWSIM's "
            "database. If parameters are missing, DWSIM may fail silently. "
            "Consider falling back to Raoult's Law to test topology first.")
    feeds_with_zero_comp = [
        s["tag"] for s in flowsheet.get("streams", [])
        if not s.get("composition") and s["tag"] not in
        {c[1] for c in flowsheet.get("connections", []) if len(c) >= 2}
    ]
    if feeds_with_zero_comp:
        hints.append(
            f"Feed streams {feeds_with_zero_comp} may have missing composition.")
    nan_streams = [t for t, s in partial.items()
                   if math.isnan(s.T_K) or math.isnan(s.P_Pa)]
    if nan_streams:
        hints.append(
            f"Streams {nan_streams} returned NaN — solver failed mid-calculation.")
    return " | ".join(hints) if hints else "No specific hint available."


# ── Auto-layout ───────────────────────────────────────────────────────────────

def _auto_layout(flowsheet: dict) -> dict[str, tuple[int, int]]:
    """
    Assign x,y positions for DWSIM's graphical canvas.
    Cosmetic only for headless execution — uses a left-to-right topological layout.
    """
    streams = [s["tag"] for s in flowsheet.get("streams", [])]
    units   = [u["tag"] for u in flowsheet.get("units", [])]
    conns   = flowsheet.get("connections", [])
    all_tags = streams + units

    # Build adjacency: tag → list of downstream tags
    downstream: dict[str, list[str]] = {t: [] for t in all_tags}
    for conn in conns:
        if len(conn) >= 2:
            downstream[conn[0]].append(conn[1])

    # Topological level assignment (BFS from roots)
    has_incoming = {c[1] for c in conns if len(c) >= 2}
    roots = [t for t in all_tags if t not in has_incoming]
    level: dict[str, int] = {}
    queue = list(roots)
    for root in queue:
        level[root] = 0
    visited = set(roots)
    while queue:
        node = queue.pop(0)
        for child in downstream.get(node, []):
            level[child] = max(level.get(child, 0), level[node] + 1)
            if child not in visited:
                visited.add(child)
                queue.append(child)

    # Assign positions: x proportional to level, y spread vertically
    x_spacing, y_spacing = 200, 150
    level_counts: dict[int, int] = {}
    positions: dict[str, tuple[int, int]] = {}
    for tag in all_tags:
        lvl = level.get(tag, 0)
        idx = level_counts.get(lvl, 0)
        positions[tag] = (lvl * x_spacing + 50, idx * y_spacing + 100)
        level_counts[lvl] = idx + 1

    return positions
