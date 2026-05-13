"""
Critic Agent: two-stage diagnosis of DWSIM execution results.

Stage 1 — Deterministic checker (no LLM, no cost):
    Runs all failure taxonomy checks against ExecutionResult.
    If all pass → CriticReport(passed=True) immediately.

Stage 2 — LLM interpreter (only on failure):
    Feeds failure signals + stream results + flowsheet to the LLM.
    Produces structured diagnosis, routing decision, and suggested fixes.
    Uses failure_taxonomy.md as system context (API-cached after first call).

CriticReport.routing tells the Orchestrator which agent to call next:
    "PASS"    → accept results, pipeline complete
    "REFINER" → send flowsheet JSON to Refiner Agent
    "THERMO"  → send back to Thermodynamics Agent
    "BASIS"   → send back to Basis Agent (compound name issue)
    "HUMAN"   → infeasible, escalate to user
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass, field

from agents.executor import ExecutionResult, StreamResult
from agents.llm import chat, DEFAULT_MODEL
from agents.physics_check import physics_validate
from context import FAILURE_TAXONOMY

# ── Failure signal ────────────────────────────────────────────────────────────

@dataclass
class FailureSignal:
    code:     str
    severity: str          # "CRITICAL" | "WARNING"
    location: str          # e.g. "stream:VAP", "global"
    evidence: str          # specific numbers/values that triggered this

# ── Critic report ─────────────────────────────────────────────────────────────

@dataclass
class CriticReport:
    passed:          bool
    routing:         str             # PASS | REFINER | THERMO | BASIS | HUMAN
    failure_codes:   list[str]       = field(default_factory=list)
    severity:        str             = "PASS"
    diagnosis:       str             = ""
    suggested_fixes: list[str]       = field(default_factory=list)
    confidence:      float           = 1.0
    iteration:       int             = 0
    signals:         list[FailureSignal] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Passed: {self.passed}",
            f"Routing: {self.routing}",
            f"Severity: {self.severity}",
        ]
        if self.failure_codes:
            lines.append(f"Codes: {self.failure_codes}")
        if self.diagnosis:
            lines.append(f"Diagnosis: {self.diagnosis}")
        if self.suggested_fixes:
            lines.append("Suggested fixes:")
            lines.extend(f"  - {f}" for f in self.suggested_fixes)
        lines.append(f"Confidence: {self.confidence:.0%}")
        return "\n".join(lines)


# ── Deterministic routing fallback (no LLM needed for clear cases) ────────────

_CODE_ROUTING: dict[str, str] = {
    "SOLVER_FAIL":        "REFINER",
    "NUMERIC_FAIL":       "THERMO",
    "MASS_BALANCE":       "REFINER",
    "UNPHYSICAL_T":       "REFINER",
    "UNPHYSICAL_P":       "REFINER",
    "ENERGY_UNPHYSICAL":  "REFINER",
    "ZERO_OUTLET":        "THERMO",
    "NO_SEPARATION":      "THERMO",
    "COMP_SUM":           "REFINER",
    "PARAM_MISSING":      "CALIBRATION",
    "SPURIOUS_SEPARATION": "THERMO",
    "INFEASIBLE":         "HUMAN",
}


# Normal boiling points [K] for WRONG_PHASE_DIR detection.
# Compounds not listed are skipped in the phase-direction check.
_NBP_K: dict[str, float] = {
    "methane": 111.7, "ethane": 184.6, "propane": 231.1,
    "n-butane": 272.7, "isobutane": 261.5,
    "n-pentane": 309.2, "isopentane": 301.0,
    "n-hexane": 341.9, "n-heptane": 371.6,
    "cyclohexane": 353.9, "benzene": 353.2,
    "toluene": 383.8, "ethylbenzene": 409.3,
    "methanol": 337.8, "ethanol": 351.4,
    "1-propanol": 370.3, "2-propanol": 355.4,
    "n-butanol": 390.7, "1-butanol": 390.7,
    "acetone": 329.2, "acetic acid": 391.0,
    "ethyl acetate": 350.3, "diethyl ether": 307.6,
    "n-octane": 398.8, "n-nonane": 423.9,
    "2-butanol": 372.7, "isobutanol": 381.0, "tert-butanol": 355.6,
    "1-pentanol": 411.2,
    "chlorobenzene": 404.9, "carbon disulfide": 319.4,
    "n-methylpyrrolidone": 475.0,
    "water": 373.15, "hydrogen": 20.3,
    "nitrogen": 77.4, "oxygen": 90.2,
    "carbon dioxide": 194.7, "hydrogen sulfide": 212.8,
    "ammonia": 239.8, "chloroform": 334.4,
    "dichloromethane": 313.0, "acetonitrile": 354.8,
    "tetrahydrofuran": 339.1, "dimethyl sulfoxide": 462.2,
}


def _estimate_bubble_point(compounds: list[str], composition: dict[str, float],
                            pressure_pa: float = 101325.0) -> float | None:
    """
    Raoult's Law bubble-point estimate using _NBP_K (lowercase keys).
    Returns None if any compound is absent from the table or pressure is extreme.
    Includes a first-order Clausius-Clapeyron pressure correction.
    """
    import math
    if not (20_000 < pressure_pa < 600_000):
        return None
    total = sum(composition.values())
    if total <= 0:
        return None
    for c in compounds:
        if c.lower() not in _NBP_K:
            return None
    t_bub = sum(
        (composition.get(c, 0.0) / total) * _NBP_K[c.lower()]
        for c in compounds
    )
    if abs(pressure_pa - 101_325.0) > 5_000:
        lnP = math.log(pressure_pa / 101_325.0)
        dHvap = 88.0 * t_bub          # Trouton's rule
        t_bub = t_bub / (1.0 - 8.314 * t_bub * lnP / dHvap)
    return round(t_bub, 1)


# ── LLM system prompt ─────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a chemical process simulation diagnostic expert. "
    "You receive DWSIM simulation results, a list of detected failure signals, "
    "and the flowsheet definition. Your job is to diagnose the root cause, "
    "decide where to route the failure for fixing, and suggest specific fixes.\n\n"
    "Output ONLY valid JSON — no markdown, no explanation:\n"
    "{\n"
    '  "diagnosis": "<one clear sentence explaining the root cause>",\n'
    '  "routing": "<REFINER|THERMO|BASIS|HUMAN|PASS>",\n'
    '  "suggested_fixes": ["<specific fix 1>", "<specific fix 2>"],\n'
    '  "confidence": <0.0-1.0>,\n'
    '  "severity": "<CRITICAL|WARNING|PASS>"\n'
    "}\n\n"
    "Routing rules:\n"
    "  REFINER → topology, conditions, units, port assignments\n"
    "  THERMO  → wrong property package, missing binary parameters\n"
    "  BASIS   → wrong compound name (DWSIM cannot find compound)\n"
    "  HUMAN   → fundamentally infeasible, cannot be fixed automatically\n"
    "  PASS    → results are physically credible despite warnings\n\n"
    "---\n"
    + FAILURE_TAXONOMY
)


# ── Critic Agent ──────────────────────────────────────────────────────────────

class CriticAgent:
    """
    Two-stage critic: deterministic checks first, LLM only on failure.

    Args:
        model: any model supported by agents/llm.py — enables benchmarking.
    """

    def __init__(self, model: str = DEFAULT_MODEL, infeasible_threshold: int = 3):
        self._model                = model
        self._infeasible_threshold = infeasible_threshold

    def critique(
        self,
        result: ExecutionResult,
        flowsheet: dict,
        iteration: int = 0,
    ) -> CriticReport:
        """
        Evaluate an ExecutionResult and return a CriticReport.
        Stage 1 runs always. Stage 2 (LLM) runs only if Stage 1 finds issues.
        """
        # Stage 1 — deterministic
        signals = _run_stage1(result, flowsheet, iteration)

        if not signals:
            return CriticReport(
                passed=True,
                routing="PASS",
                severity="PASS",
                diagnosis="All checks passed. Results are physically credible.",
                confidence=1.0,
                iteration=iteration,
            )

        # WARNING-only signals (e.g. SPURIOUS_SEPARATION) are informational —
        # they do not indicate a recoverable failure requiring agent action.
        if all(s.severity == "WARNING" for s in signals):
            return CriticReport(
                passed=True,
                routing="PASS",
                severity="PASS",
                diagnosis=(
                    "Minor warnings detected but no critical failures. "
                    "Results are physically acceptable. "
                    f"Warnings: {[s.code for s in signals]}"
                ),
                confidence=0.9,
                iteration=iteration,
                signals=signals,
            )

        # Check for infeasibility threshold
        if iteration >= self._infeasible_threshold:
            critical = [s for s in signals if s.severity == "CRITICAL"]
            if critical:
                return CriticReport(
                    passed=False,
                    routing="HUMAN",
                    failure_codes=["INFEASIBLE"],
                    severity="CRITICAL",
                    diagnosis=(
                        f"Process has not converged after {iteration} iterations. "
                        f"Persistent failures: {[s.code for s in critical]}. "
                        "Manual intervention required."),
                    suggested_fixes=[
                        "Review process feasibility with a domain expert.",
                        "Check whether the specified separation is achievable "
                        "at the given conditions.",
                        "Consider simplifying the flowsheet topology.",
                    ],
                    confidence=0.9,
                    iteration=iteration,
                    signals=signals,
                )

        # Stage 2 — LLM interpretation
        return self._run_stage2(signals, result, flowsheet, iteration)

    def _run_stage2(
        self,
        signals: list[FailureSignal],
        result: ExecutionResult,
        flowsheet: dict,
        iteration: int,
    ) -> CriticReport:
        """LLM interprets the failure signals and produces a structured report."""

        signal_text = "\n".join(
            f"  [{s.severity}] {s.code} @ {s.location}: {s.evidence}"
            for s in signals
        )
        stream_text = _format_streams(result)
        prompt = (
            f"Iteration: {iteration}\n\n"
            f"Failure signals detected:\n{signal_text}\n\n"
            f"Stream results:\n{stream_text}\n\n"
            f"Flowsheet:\n{json.dumps(flowsheet, indent=2)}\n\n"
            "Diagnose the root cause and output JSON as specified."
        )

        try:
            raw = chat(prompt, system=_SYSTEM, model=self._model)
            parsed = _parse_llm_response(raw)
        except Exception as e:
            # LLM failed — fall back to deterministic routing
            parsed = _deterministic_fallback(signals)

        # Merge Stage 1 codes with Stage 2 routing
        codes = list({s.code for s in signals})
        severity = ("CRITICAL" if any(s.severity == "CRITICAL" for s in signals)
                    else "WARNING")

        return CriticReport(
            passed=False,
            routing=parsed.get("routing", _fallback_routing(signals)),
            failure_codes=codes,
            severity=parsed.get("severity", severity),
            diagnosis=parsed.get("diagnosis", ""),
            suggested_fixes=parsed.get("suggested_fixes", []),
            confidence=float(parsed.get("confidence", 0.7)),
            iteration=iteration,
            signals=signals,
        )


# ── Stage 1: deterministic checks ────────────────────────────────────────────

def _run_stage1(
    result: ExecutionResult,
    flowsheet: dict,
    iteration: int,
) -> list[FailureSignal]:
    signals: list[FailureSignal] = []

    # SOLVER_FAIL
    if not result.solved:
        signals.append(FailureSignal(
            code="SOLVER_FAIL", severity="CRITICAL", location="global",
            evidence=f"sim.Solved=False. Solver errors: {result.solver_errors}"))

    # NUMERIC_FAIL — NaN / Inf in any stream
    feed_tags_set = {t for t, s in result.stream_results.items() if s.is_feed}
    for tag, s in result.stream_results.items():
        for fname, val in [("T_K", s.T_K), ("P_Pa", s.P_Pa),
                           ("flow_mol_s", s.flow_mol_s)]:
            if math.isnan(val) or math.isinf(val):
                signals.append(FailureSignal(
                    code="NUMERIC_FAIL", severity="CRITICAL",
                    location=f"stream:{tag}",
                    evidence=f"{fname}={val}"))
    for tag, s in result.stream_results.items():
        if not s.composition and tag not in feed_tags_set:
            signals.append(FailureSignal(
                code="NUMERIC_FAIL", severity="CRITICAL",
                location=f"stream:{tag}",
                evidence="composition dict is empty — solver did not compute phase split"))

    # UNPHYSICAL_T / UNPHYSICAL_P
    for tag, s in result.stream_results.items():
        if not (math.isnan(s.T_K) or math.isinf(s.T_K)):
            if s.T_K < 100.0:
                signals.append(FailureSignal(
                    code="UNPHYSICAL_T", severity="CRITICAL",
                    location=f"stream:{tag}",
                    evidence=f"T={s.T_K:.2f} K (below 100 K)"))
            if s.T_K > 2000.0:
                signals.append(FailureSignal(
                    code="UNPHYSICAL_T", severity="CRITICAL",
                    location=f"stream:{tag}",
                    evidence=f"T={s.T_K:.2f} K (above 2000 K — check K vs °C)"))
        if not (math.isnan(s.P_Pa) or math.isinf(s.P_Pa)):
            if s.P_Pa < 100.0:
                signals.append(FailureSignal(
                    code="UNPHYSICAL_P", severity="CRITICAL",
                    location=f"stream:{tag}",
                    evidence=f"P={s.P_Pa:.2f} Pa (near vacuum — check Pa vs bar)"))
            if s.P_Pa > 1e8:
                signals.append(FailureSignal(
                    code="UNPHYSICAL_P", severity="CRITICAL",
                    location=f"stream:{tag}",
                    evidence=f"P={s.P_Pa:.2e} Pa (above 1000 bar)"))

    # MASS_BALANCE — computed directly from stream flows (not string-matched from errors)
    conns_list = flowsheet.get("connections", [])
    conn_srcs_set = {c[0] for c in conns_list if len(c) >= 2}
    conn_dsts_set = {c[1] for c in conns_list if len(c) >= 2}
    _feed_tags  = {t for t, s in result.stream_results.items() if s.is_feed}
    _term_tags  = {t for t in result.stream_results
                   if t not in conn_srcs_set and not result.stream_results[t].is_feed}
    total_in  = sum(result.stream_results[t].flow_mol_s for t in _feed_tags
                    if t in result.stream_results)
    total_out = sum(result.stream_results[t].flow_mol_s for t in _term_tags
                    if t in result.stream_results)
    if total_in > 0 and abs(total_in - total_out) / total_in > 0.01:
        signals.append(FailureSignal(
            code="MASS_BALANCE", severity="CRITICAL",
            location="global",
            evidence=(f"feed={total_in:.4f} mol/s, "
                      f"outlets={total_out:.4f} mol/s "
                      f"(err={abs(total_in-total_out)/total_in:.1%})")))

    # ENERGY_UNPHYSICAL — heater outlet cooler than feed, or cooler outlet hotter
    conn_srcs = {c[0] for c in flowsheet.get("connections", []) if len(c) >= 2}
    feed_tags = {s["tag"] for s in flowsheet.get("streams", [])
                 if s["tag"] not in
                 {c[1] for c in flowsheet.get("connections", []) if len(c) >= 2}}
    for u in flowsheet.get("units", []):
        utype = u.get("type", "")
        t_out = u.get("T_out")
        if t_out and utype in ("Heater", "Cooler"):
            # Find inlet stream (stream that connects TO this unit)
            inlet_tag = next(
                (c[0] for c in flowsheet.get("connections", [])
                 if len(c) >= 2 and c[1] == u["tag"]), None)
            if inlet_tag and inlet_tag in result.stream_results:
                inlet_T = result.stream_results[inlet_tag].T_K
                if utype == "Heater" and t_out < inlet_T - 1.0:
                    signals.append(FailureSignal(
                        code="ENERGY_UNPHYSICAL", severity="CRITICAL",
                        location=f"unit:{u['tag']}",
                        evidence=(f"Heater T_out={t_out:.1f} K < "
                                  f"inlet T={inlet_T:.1f} K")))
                if utype == "Cooler" and t_out > inlet_T + 1.0:
                    signals.append(FailureSignal(
                        code="ENERGY_UNPHYSICAL", severity="CRITICAL",
                        location=f"unit:{u['tag']}",
                        evidence=(f"Cooler T_out={t_out:.1f} K > "
                                  f"inlet T={inlet_T:.1f} K")))

    # ZERO_OUTLET — terminal stream with zero flow
    terminal_tags = {t for t in result.stream_results
                     if t not in conn_srcs and t not in feed_tags}
    for tag in terminal_tags:
        s = result.stream_results.get(tag)
        if s and s.flow_mol_s == 0.0:
            signals.append(FailureSignal(
                code="ZERO_OUTLET", severity="WARNING",
                location=f"stream:{tag}",
                evidence="Terminal outlet flow = 0.0 mol/s"))

    # NO_SEPARATION — outlet compositions match feed (within 1%)
    feed_comps: dict[str, float] = {}
    _feed_streams = [
        result.stream_results[t]
        for t in feed_tags
        if t in result.stream_results and result.stream_results[t].flow_mol_s > 0
    ]
    if _feed_streams:
        _total_feed_flow = sum(s.flow_mol_s for s in _feed_streams)
        if _total_feed_flow > 0:
            _all_compounds = set().union(*(s.composition.keys() for s in _feed_streams))
            feed_comps = {
                c: sum(s.flow_mol_s * s.composition.get(c, 0.0) for s in _feed_streams)
                   / _total_feed_flow
                for c in _all_compounds
            }
    if feed_comps:
        for tag in terminal_tags:
            s = result.stream_results.get(tag)
            if not s or s.flow_mol_s == 0.0:
                continue
            if not s.composition:
                continue
            max_diff = max(
                abs(s.composition.get(c, 0) - feed_comps.get(c, 0))
                for c in feed_comps
            )
            if max_diff < 0.01 and len(terminal_tags) > 1:
                signals.append(FailureSignal(
                    code="NO_SEPARATION", severity="CRITICAL",
                    location=f"stream:{tag}",
                    evidence=(f"Max composition deviation from feed = "
                              f"{max_diff:.4f} — likely missing binary parameters")))

    # PARAM_MISSING — NRTL/UNIQUAC used + NO_SEPARATION + Solved=True
    pp = flowsheet.get("property_package", "")
    if pp in ("NRTL", "UNIQUAC") and result.solved:
        no_sep = [s for s in signals if s.code == "NO_SEPARATION"]
        if no_sep:
            signals.append(FailureSignal(
                code="PARAM_MISSING", severity="CRITICAL",
                location="global",
                evidence=(f"{pp} used but outlet ≈ feed — "
                          "binary interaction parameters likely absent from "
                          "DWSIM database")))

    # AZEOTROPE_IDEAL_MODEL / LLE_INCAPABLE_PACKAGE — detect Raoult's Law (or other
    # incapable package) applied to a non-ideal system. Surfaced as PARAM_MISSING so
    # the existing _CODE_ROUTING sends it to ThermoAgent without a new routing key.
    for issue in physics_validate(flowsheet):
        if issue.code in ("AZEOTROPE_IDEAL_MODEL", "LLE_INCAPABLE_PACKAGE"):
            # Known non-ideal pair with wrong package → CALIBRATION (may inject BIPs)
            signals.append(FailureSignal(
                code="PARAM_MISSING", severity="CRITICAL",
                location="global",
                evidence=issue.message))
        elif issue.code == "POLAR_IDEAL_MODEL":
            # Wrong package class (Raoult's Law on polar system) → THERMO directly.
            # Routing through CALIBRATION wastes an iteration because CalibrationAgent
            # immediately rejects non-NRTL/UNIQUAC flowsheets.
            signals.append(FailureSignal(
                code="NO_SEPARATION", severity="CRITICAL",
                location="global",
                evidence=issue.message))

    # COMP_SUM — mole fractions don't sum to ~1
    for tag, s in result.stream_results.items():
        total = sum(s.composition.values())
        if s.composition and abs(total - 1.0) > 0.02:
            signals.append(FailureSignal(
                code="COMP_SUM", severity="WARNING",
                location=f"stream:{tag}",
                evidence=f"Composition sum = {total:.4f}"))

    # WRONG_PHASE_DIR — vapour richer in heavy component than liquid
    signals.extend(_check_phase_direction(result, flowsheet))

    # NEAR_AZEOTROPE — relative volatility ≈ 1 but large composition spread detected
    signals.extend(_check_near_unity_alpha(result, flowsheet))

    return signals


def _check_phase_direction(
        result: ExecutionResult,
        flowsheet: dict,
) -> list[FailureSignal]:
    """
    Detect phase direction inversion after flash vessels.
    Compares most-volatile vs least-volatile compound fractions in vapour/liquid.
    Uses NBP ranking — only fires when 2+ compounds have known boiling points.
    """
    signals: list[FailureSignal] = []
    conns = flowsheet.get("connections", [])

    for unit in flowsheet.get("units", []):
        if unit.get("type") != "Vessel":
            continue
        utag = unit["tag"]

        # Outlet streams: connections FROM this vessel
        vap_tag = next((c[1] for c in conns
                        if len(c) >= 3 and c[0] == utag and c[2] == 0), None)
        liq_tag = next((c[1] for c in conns
                        if len(c) >= 3 and c[0] == utag and c[2] == 1), None)
        if not vap_tag or not liq_tag:
            continue

        vap = result.stream_results.get(vap_tag)
        liq = result.stream_results.get(liq_tag)
        if not vap or not liq:
            continue
        if vap.flow_mol_s == 0.0 or liq.flow_mol_s == 0.0:
            continue  # zero-flow already flagged as ZERO_OUTLET

        # Build NBP-ranked list of compounds present in both outlets
        ranked: list[tuple[float, str]] = []
        for name in vap.composition:
            nbp = _NBP_K.get(name.lower())
            if nbp is not None and name in liq.composition:
                ranked.append((nbp, name))
        if len(ranked) < 2:
            continue

        ranked.sort()
        most_vol  = ranked[0][1]   # lowest NBP → most volatile
        least_vol = ranked[-1][1]  # highest NBP → least volatile

        vap_mv = vap.composition.get(most_vol, 0.0)
        liq_mv = liq.composition.get(most_vol, 0.0)
        vap_lv = vap.composition.get(least_vol, 0.0)
        liq_lv = liq.composition.get(least_vol, 0.0)

        # Inversion: most volatile heavier in liquid AND least volatile heavier in vapour
        if liq_mv > vap_mv + 0.05 and vap_lv > liq_lv + 0.05:
            signals.append(FailureSignal(
                code="WRONG_PHASE_DIR", severity="WARNING",
                location=f"unit:{utag}",
                evidence=(
                    f"{most_vol} (NBP={ranked[0][0]:.0f}K) "
                    f"x_vap={vap_mv:.3f} < x_liq={liq_mv:.3f}; "
                    f"{least_vol} (NBP={ranked[-1][0]:.0f}K) "
                    f"x_vap={vap_lv:.3f} > x_liq={liq_lv:.3f}. "
                    "Phase outlets may be swapped (check src_port 0/1) "
                    "or property package is producing inverted VLE."
                )))

    return signals


def _check_near_unity_alpha(
        result: ExecutionResult,
        flowsheet: dict,
) -> list[FailureSignal]:
    """
    Flag when DWSIM reports large composition separation but the two most
    abundant compounds have nearly identical NBPs (relative volatility ≈ 1).

    Near-unity α means a flash should produce outlet ≈ feed. If substantial
    composition enrichment is seen instead, the property package is likely
    producing spurious separation (AutoEstimate artefact or wrong model).

    Only fires for binary/pseudo-binary systems where both NBPs are known.
    Threshold: |NBP_A - NBP_B| < 5 K but |x_vap - x_liq| > 0.15 for either.
    """
    signals: list[FailureSignal] = []
    conns = flowsheet.get("connections", [])

    for unit in flowsheet.get("units", []):
        if unit.get("type") != "Vessel":
            continue
        utag = unit["tag"]

        vap_tag = next((c[1] for c in conns
                        if len(c) >= 3 and c[0] == utag and c[2] == 0), None)
        liq_tag = next((c[1] for c in conns
                        if len(c) >= 3 and c[0] == utag and c[2] == 1), None)
        if not vap_tag or not liq_tag:
            continue

        vap = result.stream_results.get(vap_tag)
        liq = result.stream_results.get(liq_tag)
        if not vap or not liq:
            continue
        if vap.flow_mol_s == 0.0 or liq.flow_mol_s == 0.0:
            continue

        # Only proceed if we have exactly 2 dominant compounds with NBP data
        ranked: list[tuple[float, str]] = []
        for name in vap.composition:
            nbp = _NBP_K.get(name.lower())
            if nbp is not None and name in liq.composition:
                ranked.append((nbp, name))
        if len(ranked) < 2:
            continue

        ranked.sort()
        nbp_lo, comp_lo = ranked[0]
        nbp_hi, comp_hi = ranked[-1]
        delta_nbp = abs(nbp_hi - nbp_lo)

        if delta_nbp >= 5.0:
            continue  # sufficient boiling-point spread; separation is expected

        # Small ΔT_bp → relative volatility ≈ 1 → separation should be minimal
        sep_lo = abs(vap.composition.get(comp_lo, 0.0) - liq.composition.get(comp_lo, 0.0))
        sep_hi = abs(vap.composition.get(comp_hi, 0.0) - liq.composition.get(comp_hi, 0.0))
        if max(sep_lo, sep_hi) > 0.15:
            signals.append(FailureSignal(
                code="SPURIOUS_SEPARATION", severity="WARNING",
                location=f"unit:{utag}",
                evidence=(
                    f"{comp_lo} (NBP={nbp_lo:.0f}K) and {comp_hi} "
                    f"(NBP={nbp_hi:.0f}K) differ by only {delta_nbp:.1f}K "
                    f"(α ≈ 1), yet outlet compositions differ by "
                    f"{max(sep_lo, sep_hi):.2f} mole fraction. "
                    "Property package may be generating spurious separation "
                    "(AutoEstimate artefact or incorrect model)."
                )))

    return signals


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_streams(result: ExecutionResult) -> str:
    lines = []
    for tag, s in result.stream_results.items():
        marker = " [FEED]" if s.is_feed else ""
        lines.append(
            f"  {tag}{marker}: T={s.T_K:.2f}K  P={s.P_Pa:.1f}Pa  "
            f"flow={s.flow_mol_s:.4f}mol/s  comp={s.composition}")
    return "\n".join(lines) if lines else "  No stream results available."


def _parse_llm_response(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(l for l in lines if not l.strip().startswith("```"))
    return json.loads(text.strip())


def _deterministic_fallback(signals: list[FailureSignal]) -> dict:
    routing = _fallback_routing(signals)
    codes = [s.code for s in signals]
    return {
        "routing": routing,
        "diagnosis": f"Deterministic fallback. Failures: {codes}",
        "suggested_fixes": [
            f"Address {s.code} at {s.location}: {s.evidence}"
            for s in signals if s.severity == "CRITICAL"
        ],
        "confidence": 0.6,
        "severity": ("CRITICAL" if any(s.severity == "CRITICAL" for s in signals)
                     else "WARNING"),
    }


def _fallback_routing(signals: list[FailureSignal]) -> str:
    """Deterministic routing without LLM — priority order matters."""
    codes = {s.code for s in signals}
    for code in ["INFEASIBLE", "NUMERIC_FAIL", "PARAM_MISSING",
                 "NO_SEPARATION", "WRONG_PHASE_DIR", "ZERO_OUTLET",
                 "SOLVER_FAIL", "MASS_BALANCE", "ENERGY_UNPHYSICAL",
                 "UNPHYSICAL_T", "UNPHYSICAL_P", "COMP_SUM"]:
        if code in codes:
            return _CODE_ROUTING.get(code, "REFINER")
    return "REFINER"
