"""
Refiner Agent: applies targeted fixes to a failing flowsheet.

Two-stage design (mirrors the Critic):

Stage 1 — Deterministic fixes (no LLM, no cost):
    Applied for failure codes with unambiguous remediation:
    UNPHYSICAL_T   → unit conversion (°C → K) on streams and units
    UNPHYSICAL_P   → unit conversion (bar/kPa → Pa) on streams and units
    PARAM_MISSING  → fall back to Raoult's Law for topology verification
    NO_SEPARATION  → same as PARAM_MISSING when NRTL/UNIQUAC used
    ENERGY_UNPHYS  → swap T_out to be physically consistent
    COMP_SUM       → renormalise feed compositions to sum to 1.0

Stage 2 — LLM refinement (only when Stage 1 leaves unresolved signals):
    Sends flowsheet + CriticReport to LLM.
    LLM outputs updated flowsheet JSON + structured list of changes made.
    Output validated with schema.validate() + physics_validate().

RefinementResult.changes provides a per-iteration audit trail used in
the Orchestrator's convergence analysis and paper results.
"""
from __future__ import annotations
import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any

from agents import schema
from agents.critic import CriticReport, _estimate_bubble_point
from agents.llm import chat, DEFAULT_MODEL, retry_temperature
from agents.physics_check import physics_validate, has_errors

# ── Change record ─────────────────────────────────────────────────────────────

@dataclass
class RefinementChange:
    target:       str    # "stream:FEED", "unit:HT-01", "global"
    field:        str    # "T", "P", "property_package", "composition", ...
    old_value:    Any
    new_value:    Any
    reason:       str
    failure_code: str    # which CriticReport failure code triggered this


# ── Refinement result ─────────────────────────────────────────────────────────

@dataclass
class RefinementResult:
    success:           bool
    updated_flowsheet: dict
    changes:           list[RefinementChange] = field(default_factory=list)
    stage:             str = "NONE"   # "DETERMINISTIC" | "LLM" | "FAILED"
    reasoning:         str = ""

    def summary(self) -> str:
        lines = [
            f"Success: {self.success}",
            f"Stage: {self.stage}",
            f"Changes ({len(self.changes)}):",
        ]
        for c in self.changes:
            lines.append(
                f"  [{c.failure_code}] {c.target}.{c.field}: "
                f"{c.old_value!r} → {c.new_value!r}  ({c.reason})")
        if self.reasoning:
            lines.append(f"Reasoning: {self.reasoning}")
        return "\n".join(lines)


# ── LLM system prompt ─────────────────────────────────────────────────────────

_SYSTEM = """\
You are a chemical process simulation expert. You receive a DWSIM flowsheet state
and a diagnostic report. Patch ONLY the broken fields.

Output ONLY this JSON object (no markdown, no explanation):
{
  "patches": [
    {"target": "unit",   "tag": "<TAG>", "field": "<field>", "new_value": <value>},
    {"target": "stream", "tag": "<TAG>", "field": "<field>", "new_value": <value>},
    {"target": "global",                 "field": "<field>", "new_value": <value>}
  ],
  "reasoning": "<one sentence: what was changed and why>"
}

MUST: patch ONLY the broken values — do NOT include unchanged fields.
MUST: "target" is exactly one of: "unit", "stream", "global".
MUST: "tag" must match a tag shown in the flowsheet state ("unit"/"stream" targets).
MUST: temperatures in Kelvin, pressures in Pascals, flows in mol/s.
MUST: if "field" is "composition", "new_value" must be a dict summing to exactly 1.0.

Fix guide by failure code (patch only the minimum fields needed):
  SOLVER_FAIL     → reduce temperature step: lower unit T_out closer to feed T
  UNPHYSICAL_T    → add 273.15 to any T < 100K (was entered as °C)
  UNPHYSICAL_P    → multiply P by 1e5 if looks like bar, by 1000 if looks like kPa
  ENERGY_UNPHYS   → Heater: raise T_out above inlet T; Cooler: lower T_out below inlet T
  ZERO_OUTLET     → raise Heater/Cooler T_out above mixture bubble point + 15K;
                    if no heater/cooler, raise feed stream T above bubble point.
                    Bubble points at 1 atm:
                    Methane 112K | Ethane 185K | Propane 231K | n-Butane 273K |
                    Diethyl Ether 308K | n-Pentane 309K | DCM 313K |
                    Acetone 329K | Chloroform 334K | MeOH 338K | THF 339K |
                    n-Hexane 342K | EtOAc 350K | EtOH 352K | Benzene 353K |
                    Cyclohexane 354K | IPA 356K | n-PrOH 370K | Water 373K |
                    Toluene 384K | n-Butanol 391K | n-Octane 399K.
  NO_SEPARATION   → global property_package → "Raoult's Law"
  COMP_SUM        → stream composition → renormalised dict summing to 1.0
  MASS_BALANCE    → check connections list; patch stream tags to match unit ports\
"""

_FEW_SHOT_PATCHES = """\
━━━ EXAMPLE 1 — ZERO_OUTLET: HT-01 T_out below bubble point ━━━━━━━━━━━━━━━━━━
Failure: [CRITICAL] ZERO_OUTLET @ stream:VAP — Terminal outlet flow = 0.0 mol/s
Diagnosis: HT-01 T_out=310K is below methanol/water bubble point (~355K); all-liquid feed.
Flowsheet state:
  Global: property_package='NRTL'  compounds=['Methanol','Water']
  Stream FEED: T=298.15K  P=101325Pa  flow=1.0mol/s  {Methanol:0.5, Water:0.5}
  Unit HT-01 (Heater): T_out=310.0  dP=0.0
Output:
{"patches": [{"target": "unit", "tag": "HT-01", "field": "T_out", "new_value": 368.0}],
 "reasoning": "HT-01 T_out raised 310→368K — above methanol/water bubble point ~355K to produce two-phase feed at V-01."}

━━━ EXAMPLE 2 — UNPHYSICAL_T: feed T entered in Celsius ━━━━━━━━━━━━━━━━━━━━━━
Failure: [CRITICAL] UNPHYSICAL_T @ stream:FEED — T=25.0K (below 100K)
Diagnosis: FEED temperature 25K was entered in °C; must be 298.15K.
Flowsheet state:
  Stream FEED: T=25.0K  P=101325Pa  flow=1.0mol/s  {Benzene:0.5, Toluene:0.5}
Output:
{"patches": [{"target": "stream", "tag": "FEED", "field": "T", "new_value": 298.15}],
 "reasoning": "FEED T corrected 25→298.15K (value was in Celsius, +273.15)."}

━━━ EXAMPLE 3 — SOLVER_FAIL: T_out step too large ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Failure: [CRITICAL] SOLVER_FAIL — Solver failed to converge
Diagnosis: HT-01 T_out=600K is too large a temperature step from 298K; reduce to near bubble point.
Flowsheet state:
  Stream FEED: T=298.15K  P=101325Pa  flow=1.0mol/s  {Ethanol:0.5, Water:0.5}
  Unit HT-01 (Heater): T_out=600.0  dP=0.0
Output:
{"patches": [{"target": "unit", "tag": "HT-01", "field": "T_out", "new_value": 365.0}],
 "reasoning": "HT-01 T_out reduced 600→365K — smaller step reduces solver stiffness; ethanol/water bubble point ~352K."}

━━━ EXAMPLE 4 — NO_SEPARATION: NRTL without BIPs ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Failure: [CRITICAL] NO_SEPARATION @ stream:VAP — outlet composition matches feed
Diagnosis: NRTL has no binary interaction parameters for this pair; outlet equals feed.
Flowsheet state:
  Global: property_package='NRTL'  compounds=['Hexane','Heptane']
Output:
{"patches": [{"target": "global", "field": "property_package", "new_value": "Raoult's Law"}],
 "reasoning": "Switched NRTL→Raoult's Law for topology verification; NRTL lacks BIPs for hexane/heptane."}

━━━ EXAMPLE 5 — MASS_BALANCE: Mixer outlet zero because one inlet flow=0 ━━━━━
Failure: [CRITICAL] MASS_BALANCE @ unit:MX-01 — outlet flow = 0.0 mol/s
Diagnosis: FEED-B stream flow=0.0; both Mixer inlets must have non-zero flow.
Flowsheet state:
  Stream FEED-A: T=298.15K  P=101325Pa  flow=1.0mol/s  {Methanol:1.0}
  Stream FEED-B: T=298.15K  P=101325Pa  flow=0.0mol/s  {Water:1.0}
  Unit MX-01 (Mixer):
Output:
{"patches": [{"target": "stream", "tag": "FEED-B", "field": "flow", "new_value": 1.0}],
 "reasoning": "FEED-B flow raised 0→1.0 mol/s — MX-01 outlet was zero because the second inlet had no flow."}
"""


# ── Refiner Agent ─────────────────────────────────────────────────────────────

class RefinerAgent:
    """
    Applies targeted fixes to a failing flowsheet given a CriticReport.

    Args:
        model: any model string supported by agents/llm.py
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self._model = model

    def refine(
        self,
        flowsheet: dict,
        report: CriticReport,
        max_retries: int = 3,
        run_history: list | None = None,
    ) -> RefinementResult:
        """
        Apply fixes and return an updated flowsheet.
        Stage 1 always runs. Stage 2 only runs if Stage 1 leaves failures.
        """
        fs = copy.deepcopy(flowsheet)
        all_changes: list[RefinementChange] = []

        # Stage 1 — deterministic
        fs, det_changes = _apply_deterministic_fixes(fs, report)
        all_changes.extend(det_changes)

        # Only return early if every reported failure code was handled by Stage 1
        # and the resulting flowsheet passes schema validation.
        if det_changes:
            handled = {c.failure_code for c in det_changes}
            all_handled = set(report.failure_codes).issubset(handled)
            if all_handled and not schema.validate(fs):
                if not has_errors(physics_validate(fs)):
                    return RefinementResult(
                        success=True,
                        updated_flowsheet=fs,
                        changes=all_changes,
                        stage="DETERMINISTIC",
                        reasoning="Deterministic fixes resolved all detected failures.",
                    )
                # Physics errors remain after deterministic fixes — fall through to Stage 2 LLM.

        # Stage 2 — LLM patch
        base_prompt = _build_prompt(fs, report, run_history=run_history)
        prompt = base_prompt

        for attempt in range(max_retries):
            try:
                raw = chat(prompt, system=_SYSTEM, model=self._model,
                           temperature=retry_temperature(attempt), thinking=True)
                patches, reasoning = _parse_patch_response(raw)
            except Exception as e:
                prompt = _retry_prompt(base_prompt, str(e))
                continue

            updated_fs, patch_changes = _apply_patches(fs, patches)

            errors = schema.validate(updated_fs)
            if errors:
                prompt = _retry_prompt(base_prompt, "\n".join(errors))
                continue

            if updated_fs == fs:
                prompt = _retry_prompt(
                    base_prompt,
                    "The patches produced no change — the flowsheet is IDENTICAL to the input.\n"
                    "You MUST patch the broken values flagged in the failure report."
                )
                continue

            all_changes.extend(patch_changes)
            return RefinementResult(
                success=True,
                updated_flowsheet=updated_fs,
                changes=all_changes,
                stage="LLM",
                reasoning=reasoning,
            )

        return RefinementResult(
            success=False,
            updated_flowsheet=fs,
            changes=all_changes,
            stage="FAILED",
            reasoning=f"Refiner failed after {max_retries} LLM attempts.",
        )


# ── Stage 1: deterministic fixes ──────────────────────────────────────────────

def _apply_deterministic_fixes(
        flowsheet: dict,
        report: CriticReport,
) -> tuple[dict, list[RefinementChange]]:
    """Apply rule-based fixes. Returns (updated_flowsheet, changes_made)."""
    fs = flowsheet
    changes: list[RefinementChange] = []
    codes = set(report.failure_codes)

    # SOLVER_FAIL with NRTL/UNIQUAC — fall back to Raoult's Law as first resort,
    # but only when physics_validate confirms Raoult's Law is valid for the compound
    # set. For known azeotropic or LLE systems the fallback would produce wrong VLE;
    # leave SOLVER_FAIL unhandled so Stage 2 LLM receives it with full context.
    if "SOLVER_FAIL" in codes:
        current = fs.get("property_package", "")
        if current in ("NRTL", "UNIQUAC"):
            candidate = _set_global(fs, "property_package", "Raoult's Law")
            if not has_errors(physics_validate(candidate)):
                fs = candidate
                changes.append(RefinementChange(
                    target="global", field="property_package",
                    old_value=current, new_value="Raoult's Law",
                    reason=(f"SOLVER_FAIL with {current}: likely missing binary "
                            "interaction parameters. Physics validation passed — "
                            "Raoult's Law used for topology test."),
                    failure_code="SOLVER_FAIL",
                ))
            # else: physics_validate returned errors (e.g. azeotropic system) —
            # skip fallback, let Stage 2 LLM handle with full context.

    # PARAM_MISSING / NO_SEPARATION — fall back to Raoult's Law, guarded by
    # physics_validate. Emit one change record per matched code so `all_handled`
    # covers both. For azeotropic/LLE systems leave codes unhandled → Stage 2 LLM.
    if codes & {"PARAM_MISSING", "NO_SEPARATION"}:
        current = fs.get("property_package", "")
        if current in ("NRTL", "UNIQUAC"):
            candidate = _set_global(fs, "property_package", "Raoult's Law")
            if not has_errors(physics_validate(candidate)):
                fs = candidate
                for code in sorted(codes & {"PARAM_MISSING", "NO_SEPARATION"}):
                    changes.append(RefinementChange(
                        target="global", field="property_package",
                        old_value=current, new_value="Raoult's Law",
                        reason=(f"{current} lacks binary interaction parameters in DWSIM. "
                                "Physics validation passed — Raoult's Law confirms "
                                "topology before re-attempting NRTL."),
                        failure_code=code,
                    ))
            # else: azeotropic or LLE system — skip fallback, let Stage 2 LLM handle.

    # UNPHYSICAL_T — temperature unit conversion
    if "UNPHYSICAL_T" in codes:
        fs, t_changes = _fix_temperatures(fs)
        changes.extend(t_changes)

    # UNPHYSICAL_P — pressure unit conversion
    if "UNPHYSICAL_P" in codes:
        fs, p_changes = _fix_pressures(fs)
        changes.extend(p_changes)

    # ENERGY_UNPHYSICAL — heater/cooler T_out direction
    if "ENERGY_UNPHYSICAL" in codes:
        fs, e_changes = _fix_energy_direction(fs, report)
        changes.extend(e_changes)

    # COMP_SUM — renormalise compositions
    if "COMP_SUM" in codes:
        fs, c_changes = _fix_compositions(fs)
        changes.extend(c_changes)

    # ZERO_OUTLET — sub-bubble-point condition: patch T_out on Heater/Cooler
    # This handles the case where the unit outlet temperature is below the mixture
    # bubble point, producing fully-liquid feed to the flash vessel.
    if "ZERO_OUTLET" in codes:
        fs, z_changes = _fix_sub_bubble_point(fs, report)
        changes.extend(z_changes)

    # WRONG_PHASE_DIR — swap src_port 0 ↔ 1 on Vessel outlets
    if "WRONG_PHASE_DIR" in codes:
        fs, w_changes = _fix_phase_direction(fs, report)
        changes.extend(w_changes)

    return fs, changes


def _fix_temperatures(flowsheet: dict) -> tuple[dict, list[RefinementChange]]:
    """
    Convert likely °C temperatures to Kelvin where context confirms a unit-mismatch.

    Context rule: a temperature T is treated as °C (and corrected to T+273.15)
    ONLY when ALL of the following hold:
      1. T < 100 K  (below water freezing point in Kelvin — unphysical for most processes)
      2. T >= -80   (not an extreme cryogenic value specified in °C, e.g. -196 °C for LN₂)
      3. The OTHER defined temperatures in this flowsheet are all > 150 K
         (i.e. the problematic value is an outlier, not part of a consistent cryogenic set)

    This avoids corrupting legitimate cryogenic simulations (methane at 111 K,
    ethane at 184 K, LNG at 110 K) where T < 100 is physically correct.
    """
    fs = copy.deepcopy(flowsheet)
    changes = []

    # Collect all explicitly defined temperatures to determine context
    all_defined_T = []
    for s in fs.get("streams", []):
        t = s.get("T")
        if t is not None:
            all_defined_T.append(t)
    for u in fs.get("units", []):
        t = u.get("T_out")
        if t is not None:
            all_defined_T.append(t)

    def _is_celsius_error(t: float) -> bool:
        """True only if t is almost certainly a °C value where K was required."""
        if t >= 100.0:
            return False          # already in plausible Kelvin range
        if t < -80.0:
            return False          # extreme °C value (e.g. -196 °C LN₂) — leave to LLM
        other_T = [x for x in all_defined_T if x != t]

        # Case 1: no other temperatures — accept if in the ambient-process °C range
        if not other_T:
            return 0.0 <= t <= 100.0

        # Case 2: all OTHER temperatures are well into the Kelvin range (> 150 K)
        # → this value is an outlier in a K-context flowsheet → it's °C
        if all(x > 150.0 for x in other_T):
            return True

        # Case 3: ALL defined temperatures (including this one) are in [0, 100]
        # → consistent °C input (e.g. all streams specified in °C) → convert all
        if all(0.0 <= x <= 100.0 for x in other_T) and 0.0 <= t <= 100.0:
            return True

        # Case 4: mixed context — some > 150 K alongside values < 100
        # → could be a genuine cryogenic flowsheet; do not risk corrupting it
        return False

    for s in fs.get("streams", []):
        T = s.get("T")
        if T is not None and _is_celsius_error(T):
            new_T = T + 273.15
            changes.append(RefinementChange(
                target=f"stream:{s['tag']}", field="T",
                old_value=T, new_value=round(new_T, 2),
                reason=(f"T={T} is below 100 K but other stream temperatures are "
                        f"in the Kelvin range — treated as °C, converted to K (+273.15)."),
                failure_code="UNPHYSICAL_T",
            ))
            s["T"] = round(new_T, 2)

    for u in fs.get("units", []):
        T_out = u.get("T_out")
        if T_out is not None and _is_celsius_error(T_out):
            new_T = T_out + 273.15
            changes.append(RefinementChange(
                target=f"unit:{u['tag']}", field="T_out",
                old_value=T_out, new_value=round(new_T, 2),
                reason=(f"T_out={T_out} is below 100 K but other temperatures are "
                        f"in the Kelvin range — treated as °C, converted to K (+273.15)."),
                failure_code="UNPHYSICAL_T",
            ))
            u["T_out"] = round(new_T, 2)

    return fs, changes


def _fix_pressures(flowsheet: dict) -> tuple[dict, list[RefinementChange]]:
    """Convert likely bar/kPa pressures to Pascals."""
    fs = copy.deepcopy(flowsheet)
    changes = []

    def _fix_P(tag: str, obj: dict, field: str, code: str):
        P = obj.get(field)
        if P is None:
            return
        if P < 100.0:          # likely bar — multiply by 1e5
            new_P = P * 1e5
            unit = "bar"
        elif P < 10_000.0:     # likely kPa — multiply by 1000
            new_P = P * 1_000.0
            unit = "kPa"
        else:
            return
        changes.append(RefinementChange(
            target=tag, field=field,
            old_value=P, new_value=round(new_P, 1),
            reason=f"P={P} looks like {unit} — converted to Pa (×{1e5 if unit=='bar' else 1000:.0f}).",
            failure_code="UNPHYSICAL_P",
        ))
        obj[field] = round(new_P, 1)

    for s in fs.get("streams", []):
        _fix_P(f"stream:{s['tag']}", s, "P", "UNPHYSICAL_P")

    for u in fs.get("units", []):
        _fix_P(f"unit:{u['tag']}", u, "P_out", "UNPHYSICAL_P")

    return fs, changes


def _fix_energy_direction(
        flowsheet: dict,
        report: CriticReport,
) -> tuple[dict, list[RefinementChange]]:
    """
    For Heater units where T_out < inlet T (ENERGY_UNPHYSICAL),
    if the specified T_out looks like it should be above the feed, it's a unit
    conversion issue already caught by UNPHYSICAL_T. If not, swap unit type.
    """
    fs = copy.deepcopy(flowsheet)
    changes = []
    for sig in report.signals:
        if sig.code != "ENERGY_UNPHYSICAL":
            continue
        # sig.location = "unit:HT-01"
        utag = sig.location.split(":", 1)[-1]
        for u in fs.get("units", []):
            if u["tag"] != utag:
                continue
            old_type = u["type"]
            new_type = "Cooler" if old_type == "Heater" else "Heater"
            changes.append(RefinementChange(
                target=f"unit:{utag}", field="type",
                old_value=old_type, new_value=new_type,
                reason=(f"{old_type} outlet is cooler than inlet — "
                        f"swapped to {new_type}."),
                failure_code="ENERGY_UNPHYSICAL",
            ))
            u["type"] = new_type
    return fs, changes


def _fix_compositions(flowsheet: dict) -> tuple[dict, list[RefinementChange]]:
    """Renormalise feed stream compositions to sum to 1.0."""
    fs = copy.deepcopy(flowsheet)
    changes = []
    for s in fs.get("streams", []):
        comp = s.get("composition", {})
        if not comp:
            continue
        total = sum(comp.values())
        if total > 0 and abs(total - 1.0) > 0.02:
            old_comp = dict(comp)
            for k in comp:
                comp[k] = round(comp[k] / total, 6)
            changes.append(RefinementChange(
                target=f"stream:{s['tag']}", field="composition",
                old_value=old_comp, new_value=dict(comp),
                reason=f"Composition summed to {total:.4f} — renormalised to 1.0.",
                failure_code="COMP_SUM",
            ))
    return fs, changes


def _fix_sub_bubble_point(
        flowsheet: dict,
        report: CriticReport,
) -> tuple[dict, list[RefinementChange]]:
    """
    When ZERO_OUTLET fires and a Heater or Cooler is present, check whether
    T_out is below the mixture bubble point and raise it to bubble_point + 15 K.
    Only patches units whose T_out can be confirmed as below the bubble point —
    skips units where the bubble point cannot be estimated (unknown compounds).
    """
    fs = copy.deepcopy(flowsheet)
    changes = []
    compounds = fs.get("compounds", [])

    # Find the fully-specified feed stream for T, P, composition
    feed = next(
        (s for s in fs.get("streams", []) if s.get("T") is not None
         and s.get("composition")),
        None,
    )
    if feed is None:
        return fs, changes

    feed_comp = feed.get("composition", {})
    feed_p    = feed.get("P", 101_325.0)
    t_bub     = _estimate_bubble_point(compounds, feed_comp, feed_p)
    if t_bub is None:
        return fs, changes   # unknown compounds — leave for Stage 2 LLM

    for u in fs.get("units", []):
        if u.get("type") not in ("Heater", "Cooler"):
            continue
        t_out = u.get("T_out")
        if t_out is None or t_out > t_bub:
            continue
        new_t_out = round(t_bub + 15.0, 1)
        changes.append(RefinementChange(
            target=f"unit:{u['tag']}", field="T_out",
            old_value=t_out, new_value=new_t_out,
            reason=(
                f"T_out={t_out} K is at or below the estimated mixture bubble point "
                f"{t_bub} K at {feed_p:.0f} Pa — raised to {new_t_out} K to ensure "
                "two-phase feed to flash vessel."
            ),
            failure_code="ZERO_OUTLET",
        ))
        u["T_out"] = new_t_out

    return fs, changes


def _fix_phase_direction(
        flowsheet: dict,
        report: CriticReport,
) -> tuple[dict, list[RefinementChange]]:
    """Swap src_port 0 ↔ 1 on Vessel outlets where phase direction is inverted."""
    fs = copy.deepcopy(flowsheet)
    changes = []
    for sig in report.signals:
        if sig.code != "WRONG_PHASE_DIR":
            continue
        utag = sig.location.split(":", 1)[-1]
        # Find the two Vessel outlet connections and swap their src_ports
        vessel_outs = [
            i for i, c in enumerate(fs.get("connections", []))
            if len(c) >= 3 and c[0] == utag
        ]
        if len(vessel_outs) == 2:
            i0, i1 = vessel_outs
            old_p0 = fs["connections"][i0][2]
            old_p1 = fs["connections"][i1][2]
            fs["connections"][i0][2] = old_p1
            fs["connections"][i1][2] = old_p0
            changes.append(RefinementChange(
                target=f"unit:{utag}", field="connections.src_port",
                old_value=f"[{old_p0}, {old_p1}]",
                new_value=f"[{old_p1}, {old_p0}]",
                reason="Phase direction inverted — swapped vapour/liquid outlet ports.",
                failure_code="WRONG_PHASE_DIR",
            ))
    return fs, changes


# ── Helpers ───────────────────────────────────────────────────────────────────

def _set_global(flowsheet: dict, field: str, value) -> dict:
    fs = copy.deepcopy(flowsheet)
    fs[field] = value
    return fs


def _format_flowsheet_compact(flowsheet: dict) -> str:
    """Compact human-readable flowsheet state — much shorter than full JSON."""
    lines = [
        "Global:",
        f"  compounds       : {flowsheet.get('compounds', [])}",
        f"  property_package: {flowsheet.get('property_package', '')!r}",
        "",
        "Streams:",
    ]
    for s in flowsheet.get("streams", []):
        tag = s["tag"]
        if "T" in s:
            comp = s.get("composition", {})
            comp_str = ", ".join(f"{k}: {v}" for k, v in comp.items())
            lines.append(
                f"  {tag}: T={s['T']}K  P={s['P']}Pa  "
                f"flow={s.get('flow', '?')}mol/s  {{{comp_str}}}"
            )
        else:
            lines.append(f"  {tag}: (no conditions — intermediate/product)")
    lines += ["", "Units:"]
    for u in flowsheet.get("units", []):
        tag   = u["tag"]
        utype = u.get("type", "?")
        params = {k: v for k, v in u.items() if k not in ("tag", "type")}
        params_str = "  ".join(f"{k}={v}" for k, v in params.items())
        lines.append(f"  {tag} ({utype}): {params_str}")
    lines += ["", "Connections:"]
    for c in flowsheet.get("connections", []):
        if len(c) >= 4:
            lines.append(f"  {c[0]} → {c[1]}  src_port={c[2]}  dst_port={c[3]}")
    return "\n".join(lines)


def _build_prompt(
        flowsheet: dict,
        report: CriticReport,
        run_history: list | None = None,
) -> str:
    history_block = ""
    if run_history:
        lines = [
            "TRIAL HISTORY — already attempted and failed. Do NOT repeat these fixes:\n",
        ]
        for rec in run_history:
            codes = ", ".join(rec.failure_codes) if rec.failure_codes else "none"
            lines.append(
                f"  Iteration {rec.iteration}: pkg={rec.property_package}"
                f"  outcome={rec.execution_summary}  codes=[{codes}]"
                + (f"  refiner={rec.refiner_outcome}" if rec.refiner_outcome else "")
            )
            if rec.diagnosis:
                lines.append(f"    ↳ {rec.diagnosis[:200]}")
        history_block = "\n".join(lines) + "\n\n"

    signals_text = "\n".join(
        f"  [{s.severity}] {s.code} @ {s.location}: {s.evidence}"
        for s in report.signals
    )
    fixes_text = "\n".join(f"  - {f}" for f in report.suggested_fixes)

    return (
        _FEW_SHOT_PATCHES
        + "\n━━━ NOW FIX THIS FLOWSHEET ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        + history_block
        + f"FAILURE REPORT:\n"
        + f"  Diagnosis : {report.diagnosis}\n"
        + f"  Codes     : {report.failure_codes}\n"
        + f"  Signals:\n{signals_text}\n"
        + f"  Suggested fixes:\n{fixes_text}\n"
        + f"\nCurrent flowsheet state:\n{_format_flowsheet_compact(flowsheet)}\n\n"
        + 'TASK: patch ONLY the broken fields.\n'
        + 'Return ONLY: {"patches": [...], "reasoning": "<one sentence>"}'
    )


def _parse_patch_response(raw: str) -> tuple[list[dict], str]:
    """Parse patch-format LLM response. Returns (patches, reasoning)."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(ln for ln in lines if not ln.strip().startswith("```"))
    text = text.strip()

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(
            'No JSON object found. Return {"patches": [...], "reasoning": "..."}')
    parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
    if "patches" not in parsed:
        raise ValueError(
            'Missing "patches" key — return {"patches": [...], "reasoning": "..."}')

    patches = parsed["patches"]
    if not isinstance(patches, list):
        raise ValueError('"patches" must be a list')
    if not patches:
        raise ValueError('"patches" list is empty — include at least one patch')

    _VALID_TARGETS = {"unit", "stream", "global"}
    for i, p in enumerate(patches):
        if not isinstance(p, dict):
            raise ValueError(f"Patch {i} must be a dict, got {type(p).__name__}")
        for key in ("target", "field", "new_value"):
            if key not in p:
                raise ValueError(f"Patch {i} missing required key '{key}': {p}")
        if p["target"] not in _VALID_TARGETS:
            raise ValueError(
                f"Patch {i} 'target' must be one of {_VALID_TARGETS}, "
                f"got: {p['target']!r}")
        if p["target"] in ("unit", "stream") and "tag" not in p:
            raise ValueError(
                f"Patch {i} with target='{p['target']}' must include 'tag'")

    return patches, parsed.get("reasoning", "")


def _apply_patches(
        flowsheet: dict,
        patches: list[dict],
) -> tuple[dict, list[RefinementChange]]:
    """Apply a patch list to a copy of the flowsheet. Returns (updated_fs, changes)."""
    fs = copy.deepcopy(flowsheet)
    changes: list[RefinementChange] = []

    for patch in patches:
        target    = patch["target"]
        field     = patch["field"]
        new_value = patch["new_value"]
        tag       = patch.get("tag", "")

        if target == "global":
            old_value  = fs.get(field)
            fs[field]  = new_value
            changes.append(RefinementChange(
                target="global", field=field,
                old_value=old_value, new_value=new_value,
                reason=patch.get("reason", ""), failure_code="",
            ))

        elif target == "stream":
            for s in fs.get("streams", []):
                if s["tag"] == tag:
                    old_value = s.get(field)
                    s[field]  = new_value
                    changes.append(RefinementChange(
                        target=f"stream:{tag}", field=field,
                        old_value=old_value, new_value=new_value,
                        reason=patch.get("reason", ""), failure_code="",
                    ))
                    break

        elif target == "unit":
            for u in fs.get("units", []):
                if u["tag"] == tag:
                    old_value = u.get(field)
                    u[field]  = new_value
                    changes.append(RefinementChange(
                        target=f"unit:{tag}", field=field,
                        old_value=old_value, new_value=new_value,
                        reason=patch.get("reason", ""), failure_code="",
                    ))
                    break

    return fs, changes


def _retry_prompt(base: str, errors: str) -> str:
    return (
        f"CORRECTION REQUIRED — fix EXACTLY these errors:\n"
        f"  {errors}\n\n"
        f"Key rules:\n"
        f"  - 'patches' must be a non-empty list of patch objects.\n"
        f"  - Each patch: "
        f'{{\"target\": \"unit\"|\"stream\"|\"global\", '
        f'\"tag\": \"<TAG>\", \"field\": \"<field>\", \"new_value\": <value>}}\n'
        f"  - Temperatures in Kelvin, pressures in Pascals, flows in mol/s.\n"
        f"  - Do NOT return a full flowsheet — return ONLY the patch list.\n\n"
        f"Failure context and flowsheet state:\n{base}\n\n"
        'Return ONLY: {"patches": [...], "reasoning": "<one sentence>"}'
    )
