"""
Static physics compatibility checker.

Validates a flowsheet dict against known thermodynamic constraints BEFORE
sending it to DWSIM. Returns a list of PhysicsIssue objects (ERROR or WARNING).

Errors   → must be resolved before execution (will cause DWSIM failure or
           produce physically meaningless results).
Warnings → should be reviewed; execution may proceed but results need scrutiny.

Called by schema.physics_validate(flowsheet) and by the Thermo Agent after
every property package assignment.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class Severity(Enum):
    ERROR   = "ERROR"
    WARNING = "WARNING"


@dataclass
class PhysicsIssue:
    severity:    Severity
    location:    str    # e.g. "global", "unit:V-01", "stream:FEED"
    code:        str    # short machine-readable code
    message:     str

    def __str__(self) -> str:
        return f"[{self.severity.value}] {self.location} — {self.code}: {self.message}"

    @property
    def fix(self) -> str:
        """Return a specific, actionable correction instruction for this error code.

        Used by PlannerAgent._physics_retry_prompt() to tell the LLM exactly
        what field to change and to what value — not just what is wrong.
        """
        _FIXES = {
            "LIGHT_GAS_ACTIVITY_MODEL": (
                'Set "property_package" to "Peng-Robinson". '
                'Activity-coefficient models (Raoult\'s Law, NRTL, UNIQUAC) cannot '
                'represent non-condensable gases (N2, O2, CH4, H2, CO2, H2S). '
                'Peng-Robinson or Soave-Redlich-Kwong are required for any mixture '
                'containing these gases.'
            ),
            "AZEOTROPE_IDEAL_MODEL": (
                'Set "property_package" to "NRTL". '
                'Raoult\'s Law assumes ideal mixing and predicts no azeotropes. '
                'The compound pair forms a known azeotrope — NRTL with binary '
                'interaction parameters will give physically correct VLE.'
            ),
            "LLE_INCAPABLE_PACKAGE": (
                'Set "property_package" to "NRTL". '
                'The compound pair has limited mutual solubility (two liquid phases). '
                'Only NRTL or UNIQUAC can model liquid-liquid equilibrium. '
                'Do not use Raoult\'s Law, Peng-Robinson, or SRK for this system.'
            ),
            "VESSEL_LLE_INCAPABLE": (
                'Set "property_package" to "NRTL". '
                'The flash vessel may produce two liquid phases that require '
                'NRTL or UNIQUAC to model correctly.'
            ),
            "HIGH_P_ACTIVITY_MODEL": (
                'Set "property_package" to "Peng-Robinson". '
                'The unit outlet pressure exceeds the valid range of activity-coefficient '
                'models (max ~15 bar). Use Peng-Robinson or Soave-Redlich-Kwong '
                'for high-pressure VLE.'
            ),
            "PRESSURE_EXCEEDS_PACKAGE_RANGE": (
                'Either lower the stream pressure to within the package limit, '
                'or set "property_package" to "Peng-Robinson" which handles any pressure.'
            ),
            "ELECTROLYTE_UNSUPPORTED": (
                'Remove the electrolyte compound from the "compounds" list. '
                'No supported property package can model ionic species. '
                'The process must be reformulated without electrolytes.'
            ),
            "COMPRESSOR_NEEDS_EOS": (
                'Consider setting "property_package" to "Peng-Robinson" or '
                '"Soave-Redlich-Kwong" for accurate compressibility factors '
                'in the Compressor unit.'
            ),
        }
        return _FIXES.get(
            self.code,
            f"Fix {self.code}: {self.message[:120]}"
        )


# ── Compound classification ────────────────────────────────────────────────────
# Case-insensitive prefix/substring matching is used — exact DWSIM names vary.

# Polar / hydrogen-bonding liquids
_POLAR = {
    "water", "methanol", "ethanol", "1-propanol", "2-propanol", "n-propanol",
    "1-butanol", "2-butanol", "n-butanol", "isobutanol", "tert-butanol",
    "1-pentanol", "ethylene glycol", "propylene glycol", "glycerol", "glycerin",
    "acetic acid", "formic acid", "propionic acid", "butyric acid",
    "methylamine", "ethylamine", "dimethylamine", "trimethylamine",
    "ethanolamine", "diethanolamine", "triethanolamine",
    "dimethyl sulfoxide", "dmso", "n-methylpyrrolidone", "nmp",
    "acetonitrile", "formamide", "dimethylformamide", "dmf",
    "hydrogen fluoride", "ammonia",
}

# Mildly polar (esters, ethers, ketones) — can use NRTL or PR depending on context
_MILD_POLAR = {
    "acetone", "methyl ethyl ketone", "mek", "cyclohexanone",
    "ethyl acetate", "methyl acetate", "n-butyl acetate", "isopropyl acetate",
    "diethyl ether", "diisopropyl ether", "tetrahydrofuran", "thf",
    "1,4-dioxane", "methyl tert-butyl ether", "mtbe",
    "chloroform", "dichloromethane", "dcm", "carbon tetrachloride",
    "chlorobenzene", "1,2-dichloroethane", "1,2-dce",
    "carbon disulfide", "nitromethane",
    "dimethyl ether",
}

# Non-condensable / permanent gases — cannot exist as pure liquids at normal conditions
_LIGHT_GAS = {
    "hydrogen", "nitrogen", "oxygen", "argon", "helium", "neon", "krypton",
    "carbon monoxide", "nitric oxide", "nitrogen dioxide",
}

# Light condensable hydrocarbons — EOS required at elevated pressures
_LIGHT_HC = {
    "methane", "ethane", "propane", "n-butane", "isobutane", "n-pentane",
    "isopentane", "neopentane",
    "ethylene", "propylene", "1-butene", "2-butene", "isobutylene",
    "acetylene",
    "carbon dioxide",   # behaves like a light gas in VLE context
    "hydrogen sulfide",
}

# Heavier non-polar hydrocarbons
_HYDROCARBON = {
    "n-hexane", "n-heptane", "n-octane", "n-nonane", "n-decane",
    "n-undecane", "n-dodecane", "n-tetradecane", "n-hexadecane",
    "cyclohexane", "cyclopentane", "methylcyclohexane",
    "benzene", "toluene", "ethylbenzene",
    "o-xylene", "m-xylene", "p-xylene", "xylene",
    "styrene", "cumene", "naphthalene",
    "isooctane", "2-methylheptane",
}

# Electrolytes — require specialised DWSIM electrolyte packages (not yet supported)
_ELECTROLYTE = {
    "sodium chloride", "nacl", "hydrochloric acid", "hcl",
    "sodium hydroxide", "naoh", "potassium chloride", "kcl",
    "sulfuric acid", "nitric acid", "phosphoric acid",
    "calcium chloride", "ammonium chloride", "sodium sulfate",
    "potassium hydroxide", "lithium chloride",
}

# ── Known azeotropic binary pairs ──────────────────────────────────────────────
# Raoult's Law cannot predict these — it assumes ideality.
_AZEOTROPES: set[frozenset] = {
    frozenset({"ethanol",     "water"}),           # min-boiling, 95.6 mol% EtOH, 78.1°C
    frozenset({"1-propanol",  "water"}),           # min-boiling
    frozenset({"2-propanol",  "water"}),           # min-boiling, 87.7 mol% IPA, 80.3°C
    frozenset({"1-butanol",   "water"}),           # heterogeneous, min-boiling
    frozenset({"n-butanol",   "water"}),
    frozenset({"isobutanol",  "water"}),
    frozenset({"tert-butanol","water"}),
    frozenset({"ethyl acetate","water"}),          # heterogeneous, 70.4 mol% EtOAc, 70.4°C
    frozenset({"n-butyl acetate","water"}),        # heterogeneous
    frozenset({"isopropyl acetate","water"}),
    frozenset({"diethyl ether","water"}),          # min-boiling, 98.7 mol% ether
    frozenset({"tetrahydrofuran","water"}),        # min-boiling
    frozenset({"1,4-dioxane","water"}),            # max-boiling
    frozenset({"chloroform","water"}),             # heterogeneous
    frozenset({"dichloromethane","water"}),
    frozenset({"chlorobenzene","water"}),          # heterogeneous, two-phase
    frozenset({"1,2-dichloroethane","water"}),     # heterogeneous
    frozenset({"1-pentanol","water"}),             # min-boiling, partially miscible
    frozenset({"carbon disulfide","benzene"}),     # min-boiling
    frozenset({"n-hexane","ethanol"}),             # min-boiling
    frozenset({"cyclohexane","ethanol"}),          # min-boiling
    frozenset({"benzene","ethanol"}),              # min-boiling
    frozenset({"toluene","ethanol"}),
    frozenset({"benzene","water"}),                # heterogeneous
    frozenset({"toluene","water"}),                # heterogeneous
    frozenset({"n-hexane","water"}),               # immiscible — trivial LLE not azeotrope
    frozenset({"acetonitrile","water"}),           # min-boiling, 83 mol% MeCN
    frozenset({"methyl tert-butyl ether","water"}),
    frozenset({"formic acid","water"}),            # max-boiling, 77.5 mol% formic acid
    frozenset({"hydrogen fluoride","water"}),      # max-boiling, 35.6% HF
    frozenset({"acetone","chloroform"}),           # max-boiling (negative deviation)
    frozenset({"methanol","methyl acetate"}),      # min-boiling
    frozenset({"ethanol","ethyl acetate"}),        # min-boiling
    frozenset({"ethanol","cyclohexane"}),
}

# ── Known partially miscible / LLE pairs ──────────────────────────────────────
# NRTL or UNIQUAC with LLE parameters required.
_LLE_PAIRS: set[frozenset] = {
    frozenset({"n-butanol",  "water"}),
    frozenset({"1-butanol",  "water"}),
    frozenset({"isobutanol", "water"}),
    frozenset({"ethyl acetate","water"}),
    frozenset({"n-butyl acetate","water"}),
    frozenset({"n-hexane",   "water"}),
    frozenset({"n-heptane",  "water"}),
    frozenset({"n-octane",   "water"}),
    frozenset({"cyclohexane","water"}),
    frozenset({"benzene",    "water"}),
    frozenset({"toluene",    "water"}),
    frozenset({"diethyl ether","water"}),
    frozenset({"chloroform", "water"}),
    frozenset({"dichloromethane","water"}),
    frozenset({"carbon tetrachloride","water"}),
    frozenset({"phenol",     "water"}),            # partially miscible below 65°C
    frozenset({"aniline",    "water"}),
    frozenset({"furfural",   "water"}),
    frozenset({"n-hexane",   "methanol"}),
    frozenset({"n-heptane",  "methanol"}),
    frozenset({"cyclohexane","methanol"}),
    frozenset({"n-hexane",   "acetonitrile"}),
    frozenset({"1-pentanol", "water"}),
    frozenset({"chlorobenzene","water"}),
    frozenset({"1,2-dichloroethane","water"}),
    frozenset({"n-methylpyrrolidone","toluene"}),
}

# ── Property package capability profiles ──────────────────────────────────────
_PKG_PROFILES = {
    "Raoult's Law": {
        "handles_polar":         True,   # but only ideal polar — misleading for real polar
        "handles_light_gas":     False,
        "handles_lle":           False,
        "handles_azeotrope":     False,
        "handles_high_pressure": False,
        "max_pressure_Pa":       500_000,   # ~5 bar
        "min_temp_K":            200.0,
        "max_temp_K":            600.0,
        "notes": "Ideal VLE only. Fails for any non-ideal system.",
    },
    "NRTL": {
        "handles_polar":         True,
        "handles_light_gas":     False,   # activity coeff models not suitable for gases
        "handles_lle":           True,
        "handles_azeotrope":     True,
        "handles_high_pressure": False,
        "max_pressure_Pa":       1_500_000,  # ~15 bar (beyond this EOS more reliable)
        "min_temp_K":            200.0,
        "max_temp_K":            700.0,
        "requires_binary_params": True,
        "notes": "Best for polar non-ideal VLE and LLE. Needs binary parameters.",
    },
    "UNIQUAC": {
        "handles_polar":         True,
        "handles_light_gas":     False,
        "handles_lle":           True,
        "handles_azeotrope":     True,
        "handles_high_pressure": False,
        "max_pressure_Pa":       1_500_000,
        "min_temp_K":            200.0,
        "max_temp_K":            700.0,
        "requires_binary_params": True,
        "notes": "Like NRTL but size/shape based. Better for size-asymmetric mixtures.",
    },
    "Peng-Robinson": {
        "handles_polar":         False,   # poor for strongly polar liquid phase
        "handles_light_gas":     True,
        "handles_lle":           False,
        "handles_azeotrope":     False,   # limited azeotrope prediction
        "handles_high_pressure": True,
        "max_pressure_Pa":       1e8,     # 1000 bar
        "min_temp_K":            100.0,
        "max_temp_K":            2000.0,
        "notes": "Best for hydrocarbons and light gases at any pressure.",
    },
    "Soave-Redlich-Kwong": {
        "handles_polar":         False,
        "handles_light_gas":     True,
        "handles_lle":           False,
        "handles_azeotrope":     False,
        "handles_high_pressure": True,
        "max_pressure_Pa":       1e8,
        "min_temp_K":            100.0,
        "max_temp_K":            2000.0,
        "notes": "Similar to PR. Slightly better for gas-phase at high T.",
    },
    "Lee-Kesler-Plöcker": {
        "handles_polar":         False,
        "handles_light_gas":     True,
        "handles_lle":           False,
        "handles_azeotrope":     False,
        "handles_high_pressure": True,
        "max_pressure_Pa":       1e8,
        "min_temp_K":            50.0,
        "max_temp_K":            2000.0,
        "notes": "Best for cryogenic and natural gas processing.",
    },
}

# ── Unit operation thermodynamic requirements ─────────────────────────────────
_UNIT_THERMO_REQUIREMENTS = {
    "Vessel": {
        "needs_vle":   True,
        "needs_lle":   False,   # flag if LLE pair present
        "description": "Flash vessel — needs accurate VLE.",
    },
    "Heater": {
        "needs_vle":   False,
        "needs_lle":   False,
        "description": "Sensible heat only — package affects enthalpy calculation.",
    },
    "Cooler": {
        "needs_vle":   False,
        "needs_lle":   False,
        "description": "Sensible/latent heat — package affects condensation.",
    },
    "Mixer": {
        "needs_vle":   False,
        "needs_lle":   False,
        "description": "Mixing — package affects enthalpy of mixing.",
    },
    "Splitter": {
        "needs_vle":   False,
        "needs_lle":   False,
        "description": "Flow split — thermodynamics minimally involved.",
    },
    "Pump": {
        "needs_vle":   False,
        "needs_lle":   False,
        "description": "Liquid pump — package affects liquid density.",
    },
    "Compressor": {
        "needs_vle":   False,
        "needs_lle":   False,
        "description": "Gas compression — cubic EOS preferred for accurate Z-factor.",
    },
    "Expander": {
        "needs_vle":   False,
        "needs_lle":   False,
        "description": "Gas expansion — cubic EOS preferred.",
    },
}


# ── Public interface ───────────────────────────────────────────────────────────

def physics_validate(flowsheet: dict) -> list[PhysicsIssue]:
    """
    Run all physics compatibility checks on a flowsheet dict.
    Returns a list of PhysicsIssue objects (empty = no issues found).
    """
    issues: list[PhysicsIssue] = []
    compounds_lower = [c.lower() for c in flowsheet.get("compounds", [])]
    default_pkg = flowsheet.get("property_package", "")

    # Classify the compound set
    classification = _classify_compounds(compounds_lower)

    # Global package checks
    issues.extend(_check_global_package(
        default_pkg, classification, compounds_lower, flowsheet))

    # Per-stream checks
    for stream in flowsheet.get("streams", []):
        issues.extend(_check_stream(stream, default_pkg, classification))

    # Per-unit checks
    for unit in flowsheet.get("units", []):
        effective_pkg = unit.get("property_package") or default_pkg
        issues.extend(_check_unit(unit, effective_pkg, classification, compounds_lower))

    return issues


def format_issues(issues: list[PhysicsIssue]) -> str:
    if not issues:
        return "No physics issues found."
    errors   = [i for i in issues if i.severity == Severity.ERROR]
    warnings = [i for i in issues if i.severity == Severity.WARNING]
    lines = []
    if errors:
        lines.append(f"ERRORS ({len(errors)}):")
        lines.extend(f"  {i}" for i in errors)
    if warnings:
        lines.append(f"WARNINGS ({len(warnings)}):")
        lines.extend(f"  {i}" for i in warnings)
    return "\n".join(lines)


def has_errors(issues: list[PhysicsIssue]) -> bool:
    return any(i.severity == Severity.ERROR for i in issues)


# ── Internal check functions ───────────────────────────────────────────────────

def _classify_compounds(compounds_lower: list[str]) -> dict[str, set[str]]:
    """Return which classification sets each compound belongs to."""
    result: dict[str, set[str]] = {
        "polar": set(), "mild_polar": set(), "light_gas": set(),
        "light_hc": set(), "hydrocarbon": set(), "electrolyte": set(),
        "unknown": set(),
    }
    for c in compounds_lower:
        matched = False
        if _in_set(c, _ELECTROLYTE):
            result["electrolyte"].add(c); matched = True
        if _in_set(c, _LIGHT_GAS):
            result["light_gas"].add(c); matched = True
        if _in_set(c, _LIGHT_HC):
            result["light_hc"].add(c); matched = True
        if _in_set(c, _POLAR):
            result["polar"].add(c); matched = True
        if _in_set(c, _MILD_POLAR):
            result["mild_polar"].add(c); matched = True
        if _in_set(c, _HYDROCARBON):
            result["hydrocarbon"].add(c); matched = True
        if not matched:
            result["unknown"].add(c)
    return result


def _check_global_package(
        pkg: str, cls: dict, compounds_lower: list[str],
        flowsheet: dict) -> list[PhysicsIssue]:
    issues = []
    profile = _PKG_PROFILES.get(pkg)
    if not profile:
        return issues

    # Electrolytes
    if cls["electrolyte"]:
        issues.append(PhysicsIssue(
            Severity.ERROR, "global", "ELECTROLYTE_UNSUPPORTED",
            f"Compounds {cls['electrolyte']} are electrolytes. No supported "
            "property package handles electrolytes. Remove them or use a "
            "specialised electrolyte model."))

    # Light gases with activity-coefficient models
    if cls["light_gas"] and pkg in ("Raoult's Law", "NRTL", "UNIQUAC"):
        issues.append(PhysicsIssue(
            Severity.ERROR, "global", "LIGHT_GAS_ACTIVITY_MODEL",
            f"Non-condensable gases {cls['light_gas']} cannot be handled by "
            f"activity-coefficient model '{pkg}'. Use Peng-Robinson or SRK."))

    # Raoult's Law with known azeotropic pairs
    if pkg == "Raoult's Law":
        azeotropic = _find_azeotropic_pairs(compounds_lower)
        if azeotropic:
            issues.append(PhysicsIssue(
                Severity.ERROR, "global", "AZEOTROPE_IDEAL_MODEL",
                f"Known azeotropic pairs detected: {azeotropic}. Raoult's Law "
                "assumes ideal VLE and cannot predict azeotropes. Use NRTL or UNIQUAC."))

    # Raoult's Law with strongly or mildly polar mixtures
    if pkg == "Raoult's Law":
        polar_like = cls["polar"] | cls["mild_polar"]
        non_water_polar = polar_like - {"water"}
        # Polar + hydrocarbon → activity coefficients significantly > 1
        if polar_like and cls["hydrocarbon"]:
            issues.append(PhysicsIssue(
                Severity.WARNING, "global", "POLAR_IDEAL_MODEL",
                f"Polar/mild-polar compounds {polar_like} mixed with "
                f"hydrocarbons {cls['hydrocarbon']} using Raoult's Law. "
                "Activity coefficients likely >> 1 — use NRTL or UNIQUAC."))
        # Two or more different polar-type compounds → non-ideal interactions
        elif len(non_water_polar) >= 2:
            issues.append(PhysicsIssue(
                Severity.WARNING, "global", "POLAR_IDEAL_MODEL",
                f"Multiple polar/mild-polar compounds {polar_like} with "
                "Raoult's Law. Non-ideal interactions expected — "
                "consider NRTL or UNIQUAC."))
        elif cls["polar"] and (cls["polar"] - {"water"}):
            issues.append(PhysicsIssue(
                Severity.WARNING, "global", "POLAR_IDEAL_MODEL",
                f"Polar compounds {cls['polar']} present with Raoult's Law. "
                "Significant non-ideality likely — consider NRTL or UNIQUAC."))

    # Cubic EOS with dominant polar liquid phase
    if pkg in ("Peng-Robinson", "Soave-Redlich-Kwong"):
        if cls["polar"] and not cls["light_hc"] and not cls["light_gas"]:
            issues.append(PhysicsIssue(
                Severity.WARNING, "global", "CUBIC_EOS_POLAR_LIQUID",
                f"Cubic EOS '{pkg}' applied to polar-dominant liquid system "
                f"{cls['polar']}. Liquid-phase activity not well represented. "
                "Consider NRTL or UNIQUAC for accurate liquid VLE."))

    # LLE pairs present without LLE-capable package
    if not profile.get("handles_lle", False):
        lle = _find_lle_pairs(compounds_lower)
        if lle:
            issues.append(PhysicsIssue(
                Severity.ERROR, "global", "LLE_INCAPABLE_PACKAGE",
                f"Known partially miscible pairs {lle} present but '{pkg}' "
                "cannot model liquid-liquid equilibrium. Use NRTL or UNIQUAC."))

    # Unknown compounds — cannot verify compatibility
    if cls["unknown"]:
        issues.append(PhysicsIssue(
            Severity.WARNING, "global", "UNKNOWN_COMPOUNDS",
            f"Compounds {cls['unknown']} not in physics checker database. "
            "Verify manually that '{pkg}' is appropriate."))

    return issues


def _check_stream(stream: dict, pkg: str, cls: dict) -> list[PhysicsIssue]:
    issues = []
    tag = stream.get("tag", "<unnamed>")
    profile = _PKG_PROFILES.get(pkg, {})

    T = stream.get("T")
    P = stream.get("P")

    if T is not None and profile:
        if T < profile.get("min_temp_K", 0):
            issues.append(PhysicsIssue(
                Severity.WARNING, f"stream:{tag}", "TEMP_BELOW_PACKAGE_RANGE",
                f"T={T} K is below the valid range for '{pkg}' "
                f"(min {profile['min_temp_K']} K)."))
        if T > profile.get("max_temp_K", 1e9):
            issues.append(PhysicsIssue(
                Severity.WARNING, f"stream:{tag}", "TEMP_ABOVE_PACKAGE_RANGE",
                f"T={T} K is above the valid range for '{pkg}' "
                f"(max {profile['max_temp_K']} K)."))

    if P is not None and profile:
        max_P = profile.get("max_pressure_Pa", 1e9)
        if P > max_P:
            sev = (Severity.ERROR
                   if pkg in ("Raoult's Law", "NRTL", "UNIQUAC")
                   else Severity.WARNING)
            issues.append(PhysicsIssue(
                sev, f"stream:{tag}", "PRESSURE_EXCEEDS_PACKAGE_RANGE",
                f"P={P/1e5:.1f} bar exceeds recommended range for '{pkg}' "
                f"({max_P/1e5:.0f} bar). "
                f"{'Use Peng-Robinson or SRK.' if sev == Severity.ERROR else ''}"))

    return issues


def _check_unit(unit: dict, pkg: str, cls: dict,
                compounds_lower: list[str]) -> list[PhysicsIssue]:
    issues = []
    tag  = unit.get("tag", "<unnamed>")
    utype = unit.get("type", "")
    profile = _PKG_PROFILES.get(pkg, {})
    req = _UNIT_THERMO_REQUIREMENTS.get(utype, {})

    # Compressor/Expander strongly prefer cubic EOS
    if utype in ("Compressor", "Expander"):
        if pkg in ("Raoult's Law", "NRTL", "UNIQUAC"):
            issues.append(PhysicsIssue(
                Severity.WARNING, f"unit:{tag}", "COMPRESSOR_NEEDS_EOS",
                f"Compressor/Expander '{tag}' uses '{pkg}'. Cubic EOS "
                "(Peng-Robinson or SRK) gives more accurate compressibility "
                "factors and departure functions for gas-phase equipment."))

    # Vessel with LLE pair but non-LLE package
    if utype == "Vessel" and not profile.get("handles_lle", False):
        lle = _find_lle_pairs(compounds_lower)
        if lle:
            issues.append(PhysicsIssue(
                Severity.ERROR, f"unit:{tag}", "VESSEL_LLE_INCAPABLE",
                f"Flash vessel '{tag}' may produce two liquid phases "
                f"({lle}) but '{pkg}' cannot model LLE. "
                "Use NRTL or UNIQUAC."))

    # High-pressure unit with activity model
    P_out = unit.get("P_out")
    if P_out and not profile.get("handles_high_pressure", False):
        if P_out > profile.get("max_pressure_Pa", 1e9):
            issues.append(PhysicsIssue(
                Severity.ERROR, f"unit:{tag}", "HIGH_P_ACTIVITY_MODEL",
                f"Unit '{tag}' outlet at {P_out/1e5:.1f} bar but '{pkg}' "
                "is not suitable for high-pressure VLE. Use PR or SRK."))

    # Package requires binary parameters — flag for rare compound pairs
    if profile.get("requires_binary_params") and cls["unknown"]:
        issues.append(PhysicsIssue(
            Severity.WARNING, f"unit:{tag}", "BINARY_PARAMS_UNVERIFIED",
            f"'{pkg}' requires binary interaction parameters. Unknown "
            f"compounds {cls['unknown']} may lack parameters in DWSIM's "
            "database — verify or provide regressed parameters."))

    return issues


# ── Pair-matching helpers ──────────────────────────────────────────────────────

def _find_azeotropic_pairs(compounds_lower: list[str]) -> list[str]:
    found = []
    for pair in _AZEOTROPES:
        pair_list = list(pair)
        if (any(_fuzzy(pair_list[0], c) for c in compounds_lower) and
                any(_fuzzy(pair_list[1], c) for c in compounds_lower)):
            found.append(f"({pair_list[0]}/{pair_list[1]})")
    return found


def _find_lle_pairs(compounds_lower: list[str]) -> list[str]:
    found = []
    for pair in _LLE_PAIRS:
        pair_list = list(pair)
        if (any(_fuzzy(pair_list[0], c) for c in compounds_lower) and
                any(_fuzzy(pair_list[1], c) for c in compounds_lower)):
            found.append(f"({pair_list[0]}/{pair_list[1]})")
    return found


def _in_set(compound: str, compound_set: set[str]) -> bool:
    return any(_fuzzy(compound, s) for s in compound_set)


def _fuzzy(a: str, b: str) -> bool:
    """Case-insensitive exact match. Substring matching caused false positives
    (e.g. 'methane' inside 'dimethylamine'). Compound normalisation is the
    Basis Agent's job — physics_check assumes standardised DWSIM names."""
    return a.lower().strip() == b.lower().strip()
