"""
Shared chemical data for zero-LLM rule-based decisions.

Imported by ThermoAgent (package selection) and ConditionAgent (condition
estimation). No agent imports are allowed here — this module is a leaf.
"""
from __future__ import annotations
import itertools
import math

# ── Normal boiling points at 1 atm (K) ───────────────────────────────────────
# Covers all DWSIM compound names used in the benchmark suite plus common aliases.

NBP_K: dict[str, float] = {
    # Water
    "Water":                373.15,
    # Alcohols
    "Methanol":             337.85,
    "Ethanol":              351.44,
    "1-Propanol":           370.35,
    "n-Propanol":           370.35,
    "2-Propanol":           355.39,
    "Isopropanol":          355.39,
    "1-Butanol":            390.81,
    "n-Butanol":            390.81,
    "2-Butanol":            372.66,
    "Isobutanol":           381.04,
    "tert-Butanol":         355.57,
    "1-Pentanol":           411.16,
    "Ethylene Glycol":      470.45,
    "Glycerol":             563.15,
    # Ketones
    "Acetone":              329.15,
    "Methyl Ethyl Ketone":  352.79,
    "Cyclohexanone":        428.58,
    "Methyl Isobutyl Ketone": 389.65,
    "Acetophenone":         475.15,
    # Esters
    "Ethyl Acetate":        350.26,
    "Methyl Acetate":       330.09,
    "Butyl Acetate":        399.15,
    "Isopropyl Acetate":    361.65,
    # Ethers
    "Diethyl Ether":        307.58,
    "MTBE":                 328.35,
    "Tetrahydrofuran":      339.12,
    "1,4-Dioxane":          374.47,
    "Diisopropyl Ether":    341.45,
    # Chlorinated
    "Chloroform":           334.35,
    "Dichloromethane":      312.95,
    "Carbon Tetrachloride": 349.85,
    "1,2-Dichloroethane":   356.65,
    "Chlorobenzene":        404.87,
    # Aromatics
    "Benzene":              353.25,
    "Toluene":              383.78,
    "o-Xylene":             417.58,
    "m-Xylene":             412.27,
    "p-Xylene":             411.51,
    "Ethylbenzene":         409.35,
    "Styrene":              418.31,
    "Naphthalene":          491.16,
    # Alkanes (ambient liquid)
    "n-Pentane":            309.22,
    "n-Hexane":             341.88,
    "n-Heptane":            371.58,
    "n-Octane":             398.82,
    "Cyclohexane":          353.87,
    "Methylcyclohexane":    374.08,
    # Light gases / cryogenic
    "Methane":              111.66,
    "Ethane":               184.55,
    "Propane":              231.11,
    "n-Butane":             272.65,
    "i-Butane":             261.43,
    "Isobutane":            261.43,
    "Nitrogen":              77.36,
    "Oxygen":                90.19,
    "Carbon Dioxide":       194.65,
    "Hydrogen Sulfide":     212.84,
    "Hydrogen":              20.28,
    "Argon":                 87.30,
    "Ammonia":              239.82,
    # Polar other
    "Acetic Acid":          391.15,
    "Formic Acid":          373.65,
    "Acetonitrile":         354.75,
    "DMSO":                 462.15,
}

# ── Compound classes (DWSIM name → class string) ─────────────────────────────
# "light_gas" compounds are always best handled by Peng-Robinson regardless of P.
# "alkane" are ambient-liquid alkanes (C5+) where Raoult's Law is valid for
# all-alkane mixtures.

COMPOUND_CLASS: dict[str, str] = {
    "Water":                "water",
    # Alcohols
    "Methanol":             "alcohol",
    "Ethanol":              "alcohol",
    "1-Propanol":           "alcohol",
    "n-Propanol":           "alcohol",
    "2-Propanol":           "alcohol",
    "Isopropanol":          "alcohol",
    "1-Butanol":            "alcohol",
    "n-Butanol":            "alcohol",
    "2-Butanol":            "alcohol",
    "Isobutanol":           "alcohol",
    "tert-Butanol":         "alcohol",
    "1-Pentanol":           "alcohol",
    "Ethylene Glycol":      "alcohol",
    "Glycerol":             "alcohol",
    # Ketones
    "Acetone":              "ketone",
    "Methyl Ethyl Ketone":  "ketone",
    "Cyclohexanone":        "ketone",
    "Methyl Isobutyl Ketone": "ketone",
    "Acetophenone":         "ketone",
    # Esters
    "Ethyl Acetate":        "ester",
    "Methyl Acetate":       "ester",
    "Butyl Acetate":        "ester",
    "Isopropyl Acetate":    "ester",
    # Ethers
    "Diethyl Ether":        "ether",
    "MTBE":                 "ether",
    "Tetrahydrofuran":      "ether",
    "1,4-Dioxane":          "ether",
    "Diisopropyl Ether":    "ether",
    # Chlorinated
    "Chloroform":           "chlorinated",
    "Dichloromethane":      "chlorinated",
    "Carbon Tetrachloride": "chlorinated",
    "1,2-Dichloroethane":   "chlorinated",
    "Chlorobenzene":        "chlorinated",
    # Aromatics
    "Benzene":              "aromatic",
    "Toluene":              "aromatic",
    "o-Xylene":             "aromatic",
    "m-Xylene":             "aromatic",
    "p-Xylene":             "aromatic",
    "Ethylbenzene":         "aromatic",
    "Styrene":              "aromatic",
    "Naphthalene":          "aromatic",
    # Alkanes (ambient-liquid, C5+) — Raoult's Law valid for all-alkane mixtures
    "n-Pentane":            "alkane",
    "n-Hexane":             "alkane",
    "n-Heptane":            "alkane",
    "n-Octane":             "alkane",
    "Cyclohexane":          "alkane",
    "Methylcyclohexane":    "alkane",
    # Light gases / low-boiling alkanes — always EOS
    "Methane":              "light_gas",
    "Ethane":               "light_gas",
    "Propane":              "light_gas",
    "n-Butane":             "light_gas",
    "i-Butane":             "light_gas",
    "Isobutane":            "light_gas",
    "Nitrogen":             "light_gas",
    "Oxygen":               "light_gas",
    "Carbon Dioxide":       "light_gas",
    "Hydrogen Sulfide":     "light_gas",
    "Hydrogen":             "light_gas",
    "Argon":                "light_gas",
    # Polar other
    "Acetic Acid":          "polar_other",
    "Formic Acid":          "polar_other",
    "Acetonitrile":         "polar_other",
    "DMSO":                 "polar_other",
    "Ammonia":              "polar_other",
}

# ── Known azeotropes — any pair in this set REQUIRES NRTL / UNIQUAC ──────────
KNOWN_AZEOTROPE_PAIRS: frozenset[frozenset] = frozenset({
    frozenset(["Ethanol",        "Water"]),
    frozenset(["Methanol",       "Water"]),
    frozenset(["1-Propanol",     "Water"]),
    frozenset(["n-Propanol",     "Water"]),
    frozenset(["2-Propanol",     "Water"]),
    frozenset(["Isopropanol",    "Water"]),
    frozenset(["1-Butanol",      "Water"]),
    frozenset(["n-Butanol",      "Water"]),
    frozenset(["Ethyl Acetate",  "Ethanol"]),
    frozenset(["Ethyl Acetate",  "Water"]),
    frozenset(["Acetone",        "Chloroform"]),
    frozenset(["Acetone",        "Methanol"]),
    frozenset(["Diethyl Ether",  "Water"]),
    frozenset(["n-Hexane",       "Ethanol"]),
    frozenset(["Benzene",        "Cyclohexane"]),
    frozenset(["Tetrahydrofuran","Water"]),
    frozenset(["Acetonitrile",   "Water"]),
    frozenset(["Chloroform",     "Acetone"]),   # duplicate direction
})


# ── Bubble-point estimator (Raoult's Law / NBP linear interpolation) ─────────

def estimate_bubble_point(
        compounds:   list[str],
        composition: dict[str, float],
        pressure_pa: float = 101_325.0,
) -> float | None:
    """
    Estimate the mixture bubble point using linear mole-fraction interpolation
    of normal boiling points (Raoult's Law approximation).

    Returns None if any compound is missing from NBP_K or if pressure is outside
    the reliable range (factor-of-5 from 1 atm).  Only reliable for quick
    pre-flight estimates — not a substitute for rigorous VLE calculation.
    """
    if not (20_000 < pressure_pa < 600_000):
        return None
    total = sum(composition.values())
    if total <= 0:
        return None
    for c in compounds:
        if c not in NBP_K:
            return None
    t_bub = sum((composition.get(c, 0.0) / total) * NBP_K[c] for c in compounds)
    # First-order Clausius-Clapeyron pressure correction
    if abs(pressure_pa - 101_325.0) > 5_000:
        lnP = math.log(pressure_pa / 101_325.0)
        dHvap = 88.0 * t_bub       # Trouton's rule
        t_bub = t_bub / (1.0 - 8.314 * t_bub * lnP / dHvap)
    return round(t_bub, 1)


# ── Rule-based package selection ─────────────────────────────────────────────

def rule_based_package_select(
        compounds: list[str],
) -> tuple[str, str] | None:
    """
    Apply compound-class hard rules to select a property package.

    Returns (package_name, one_sentence_reasoning) when the rules
    unambiguously determine the package, or None when the compound set
    contains unknown compounds or the rules are genuinely ambiguous
    (callers should fall through to an LLM in that case).

    Rules mirror the HARD RULES section of ThermoAgent._SYSTEM exactly so
    that this function and the LLM prompt agree.
    """
    classes = {c: COMPOUND_CLASS.get(c) for c in compounds}

    # Unknown compound → can't apply hard rules
    if any(v is None for v in classes.values()):
        return None

    compound_set = frozenset(compounds)
    class_set    = set(classes.values())

    # ── Single component ──────────────────────────────────────────────────────
    if len(compounds) == 1:
        c   = compounds[0]
        cls = classes[c]
        if cls == "light_gas":
            return ("Peng-Robinson",
                    f"{c} is a light gas/alkane — Peng-Robinson EOS required.")
        return ("Raoult's Law",
                f"Single component ({c}) — Raoult's Law (trivial).")

    # ── Known azeotropes → NRTL ───────────────────────────────────────────────
    for a, b in itertools.combinations(compounds, 2):
        if frozenset([a, b]) in KNOWN_AZEOTROPE_PAIRS:
            return ("NRTL",
                    f"{a}/{b} forms a known azeotrope — NRTL required "
                    "(HARD RULE: azeotropic system).")

    # ── All chemically similar ambient-liquid alkanes → Raoult's Law ─────────
    # Check BEFORE the EOS branch so C5+ alkanes (n-hexane, n-heptane, …) are
    # not accidentally routed to Peng-Robinson.
    if class_set == {"alkane"}:
        return ("Raoult's Law",
                "All compounds are similar alkanes (C5+) — Raoult's Law valid.")

    # ── Light gases / mixed light-gas+alkane → Peng-Robinson ─────────────────
    if class_set.issubset({"light_gas"}):
        return ("Peng-Robinson",
                "All compounds are light gases — cubic EOS required.")
    if class_set.issubset({"light_gas", "alkane"}) and "light_gas" in class_set:
        return ("Peng-Robinson",
                "Light hydrocarbons (includes gases) — cubic EOS required.")
    if class_set == {"aromatic"}:
        return ("Raoult's Law",
                "All compounds are similar aromatics — Raoult's Law valid.")

    # ── Forbidden Raoult's Law pairs → NRTL ──────────────────────────────────
    for c, cls in classes.items():
        other_classes = {classes[c2] for c2 in compounds if c2 != c}

        if cls == "alcohol":
            forbidden = other_classes & {
                "water", "alkane", "aromatic", "ester", "ether", "chlorinated"}
            if forbidden:
                return ("NRTL",
                        f"{c} (alcohol) mixed with {sorted(forbidden)[0]} — "
                        "Raoult's Law forbidden (HARD RULE).")

        if cls == "ketone":
            forbidden = other_classes & {"water", "chlorinated"}
            if forbidden:
                return ("NRTL",
                        f"{c} (ketone) mixed with {sorted(forbidden)[0]} — "
                        "non-ideal VLE.")

        if cls == "ester":
            forbidden = other_classes & {"water", "alcohol", "alkane", "aromatic"}
            if forbidden:
                return ("NRTL",
                        f"{c} (ester) — non-ideal pair detected.")

        if cls == "ether":
            forbidden = other_classes & {"water", "alcohol"}
            if forbidden:
                return ("NRTL",
                        f"{c} (ether) mixed with {sorted(forbidden)[0]} — "
                        "non-ideal VLE.")

        if cls == "chlorinated":
            forbidden = other_classes & {"ketone", "ether", "aromatic"}
            if forbidden:
                return ("NRTL",
                        f"{c} (chlorinated) — non-ideal interactions.")

        if cls == "polar_other":
            return ("NRTL",
                    f"{c} is a strongly polar compound — NRTL required.")

    # ── Any polar compound with water ─────────────────────────────────────────
    if "water" in class_set and len(class_set) > 1:
        return ("NRTL",
                "Water present in a mixture — non-ideal VLE (NRTL).")

    # ── Aromatic + alkane (near-ideal but different classes) → Raoult's Law ──
    if class_set.issubset({"aromatic", "alkane"}):
        return ("Raoult's Law",
                "Non-polar aromatic/alkane mixture — Raoult's Law acceptable.")

    # ── Ambiguous / mixed non-polar → Raoult's Law fallback ──────────────────
    if not (class_set & {"alcohol", "ketone", "ester", "ether",
                          "chlorinated", "polar_other", "water"}):
        return ("Raoult's Law",
                "No polar or hydrogen-bonding compounds — Raoult's Law.")

    # ── Could not determine unambiguously → fall through to LLM ──────────────
    return None
