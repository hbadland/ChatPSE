"""
Domain-informed default operating temperatures for reactors when the process
description specifies none (it names the reaction but no temperature).

A single-pass conversion reactor with no temperature spec otherwise defaults to a
crude feed_T+300 K (≈325 °C) regardless of chemistry — wrong for e.g. steam
methane reforming (~800 °C) — and DWSIM then solves it adiabatically, collapsing
the whole downstream train to an absurd temperature.  This table supplies a
physically-defensible operating temperature keyed on the reaction TYPE (matched
against the reaction stoichiometry string and the unit's role text).

Configurable: edit REACTION_TEMPERATURE_DEFAULTS.  Each entry carries a cited
basis, not a magic number.  Values are mid-range representative operating points.
"""
from __future__ import annotations
import re
from typing import Tuple

from agents.compound_normalize import canonicalize_compound


# ── Stage 3b: stoichiometry-based reaction-type classification ─────────────────
# Primary signal: the reactant/product COMPOUND SETS parsed from the stoichiometry
# string (robust to coefficients, ordering, and name casing — compounds are
# canonicalised to DWSIM keys).  A signature matches when its reactant set AND
# product set are both subsets of the parsed reaction's sets (extra species OK).
# Role text is only a secondary/fallback signal.

def _canon(name: str) -> str:
    return canonicalize_compound(name)[0]

# (reaction_type, required reactant compounds, required product compounds)
REACTION_SIGNATURES: list[tuple[str, frozenset, frozenset]] = [
    ("steam_methane_reforming",
     frozenset({_canon("Methane"), _canon("Water")}),
     frozenset({_canon("Carbon monoxide"), _canon("Hydrogen")})),
    ("dry_methane_reforming",
     frozenset({_canon("Methane"), _canon("Carbon dioxide")}),
     frozenset({_canon("Carbon monoxide"), _canon("Hydrogen")})),
    ("water_gas_shift",
     frozenset({_canon("Carbon monoxide"), _canon("Water")}),
     frozenset({_canon("Carbon dioxide"), _canon("Hydrogen")})),
    ("reverse_water_gas_shift",
     frozenset({_canon("Carbon dioxide"), _canon("Hydrogen")}),
     frozenset({_canon("Carbon monoxide"), _canon("Water")})),
    ("co_methanation",
     frozenset({_canon("Carbon monoxide"), _canon("Hydrogen")}),
     frozenset({_canon("Methane"), _canon("Water")})),
    ("co2_methanation",
     frozenset({_canon("Carbon dioxide"), _canon("Hydrogen")}),
     frozenset({_canon("Methane"), _canon("Water")})),
    ("ammonia_synthesis",
     frozenset({_canon("Nitrogen"), _canon("Hydrogen")}),
     frozenset({_canon("Ammonia")})),
    # Chlorination and EDC pyrolysis have distinct reactant sets, so classifying
    # them by stoichiometry (not the shared text 'dichloroethane') keeps the
    # pyrolysis reactor from being mislabelled as chlorination.
    ("chlorination",
     frozenset({_canon("Ethylene"), _canon("Chlorine")}),
     frozenset({_canon("1,2-dichloroethane")})),
    ("edc_pyrolysis",
     frozenset({_canon("1,2-dichloroethane")}),
     frozenset({_canon("Vinyl chloride"), _canon("Hydrogen chloride")})),
    ("hydrodealkylation",
     frozenset({_canon("Toluene"), _canon("Hydrogen")}),
     frozenset({_canon("Benzene"), _canon("Methane")})),
]

# Secondary signal — role/reaction text keywords → reaction_type (fallback only).
_ROLE_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("steam_methane_reforming", ("steam methane reform", "steam reform", "reform", "smr")),
    ("dry_methane_reforming",   ("dry reform",)),
    ("water_gas_shift",         ("water-gas shift", "water gas shift", "wgs", "shift")),
    ("co_methanation",          ("methanation", "methanate", "sabatier")),
    ("ammonia_synthesis",       ("ammonia synthesis", "haber")),
    ("hydrodealkylation",       ("hydrodealkylation", "dealkylation")),
    # EDC pyrolysis before chlorination; 'dichloroethane' dropped from chlorination
    # because it appears in BOTH reactions' strings (product vs reactant).
    ("edc_pyrolysis",           ("edc pyrolysis", "pyrolysis of edc", "vinyl chloride",
                                 "cracking of edc")),
    ("chlorination",            ("chlorination", "direct chlorination")),
    ("combustion",              ("combustion", "oxidation", "furnace", "burner")),
]


def _parse_reaction_sets(reaction: str) -> tuple[frozenset, frozenset]:
    """Parse 'A + 3 B -> C + D' into (reactant_set, product_set) of canonical names."""
    if not reaction:
        return frozenset(), frozenset()
    arrow = "->" if "->" in reaction else ("→" if "→" in reaction else None)
    if arrow is None:
        return frozenset(), frozenset()
    lhs, rhs = reaction.split(arrow, 1)

    def side(s: str) -> frozenset:
        out = set()
        for term in s.split("+"):
            t = re.sub(r"^\s*\d+\s+", "", term.strip())   # strip leading coefficient
            if t:
                out.add(_canon(t))
        return frozenset(out)

    return side(lhs), side(rhs)


# ── Stage 3c: reaction_type → target outlet temperature (approved table) ──────
# (target_temperature_K, plausible_range, cited_source)
TARGET_TEMPERATURE_BY_TYPE: dict[str, tuple[float, str, str]] = {
    "steam_methane_reforming": (1073.15, "750-900 C",
        "Rostrup-Nielsen, Catal. Sci. Technol. 5 (1984); reformer outlet 800-900 C"),
    "dry_methane_reforming":   (1123.15, "800-950 C",
        "Pakhare & Spivey, Chem. Soc. Rev. 43 (2014) 7813"),
    "water_gas_shift":         (623.15, "310-450 C (HTS, Fe-Cr)",
        "Newsome, Catal. Rev.-Sci. Eng. 21 (1980) 275 (HTS default)"),
    "water_gas_shift_lts":     (473.15, "190-250 C (LTS, Cu-Zn)",
        "Ratnasamy & Wagner, Catal. Rev. 51 (2009) 325"),
    "reverse_water_gas_shift": (873.15, "500-700 C",
        "Daza & Kuhn, RSC Adv. 6 (2016) 49675"),
    "co_methanation":          (573.15, "250-400 C (Ni)",
        "Kopyscinski et al., Fuel 89 (2010) 1763"),
    "co2_methanation":         (573.15, "250-400 C (Ni, Sabatier)",
        "Ronsch et al., Fuel 166 (2016) 276"),
    "ammonia_synthesis":       (723.15, "400-500 C (Fe)",
        "Appl, 'Ammonia', Ullmann's Encyclopedia of Industrial Chemistry"),
    "hydrodealkylation":       (923.15, "600-750 C",
        "thermal toluene HDA (Perry's Chemical Engineers' Handbook)"),
    "chlorination":            (363.15, "50-90 C (liquid phase)",
        "direct chlorination of ethylene to EDC (Ullmann's, Chlorinated Hydrocarbons)"),
    "edc_pyrolysis":           (773.15, "480-550 C",
        "EDC (1,2-dichloroethane) pyrolysis to vinyl chloride ~500 C "
        "(Ullmann's, Vinyl Chloride)"),
    "combustion":              (1273.15, ">=1000 C",
        "Perry's Chemical Engineers' Handbook (combustion/partial oxidation)"),
    "generic":                 (623.15, "moderate catalytic default",
        "unresolved reaction type - conservative moderate default"),
}

# WGS is HTS by default; select the LTS entry only on an explicit low-T marker.
_LTS_MARKERS = ("low-temperature", "low temperature", "lts", "cu-zn", "copper-zinc")


def template_temperature(reaction: str = "", role: str = "") -> dict:
    """
    Target outlet temperature for a reactor whose temperature is unspecified.
    Returns {temperature_K, reaction_type, range, basis}.  Uses the Stage-3b
    stoichiometry classifier; water_gas_shift defaults to HTS (623 K) unless the
    reaction/role text explicitly indicates low-temperature shift.
    """
    rtype = classify_reaction(reaction, role)["reaction_type"]
    if rtype == "water_gas_shift" and any(
            m in f" {reaction} {role} ".lower() for m in _LTS_MARKERS):
        rtype = "water_gas_shift_lts"
    T_K, rng, src = TARGET_TEMPERATURE_BY_TYPE.get(
        rtype, TARGET_TEMPERATURE_BY_TYPE["generic"])
    return {"temperature_K": T_K, "reaction_type": rtype, "range": rng, "basis": src}


def classify_reaction(reaction: str = "", role: str = "") -> dict:
    """
    Classify a reactor's reaction. Returns:
      {reaction_type, method('stoichiometry'|'role_text'|'fallback'),
       reactants:[...], products:[...]}
    Stoichiometry (compound sets) is authoritative; role text is a fallback.
    """
    R, P = _parse_reaction_sets(reaction)
    for rtype, sig_R, sig_P in REACTION_SIGNATURES:
        if sig_R <= R and sig_P <= P:
            return {"reaction_type": rtype, "method": "stoichiometry",
                    "reactants": sorted(R), "products": sorted(P)}
    text = f" {reaction} {role} ".lower()
    for rtype, kws in _ROLE_KEYWORDS:
        if any(k in text for k in kws):
            return {"reaction_type": rtype, "method": "role_text",
                    "reactants": sorted(R), "products": sorted(P)}
    return {"reaction_type": "generic", "method": "fallback",
            "reactants": sorted(R), "products": sorted(P)}


# (keyword aliases, representative T [K], basis)
REACTION_TEMPERATURE_DEFAULTS: list[tuple[tuple[str, ...], float, str]] = [
    (("steam methane reform", "steam reform", "methane reform", "dry reform",
      "reformer", "reforming", "smr"),
     1073.15,
     "Steam/dry methane reforming 750–900 °C (strongly endothermic, Ni catalyst); "
     "~800 °C representative — Rostrup-Nielsen, 'Catalytic Steam Reforming' (1984)."),

    (("water-gas shift", "water gas shift", "shift reactor", "wgs", " shift"),
     623.15,
     "High-temperature water-gas shift 300–450 °C over Fe-Cr; ~350 °C representative "
     "— Newsome, Catal. Rev.-Sci. Eng. 21 (1980) 275."),

    (("methanation", "methanate", "sabatier"),
     573.15,
     "CO/CO2 methanation 250–400 °C over Ni; ~300 °C representative."),

    (("ammonia synthesis", "haber", "haber-bosch"),
     723.15,
     "Ammonia synthesis 400–500 °C over promoted Fe; ~450 °C representative."),

    (("hydrodealkylation", "dealkylation", "dehydrogenation", "pyrolysis",
      "cracking", "thermal crack"),
     923.15,
     "Thermal dealkylation/cracking 600–750 °C; ~650 °C representative "
     "(e.g. toluene HDA)."),

    (("combustion", "oxidation", "burner", "furnace", "incinerat"),
     1273.15,
     "Combustion/partial oxidation ≳1000 °C; ~1000 °C representative."),

    (("direct chlorination", "chlorination", "dichloroethane", "edc"),
     363.15,
     "Liquid-phase direct chlorination of ethylene to EDC ~50–90 °C; "
     "~90 °C representative."),

    (("pyrolysis of edc", "vinyl chloride", "vcm", "cracking of edc"),
     773.15,
     "EDC pyrolysis to vinyl chloride 480–550 °C; ~500 °C representative."),
]

# Fallback when the reaction type cannot be identified: a generic catalytic /
# moderately-exothermic operating point (~350 °C), NOT feed_T-relative.
DEFAULT_REACTOR_TEMPERATURE_K = 623.15
DEFAULT_REACTOR_BASIS = ("generic reactor default (reaction type not identified); "
                         "~350 °C moderate catalytic operating point")


def default_reactor_temperature(reaction: str = "", role: str = "") -> Tuple[float, str]:
    """
    Return (temperature_K, basis) for a reactor whose temperature is unspecified,
    inferred from the reaction stoichiometry string and/or the unit role text.
    Falls back to DEFAULT_REACTOR_TEMPERATURE_K when the type is unknown.
    """
    text = f" {reaction} {role} ".lower()
    for keywords, T_K, basis in REACTION_TEMPERATURE_DEFAULTS:
        if any(k in text for k in keywords):
            return T_K, basis
    return DEFAULT_REACTOR_TEMPERATURE_K, DEFAULT_REACTOR_BASIS
