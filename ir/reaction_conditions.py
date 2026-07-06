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
from typing import Tuple

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
