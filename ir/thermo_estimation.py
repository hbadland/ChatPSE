"""
Thermodynamic estimation — Antoine equation, iterative Raoult's Law, bubble/boiling point.

Public API:
    boiling_point_K(compound, pressure_pa)   — single compound, Antoine-first
    bubble_point_K(compounds, pressure_pa)   — mixture (azeotrope > Antoine > NBP fallback)
"""
from __future__ import annotations

import math
from typing import Optional

# ── Antoine coefficients ───────────────────────────────────────────────────────
# Form: log10(P* [mmHg]) = A - B / (C + T [°C])
# Source: Perry's Chemical Engineers' Handbook 8th ed. / NIST WebBook
_ANTOINE: dict[str, tuple[float, float, float]] = {
    "Water":             (8.07131,  1730.63,  233.426),
    "Methanol":          (7.87863,  1473.11,  230.000),
    "Ethanol":           (8.04494,  1554.30,  222.650),
    "n-Propanol":        (8.37895,  1788.02,  227.438),
    "1-Propanol":        (8.37895,  1788.02,  227.438),
    "Isopropanol":       (8.11778,  1580.92,  219.617),
    "2-Propanol":        (8.11778,  1580.92,  219.617),
    "n-Butanol":         (7.83637,  1558.19,  196.881),
    "1-Butanol":         (7.83637,  1558.19,  196.881),
    "Acetone":           (7.11714,  1210.60,  229.664),
    "Benzene":           (6.90565,  1211.03,  220.790),
    "Toluene":           (6.95334,  1343.94,  219.377),
    "Cyclohexane":       (6.84498,  1203.526, 222.863),
    "o-Xylene":          (6.99891,  1474.680, 213.690),
    "m-Xylene":          (7.00908,  1462.266, 215.105),
    "p-Xylene":          (6.98760,  1451.795, 215.118),
    "Xylene":            (7.00908,  1462.266, 215.105),  # generic: m-Xylene representative
    "Ethylbenzene":      (6.95747,  1424.255, 213.206),
    "Styrene":           (7.14016,  1574.510, 224.660),
    "n-Hexane":          (6.87601,  1171.17,  224.408),
    "n-Heptane":         (6.89385,  1264.55,  216.640),
    "n-Octane":          (6.91868,  1351.756, 209.155),
    "Methane":           (6.61184,   389.93,  266.000),
    "Ethane":            (6.80896,   656.40,  256.000),
    "Ethylene":          (6.74756,   585.000, 255.000),
    "Ethene":            (6.74756,   585.000, 255.000),
    "Propane":           (6.80338,   803.81,  246.990),
    "Propylene":         (6.82173,   785.000, 247.000),
    "Propene":           (6.82173,   785.000, 247.000),
    "1-Butene":          (6.84290,   926.100, 240.000),
    "Isobutylene":       (6.84290,   923.200, 240.000),
    "2-Methylpropene":   (6.84290,   923.200, 240.000),
    "n-Butane":          (6.80896,   935.86,  238.730),
    "i-Butane":          (6.74808,   882.80,  240.000),
    "Isobutane":         (6.74808,   882.80,  240.000),
    "n-Pentane":         (6.85221,  1064.63,  232.000),
    "Carbon Dioxide":    (6.81228,  1301.68,   -3.494),
    "Chloroform":        (6.95465,  1170.97,  226.232),
    "Dichloromethane":   (7.09915,  1138.91,  231.000),
    "Ethyl Acetate":     (7.10179,  1244.95,  217.880),
    "Methyl Acetate":    (7.06524,  1157.630, 219.726),
    "Diethyl Ether":     (6.92032,  1064.070, 228.800),
    "Methyl Ethyl Ketone": (7.06356, 1261.34,  221.970),
    "2-Butanone":          (7.06356, 1261.34,  221.970),
    "Acetonitrile":      (7.11988,  1285.70,  224.365),
    "Acetic Acid":       (7.38782,  1533.313, 222.309),
    "Formic Acid":       (6.94459,  1295.270, 218.000),
    "Ethylene Glycol":   (8.09083,  2088.936, 203.454),
    "Ammonia":           (7.55466,  1002.71,  247.885),
    "Hydrogen Sulfide":  (6.99392,   867.81,  240.000),
    "Nitrogen":          (6.49457,   255.68,  266.550),
    "Oxygen":            (6.69144,   319.01,  266.700),
    "Hydrogen":          (5.82320,   122.92,  -20.000),
}

# Normal boiling points (K) at 1 atm — fallback when Antoine unavailable
_NBP_K: dict[str, float] = {
    "Methanol": 337.85, "Ethanol": 351.44, "n-Propanol": 370.35,
    "1-Propanol": 370.35, "Isopropanol": 355.39, "2-Propanol": 355.39,
    "n-Butanol": 390.81, "1-Butanol": 390.81, "Water": 373.15,
    "Acetone": 329.15, "Benzene": 353.25, "Toluene": 383.78,
    "Cyclohexane": 353.87,
    "o-Xylene": 417.6, "m-Xylene": 412.3, "p-Xylene": 411.5, "Xylene": 412.3,
    "n-Hexane": 341.88, "n-Heptane": 371.58, "Methane": 111.66,
    "Ethane": 184.55, "Ethylene": 169.38, "Ethene": 169.38,
    "Propane": 231.11, "Propylene": 225.46, "Propene": 225.46, "n-Butane": 272.65,
    "i-Butane": 261.43, "Isobutane": 261.43, "n-Pentane": 309.21,
    "Carbon Dioxide": 194.65, "Chloroform": 334.35,
    "Dichloromethane": 312.95, "Ethyl Acetate": 350.26,
    "Methyl Ethyl Ketone": 352.74, "2-Butanone": 352.74,
    "Acetonitrile": 354.75, "Nitrogen": 77.36, "Oxygen": 90.19,
    "Hydrogen": 20.27, "Ammonia": 239.72, "Hydrogen Sulfide": 212.85,
    # Aromatics
    "Ethylbenzene": 409.36, "Styrene": 418.31,
    "Cumene": 425.57, "Isopropylbenzene": 425.57,
    # Alkanes
    "n-Octane": 398.83,
    # Olefins
    "1-Butene": 266.89, "Isobutylene": 266.25, "2-Methylpropene": 266.25,
    # Esters / Ethers
    "Methyl Acetate": 330.02,
    "Diethyl Ether": 307.65,
    "Dimethyl Ether": 248.29,
    # Acids / Aldehydes
    "Acetic Acid": 391.15, "Ethanoic Acid": 391.15,
    "Formic Acid": 373.75, "Methanoic Acid": 373.75,
    "Acetaldehyde": 293.55, "Ethanal": 293.55,
    # Alcohols / Glycols
    "Isobutanol": 381.04, "2-Methyl-1-propanol": 381.04,
    "Ethylene Glycol": 470.45,
    # Ketones
    "Methyl Isobutyl Ketone": 389.65, "4-Methylpentan-2-one": 389.65,
}

# Minimum azeotropic boiling points (K) at 1 atm for known binary pairs.
# These take priority over Raoult's Law for strongly non-ideal systems.
_AZEOTROPE_BP: dict[frozenset, float] = {
    frozenset({"Ethanol", "Water"}):         351.3,
    frozenset({"Methanol", "Water"}):        337.9,
    frozenset({"Isopropanol", "Water"}):     353.4,
    frozenset({"2-Propanol", "Water"}):      353.4,
    frozenset({"n-Propanol", "Water"}):      360.2,
    frozenset({"1-Propanol", "Water"}):      360.2,
    frozenset({"n-Butanol", "Water"}):       365.5,
    frozenset({"1-Butanol", "Water"}):       365.5,
    frozenset({"Acetone", "Methanol"}):      328.7,
    frozenset({"Ethyl Acetate", "Ethanol"}): 345.1,
    frozenset({"Ethyl Acetate", "Water"}):   343.7,
    frozenset({"Acetonitrile", "Water"}):    349.9,
}


# ── Public API ─────────────────────────────────────────────────────────────────

# ── Case-insensitive compound name resolution ──────────────────────────────────
# BasisAgent/DWSIM may produce names with different capitalisation (e.g.
# "methyl ethyl ketone" vs "Methyl Ethyl Ketone").  Build a lowercase→canonical
# index at import time so all lookups are case-insensitive.
_LOWER_TO_CANONICAL: dict[str, str] = {
    name.lower(): name
    for name in (*_ANTOINE, *_NBP_K)
}


# Short acronym aliases not derivable from lowercasing the canonical name.
_LOWER_TO_CANONICAL.update({
    "ipa":  "Isopropanol",
    "dme":  "Dimethyl Ether",
    "dee":  "Diethyl Ether",
    "mek":  "Methyl Ethyl Ketone",
    "mibk": "Methyl Isobutyl Ketone",
    "co2":  "Carbon Dioxide",
})


def _resolve(name: str) -> str:
    """Return the canonical compound name stored in _ANTOINE/_NBP_K."""
    return _LOWER_TO_CANONICAL.get(name.lower(), name)


# ── Module-level caches ────────────────────────────────────────────────────────
# Key for bubble_point_K: (sorted_compounds_tuple, pressure_bucket)
# Pressure is bucketed to nearest 1000 Pa to reduce cache misses from float noise.
_BP_CACHE: dict[tuple, Optional[float]] = {}
_BOILING_CACHE: dict[tuple, Optional[float]] = {}


def boiling_point_K(
    compound:    str,
    pressure_pa: float = 101_325.0,
) -> Optional[float]:
    """
    Single-compound boiling point at the given pressure (cached).
    Antoine equation first; falls back to NBP + Clausius-Clapeyron.
    """
    compound = _resolve(compound)
    key = (compound, round(pressure_pa / 1000) * 1000)
    if key in _BOILING_CACHE:
        return _BOILING_CACHE[key]
    if compound in _ANTOINE:
        result = _antoine_boiling_K(*_ANTOINE[compound], pressure_pa)
    elif compound in _NBP_K:
        result = _clausius_clapeyron(_NBP_K[compound], pressure_pa)
    else:
        result = None
    _BOILING_CACHE[key] = result
    return result


def bubble_point_K(
    compounds:   list[str],
    pressure_pa: float = 101_325.0,
) -> Optional[float]:
    """
    Mixture bubble point (K) — result is cached by (sorted compounds, pressure bucket).

    Priority:
      1. Azeotrope table  — known non-ideal binary pairs
      2. Antoine + Raoult — iterative solver, equal-mole assumption
      3. NBP average      — simple average with Clausius-Clapeyron
      4. None             — insufficient data
    """
    if not compounds:
        return None

    resolved = [_resolve(c) for c in compounds]
    key = (tuple(sorted(resolved)), round(pressure_pa / 1000) * 1000)
    if key in _BP_CACHE:
        return _BP_CACHE[key]

    comp_set = frozenset(resolved)

    # 1. Azeotrope table
    result: Optional[float] = None
    for pair, azeo_bp in _AZEOTROPE_BP.items():
        if pair.issubset(comp_set):
            result = _clausius_clapeyron(azeo_bp, pressure_pa)
            break

    # 2. Antoine iterative Raoult's Law
    if result is None:
        result = _raoult_bubble_point(resolved, pressure_pa)

    # 3. NBP arithmetic average
    if result is None:
        nbps = [_NBP_K[c] for c in resolved if c in _NBP_K]
        if len(nbps) == len(resolved):
            result = _clausius_clapeyron(sum(nbps) / len(nbps), pressure_pa)

    _BP_CACHE[key] = result
    return result


def clear_cache() -> None:
    """Clear all thermodynamic estimate caches (useful between benchmark runs)."""
    _BP_CACHE.clear()
    _BOILING_CACHE.clear()


# ── Private helpers ────────────────────────────────────────────────────────────

def _raoult_bubble_point(
    compounds:   list[str],
    pressure_pa: float,
    max_iter:    int   = 60,
    tol_pa:      float = 100.0,
) -> Optional[float]:
    """
    Iterative bubble-point solve via Antoine + Raoult: sum_i(xi * P*_i(T)) = P.
    Assumes equimolar feed. Newton's method for rapid convergence.
    """
    compounds = [_resolve(c) for c in compounds]
    if not all(c in _ANTOINE for c in compounds):
        return None

    n    = len(compounds)
    xi   = 1.0 / n
    coef = [_ANTOINE[c] for c in compounds]

    # Initial guess: average individual boiling points
    T = sum(
        _antoine_boiling_K(A, B, C, pressure_pa) or 373.15
        for A, B, C in coef
    ) / n

    for _ in range(max_iter):
        p_stars = [_pstar_pa(A, B, C, T) for A, B, C in coef]
        total   = xi * sum(p_stars)
        err     = total - pressure_pa

        # d(total)/dT from Antoine: sum xi * dP*/dT
        dpdT = xi * sum(
            math.log(10.0) * (B / (C + T - 273.15) ** 2) * p
            for (A, B, C), p in zip(coef, p_stars)
        )
        if abs(dpdT) < 1e-12:
            break
        T -= err / dpdT
        if abs(err) < tol_pa:
            break

    return round(T, 2)


def _antoine_boiling_K(
    A: float, B: float, C: float, pressure_pa: float
) -> Optional[float]:
    p_mmhg = pressure_pa / 133.322
    denom  = A - math.log10(max(p_mmhg, 1e-6))
    if abs(denom) < 1e-9:
        return None
    return round(B / denom - C + 273.15, 2)


def _pstar_pa(A: float, B: float, C: float, T_K: float) -> float:
    """Vapour pressure in Pa at temperature T_K via Antoine equation."""
    return 133.322 * 10.0 ** (A - B / (C + T_K - 273.15))


def _clausius_clapeyron(nbp_K: float, pressure_pa: float) -> float:
    """Clausius-Clapeyron pressure correction applied to a normal boiling point."""
    if abs(pressure_pa - 101_325.0) < 5_000:
        return round(nbp_K, 1)
    ln_p  = math.log(pressure_pa / 101_325.0)
    dHvap = 88.0 * nbp_K          # Trouton's rule: ΔHvap ≈ 88 R Tb
    denom = 1.0 - (8.314 * nbp_K * ln_p) / dHvap
    if denom > 0:
        return round(nbp_K / denom, 1)
    return round(nbp_K, 1)
