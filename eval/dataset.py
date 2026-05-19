"""
Dataset split management — single authoritative source for all partitions.

Split design (60 cases total):
  DEV     (42, 70%) — E + M + H + U01-U05 + AMB01-AMB05
                      Only set allowed for prompt tuning / scoring adjustments.
  HOLDOUT  (9, 15%) — U06-U08, AMB06, ADV01, ADV02, EDGE01, EDGE02, EDGE07
                      MUST NOT be used during development.
                      Run periodically to detect overfitting.
  STRESS   (9, 15%) — ADV03-ADV07, EDGE03-EDGE06
                      Intentionally difficult; used to analyse failure modes only.

FROZEN (12, subset across DEV + HOLDOUT):
  These cases may NEVER be edited or used for any tuning.
  They will form part of the final evaluation set for the paper.

All splits are determined by explicit ID sets — no random sampling —
so they are fully reproducible with no seed dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from eval.benchmark_cases import BenchmarkCase, BENCHMARK_CASES, CASES_BY_ID

# ── Split assignment (explicit, not random) ────────────────────────────────────

SPLIT_DEV      = "dev"
SPLIT_HOLDOUT  = "holdout"
SPLIT_STRESS   = "stress"
SplitName = Literal["dev", "holdout", "stress", "all"]

_DEV_IDS: frozenset[str] = frozenset({
    # Easy (all 10)
    "E01", "E02", "E03", "E04", "E05",
    "E06", "E07", "E08", "E09", "E10",
    # Medium (all 12)
    "M01", "M02", "M03", "M04", "M05", "M06",
    "M07", "M08", "M09", "M10", "M11", "M12",
    # Hard (all 10)
    "H01", "H02", "H03", "H04", "H05",
    "H06", "H07", "H08", "H09", "H10",
    # Underspecified (first 5)
    "U01", "U02", "U03", "U04", "U05",
    # Ambiguous (first 5)
    "AMB01", "AMB02", "AMB03", "AMB04", "AMB05",
})

_HOLDOUT_IDS: frozenset[str] = frozenset({
    "U06", "U07", "U08",        # remaining underspec
    "AMB06",                    # remaining ambiguous
    "ADV01", "ADV02",           # first 2 adversarial
    "EDGE01", "EDGE02", "EDGE07",  # 3 edge (varied archetypes)
})

_STRESS_IDS: frozenset[str] = frozenset({
    "ADV03", "ADV04", "ADV05", "ADV06", "ADV07",  # remaining adversarial
    "EDGE03", "EDGE04", "EDGE05", "EDGE06",        # remaining edge
})

# Sanity: assert disjoint and total = 60
assert len(_DEV_IDS & _HOLDOUT_IDS) == 0, "DEV ∩ HOLDOUT non-empty"
assert len(_DEV_IDS & _STRESS_IDS)  == 0, "DEV ∩ STRESS non-empty"
assert len(_HOLDOUT_IDS & _STRESS_IDS) == 0, "HOLDOUT ∩ STRESS non-empty"
assert len(_DEV_IDS) + len(_HOLDOUT_IDS) + len(_STRESS_IDS) == 60

# ── Frozen set (never edit, never use for tuning) ──────────────────────────────
# Chosen to span all difficulty tiers and thermodynamic archetypes.
# Must include at least one HOLDOUT case so the frozen set is never inadvertently
# used for development (a HOLDOUT case caught using it would be a violation).

_FROZEN_IDS: frozenset[str] = frozenset({
    # Easy — baseline sanity
    "E01",   # heat ethanol-water (NRTL, simplest polar case)
    "E05",   # benzene-toluene flash (PR, non-polar VLE)
    # Medium — BIP and multi-unit
    "M01",   # ethanol-water NRTL flash (BIP injection)
    "M07",   # compress-cool-flash propane
    # Hard — complex topology
    "H01",   # ternary acetone-methanol-water recycle
    "H03",   # cryogenic air (Lee-Kesler-Plöcker)
    # Underspec — incomplete-description handling
    "U04",   # "Flash separate the mixture" (minimal)
    # Ambiguous — multi-interpretation
    "AMB03", # "Separate components at low temperature"
    # Holdout cases in frozen (ensures at least one holdout is frozen)
    "U07",   # "Mix feeds and heat" (holdout, underspec)
    "AMB06", # "Recover solvent from acetone mixture" (holdout, ambiguous)
    "ADV01", # "Cool ethanol to 200°C" (holdout, adversarial)
    "EDGE01", # supercritical CO2 (holdout, edge)
})

# Frozen must be subset of DEV ∪ HOLDOUT (never from stress)
assert _FROZEN_IDS <= (_DEV_IDS | _HOLDOUT_IDS), "Frozen case in STRESS set"
assert len(_FROZEN_IDS) == 12


# ── Accessor functions ─────────────────────────────────────────────────────────

def get_cases(split: SplitName = "all") -> list[BenchmarkCase]:
    """Return cases for the requested split."""
    if split == "all":
        return list(BENCHMARK_CASES)
    id_set = {
        "dev":     _DEV_IDS,
        "holdout": _HOLDOUT_IDS,
        "stress":  _STRESS_IDS,
    }[split]
    return [c for c in BENCHMARK_CASES if c.case_id in id_set]


def get_frozen() -> list[BenchmarkCase]:
    """Return the frozen evaluation cases (never modify or optimise against)."""
    return [c for c in BENCHMARK_CASES if c.case_id in _FROZEN_IDS]


def split_of(case_id: str) -> str:
    """Return the split name for a given case ID."""
    if case_id in _DEV_IDS:
        return "dev"
    if case_id in _HOLDOUT_IDS:
        return "holdout"
    if case_id in _STRESS_IDS:
        return "stress"
    return "unknown"


def is_frozen(case_id: str) -> bool:
    return case_id in _FROZEN_IDS


def split_summary() -> dict:
    """Return a summary dict for reporting."""
    return {
        "dev":     {"n": len(_DEV_IDS),     "ids": sorted(_DEV_IDS)},
        "holdout": {"n": len(_HOLDOUT_IDS), "ids": sorted(_HOLDOUT_IDS)},
        "stress":  {"n": len(_STRESS_IDS),  "ids": sorted(_STRESS_IDS)},
        "frozen":  {"n": len(_FROZEN_IDS),  "ids": sorted(_FROZEN_IDS)},
        "total":   60,
    }
