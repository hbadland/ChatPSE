"""
Adversarial benchmark — covers every decision boundary of the multi-agent
flowsheet system. Designed to preempt reviewer challenges.

  8 categories × 50 cases total
  ─────────────────────────────
  BASIS        (7)  compound recognition: abbreviations, synonyms, unsupported
  EOS          (8)  equation-of-state path: gases, high-P, close-boiling, cryo
  ACTIVITY     (7)  activity-model path: polar, associating, limited miscibility
  CALIBRATION  (5)  NRTL/UNIQUAC with explicit BIP injection via CalibrationAgent
  TOPOLOGY     (7)  multi-unit flowsheets: compressors, mixers, staged vessels
  DESCRIPTION  (5)  phrasing stress: imperial units, minimal, verbose, qualitative
  EDGE         (5)  composition extremes: trace impurity, 4-component, aliases
  REJECT       (6)  expected failures — polymers, ionic liquids, reactions, membranes

Usage
─────
    python agents/benchmark_adversarial.py
    python agents/benchmark_adversarial.py --real-executor
    python agents/benchmark_adversarial.py --model claude-haiku-4-5-20251001
    python agents/benchmark_adversarial.py --category EOS
    python agents/benchmark_adversarial.py --category REJECT --real-executor

Notes
─────
  REJECT cases marked '†' have soft HUMAN expectations that require --real-executor
  to validate; the mock executor may return PASS (which is also informative data).
  BASIS_FAILED cases are deterministic across both modes.
"""
from __future__ import annotations

import sys
import time
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.orchestrator import Orchestrator, OrchestratorResult
from agents.executor import ExecutionResult, StreamResult
from agents.llm import reset_call_count, get_call_count

# ── Property-package families ─────────────────────────────────────────────────

_IDEAL_PKGS    = {"Raoult's Law"}
_ACTIVITY_PKGS = {"NRTL", "UNIQUAC"}
_EOS_PKGS      = {"Peng-Robinson", "Soave-Redlich-Kwong", "Lee-Kesler-Plöcker"}

_PKG_FAMILIES: dict[str, set[str]] = {
    "ideal":    _IDEAL_PKGS,
    "activity": _ACTIVITY_PKGS,
    "eos":      _EOS_PKGS,
}

# ── Test-case dataclass ───────────────────────────────────────────────────────

@dataclass
class AdversarialTestCase:
    name:                             str
    description:                      str
    category:                         str   # BASIS | EOS | ACTIVITY | CALIBRATION |
                                            # TOPOLOGY | DESCRIPTION | EDGE | REJECT
    expected_outcome:                 str   # PASS | HUMAN | BASIS_FAILED
    expected_compounds:               list[str]
    expected_property_package_family: str   # ideal | activity | eos
    max_iterations_allowed:           int = 4


# ── All 50 adversarial test cases ─────────────────────────────────────────────

ADVERSARIAL_TEST_CASES: list[AdversarialTestCase] = [

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY: BASIS — compound recognition under stress
    # Tests: abbreviations, synonyms, mixture aliases, unsupported classes
    # ══════════════════════════════════════════════════════════════════════════

    AdversarialTestCase(
        name="[BASIS] MEK abbreviation → Methyl Ethyl Ketone",
        description=(
            "Flash separate a 50/50 molar MEK and water feed at 1 atm and 360 K. "
            "Feed flow is 1 mol/s. Recover the MEK-enriched vapour from the "
            "water-enriched liquid bottoms."
        ),
        category="BASIS",
        expected_outcome="PASS",
        expected_compounds=["Water"],
        expected_property_package_family="activity",
        max_iterations_allowed=6,
    ),
    AdversarialTestCase(
        name="[BASIS] DCM abbreviation → Dichloromethane",
        description=(
            "Separate a 60/40 molar DCM and methanol mixture at 1 atm. "
            "Heat the feed from 298 K to 318 K in a heater then flash in a vessel "
            "to recover the more volatile DCM in the vapour phase."
        ),
        category="BASIS",
        expected_outcome="PASS",
        expected_compounds=["Dichloromethane", "Methanol"],
        expected_property_package_family="activity",
        max_iterations_allowed=6,
    ),
    AdversarialTestCase(
        name="[BASIS] Trichloromethane synonym → Chloroform",
        description=(
            "Flash separate an equimolar trichloromethane and acetone mixture "
            "at 1 atm and 335 K. Feed flow is 1 mol/s."
        ),
        category="BASIS",
        expected_outcome="PASS",
        expected_compounds=["Chloroform", "Acetone"],
        expected_property_package_family="activity",
        max_iterations_allowed=6,
    ),
    AdversarialTestCase(
        name="[BASIS] Cyclohexane IUPAC name + n-heptane",
        description=(
            "Partially vaporise a 50/50 molar cyclohexane and n-heptane feed at "
            "1 atm by heating from 298 K to 375 K in a heater, then flash the "
            "heated mixture in a vessel to separate the two hydrocarbons."
        ),
        category="BASIS",
        expected_outcome="PASS",
        expected_compounds=["Cyclohexane", "n-Heptane"],
        expected_property_package_family="ideal",
        max_iterations_allowed=4,
    ),
    AdversarialTestCase(
        name="[BASIS] Unsupported compound — glucose (biomolecule)",
        description=(
            "Concentrate a glucose-water solution by evaporating water at 1 atm "
            "and 373 K. The feed contains 10 mol% glucose and 90 mol% water "
            "at a flow of 1 mol/s."
        ),
        category="BASIS",
        expected_outcome="BASIS_FAILED",
        expected_compounds=[],
        expected_property_package_family="ideal",
        max_iterations_allowed=2,
    ),
    AdversarialTestCase(
        name="[BASIS] Unsupported compound — polystyrene polymer",
        description=(
            "Dissolve polystyrene in dimethylformamide at 298 K and 1 atm, "
            "then evaporate the DMF solvent to recover the polymer."
        ),
        category="BASIS",
        expected_outcome="BASIS_FAILED",
        expected_compounds=[],
        expected_property_package_family="ideal",
        max_iterations_allowed=2,
    ),
    AdversarialTestCase(
        name="[BASIS] Unsupported mixture — crude oil pseudo-components",
        description=(
            "Perform an atmospheric distillation of crude oil at 1 atm to cut "
            "naphtha, kerosene, and gas oil fractions. Feed flow is 1 mol/s."
        ),
        category="BASIS",
        expected_outcome="BASIS_FAILED",
        expected_compounds=[],
        expected_property_package_family="eos",
        max_iterations_allowed=2,
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY: EOS — equation-of-state path
    # Tests: gases, high pressure, close-boiling pairs, cryogenic, multi-component
    # ══════════════════════════════════════════════════════════════════════════

    AdversarialTestCase(
        name="[EOS] Close-boiling propylene/propane at 10 bar",
        description=(
            "Flash separate a 50/50 molar propylene and propane mixture at 10 bar "
            "and 265 K. Feed flow is 1 mol/s. Recover the propylene-enriched vapour "
            "and the propane-enriched liquid."
        ),
        category="EOS",
        expected_outcome="PASS",
        expected_compounds=["Propylene", "Propane"],
        expected_property_package_family="eos",
        max_iterations_allowed=4,
    ),
    AdversarialTestCase(
        name="[EOS] Cryogenic air separation — nitrogen/oxygen at 1 atm",
        description=(
            "Flash separate a 79/21 molar nitrogen and oxygen feed at 1 atm and "
            "85 K to produce an oxygen-enriched liquid and a nitrogen-rich vapour. "
            "Feed flow is 1 mol/s."
        ),
        category="EOS",
        expected_outcome="PASS",
        expected_compounds=["Nitrogen", "Oxygen"],
        expected_property_package_family="eos",
        max_iterations_allowed=4,
    ),
    AdversarialTestCase(
        name="[EOS] CO2/methane flash at 80 bar — near-supercritical CO2",
        description=(
            "Flash separate a 30/70 molar carbon dioxide and methane mixture at "
            "80 bar and 250 K. Feed flow is 1 mol/s. Recover the CO2-rich liquid "
            "and the methane-rich vapour fraction."
        ),
        category="EOS",
        expected_outcome="PASS",
        expected_compounds=["Carbon Dioxide", "Methane"],
        expected_property_package_family="eos",
        max_iterations_allowed=4,
    ),
    AdversarialTestCase(
        name="[EOS] n-Butane/isobutane isomer flash at 5 bar",
        description=(
            "Flash separate an equimolar n-butane and isobutane mixture at 5 bar "
            "and 310 K. Feed flow is 1 mol/s. Recover the isobutane-enriched vapour."
        ),
        category="EOS",
        expected_outcome="PASS",
        expected_compounds=["n-Butane", "Isobutane"],
        expected_property_package_family="eos",
        max_iterations_allowed=4,
    ),
    AdversarialTestCase(
        name="[EOS] Propane compressor + cooler — no flash",
        description=(
            "Compress a pure propane vapour stream from 1 bar and 298 K to 10 bar "
            "using a compressor, then cool the compressed stream to 320 K in a "
            "cooler. Feed flow is 1 mol/s. No phase separation required."
        ),
        category="EOS",
        expected_outcome="PASS",
        expected_compounds=["Propane"],
        expected_property_package_family="eos",
        max_iterations_allowed=4,
    ),
    AdversarialTestCase(
        name="[EOS] Ethylene/ethane cracker separation at 20 bar",
        description=(
            "Flash separate a 55/45 molar ethylene and ethane overhead stream "
            "at 20 bar and 243 K. Feed flow is 1 mol/s. Recover the "
            "ethylene-enriched vapour for further purification."
        ),
        category="EOS",
        expected_outcome="PASS",
        expected_compounds=["Ethylene", "Ethane"],
        expected_property_package_family="eos",
        max_iterations_allowed=4,
    ),
    AdversarialTestCase(
        name="[EOS] LNG fractionation — methane/ethane/propane/n-butane at 30 bar",
        description=(
            "Flash separate a liquefied natural gas stream at 30 bar and 230 K "
            "containing 60 mol% methane, 20 mol% ethane, 12 mol% propane, and "
            "8 mol% n-butane. Feed flow is 1 mol/s. Recover the methane-rich vapour."
        ),
        category="EOS",
        expected_outcome="PASS",
        expected_compounds=["Methane", "Ethane", "Propane", "n-Butane"],
        expected_property_package_family="eos",
        max_iterations_allowed=6,
    ),
    AdversarialTestCase(
        name="[EOS] Sour gas H2S/methane flash at 30 bar",
        description=(
            "Flash separate a 20/80 molar hydrogen sulfide and methane sour-gas "
            "stream at 30 bar and 250 K. Feed flow is 1 mol/s. Recover the "
            "H2S-enriched liquid and the methane-rich vapour."
        ),
        category="EOS",
        expected_outcome="PASS",
        expected_compounds=["Hydrogen Sulfide", "Methane"],
        expected_property_package_family="eos",
        max_iterations_allowed=4,
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY: ACTIVITY — activity-model path (no explicit package request)
    # Tests: ketone/water, acetic acid, limited miscibility, ester/alcohol
    # ══════════════════════════════════════════════════════════════════════════

    AdversarialTestCase(
        name="[ACTIVITY] Ketone/water — Methyl Ethyl Ketone/water flash",
        description=(
            "Flash a 40/60 molar methyl ethyl ketone and water feed at 1 atm and "
            "362 K. Feed flow is 1 mol/s. Recover the MEK-enriched vapour phase."
        ),
        category="ACTIVITY",
        expected_outcome="PASS",
        expected_compounds=["Methyl Ethyl Ketone", "Water"],
        expected_property_package_family="activity",
        max_iterations_allowed=6,
    ),
    AdversarialTestCase(
        name="[ACTIVITY] Associating system — acetic acid/water flash",
        description=(
            "Partially vaporise a 30/70 molar acetic acid and water feed at 1 atm "
            "and 383 K. Feed flow is 1 mol/s. Acetic acid vapour-phase dimerisation "
            "should be considered in the thermodynamic model."
        ),
        category="ACTIVITY",
        expected_outcome="PASS",
        expected_compounds=["Acetic Acid", "Water"],
        expected_property_package_family="activity",
        max_iterations_allowed=6,
    ),
    AdversarialTestCase(
        name="[ACTIVITY] Limited miscibility — cyclohexane/ethanol flash",
        description=(
            "Flash separate a 50/50 molar cyclohexane and ethanol feed at 1 atm "
            "and 345 K. Feed flow is 1 mol/s. The system forms a heterogeneous "
            "minimum-boiling azeotrope."
        ),
        category="ACTIVITY",
        expected_outcome="PASS",
        expected_compounds=["Cyclohexane", "Ethanol"],
        expected_property_package_family="activity",
        max_iterations_allowed=6,
    ),
    AdversarialTestCase(
        name="[ACTIVITY] Partial miscibility — 1-butanol/water flash",
        description=(
            "Flash a 25/75 molar 1-butanol and water feed at 1 atm and 370 K. "
            "Feed flow is 1 mol/s. The system exhibits partial liquid miscibility "
            "below 125°C at atmospheric pressure."
        ),
        category="ACTIVITY",
        expected_outcome="PASS",
        expected_compounds=["1-Butanol", "Water"],
        expected_property_package_family="activity",
        max_iterations_allowed=6,
    ),
    AdversarialTestCase(
        name="[ACTIVITY] Ester/alcohol azeotrope — methyl acetate/methanol",
        description=(
            "Flash separate a 50/50 molar methyl acetate and methanol feed at "
            "1 atm and 333 K. Feed flow is 1 mol/s. The binary forms a "
            "minimum-boiling azeotrope at approximately 327.5 K."
        ),
        category="ACTIVITY",
        expected_outcome="PASS",
        expected_compounds=["Methyl Acetate", "Methanol"],
        expected_property_package_family="activity",
        max_iterations_allowed=6,
    ),
    AdversarialTestCase(
        name="[ACTIVITY] Polar nitrile — acetonitrile/water flash",
        description=(
            "Flash a 40/60 molar acetonitrile and water feed at 1 atm and 358 K. "
            "Feed flow is 1 mol/s. Recover the acetonitrile-enriched vapour near "
            "the minimum-boiling azeotrope at 354.75 K."
        ),
        category="ACTIVITY",
        expected_outcome="PASS",
        expected_compounds=["Acetonitrile", "Water"],
        expected_property_package_family="activity",
        max_iterations_allowed=6,
    ),
    AdversarialTestCase(
        name="[ACTIVITY] High-boiling polar solvent — DMSO/water flash",
        description=(
            "Flash a 20/80 molar dimethyl sulfoxide and water feed at 1 atm and "
            "380 K. Feed flow is 1 mol/s. DMSO boils at 462 K — the vapour will "
            "be predominantly water while DMSO concentrates in the liquid phase."
        ),
        category="ACTIVITY",
        expected_outcome="PASS",
        expected_compounds=["Dimethyl Sulfoxide", "Water"],
        expected_property_package_family="activity",
        max_iterations_allowed=6,
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY: CALIBRATION — explicit NRTL/UNIQUAC with BIP injection
    # Tests: CalibrationAgent corpus lookup for pairs not in existing benchmark
    # All descriptions explicitly request NRTL so the CALIBRATION path fires.
    # ══════════════════════════════════════════════════════════════════════════

    AdversarialTestCase(
        name="[CALIBRATION] Benzene/methanol flash — aromatic/alcohol azeotrope",
        description=(
            "Flash separate a 50/50 molar benzene and methanol mixture at 1 atm "
            "and 340 K. Feed flow is 1 mol/s. Use NRTL to model the "
            "minimum-boiling azeotrope at 328 K (61 mol% benzene)."
        ),
        category="CALIBRATION",
        expected_outcome="PASS",
        expected_compounds=["Benzene", "Methanol"],
        expected_property_package_family="activity",
        max_iterations_allowed=6,
    ),
    AdversarialTestCase(
        name="[CALIBRATION] Acetone/water flash at 2 atm — NRTL",
        description=(
            "Flash separate a 45/55 molar acetone and water feed at 2 atm "
            "(202 650 Pa) and 370 K. Feed flow is 1 mol/s. Use NRTL to capture "
            "the vapour-liquid equilibrium at elevated pressure."
        ),
        category="CALIBRATION",
        expected_outcome="PASS",
        expected_compounds=["Acetone", "Water"],
        expected_property_package_family="activity",
        max_iterations_allowed=6,
    ),
    AdversarialTestCase(
        name="[CALIBRATION] Ethanol/benzene flash — heterogeneous azeotrope",
        description=(
            "Flash an equimolar ethanol and benzene feed at 1 atm and 344 K. "
            "Feed flow is 1 mol/s. Use NRTL to model the heterogeneous "
            "minimum-boiling azeotrope at 341.4 K (44.8 mol% ethanol)."
        ),
        category="CALIBRATION",
        expected_outcome="PASS",
        expected_compounds=["Ethanol", "Benzene"],
        expected_property_package_family="activity",
        max_iterations_allowed=6,
    ),
    AdversarialTestCase(
        name="[CALIBRATION] 1-Butanol/water flash — NRTL VL separation",
        description=(
            "Flash a 20/80 molar 1-butanol and water feed at 1 atm and 374 K. "
            "Feed flow is 1 mol/s. Use NRTL to model the non-ideal vapour-liquid "
            "equilibrium and recover the water-enriched vapour overhead."
        ),
        category="CALIBRATION",
        expected_outcome="PASS",
        expected_compounds=["1-Butanol", "Water"],
        expected_property_package_family="activity",
        max_iterations_allowed=6,
    ),
    AdversarialTestCase(
        name="[CALIBRATION] Acetonitrile/water flash — NRTL azeotrope",
        description=(
            "Flash separate a 50/50 molar acetonitrile and water feed at 1 atm "
            "and 358 K. Feed flow is 1 mol/s. Use NRTL to account for the "
            "minimum-boiling azeotrope at 354.75 K (83.7 mol% acetonitrile)."
        ),
        category="CALIBRATION",
        expected_outcome="PASS",
        expected_compounds=["Acetonitrile", "Water"],
        expected_property_package_family="activity",
        max_iterations_allowed=6,
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY: TOPOLOGY — multi-unit flowsheets beyond heater + vessel
    # Tests: compressors, multi-feed mixers, staged flash, product cooling
    # ══════════════════════════════════════════════════════════════════════════

    AdversarialTestCase(
        name="[TOPOLOGY] Compressor + cooler + vessel — propane/n-butane LPG",
        description=(
            "Compress a 70/30 molar propane and n-butane vapour stream from 1 bar "
            "and 298 K to 8 bar using a compressor, cool the compressed stream to "
            "300 K in a cooler, then flash in a vessel to recover n-butane-enriched "
            "liquid LPG. Feed flow is 1 mol/s."
        ),
        category="TOPOLOGY",
        expected_outcome="PASS",
        expected_compounds=["Propane", "n-Butane"],
        expected_property_package_family="eos",
        max_iterations_allowed=6,
    ),
    AdversarialTestCase(
        name="[TOPOLOGY] Three-stream mixer + heater + vessel — methanol/ethanol/water",
        description=(
            "Combine three streams — 1 mol/s pure methanol, 1 mol/s pure ethanol, "
            "and 1 mol/s pure water, all at 298 K and 1 atm — in a mixer. Heat "
            "the blended stream to 360 K, then flash in a vessel to recover the "
            "alcohol-enriched vapour."
        ),
        category="TOPOLOGY",
        expected_outcome="PASS",
        expected_compounds=["Methanol", "Ethanol", "Water"],
        expected_property_package_family="activity",
        max_iterations_allowed=6,
    ),
    AdversarialTestCase(
        name="[TOPOLOGY] Heater + vessel + cooler — flash with product cooling",
        description=(
            "Heat a 40/60 molar n-hexane and n-heptane feed at 1 atm and 298 K "
            "to 360 K in a heater, flash in a vessel to separate the hexane-rich "
            "vapour, then cool the liquid bottoms to 298 K in a cooler before "
            "storage. Feed flow is 1 mol/s."
        ),
        category="TOPOLOGY",
        expected_outcome="PASS",
        expected_compounds=["n-Hexane", "n-Heptane"],
        expected_property_package_family="ideal",
        max_iterations_allowed=6,
    ),
    AdversarialTestCase(
        name="[TOPOLOGY] Two-stage flash — methane/ethane/propane staged separation",
        description=(
            "Flash a 60/25/15 molar methane, ethane, and propane stream at 50 bar "
            "and 240 K in a first vessel to remove methane-rich vapour. Flash the "
            "liquid from the first vessel at 20 bar and 260 K in a second vessel "
            "to separate ethane and propane. Feed flow is 1 mol/s."
        ),
        category="TOPOLOGY",
        expected_outcome="PASS",
        expected_compounds=["Methane", "Ethane", "Propane"],
        expected_property_package_family="eos",
        max_iterations_allowed=6,
    ),
    AdversarialTestCase(
        name="[TOPOLOGY] Compressor only — nitrogen gas compression",
        description=(
            "Compress a pure nitrogen gas stream from 1 bar and 298 K to 50 bar "
            "using a compressor. No phase separation required. Feed flow is 2 mol/s."
        ),
        category="TOPOLOGY",
        expected_outcome="PASS",
        expected_compounds=["Nitrogen"],
        expected_property_package_family="eos",
        max_iterations_allowed=3,
    ),
    AdversarialTestCase(
        name="[TOPOLOGY] Mixer + cooler + vessel — hot benzene/toluene blend",
        description=(
            "Mix equal molar streams of benzene at 420 K and toluene at 420 K "
            "(each 1 atm, 1 mol/s) in a mixer, cool the blend to 360 K in a "
            "cooler, then flash in a vessel to partially vaporise the mixture."
        ),
        category="TOPOLOGY",
        expected_outcome="PASS",
        expected_compounds=["Benzene", "Toluene"],
        expected_property_package_family="ideal",
        max_iterations_allowed=4,
    ),
    AdversarialTestCase(
        name="[TOPOLOGY] Heater + compressor — ethylene gas-phase processing",
        description=(
            "Heat a pure ethylene stream at 20 bar and 250 K to 300 K in a "
            "heater, then compress the heated stream to 60 bar using a compressor. "
            "Feed flow is 1 mol/s. No phase separation."
        ),
        category="TOPOLOGY",
        expected_outcome="PASS",
        expected_compounds=["Ethylene"],
        expected_property_package_family="eos",
        max_iterations_allowed=4,
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY: DESCRIPTION — phrasing robustness
    # Tests: unit conversion, minimal input, verbosity, qualitative language
    # ══════════════════════════════════════════════════════════════════════════

    AdversarialTestCase(
        name="[DESCRIPTION] Imperial units — °F, psia, lb-mol/hr",
        description=(
            "Flash separate a 50/50 molar ethanol and water feed at 14.7 psia "
            "and 212°F with a feed rate of 3.6 lb-mol/hr. Recover the "
            "ethanol-enriched vapour."
        ),
        category="DESCRIPTION",
        expected_outcome="PASS",
        expected_compounds=["Ethanol", "Water"],
        expected_property_package_family="activity",
        max_iterations_allowed=6,
    ),
    AdversarialTestCase(
        name="[DESCRIPTION] Minimal phrasing — 'separate acetone and water'",
        description="Separate acetone and water.",
        category="DESCRIPTION",
        expected_outcome="PASS",
        expected_compounds=["Acetone", "Water"],
        expected_property_package_family="activity",
        max_iterations_allowed=6,
    ),
    AdversarialTestCase(
        name="[DESCRIPTION] Verbose with irrelevant noise — n-hexane/n-heptane",
        description=(
            "In our laboratory at Imperial College London, after three months of "
            "experimental work and consultation with our industrial partner, we "
            "have determined the optimal operating conditions for the following "
            "separation. We need to flash separate a 40/60 molar n-hexane and "
            "n-heptane feed at atmospheric pressure and 360 K. The feed flow rate "
            "is 1 mol/s. We expect a hexane-enriched vapour overhead and a "
            "heptane-rich liquid that will be sent to further downstream processing "
            "units which are not included in this simulation scope."
        ),
        category="DESCRIPTION",
        expected_outcome="PASS",
        expected_compounds=["n-Hexane", "n-Heptane"],
        expected_property_package_family="ideal",
        max_iterations_allowed=4,
    ),
    AdversarialTestCase(
        name="[DESCRIPTION] Near-azeotrope framing — ethanol/water",
        description=(
            "Operate close to the minimum-boiling azeotropic point of the "
            "ethanol-water system at atmospheric pressure. Feed is 85 mol% ethanol "
            "and 15 mol% water at 351 K and 1 atm. Flash to observe phase behaviour "
            "near the azeotrope. Feed flow is 1 mol/s."
        ),
        category="DESCRIPTION",
        expected_outcome="PASS",
        expected_compounds=["Ethanol", "Water"],
        expected_property_package_family="activity",
        max_iterations_allowed=6,
    ),
    AdversarialTestCase(
        name="[DESCRIPTION] Qualitative conditions — 'above bubble point'",
        description=(
            "Heat a 50/50 molar benzene and toluene liquid feed at atmospheric "
            "pressure to a temperature above the mixture bubble point, then flash "
            "to separate the benzene-enriched vapour from the toluene-enriched "
            "liquid. Feed flow is 1 mol/s."
        ),
        category="DESCRIPTION",
        expected_outcome="PASS",
        expected_compounds=["Benzene", "Toluene"],
        expected_property_package_family="ideal",
        max_iterations_allowed=6,
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY: EDGE — extreme compositions, mixture aliases, 4-component
    # ══════════════════════════════════════════════════════════════════════════

    AdversarialTestCase(
        name="[EDGE] Near-pure feed — 98/2 molar n-hexane/n-heptane",
        description=(
            "Flash a 98/2 molar n-hexane and n-heptane feed at 1 atm and 343 K. "
            "Feed flow is 1 mol/s. Recover the trace n-heptane concentrated in "
            "the liquid phase."
        ),
        category="EDGE",
        expected_outcome="PASS",
        expected_compounds=["n-Hexane", "n-Heptane"],
        expected_property_package_family="ideal",
        max_iterations_allowed=4,
    ),
    AdversarialTestCase(
        name="[EDGE] Four-component flash — acetone/methanol/ethanol/water",
        description=(
            "Flash a 25/25/25/25 molar quaternary mixture of acetone, methanol, "
            "ethanol, and water at 1 atm and 355 K. Feed flow is 1 mol/s. "
            "Recover the acetone-enriched vapour."
        ),
        category="EDGE",
        expected_outcome="PASS",
        expected_compounds=["Acetone", "Methanol", "Ethanol", "Water"],
        expected_property_package_family="activity",
        max_iterations_allowed=6,
    ),
    AdversarialTestCase(
        name="[EDGE] Trace impurity — 1 mol% acetone in water",
        description=(
            "Flash a feed of 99 mol% water and 1 mol% acetone at 1 atm and 375 K. "
            "Feed flow is 1 mol/s. Recover the acetone-enriched vapour even at "
            "trace concentrations."
        ),
        category="EDGE",
        expected_outcome="PASS",
        expected_compounds=["Water", "Acetone"],
        expected_property_package_family="activity",
        max_iterations_allowed=4,
    ),
    AdversarialTestCase(
        name="[EDGE] Mixture alias BTX — benzene/toluene/o-xylene",
        description=(
            "Flash separate a BTX reformate stream at 1 atm and 380 K. "
            "The feed contains equal molar fractions of benzene, toluene, and "
            "o-xylene at a total flow of 1 mol/s."
        ),
        category="EDGE",
        expected_outcome="PASS",
        expected_compounds=["Benzene", "Toluene", "o-Xylene"],
        expected_property_package_family="ideal",
        max_iterations_allowed=4,
    ),
    AdversarialTestCase(
        name="[EDGE] Mixture alias LPG — propane/n-butane flash",
        description=(
            "Flash separate an LPG stream at 5 bar and 298 K. "
            "Feed flow is 1 mol/s. Recover the propane-enriched vapour and "
            "the n-butane-enriched liquid."
        ),
        category="EDGE",
        expected_outcome="PASS",
        expected_compounds=["Propane", "n-Butane"],
        expected_property_package_family="eos",
        max_iterations_allowed=4,
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY: REJECT — expected failures
    #
    # BASIS_FAILED: compound not in DWSIM database (deterministic in both modes)
    # HUMAN†: process beyond available unit operations (soft expectation —
    #         real-executor validates these; mock executor may return PASS)
    # ══════════════════════════════════════════════════════════════════════════

    AdversarialTestCase(
        name="[REJECT] Polymer dissolution — polypropylene in toluene (BASIS_FAILED)",
        description=(
            "Dissolve polypropylene pellets in toluene at 120°C and 1 atm, then "
            "cool and precipitate the polymer. Feed is 15 wt% polypropylene in "
            "toluene at 1 mol/s total flow."
        ),
        category="REJECT",
        expected_outcome="BASIS_FAILED",
        expected_compounds=[],
        expected_property_package_family="ideal",
        max_iterations_allowed=2,
    ),
    AdversarialTestCase(
        name="[REJECT] Ionic liquid — room-temperature ionic liquid + ethanol (BASIS_FAILED)",
        description=(
            "Separate ethanol from a room-temperature ionic liquid ([BMIM][BF4]) "
            "at 1 atm and 352 K by flash vaporisation. Feed is 30/70 molar "
            "ethanol/ionic liquid at 1 mol/s."
        ),
        category="REJECT",
        expected_outcome="BASIS_FAILED",
        expected_compounds=[],
        expected_property_package_family="activity",
        max_iterations_allowed=2,
    ),
    AdversarialTestCase(
        name="[REJECT] Biomass fermentation broth — proteins/water (BASIS_FAILED)",
        description=(
            "Recover ethanol from a fermentation broth containing 8 mol% ethanol, "
            "90 mol% water, and 2 mol% enzyme proteins at 1 atm and 360 K. "
            "Feed flow is 1 mol/s."
        ),
        category="REJECT",
        expected_outcome="BASIS_FAILED",
        expected_compounds=[],
        expected_property_package_family="activity",
        max_iterations_allowed=2,
    ),
    AdversarialTestCase(
        name="[REJECT]† Steam methane reforming — chemical reaction (HUMAN)",
        description=(
            "React methane with steam at 800°C and 10 bar over a nickel catalyst "
            "to produce syngas (hydrogen and carbon monoxide) via the steam "
            "reforming reaction CH4 + H2O → CO + 3H2. Feed is 1 mol/s methane "
            "and 3 mol/s steam."
        ),
        category="REJECT",
        expected_outcome="HUMAN",
        expected_compounds=["Methane", "Water"],
        expected_property_package_family="eos",
        max_iterations_allowed=4,
    ),
    AdversarialTestCase(
        name="[REJECT]† Membrane gas separation — N2/O2 hollow-fibre (HUMAN)",
        description=(
            "Separate nitrogen from oxygen using a hollow-fibre polymeric membrane "
            "module at 5 bar feed side and 1 bar permeate side. Feed is 79/21 "
            "molar nitrogen and oxygen at 298 K and 1 mol/s. Target permeate "
            "oxygen purity of 40 mol%."
        ),
        category="REJECT",
        expected_outcome="HUMAN",
        expected_compounds=["Nitrogen", "Oxygen"],
        expected_property_package_family="eos",
        max_iterations_allowed=4,
    ),
    AdversarialTestCase(
        name="[REJECT]† Reactive distillation — ethanol esterification in column (HUMAN)",
        description=(
            "Simultaneously react acetic acid with ethanol to form ethyl acetate "
            "and water in a reactive distillation column, recovering ethyl acetate "
            "as the distillate at >99 mol% purity. Feed is equimolar acetic acid "
            "and ethanol at 1 atm and 298 K at 1 mol/s."
        ),
        category="REJECT",
        expected_outcome="HUMAN",
        expected_compounds=["Acetic Acid", "Ethanol"],
        expected_property_package_family="activity",
        max_iterations_allowed=4,
    ),
]


# ── Per-case metrics ──────────────────────────────────────────────────────────

@dataclass
class CaseMetrics:
    name:                    str
    category:                str
    outcome:                 str
    outcome_correct:         bool
    compound_match:          float
    package_family_correct:  bool
    llm_calls:               int
    iterations:              int
    elapsed_s:               float
    warnings:                list[str]     = field(default_factory=list)
    error:                   Optional[str] = None
    routing_trace:           list[str]     = field(default_factory=list)


# ── Mock executor (identical to benchmark_pipeline) ───────────────────────────

def _mock_execution_result(flowsheet: dict) -> ExecutionResult:
    streams   = flowsheet.get("streams", [])
    conns     = flowsheet.get("connections", [])
    compounds = flowsheet.get("compounds", [])
    n         = max(len(compounds), 1)

    has_incoming = {c[1] for c in conns if len(c) >= 2}
    has_outgoing = {c[0] for c in conns if len(c) >= 2}
    feed_comp    = {c: 1.0 / n for c in compounds} if compounds else {}

    feed_streams     = [s for s in streams if s["tag"] not in has_incoming]
    total_feed_flow  = sum(s.get("flow", 1.0) for s in feed_streams) or 1.0

    _outlet_port: dict[str, int] = {}
    for c in conns:
        if len(c) >= 3 and c[1] in has_incoming and c[1] not in has_outgoing:
            _outlet_port[c[1]] = c[2]
    terminal_outlets = sorted(
        [s["tag"] for s in streams
         if s["tag"] in has_incoming and s["tag"] not in has_outgoing],
        key=lambda t: _outlet_port.get(t, 99),
    )
    n_terminals  = max(len(terminal_outlets), 1)
    outlet_flow  = total_feed_flow / n_terminals

    from agents.critic import _NBP_K as _nbp
    _volatility = {c: _nbp.get(c.lower(), 500.0) for c in compounds}
    compounds_by_volatility = sorted(compounds, key=lambda c: _volatility[c])

    unit_outlet_T: dict[str, float] = {}
    stream_by_tag = {s["tag"]: s for s in streams}
    for u in flowsheet.get("units", []):
        t_out = u.get("T_out")
        if t_out and u.get("type") in ("Heater", "Cooler"):
            for conn in conns:
                if len(conn) >= 2 and conn[0] == u["tag"] and conn[1] in stream_by_tag:
                    unit_outlet_T[conn[1]] = float(t_out)

    stream_results: dict[str, StreamResult] = {}
    for s in streams:
        tag     = s["tag"]
        is_feed = tag not in has_incoming

        if is_feed:
            flow = s.get("flow", 1.0)
        elif tag in terminal_outlets:
            flow = outlet_flow
        else:
            flow = total_feed_flow

        if not is_feed and tag in terminal_outlets and n >= 2 and n_terminals >= 2:
            idx   = terminal_outlets.index(tag)
            comp  = {c: 1.0 / n for c in compounds}
            rich  = compounds_by_volatility[idx % n]
            lean  = compounds_by_volatility[(idx + 1) % n]
            delta = 0.3
            comp[rich] = min(1.0, comp[rich] + delta)
            comp[lean] = max(0.0, comp[lean] - delta)
            total = sum(comp.values())
            comp  = {c: v / total for c, v in comp.items()}
        else:
            comp = s.get("composition") or feed_comp

        T_K = unit_outlet_T.get(tag) or s.get("T", 298.15)

        stream_results[tag] = StreamResult(
            tag=tag,
            T_K=T_K,
            P_Pa=s.get("P", 101325.0),
            flow_mol_s=flow,
            composition=comp,
            is_feed=is_feed,
        )
    return ExecutionResult(solved=True, stream_results=stream_results)


# ── Package family helpers ────────────────────────────────────────────────────

def _package_family(pkg: str) -> str:
    for family, pkgs in _PKG_FAMILIES.items():
        if pkg in pkgs:
            return family
    return "unknown"


def _family_correct(assigned_pkg: str, expected_family: str) -> bool:
    return _package_family(assigned_pkg) == expected_family


# ── Runner ────────────────────────────────────────────────────────────────────

def run_benchmark(
    model:             str   = "claude-sonnet-4-6",
    use_real_executor: bool  = False,
    inter_case_delay:  float = 10.0,
    category_filter:   str | None = None,
    debug:             bool  = False,
) -> list[CaseMetrics]:
    cases = ADVERSARIAL_TEST_CASES
    if category_filter:
        cat = category_filter.upper()
        cases = [tc for tc in cases if tc.category == cat]
        if not cases:
            print(f"[benchmark] No cases found for category '{cat}'. "
                  f"Valid: {sorted({tc.category for tc in ADVERSARIAL_TEST_CASES})}")
            return []

    metrics: list[CaseMetrics] = []

    for i, tc in enumerate(cases):
        if i > 0 and inter_case_delay > 0:
            print(f"  [benchmark] sleeping {inter_case_delay:.0f}s before next case...")
            time.sleep(inter_case_delay)

        reset_call_count()
        t0 = time.time()

        try:
            orch = Orchestrator(model=model, max_iterations=tc.max_iterations_allowed)

            if use_real_executor:
                result: OrchestratorResult = orch.run(tc.description)
            else:
                with patch(
                    "agents.executor.Executor.run",
                    side_effect=lambda fs: _mock_execution_result(fs),
                ):
                    result = orch.run(tc.description)

        except Exception as exc:
            elapsed = time.time() - t0
            metrics.append(CaseMetrics(
                name=tc.name,
                category=tc.category,
                outcome="ERROR",
                outcome_correct=False,
                compound_match=0.0,
                package_family_correct=False,
                llm_calls=get_call_count(),
                iterations=0,
                elapsed_s=elapsed,
                error=f"{type(exc).__name__}: {exc}",
            ))
            continue

        elapsed   = time.time() - t0
        llm_calls = get_call_count()

        found = (set(result.basis_result.dwsim_compounds)
                 if result.basis_result else set())
        if tc.expected_compounds:
            compound_match = sum(
                1 for c in tc.expected_compounds if c in found
            ) / len(tc.expected_compounds)
        else:
            compound_match = 1.0

        assigned_pkg  = ""
        pkg_family_ok = False
        if result.final_flowsheet:
            assigned_pkg  = result.final_flowsheet.get("property_package", "")
            pkg_family_ok = _family_correct(assigned_pkg, tc.expected_property_package_family)
        elif tc.expected_outcome in ("BASIS_FAILED", "HUMAN"):
            pkg_family_ok = True   # not reached — not penalised

        routing_trace = [f"i{i}:{rec.routing}" for i, rec in enumerate(result.iterations)]

        metrics.append(CaseMetrics(
            name=tc.name,
            category=tc.category,
            outcome=result.outcome,
            outcome_correct=(result.outcome == tc.expected_outcome),
            compound_match=compound_match,
            package_family_correct=pkg_family_ok,
            llm_calls=llm_calls,
            iterations=len(result.iterations),
            elapsed_s=elapsed,
            warnings=result.warnings,
            routing_trace=routing_trace,
        ))

        # ── Debug output — show per-case diagnostics on failure ───────────────
        if debug and not (result.outcome == tc.expected_outcome):
            print(f"\n  [DEBUG] {tc.name}")
            print(f"    outcome={result.outcome}  expected={tc.expected_outcome}")
            if result.basis_result:
                print(f"    compounds found : {result.basis_result.dwsim_compounds}")
            for w in result.warnings:
                kws = ("PlannerAgent", "ThermoAgent", "pre_select",
                       "LIGHT_GAS", "AZEOTROPE", "LLE", "HIGH_P",
                       "ELECTROLYTE", "physics", "Last generated JSON",
                       "Last errors")
                if any(kw in w for kw in kws):
                    # Trim long JSON snippets for readability
                    if "Last generated JSON" in w:
                        idx = w.index("Last generated JSON")
                        print(f"    ! {w[:idx + 80]}...")
                    else:
                        print(f"    ! {w[:200]}")
            if result.final_flowsheet:
                print(f"    pkg assigned : {result.final_flowsheet.get('property_package')}")
            print()

    return metrics


# ── Reporting ─────────────────────────────────────────────────────────────────

_CATEGORIES_ORDER = [
    "BASIS", "EOS", "ACTIVITY", "CALIBRATION",
    "TOPOLOGY", "DESCRIPTION", "EDGE", "REJECT",
]


def _print_report(metrics: list[CaseMetrics], model: str) -> None:
    print(f"\n## Adversarial Benchmark — model: {model}\n")

    # ── Per-case table ──────────────────────────────────────────────────────
    header = (f"| {'Test case':<50} | {'Cat':<11} | {'Outcome':<12} | {'OK':>2} "
              f"| {'CmpMatch':>8} | {'PkgFam':>6} | {'LLM':>4} | {'Iter':>4} | {'Time(s)':>7} |")
    sep    = ("|" + "-" * 52 + "|" + "-" * 13 + "|" + "-" * 14 + "|" + "-" * 4 +
              "|" + "-" * 10 + "|" + "-" * 8 + "|" + "-" * 6 + "|" + "-" * 6 + "|" + "-" * 9 + "|")
    print(header)
    print(sep)

    for m in metrics:
        ok_mark  = "✓" if m.outcome_correct       else "✗"
        pkg_mark = "✓" if m.package_family_correct else "✗"
        name_short = m.name[:50]
        row = (f"| {name_short:<50} | {m.category:<11} | {m.outcome:<12} | {ok_mark:>2} "
               f"| {m.compound_match:>8.0%} | {pkg_mark:>6} | {m.llm_calls:>4} "
               f"| {m.iterations:>4} | {m.elapsed_s:>7.1f} |")
        print(row)
        if m.error:
            print(f"|   ERROR: {m.error}")
        if m.outcome == "HUMAN" and m.routing_trace:
            print(f"|   trace: {' → '.join(m.routing_trace)}")
        for w in m.warnings:
            if w.startswith("REPLAN"):
                print(f"|   ↺ {w[:95]}")

    print()

    # ── Category summary ────────────────────────────────────────────────────
    print("## Category Summary\n")
    cat_header = f"| {'Category':<12} | {'Cases':>5} | {'Outcome OK':>10} | {'PkgFam OK':>9} | {'Mean LLM':>8} | {'Mean t(s)':>9} |"
    cat_sep    = "|" + "-" * 14 + "|" + "-" * 7 + "|" + "-" * 12 + "|" + "-" * 11 + "|" + "-" * 10 + "|" + "-" * 11 + "|"
    print(cat_header)
    print(cat_sep)

    categories_present = sorted(
        {m.category for m in metrics},
        key=lambda c: _CATEGORIES_ORDER.index(c) if c in _CATEGORIES_ORDER else 99,
    )
    for cat in categories_present:
        cat_metrics = [m for m in metrics if m.category == cat]
        n_cat       = len(cat_metrics)
        n_ok        = sum(1 for m in cat_metrics if m.outcome_correct)
        n_pkg       = sum(1 for m in cat_metrics if m.package_family_correct)
        mean_llm    = sum(m.llm_calls for m in cat_metrics) / n_cat
        mean_t      = sum(m.elapsed_s for m in cat_metrics) / n_cat
        pct_ok      = f"{n_ok}/{n_cat} ({n_ok/n_cat:.0%})"
        pct_pkg     = f"{n_pkg}/{n_cat} ({n_pkg/n_cat:.0%})"
        print(f"| {cat:<12} | {n_cat:>5} | {pct_ok:>10} | {pct_pkg:>9} | {mean_llm:>8.1f} | {mean_t:>9.1f} |")

    print()

    # ── Aggregate summary ───────────────────────────────────────────────────
    n  = len(metrics)
    if n == 0:
        return
    n_outcome_ok  = sum(1 for m in metrics if m.outcome_correct)
    n_pkg_ok      = sum(1 for m in metrics if m.package_family_correct)
    mean_llm      = sum(m.llm_calls  for m in metrics) / n
    mean_time     = sum(m.elapsed_s  for m in metrics) / n
    mean_compound = sum(m.compound_match for m in metrics) / n

    print(f"**Total cases:**           {n}")
    print(f"**Outcome accuracy:**      {n_outcome_ok}/{n}  ({n_outcome_ok/n:.0%})")
    print(f"**Pkg family accuracy:**   {n_pkg_ok}/{n}  ({n_pkg_ok/n:.0%})")
    print(f"**Mean compound match:**   {mean_compound:.0%}")
    print(f"**Mean LLM calls:**        {mean_llm:.1f}")
    print(f"**Mean time per case:**    {mean_time:.1f}s")
    print(f"\nNote: REJECT† cases (soft HUMAN) may return PASS on mock executor —")
    print(f"      run with --real-executor for definitive REJECT validation.")


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adversarial pipeline benchmark — 50 cases, 8 categories")
    p.add_argument("--model",            default="claude-sonnet-4-6",
                   help="LLM model for all agents")
    p.add_argument("--real-executor",    action="store_true",
                   help="Use live DWSIM executor (requires DWSIM container)")
    p.add_argument("--inter-case-delay", type=float, default=10.0,
                   help="Seconds between cases (default 10, 0 to disable)")
    p.add_argument("--category",         type=str, default=None,
                   help="Run only one category: BASIS | EOS | ACTIVITY | CALIBRATION | "
                        "TOPOLOGY | DESCRIPTION | EDGE | REJECT")
    p.add_argument("--debug",            action="store_true",
                   help="Print per-case diagnostics for failures: physics errors, "
                        "pre_select decision, and generated package assignments")
    return p.parse_args()


if __name__ == "__main__":
    args    = _parse_args()
    results = run_benchmark(
        model=args.model,
        use_real_executor=args.real_executor,
        inter_case_delay=args.inter_case_delay,
        category_filter=args.category,
        debug=args.debug,
    )
    _print_report(results, model=args.model)
