"""
Benchmark dataset — 60 test cases for systematic evaluation.

Categories:
  E    (10) — Easy:          2-3 units, single-phase or simple VLE, well-specified
  M    (12) — Medium:        3-5 units, flash separation, BIP-requiring thermo
  H    (10) — Hard:          multi-step, recycle, azeotrope, high-pressure, ternary
  U    ( 8) — Underspecified: missing T, P, or composition
  AMB  ( 6) — Ambiguous:     multiple valid interpretations
  ADV  ( 7) — Adversarial:   contradictory, invalid physics, unknown compounds
  EDGE ( 7) — Edge thermo:   cryogenic, supercritical, near-critical, toxic systems

For adversarial and underspecified cases, expected_units may be empty (system
should handle gracefully without crashing) and expected_pkg may be '' (any or
unknown).  The evaluation checks that the pipeline does not raise unhandled
exceptions and returns a structured result with appropriate limitations.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BenchmarkCase:
    case_id:        str
    description:    str
    compounds:      list[str]
    tier:           str           # "easy" | "medium" | "hard" | "underspec" |
    #                                "ambiguous" | "adversarial" | "edge"
    expected_pkg:   str           # "" = any / unknown
    expected_units: list[str]     # subset check; [] = don't check
    notes:          str = ""
    # For adversarial/invalid cases:
    expect_failure: bool = False  # True = system should detect and flag, not crash


# ──────────────────────────────────────────────────────────────────────────────
# E: Easy (10) — well-specified, 1-3 units, straightforward thermodynamics
# ──────────────────────────────────────────────────────────────────────────────

EASY_CASES: list[BenchmarkCase] = [
    BenchmarkCase(
        case_id="E01",
        description="Heat an ethanol and water feed mixture from 25°C to 80°C",
        compounds=["ethanol", "water"],
        tier="easy", expected_pkg="NRTL",
        expected_units=["Heater"],
        notes="Minimal case: one unit, polar mixture",
    ),
    BenchmarkCase(
        case_id="E02",
        description="Compress pure methane gas from atmospheric pressure to 5 bar",
        compounds=["methane"],
        tier="easy", expected_pkg="Peng-Robinson",
        expected_units=["Compressor"],
        notes="Single-compound gas compression",
    ),
    BenchmarkCase(
        case_id="E03",
        description="Pump liquid water from 1 atm to 10 atm",
        compounds=["water"],
        tier="easy", expected_pkg="Peng-Robinson",
        expected_units=["Pump"],
        notes="Pure liquid pumping",
    ),
    BenchmarkCase(
        case_id="E04",
        description="Cool a hot acetone vapour stream from 150°C to 40°C",
        compounds=["acetone"],
        tier="easy", expected_pkg="Peng-Robinson",
        expected_units=["Cooler"],
        notes="Single cooler, pure component",
    ),
    BenchmarkCase(
        case_id="E05",
        description="Heat a benzene and toluene mixture to 100°C then flash "
                    "separate the vapour from the liquid",
        compounds=["benzene", "toluene"],
        tier="easy", expected_pkg="Peng-Robinson",
        expected_units=["Heater", "Vessel"],
        notes="Classic non-polar VLE; no BIPs needed",
    ),
    BenchmarkCase(
        case_id="E06",
        description="Mix two liquid water streams at different temperatures "
                    "to produce a single mixed stream",
        compounds=["water"],
        tier="easy", expected_pkg="Peng-Robinson",
        expected_units=["Mixer"],
        notes="Single mixer, pure component",
    ),
    BenchmarkCase(
        case_id="E07",
        description="Expand high-pressure nitrogen gas from 50 bar to 1 bar "
                    "through an expander turbine",
        compounds=["nitrogen"],
        tier="easy", expected_pkg="Peng-Robinson",
        expected_units=["Expander"],
        notes="Single expander, cryogenic gas",
    ),
    BenchmarkCase(
        case_id="E08",
        description="Preheat a propane feed stream from 20°C to 60°C before "
                    "further processing",
        compounds=["propane"],
        tier="easy", expected_pkg="Peng-Robinson",
        expected_units=["Heater"],
        notes="Single heater, light hydrocarbon",
    ),
    BenchmarkCase(
        case_id="E09",
        description="Cool and partially condense a propylene gas stream from "
                    "80°C to 20°C at 5 bar",
        compounds=["propane"],
        tier="easy", expected_pkg="Peng-Robinson",
        expected_units=["Cooler"],
        notes="Partial condensation; note: using propane as proxy for propylene",
    ),
    BenchmarkCase(
        case_id="E10",
        description="Pump liquid methanol from ambient pressure to 20 bar",
        compounds=["methanol"],
        tier="easy", expected_pkg="Peng-Robinson",
        expected_units=["Pump"],
        notes="Liquid pump, polar component",
    ),
]

# ──────────────────────────────────────────────────────────────────────────────
# M: Medium (12) — 3-5 units, flash separation, BIP-requiring thermo
# ──────────────────────────────────────────────────────────────────────────────

MEDIUM_CASES: list[BenchmarkCase] = [
    BenchmarkCase(
        case_id="M01",
        description="Heat an ethanol-water feed to 78°C then flash separate "
                    "the vapour from the liquid",
        compounds=["ethanol", "water"],
        tier="medium", expected_pkg="NRTL",
        expected_units=["Heater", "Vessel"],
        notes="NRTL azeotrope; BIP injection required",
    ),
    BenchmarkCase(
        case_id="M02",
        description="Separate acetone and methanol by heating the feed to 60°C "
                    "then flashing",
        compounds=["acetone", "methanol"],
        tier="medium", expected_pkg="NRTL",
        expected_units=["Heater", "Vessel"],
        notes="Polar mixture; acetone-methanol azeotrope",
    ),
    BenchmarkCase(
        case_id="M03",
        description="Cool a methane-ethane-propane natural gas mixture from 50°C "
                    "to -30°C at 20 bar to liquefy heavier components, then flash",
        compounds=["methane", "ethane", "propane"],
        tier="medium", expected_pkg="Peng-Robinson",
        expected_units=["Cooler", "Vessel"],
        notes="High-pressure light gases; PR appropriate",
    ),
    BenchmarkCase(
        case_id="M04",
        description="Pump liquid ethyl acetate and ethanol mixture to 3 bar, "
                    "heat to 80°C, then flash separate",
        compounds=["ethyl acetate", "ethanol"],
        tier="medium", expected_pkg="NRTL",
        expected_units=["Pump", "Heater", "Vessel"],
        notes="Pump + heater + flash with ester/alcohol mixture",
    ),
    BenchmarkCase(
        case_id="M05",
        description="Mix two feed streams of isopropanol and water, heat the "
                    "combined stream to 85°C, then flash separate",
        compounds=["isopropanol", "water"],
        tier="medium", expected_pkg="NRTL",
        expected_units=["Mixer", "Heater", "Vessel"],
        notes="Mixer insertion; isopropanol-water azeotrope",
    ),
    BenchmarkCase(
        case_id="M06",
        description="Heat a 1-propanol and water mixture from 20°C to 97°C, "
                    "then flash to separate propanol-rich vapour from water-rich liquid",
        compounds=["1-propanol", "water"],
        tier="medium", expected_pkg="NRTL",
        expected_units=["Heater", "Vessel"],
        notes="1-propanol/water azeotrope at 97.2°C",
    ),
    BenchmarkCase(
        case_id="M07",
        description="Compress propane feed to 10 bar, cool to 30°C to partially "
                    "condense, then flash to separate liquid and vapour phases",
        compounds=["propane"],
        tier="medium", expected_pkg="Peng-Robinson",
        expected_units=["Compressor", "Cooler", "Vessel"],
        notes="LPG processing: compress-cool-flash",
    ),
    BenchmarkCase(
        case_id="M08",
        description="Heat a chloroform and methanol mixture to 55°C then flash "
                    "to achieve phase separation",
        compounds=["chloroform", "methanol"],
        tier="medium", expected_pkg="NRTL",
        expected_units=["Heater", "Vessel"],
        notes="Chlorinated solvent + alcohol; NRTL needed",
    ),
    BenchmarkCase(
        case_id="M09",
        description="Mix two feeds — one of pure acetone and one of methyl ethyl "
                    "ketone — then heat the combined stream to 80°C and flash separate",
        compounds=["acetone", "methyl ethyl ketone"],
        tier="medium", expected_pkg="NRTL",
        expected_units=["Mixer", "Heater", "Vessel"],
        notes="Ketone mixture; mixer + heater + flash",
    ),
    BenchmarkCase(
        case_id="M10",
        description="Pump liquid ethanol to 5 bar, heat to 80°C, then flash "
                    "separate ethanol vapour from the liquid",
        compounds=["ethanol", "water"],
        tier="medium", expected_pkg="NRTL",
        expected_units=["Pump", "Heater", "Vessel"],
        notes="Ethanol/water with upstream pump",
    ),
    BenchmarkCase(
        case_id="M11",
        description="Cool a carbon dioxide stream from 100°C to 30°C at 50 bar "
                    "to partially condense the CO2",
        compounds=["carbon dioxide"],
        tier="medium", expected_pkg="Peng-Robinson",
        expected_units=["Cooler"],
        notes="Near-critical CO2; high-pressure single component",
    ),
    BenchmarkCase(
        case_id="M12",
        description="Heat diethyl ether and ethanol mixture to 40°C then flash "
                    "to separate the more volatile ether vapour",
        compounds=["diethyl ether", "ethanol"],
        tier="medium", expected_pkg="NRTL",
        expected_units=["Heater", "Vessel"],
        notes="Ether/alcohol; NRTL for polarity",
    ),
]

# ──────────────────────────────────────────────────────────────────────────────
# H: Hard (10) — multi-step, recycle, ternary, azeotrope, high-pressure
# ──────────────────────────────────────────────────────────────────────────────

HARD_CASES: list[BenchmarkCase] = [
    BenchmarkCase(
        case_id="H01",
        description="Separate a ternary acetone-methanol-water mixture: heat feed "
                    "to 65°C, flash to remove low-boiling vapour, cool and recycle "
                    "the liquid back to the feed mixer",
        compounds=["acetone", "methanol", "water"],
        tier="hard", expected_pkg="NRTL",
        expected_units=["Heater", "Vessel", "Cooler"],
        notes="Ternary polar; recycle path expected",
    ),
    BenchmarkCase(
        case_id="H02",
        description="Compress a CO2-rich gas stream from 1 bar to 80 bar in two "
                    "stages with intercooling between stages to remove heat of compression",
        compounds=["carbon dioxide"],
        tier="hard", expected_pkg="Peng-Robinson",
        expected_units=["Compressor", "Cooler"],
        notes="Two-stage compression with intercooler; topology test",
    ),
    BenchmarkCase(
        case_id="H03",
        description="Cryogenic separation of air: compress nitrogen-oxygen mixture "
                    "to 50 bar, cool to -150°C, expand through a turbine to produce "
                    "work, then flash separate liquid oxygen from gaseous nitrogen",
        compounds=["nitrogen", "oxygen"],
        tier="hard", expected_pkg="Lee-Kesler-Plöcker",
        expected_units=["Compressor", "Cooler", "Expander", "Vessel"],
        notes="Cryogenic; Lee-Kesler-Plöcker required",
    ),
    BenchmarkCase(
        case_id="H04",
        description="Separate tetrahydrofuran and water by heating to 70°C, "
                    "flashing the vapour, then cooling the vapour product to recover THF",
        compounds=["tetrahydrofuran", "water"],
        tier="hard", expected_pkg="NRTL",
        expected_units=["Heater", "Vessel", "Cooler"],
        notes="THF-water azeotrope; BIP injection required",
    ),
    BenchmarkCase(
        case_id="H05",
        description="Process an n-hexane and ethanol mixture: pump to 5 bar, "
                    "heat to 70°C, flash to split vapour and liquid, then split "
                    "the liquid product into two product streams",
        compounds=["n-hexane", "ethanol"],
        tier="hard", expected_pkg="NRTL",
        expected_units=["Pump", "Heater", "Vessel", "Splitter"],
        notes="Splitter insertion; hexane-ethanol heterogeneous azeotrope",
    ),
    BenchmarkCase(
        case_id="H06",
        description="Compress a nitrogen-methane mixture from 1 bar to 40 bar "
                    "in two stages, cooling after the first stage to 30°C before "
                    "the second stage, then flash to separate liquids",
        compounds=["nitrogen", "methane"],
        tier="hard", expected_pkg="Peng-Robinson",
        expected_units=["Compressor", "Cooler", "Compressor", "Vessel"],
        notes="Two compressors with intercooler + final flash",
    ),
    BenchmarkCase(
        case_id="H07",
        description="Ethanol dehydration simulation: heat feed to 80°C, flash "
                    "to remove bulk water, heat vapour to 120°C for further purification",
        compounds=["ethanol", "water"],
        tier="hard", expected_pkg="NRTL",
        expected_units=["Heater", "Vessel", "Heater"],
        notes="Two heaters with intermediate flash; multi-step topology",
    ),
    BenchmarkCase(
        case_id="H08",
        description="Methanol synthesis loop: compress syngas to 80 bar, heat to "
                    "250°C, flash separate methanol from unreacted gases, pump liquid "
                    "methanol to storage pressure, and recycle unreacted gas back to feed",
        compounds=["methanol", "carbon dioxide", "methane"],
        tier="hard", expected_pkg="Peng-Robinson",
        expected_units=["Compressor", "Heater", "Vessel", "Pump"],
        notes="Industrial loop with recycle; complex topology",
    ),
    BenchmarkCase(
        case_id="H09",
        description="Separate air into nitrogen, oxygen and argon: compress to 200 bar, "
                    "cool to -170°C, expand and flash three times to separate components",
        compounds=["nitrogen", "oxygen", "argon"],
        tier="hard", expected_pkg="Lee-Kesler-Plöcker",
        expected_units=["Compressor", "Cooler", "Expander", "Vessel"],
        notes="Ternary cryogenic; LKP and multi-flash",
    ),
    BenchmarkCase(
        case_id="H10",
        description="CO2 liquefaction with heat integration: compress CO2 to 73 bar, "
                    "cool feed using a cold product stream, then condense and expand "
                    "through a turbine to produce liquid CO2",
        compounds=["carbon dioxide"],
        tier="hard", expected_pkg="Peng-Robinson",
        expected_units=["Compressor", "Cooler", "Expander"],
        notes="Supercritical CO2; heat integration implied",
    ),
]

# ──────────────────────────────────────────────────────────────────────────────
# U: Underspecified (8) — missing critical information
# ──────────────────────────────────────────────────────────────────────────────

UNDERSPEC_CASES: list[BenchmarkCase] = [
    BenchmarkCase(
        case_id="U01",
        description="Heat methanol",
        compounds=["methanol"],
        tier="underspec", expected_pkg="Peng-Robinson",
        expected_units=["Heater"],
        notes="No target temperature — system should use default or flag",
    ),
    BenchmarkCase(
        case_id="U02",
        description="Separate ethanol and water",
        compounds=["ethanol", "water"],
        tier="underspec", expected_pkg="NRTL",
        expected_units=["Vessel"],
        notes="No conditions specified — system should infer or flag missing T/P",
    ),
    BenchmarkCase(
        case_id="U03",
        description="Process a natural gas stream by removing liquids",
        compounds=["methane", "ethane", "propane"],
        tier="underspec", expected_pkg="Peng-Robinson",
        expected_units=["Cooler", "Vessel"],
        notes="No composition or conditions specified",
    ),
    BenchmarkCase(
        case_id="U04",
        description="Flash separate the mixture",
        compounds=["acetone", "water"],
        tier="underspec", expected_pkg="NRTL",
        expected_units=["Vessel"],
        notes="Minimal description; no upstream conditioning specified",
    ),
    BenchmarkCase(
        case_id="U05",
        description="Compress the gas feed to high pressure",
        compounds=["methane"],
        tier="underspec", expected_pkg="Peng-Robinson",
        expected_units=["Compressor"],
        notes="No target pressure — system should use default or flag",
    ),
    BenchmarkCase(
        case_id="U06",
        description="Cool and condense the overhead stream",
        compounds=["ethanol", "water"],
        tier="underspec", expected_pkg="NRTL",
        expected_units=["Cooler"],
        notes="No compounds or inlet T specified",
    ),
    BenchmarkCase(
        case_id="U07",
        description="Mix the two feeds and then heat the combined stream",
        compounds=["methanol", "water"],
        tier="underspec", expected_pkg="NRTL",
        expected_units=["Mixer", "Heater"],
        notes="No quantities or target temperature given",
    ),
    BenchmarkCase(
        case_id="U08",
        description="Distil the acetone-water mixture to recover pure acetone",
        compounds=["acetone", "water"],
        tier="underspec", expected_pkg="NRTL",
        expected_units=["Vessel"],
        notes="Describes a distillation column (not in unit set) — system must approximate",
    ),
]

# ──────────────────────────────────────────────────────────────────────────────
# AMB: Ambiguous (6) — multiple valid interpretations
# ──────────────────────────────────────────────────────────────────────────────

AMBIGUOUS_CASES: list[BenchmarkCase] = [
    BenchmarkCase(
        case_id="AMB01",
        description="Heat and then cool an ethanol stream",
        compounds=["ethanol"],
        tier="ambiguous", expected_pkg="Peng-Robinson",
        expected_units=["Heater", "Cooler"],
        notes="Could be heat-then-cool (two units) or a temperature cycle",
    ),
    BenchmarkCase(
        case_id="AMB02",
        description="Process a hot gas stream of methane",
        compounds=["methane"],
        tier="ambiguous", expected_pkg="Peng-Robinson",
        expected_units=[],
        notes="Ambiguous: cool it? expand it? separate it?",
    ),
    BenchmarkCase(
        case_id="AMB03",
        description="Separate components at low temperature from a propane stream",
        compounds=["propane"],
        tier="ambiguous", expected_pkg="Peng-Robinson",
        expected_units=["Vessel"],
        notes="Ambiguous: flash? condensation? which components?",
    ),
    BenchmarkCase(
        case_id="AMB04",
        description="Pressurize and heat the ethanol-water feed",
        compounds=["ethanol", "water"],
        tier="ambiguous", expected_pkg="NRTL",
        expected_units=["Heater"],
        notes="Ambiguous: Pump+Heater (liquid) or Compressor+Heater (vapour)?",
    ),
    BenchmarkCase(
        case_id="AMB05",
        description="Mix and separate a methanol-water stream",
        compounds=["methanol", "water"],
        tier="ambiguous", expected_pkg="NRTL",
        expected_units=["Mixer", "Vessel"],
        notes="Underdetermined: mix what? separate how?",
    ),
    BenchmarkCase(
        case_id="AMB06",
        description="Recover solvent from the acetone mixture",
        compounds=["acetone", "water"],
        tier="ambiguous", expected_pkg="NRTL",
        expected_units=["Vessel"],
        notes="Recovery implies separation but does not specify method",
    ),
]

# ──────────────────────────────────────────────────────────────────────────────
# ADV: Adversarial (7) — contradictory, invalid physics, unknown compounds
# ──────────────────────────────────────────────────────────────────────────────

ADVERSARIAL_CASES: list[BenchmarkCase] = [
    BenchmarkCase(
        case_id="ADV01",
        description="Cool an ethanol stream from 25°C to 200°C",
        compounds=["ethanol"],
        tier="adversarial", expected_pkg="Peng-Robinson",
        expected_units=["Cooler"],
        notes="T_out > T_in for a Cooler — thermodynamic violation",
        expect_failure=True,
    ),
    BenchmarkCase(
        case_id="ADV02",
        description="Compress liquid water using a compressor from 1 atm to 50 atm",
        compounds=["water"],
        tier="adversarial", expected_pkg="Peng-Robinson",
        expected_units=["Compressor"],
        notes="Compressor expects vapour; liquid → compressor is invalid",
        expect_failure=True,
    ),
    BenchmarkCase(
        case_id="ADV03",
        description="Pump a methane gas stream to 10 bar",
        compounds=["methane"],
        tier="adversarial", expected_pkg="Peng-Robinson",
        expected_units=["Pump"],
        notes="Pump expects liquid; vapour → pump is invalid",
        expect_failure=True,
    ),
    BenchmarkCase(
        case_id="ADV04",
        description="Flash benzene at -300°C to separate vapour and liquid",
        compounds=["benzene"],
        tier="adversarial", expected_pkg="Peng-Robinson",
        expected_units=["Vessel"],
        notes="T = -300°C = -27 K — physically impossible",
        expect_failure=True,
    ),
    BenchmarkCase(
        case_id="ADV05",
        description="Separate a stream of unobtainium and phlogiston by heating",
        compounds=["water"],
        tier="adversarial", expected_pkg="",
        expected_units=[],
        notes="Non-existent compounds — system should use fallback compound or flag",
        expect_failure=True,
    ),
    BenchmarkCase(
        case_id="ADV06",
        description="Expand water through an expander with efficiency of 2.5",
        compounds=["water"],
        tier="adversarial", expected_pkg="Peng-Robinson",
        expected_units=["Expander"],
        notes="efficiency=2.5 violates thermodynamic limits",
        expect_failure=True,
    ),
    BenchmarkCase(
        case_id="ADV07",
        description="",
        compounds=["ethanol"],
        tier="adversarial", expected_pkg="",
        expected_units=[],
        notes="Empty description — system must not crash",
        expect_failure=True,
    ),
]

# ──────────────────────────────────────────────────────────────────────────────
# EDGE: Edge thermodynamic (7) — cryogenic, supercritical, near-critical, etc.
# ──────────────────────────────────────────────────────────────────────────────

EDGE_CASES: list[BenchmarkCase] = [
    BenchmarkCase(
        case_id="EDGE01",
        description="Process a supercritical CO2 stream at 315 K and 80 bar: "
                    "expand it through a turbine then flash to recover liquid CO2",
        compounds=["carbon dioxide"],
        tier="edge", expected_pkg="Peng-Robinson",
        expected_units=["Expander", "Vessel"],
        notes="Supercritical CO2; Tc=304 K, Pc=73.8 bar",
    ),
    BenchmarkCase(
        case_id="EDGE02",
        description="Cool a hydrogen stream from -200°C to -250°C to liquefy it, "
                    "then pump the liquid hydrogen to 20 bar",
        compounds=["hydrogen"],
        tier="edge", expected_pkg="Lee-Kesler-Plöcker",
        expected_units=["Cooler", "Pump"],
        notes="Cryogenic liquid hydrogen; very low T",
    ),
    BenchmarkCase(
        case_id="EDGE03",
        description="Heat a water and acetic acid mixture to 100°C then flash "
                    "to separate the more volatile acetic acid vapour",
        compounds=["acetic acid", "water"],
        tier="edge", expected_pkg="NRTL",
        expected_units=["Heater", "Vessel"],
        notes="Strong H-bonding; acetic acid dimerisation; NRTL required",
    ),
    BenchmarkCase(
        case_id="EDGE04",
        description="Heat chloroform and acetone mixture to 63°C then flash — "
                    "this pair forms a maximum-boiling azeotrope",
        compounds=["chloroform", "acetone"],
        tier="edge", expected_pkg="NRTL",
        expected_units=["Heater", "Vessel"],
        notes="Maximum-boiling azeotrope (rare); NRTL required",
    ),
    BenchmarkCase(
        case_id="EDGE05",
        description="Separate near-azeotropic ethyl acetate and ethanol by heating "
                    "to 72°C then flashing under slight vacuum at 80 kPa",
        compounds=["ethyl acetate", "ethanol"],
        tier="edge", expected_pkg="NRTL",
        expected_units=["Heater", "Vessel"],
        notes="Near-azeotrope; subatmospheric pressure",
    ),
    BenchmarkCase(
        case_id="EDGE06",
        description="Process a hydrogen sulfide and water stream: cool from 60°C "
                    "to 20°C then flash to separate H2S-rich vapour from water",
        compounds=["hydrogen sulfide", "water"],
        tier="edge", expected_pkg="NRTL",
        expected_units=["Cooler", "Vessel"],
        notes="H2S/water; NRTL with safety implications; acid gas",
    ),
    BenchmarkCase(
        case_id="EDGE07",
        description="Separate a trace-oxygen nitrogen stream (99.9% N2, 0.1% O2) "
                    "by cooling to -190°C at 4 bar, expanding and flashing",
        compounds=["nitrogen", "oxygen"],
        tier="edge", expected_pkg="Lee-Kesler-Plöcker",
        expected_units=["Cooler", "Expander", "Vessel"],
        notes="High-purity cryogenic nitrogen; trace component tracking",
    ),
]


# ── Combined dataset ───────────────────────────────────────────────────────────

BENCHMARK_CASES: list[BenchmarkCase] = (
    EASY_CASES
    + MEDIUM_CASES
    + HARD_CASES
    + UNDERSPEC_CASES
    + AMBIGUOUS_CASES
    + ADVERSARIAL_CASES
    + EDGE_CASES
)

# Convenience lookup
CASES_BY_ID: dict[str, BenchmarkCase] = {c.case_id: c for c in BENCHMARK_CASES}

# Category groups for stratified evaluation
CASE_TIERS: dict[str, list[BenchmarkCase]] = {
    "easy":        EASY_CASES,
    "medium":      MEDIUM_CASES,
    "hard":        HARD_CASES,
    "underspec":   UNDERSPEC_CASES,
    "ambiguous":   AMBIGUOUS_CASES,
    "adversarial": ADVERSARIAL_CASES,
    "edge":        EDGE_CASES,
}
