# Appendix X — Capability Benchmark: Reference Stream Provenance

This appendix documents the origin, construction, and validation of every reference stream condition used to score the small-flowsheet capability benchmark. Its purpose is to establish that the reference values are (a) independently constructed (not harvested from the system under test — non-circular), (b) physically validated against literature/first-principles where feasible, and (c) characterised for precision relative to the scoring tolerances.

## X.1 How the references were constructed

The capability-benchmark cases are standard small (2–4 unit) processes drawn from the process-description set, selected for being schema-representable and thermodynamically well-posed. **Reference stream conditions were not taken from the system pipeline's own output** (which would be circular). Instead, for each case the *correct* flowsheet was specified independently (units, connectivity, and operating conditions determined from the process description by expert judgement) and constructed directly in DWSIM via the simulator wrapper, bypassing the system's extraction and IR-construction stages (`VARIANT_B` / reference-injection disabled). The specified flowsheet was solved in DWSIM, and the resulting stream conditions (T, P, vapour fraction, composition, molar flow) were recorded as the reference.

Each reference is therefore an **expert-specified DWSIM solution** — not experimental ground truth. Because the system under test is scored by comparing its DWSIM output against this reference DWSIM output (same solver, same property models), the comparison isolates *flowsheet-construction correctness* (did the system build the right units, connectivity, and conditions) from *simulator thermodynamic accuracy* (which cancels in a DWSIM-to-DWSIM comparison). This is the appropriate measurement for a flowsheet-generation system: it tests whether the system reproduces the correct flowsheet, not whether DWSIM's thermodynamics match reality.

## X.2 Precision characterisation and scoring policy

Reference values divide into two classes with different precision:

- **Set-point quantities (exact):** operating conditions specified by the process description and imposed on the DWSIM model — e.g. a compressor discharge pressure ("compress to 15 bar"), a cooler/heater outlet temperature ("cool to 25 °C", "heat to 95 °C"). These are exact by construction; the reference value is the specified set-point. Stream T and P at controlled-unit outlets fall here.
- **Computed quantities (model-dependent):** conditions DWSIM computes from the thermodynamic model — e.g. flash vapour fraction, post-flash phase compositions, compressor discharge temperature. These carry the property model's uncertainty.

Scoring policy follows this split. **Pass/fail gating keys on the CRITICAL T and P checks** (the exact set-point references); vapour-fraction agreement is a **non-gating WARNING**. A vapour-fraction warning therefore signals only a small deviation in a computed quantity, not an incorrect build.

A vapour-fraction stability analysis (§X.4) further identifies which computed vf references are precise to the ±0.05 scoring tolerance and which are model-sensitive; the latter are treated as secondary.

## X.3 Per-case provenance

For each case: the process, the specified reference flowsheet, the basis of each specified condition, the property package, and the validation performed. Each entry closes with the exact per-stream reference table (T, P, vapour fraction, composition, molar flow), transcribed directly from the corresponding `*_reference.json` — temperatures converted K→°C, pressures Pa→bar.

### SAN_04 — Propane compression + condensation
- **Process:** Compress propane vapour from 1 bar to 10 bar, then cool the compressed gas to 25 °C.
- **Reference flowsheet:** FEED → CP-01 (Compressor) → COMP → CL-01 (Cooler) → PROD.
- **Specified conditions & basis:** FEED = propane, 25 °C / 1 bar / 1.0 mol/s vapour (feed T unstated → 25 °C ambient assumption; 1 mol/s basis). CP-01: P_out = 10 bar (from description); adiabatic efficiency η = 0.75 (unstated → standard value; discharge T depends on it). CL-01: T_out = 25 °C (from description), ΔP = 0.
- **Property package:** Peng-Robinson (pure hydrocarbon).
- **Validation:** Compressor discharge T = 129 °C (DWSIM, real-gas PR); ideal-gas isentropic estimate with η = 0.75 gives ~143 °C — DWSIM 13 °C lower, consistent with real-gas / temperature-dependent Cp reducing the rise (correct magnitude and direction). Phase behaviour cross-checked vs NIST: propane Psat(25 °C) ≈ 9.5 bar; FEED at 1 bar < 9.5 → vapour ✓; PROD at 10 bar > 9.5 → liquid (full condensation) ✓. Mass balance 1.0 mol/s conserved ✓.
- **Precision caveat:** PROD (25 °C, 10 bar) sits ~2 °C below the propane bubble point — borderline; vf treated as secondary here (a small cooling deviation would begin flashing it).

| Stream | T (°C) | P (bar) | Vapour fraction | Composition (mole fractions) | Molar flow (mol/s) |
|---|---|---|---|---|---|
| FEED | 25.0 | 1.00 | 1.0000 | propane 1.0 | 1.0000 |
| COMP | 129.4 | 10.00 | 1.0000 | propane 1.0 | 1.0000 |
| PROD | 25.0 | 10.00 | 0.0000 | propane 1.0 | 1.0000 |

### GEN_03 — n-Heptane compression
- **Process:** Compress n-heptane vapour from 0.2 bar to 2 bar, cool to 40 °C.
- **Reference flowsheet:** FEED → CP-01 (Compressor) → COMP → CL-01 (Cooler) → PROD.
- **Specified conditions & basis:** FEED = n-heptane, 60 °C / 0.2 bar (feed 60 °C so it is vapour at 0.2 bar — n-heptane boils ~51 °C at 0.2 bar; stated because it drives the result). CP-01: P_out = 2 bar, η = 0.75. CL-01: T_out = 40 °C.
- **Property package:** Peng-Robinson.
- **Validation:** COMP two-phase (vf ≈ 0.89, at the 2-bar dew point); PROD liquid (vf = 0). NIST cross-check: n-heptane Psat(40 °C) ≈ 0.12 bar ≪ 2 bar → liquid ✓. Naturally exercises a two-phase intermediate.

| Stream | T (°C) | P (bar) | Vapour fraction | Composition (mole fractions) | Molar flow (mol/s) |
|---|---|---|---|---|---|
| FEED | 60.0 | 0.20 | 1.0000 | n-heptane 1.0 | 1.0000 |
| COMP | 123.3 | 2.00 | 0.8890 | n-heptane 1.0 | 1.0000 |
| PROD | 40.0 | 2.00 | 0.0000 | n-heptane 1.0 | 1.0000 |

### EASY_04 — Propylene compression + condensation
- **Process:** Compress propylene, cool to 30 °C to condense.
- **Reference flowsheet:** FEED → CP-01 (Compressor, 15 bar) → CL-01 (Cooler, 30 °C) → PROD.
- **Specified conditions & basis:** CP-01 P_out = 15 bar. *Original description specified 12 bar, which does NOT condense at 30 °C (propylene Psat(30 °C) ≈ 13 bar > 12 bar); condition corrected to 15 bar so the described condensation occurs.* (Correction documented — original process condition was thermodynamically inconsistent.)
- **Property package:** Peng-Robinson.
- **Validation:** PROD liquid (vf = 0) at 30 °C; propylene Psat(30 °C) ≈ 13 bar < 15 bar → condenses ✓.

| Stream | T (°C) | P (bar) | Vapour fraction | Composition (mole fractions) | Molar flow (mol/s) |
|---|---|---|---|---|---|
| FEED | 25.0 | 1.00 | 1.0000 | propylene 1.0 | 1.0000 |
| COMP | 168.6 | 15.00 | 1.0000 | propylene 1.0 | 1.0000 |
| PROD | 30.0 | 15.00 | 0.0000 | propylene 1.0 | 1.0000 |

### EASY_02 — Propane condensation + pumping
- **Process:** Cool propane to condense, then pump.
- **Reference flowsheet:** FEED → CL-01 (Cooler, condense at 10 bar to 20 °C) → PM-01 (Pump, 10 → 20 bar) → PROD.
- **Specified conditions & basis:** *Original description specified cool at 2 bar / 0 °C then pump — but propane boils at −25 °C at 2 bar, so 0 °C / 2 bar is superheated vapour with no liquid to pump (thermodynamically inconsistent). Restructured to condense at 10 bar (20 °C, below the 27 °C bubble point) so there is genuine liquid to pump.* (Correction documented.)
- **Property package:** Peng-Robinson.
- **Validation:** COOL liquid (vf = 0) at 10 bar / 20 °C; genuine liquid pumped 10 → 20 bar ✓.

| Stream | T (°C) | P (bar) | Vapour fraction | Composition (mole fractions) | Molar flow (mol/s) |
|---|---|---|---|---|---|
| FEED | 50.0 | 10.00 | 1.0000 | propane 1.0 | 1.0000 |
| COOL | 20.0 | 10.00 | 0.0000 | propane 1.0 | 1.0000 |
| PROD | 21.1 | 20.00 | 0.0000 | propane 1.0 | 1.0000 |

### SAN_03 — Benzene/toluene partial vaporisation + flash
- **Process:** Heat a benzene/toluene mixture and flash.
- **Reference flowsheet:** FEED → HT-01 (Heater, 95 °C) → HOT → V-01 (Flash Vessel) → VAP + LIQ.
- **Specified conditions & basis:** Heater outlet 95 °C, chosen to place the flash inlet in a genuine two-phase state (vf ≈ 0.45) so vapour-fraction scoring is exercised. *Original description said 100 °C, which would be all-vapour; corrected to 95 °C to match the two-phase reference.*
- **Property package:** Peng-Robinson.
- **Validation:** 95 °C lies between the mixture bubble point (~92 °C) and dew point (~98 °C) → two-phase ✓; HOT vf = 0.45.
- **Precision caveat (see §X.4):** benzene/toluene is close-boiling (~6 °C two-phase window); flash vf is a steep function of temperature (Δvf ≈ 0.29 per ±2 °C). **Reference vf is NOT precise to the ±0.05 tolerance — treated as secondary/non-gating.** T and P references remain exact.

| Stream | T (°C) | P (bar) | Vapour fraction | Composition (mole fractions) | Molar flow (mol/s) |
|---|---|---|---|---|---|
| FEED | 25.0 | 1.01 | 0.0000 | benzene 0.5 / toluene 0.5 | 1.0000 |
| HOT | 95.0 | 1.01 | 0.4500 | benzene 0.5 / toluene 0.5 | 1.0000 |
| VAP | 95.0 | 1.01 | 1.0000 | benzene 0.6186 / toluene 0.3814 | 0.4500 |
| LIQ | 95.0 | 1.01 | 0.0000 | benzene 0.4029 / toluene 0.5971 | 0.5500 |

### GEN_01 — n-Hexane/n-heptane partial vaporisation + flash
- **Process:** Heat a hexane/heptane mixture and flash.
- **Reference flowsheet:** FEED → HT-01 (Heater, 85 °C) → HOT → V-01 (Flash Vessel) → VAP + LIQ.
- **Specified conditions & basis:** Heater outlet 85 °C, for a two-phase flash (vf ≈ 0.56). *Original description said 80 °C, which would be all-liquid; corrected to 85 °C.*
- **Property package:** Peng-Robinson.
- **Validation:** 85 °C between bubble (~81 °C) and dew (~89 °C) → two-phase ✓; HOT vf = 0.56.
- **Precision caveat (see §X.4):** close-boiling (~8 °C window); Δvf ≈ 0.34 per ±2 °C. **Reference vf treated as secondary/non-gating.** T/P exact.

| Stream | T (°C) | P (bar) | Vapour fraction | Composition (mole fractions) | Molar flow (mol/s) |
|---|---|---|---|---|---|
| FEED | 25.0 | 1.01 | 0.0000 | n-hexane 0.5 / n-heptane 0.5 | 1.0000 |
| HOT | 85.0 | 1.01 | 0.5600 | n-hexane 0.5 / n-heptane 0.5 | 1.0000 |
| VAP | 85.0 | 1.01 | 1.0000 | n-hexane 0.5901 / n-heptane 0.4099 | 0.5600 |
| LIQ | 85.0 | 1.01 | 0.0000 | n-hexane 0.3854 / n-heptane 0.6146 | 0.4400 |

### EASY_01 — Acetone/water partial vaporisation + flash
- **Process:** Heat an equimolar (50/50) acetone/water mixture and flash.
- **Reference flowsheet:** FEED → HT-01 (Heater, 70 °C) → HOT → V-01 (Flash Vessel) → VAP + LIQ.
- **Specified conditions & basis:** Feed 25 °C, 50/50 acetone/water; heater outlet 70 °C for a partial vaporisation (vf ≈ 0.68) yielding an acetone-enriched vapour. *An acetone-rich 70/30 feed was originally specified, but it has no stable mid-range flash at 1 atm — 70/30 flashes almost completely at 70 °C (vf ≈ 0.99) and its bubble point is so close that ±2 °C spans all-liquid to two-thirds-vapour. The equimolar composition gives the robust, wide-window two-phase flash the case is meant to demonstrate; feed corrected to 50/50.* (Correction documented.)
- **Property package:** NRTL (non-ideal acetone/water — exercises activity-model selection).
- **Validation:** Partial vaporisation with acetone-enriched vapour (VAP acetone ≈ 0.71) ✓; HOT vf = 0.68; overall component balance closes to <1e-5.
- **Precision (see §X.4):** wide-boiling (acetone/water ~44 °C apart); gentle T–vf slope (Δvf ≈ 0.038 per ±2 °C, within ±0.05). **Reference vf IS precise to tolerance.**

| Stream | T (°C) | P (bar) | Vapour fraction | Composition (mole fractions) | Molar flow (mol/s) |
|---|---|---|---|---|---|
| FEED | 25.0 | 1.01 | 0.0000 | acetone 0.5 / water 0.5 | 1.0000 |
| HOT | 70.0 | 1.01 | 0.6754 | acetone 0.5 / water 0.5 | 1.0000 |
| VAP | 70.0 | 1.01 | 1.0000 | acetone 0.7065 / water 0.2935 | 0.6754 |
| LIQ | 70.0 | 1.01 | 0.0000 | acetone 0.0705 / water 0.9295 | 0.3246 |

---

*The entries below are the extended capability series (P·/F·/C·/… IDs) — additional independently-built references that broaden unit-type and validation-route coverage. Same construction and validation discipline as above.*

### P1 — Two-stage nitrogen compression with intercooling
- **Process:** Nitrogen compressed 1 → 5 bar, cooled to 40 °C, compressed 5 → 25 bar, cooled again to 40 °C.
- **Reference flowsheet:** FEED → CP-01 (Compressor, 5 bar) → INT1 → CL-01 (Cooler, 40 °C) → COOLED → CP-02 (Compressor, 25 bar) → INT2 → CL-02 (Cooler, 40 °C) → PROD.
- **Specified conditions & basis:** FEED = N₂, 25 °C / 1 bar / 1.0 mol/s (feed T ambient assumption; 1 mol/s basis). CP-01 P_out = 5 bar, CP-02 P_out = 25 bar — **both intermediate pressures explicit**; η = 0.75 each (unstated → standard). CL-01 / CL-02 T_out = 40 °C, ΔP = 0.
- **Property package:** Peng-Robinson (light permanent gas; real-gas compression).
- **Validation (per-stage isentropic-with-efficiency, γ = 1.40, η = 0.75):** stage 1 → 257.1 °C, stage 2 → 283.8 °C; DWSIM 254.9 °C / 281.2 °C, ~2.5 °C below the ideal-gas estimate (real-gas / T-dependent Cp) — correct direction and magnitude. N₂ supercritical (T_c = 126 K) → vf = 1 throughout; mass balance 1.0 mol/s conserved.
- **Purpose:** directly targets the stage-collapse fault (cf. VAL_01). Both intermediate pressures are stated, so a system that fuses 1 → 25 bar into a single stage puts INT1/COOLED at 25 bar instead of 5 — an unmissable pressure error. No vf caveat (single-phase vapour throughout; vf trivially robust).

| Stream | T (°C) | P (bar) | Vapour fraction | Composition (mole fractions) | Molar flow (mol/s) |
|---|---|---|---|---|---|
| FEED | 25.0 | 1.00 | 1.0000 | nitrogen 1.0 | 1.0000 |
| INT1 | 254.9 | 5.00 | 1.0000 | nitrogen 1.0 | 1.0000 |
| COOLED | 40.0 | 5.00 | 1.0000 | nitrogen 1.0 | 1.0000 |
| INT2 | 281.2 | 25.00 | 1.0000 | nitrogen 1.0 | 1.0000 |
| PROD | 40.0 | 25.00 | 1.0000 | nitrogen 1.0 | 1.0000 |

### F1 — n-Pentane/n-octane partial vaporisation + flash
- **Process:** Heat an equimolar n-pentane/n-octane mixture to 75 °C and flash at 1 bar.
- **Reference flowsheet:** FEED → HT-01 (Heater, 75 °C) → HOT → V-01 (Flash Vessel) → VAP + LIQ.
- **Specified conditions & basis:** FEED = 50/50 n-pentane/n-octane, 25 °C / 1 bar / 1.0 mol/s; heater outlet 75 °C, chosen to give a mid-range two-phase flash (vf ≈ 0.38 ∈ [0.25, 0.75]).
- **Property package:** Peng-Robinson (near-ideal alkanes).
- **Validation (Antoine + Raoult + Rachford–Rice):** P°ₚₑₙₜₐₙₑ 3.234 / P°ₒ𝒸ₜₐₙₑ 0.193 bar → hand vf 0.396 vs DWSIM 0.384 (Δ 0.012, well within ±0.05); vapour y pentane 0.858 vs 0.851 — same direction. Mass balance closes exactly.
- **Precision (see §X.4):** wide-boiling (pentane 36 °C / octane 126 °C, ~90 °C apart); gentle T–vf slope (Δvf ≈ 0.031 per ±2 °C, within ±0.05). **Reference vf IS precise to tolerance** — the wide-boiling counterpart demonstrating the reliable-vf regime that the close-boiling SAN_03/GEN_01 do not.

| Stream | T (°C) | P (bar) | Vapour fraction | Composition (mole fractions) | Molar flow (mol/s) |
|---|---|---|---|---|---|
| FEED | 25.0 | 1.00 | 0.0000 | n-pentane 0.5 / n-octane 0.5 | 1.0000 |
| HOT | 75.0 | 1.00 | 0.3840 | n-pentane 0.5 / n-octane 0.5 | 1.0000 |
| VAP | 75.0 | 1.00 | 1.0000 | n-pentane 0.8508 / n-octane 0.1492 | 0.3840 |
| LIQ | 75.0 | 1.00 | 0.0000 | n-pentane 0.2814 / n-octane 0.7186 | 0.6160 |

### C1 — Benzene/toluene shortcut (FUG) distillation column
- **Process:** Separate an equimolar benzene/toluene mixture in a distillation column at atmospheric pressure to 98% benzene in the distillate and 98% toluene in the bottoms.
- **Reference flowsheet:** FEED → COL-01 (ShortcutColumn) → DIST + BOT.
- **Specified conditions & basis:** FEED = 50/50 benzene/toluene, **saturated liquid (q = 1)** at the 1 atm bubble point (92 °C) / 1.0 mol/s. COL-01: light key = benzene, heavy key = toluene; LK-in-bottoms = 0.02, HK-in-distillate = 0.02; condenser and reboiler at 1 atm. Reflux via the **two-pass logic** — a seed R below R_min is set, the solver reads the computed R_min and sets R = 1.3 × R_min.
- **Property package:** Peng-Robinson (near-ideal aromatics).
- **Validation (Fenske / Underwood):** α (NIST Antoine, 95 °C column-average) = 2.48 → **Fenske N_min = 8.58 (hand) vs 8.77 (DWSIM)**; **Underwood R_min = 1.26 (hand) vs 1.30 (DWSIM)** — within ~2–3.5%, standard textbook values, matching the Stage-A probe (8.76 / 1.30). Final R = 1.695 = 1.300 × R_min (two-pass fired). **Mass balance closes exactly (0.00%)** — see §X.5.
- **vf note:** DIST/BOT are saturated-liquid products (vf ≈ 0; total condenser, reboiler liquid). Column temperatures are computed from the FUG specs, not specified, so there is no flash-temperature set-point to perturb — the ±2 °C vf-stability test does not apply, and vf rounds to 0 for both products.

| Stream | T (°C) | P (bar) | Vapour fraction | Composition (mole fractions) | Molar flow (mol/s) |
|---|---|---|---|---|---|
| FEED | 92.0 | 1.01 | 0.0004 | benzene 0.5 / toluene 0.5 | 1.0000 |
| DIST | 80.1 | 1.01 | 0.0009 | benzene 0.98 / toluene 0.02 | 0.5000 |
| BOT | 109.7 | 1.01 | 0.0032 | benzene 0.02 / toluene 0.98 | 0.5000 |

## X.4 Vapour-fraction reference precision (stability analysis)

Each flash reference vf was tested for stability by rebuilding the DWSIM model under small perturbations (flash-temperature ±2 °C; feed-temperature ±2 °C).

| Case | Mixture | Base vf | Δvf per ±2 °C flash-T | Δvf per ±2 °C feed-T | Verdict |
|---|---|---|---|---|---|
| SAN_03 | benzene/toluene | 0.450 | 0.294 | 0.000 | model-sensitive → vf secondary |
| GEN_01 | hexane/heptane | 0.560 | 0.343 | 0.000 | model-sensitive → vf secondary |
| EASY_01 | acetone/water | 0.675 | 0.038 | 0.000 | precise-to-tolerance |
| F1 | pentane/octane | 0.384 | 0.031 | 0.000 | precise-to-tolerance |

**Findings:** (1) Feed-temperature perturbation moves flash vf by exactly zero — the flash sits at the heater-outlet set-point regardless of feed state, so feed-condition assumptions never threaten the vf reference. (2) Flash-temperature sensitivity scales with how close-boiling the mixture is: the close-boiling binaries (SAN_03, GEN_01, ~6–8 °C two-phase windows) have steep vf-vs-T curves (a ±2 °C deviation swings vf by ~0.3, well beyond the ±0.05 tolerance), so their vf reference is meaningful only if the build matches the flash temperature to ~0.3 °C; the wide-boiling cases (EASY_01 acetone/water ~44 °C apart; F1 pentane/octane ~90 °C apart) have gentle slopes and tolerance-precise vf. Pure-component condensation and permanent-gas compression cases (vf = 0 or 1) are robust by construction, except SAN_04 (borderline, treated as secondary).

## X.5 Integrity notes

- References are independently constructed (expert-specified DWSIM models), never harvested from the system under test; reference-injection into extraction (`VARIANT_B`) is disabled during scoring.
- Self-consistency verified: feeding each reference back through the scoring path yields 0.0 MAPE on T, P, and vf for **every** case (every stream matches), confirming the references and scoring pipeline agree and no min-match gate is triggered.
- **Shortcut (FUG) column mass balance closes exactly.** For the column cases (C1 and its variants), the Fenske–Underwood–Gilliland distillate/bottoms split conserves every component to <1e-6 — there is **no** inherent imbalance floor limiting achievable MAPE. The ~4% imbalance seen in an early column probe was a wrapper bug (the mass-based stream property `PROP_MS_2` was read as molar flow, so the D/B split was mass-weighted), since fixed; it was never a limitation of the shortcut method. Column references therefore carry exact molar splits, and the reflux is set by the two-pass logic (R = 1.3 × R_min).
- Four cases required correction of thermodynamically-inconsistent conditions in the original process descriptions: EASY_04 (12→15 bar), EASY_02 (restructured to condense at 10 bar before pumping), SAN_03 (100→95 °C flash), GEN_01 (80→85 °C flash), and EASY_01 (feed 70/30→50/50, since 70/30 acetone/water has no stable mid-range flash at 1 atm); all are documented above with their physical justification. These corrections were made to render physically-inconsistent original specifications self-consistent — **not** to make the cases easier: the corrected description simply gives the system a thermodynamically consistent set-point, and the test remains whether the system builds the correct flowsheet for that specification.
- Scoring gates on exact set-point T/P; vf is non-gating and, for the two close-boiling flashes plus SAN_04, treated as secondary per the stability analysis.
