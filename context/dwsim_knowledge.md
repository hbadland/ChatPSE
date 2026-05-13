# DWSIM Simulation Knowledge Base

Reference for Planner, Thermodynamics, and Critic agents.
Contains DWSIM-specific rules and constraints — not general chemistry.

---

## Topology Rules

- **Unit ops never connect directly.** Every connection between two unit operations must pass through a MaterialStream. Violating this causes a hard DWSIM error.
- **Feed streams** have no incoming connections. They must carry T, P, flow, and composition.
- **Intermediate streams** link unit ops. They need only a tag — T, P, flow, composition are calculated.
- **Terminal streams** have no outgoing connections. They carry solver results.

### Connection port conventions
| src_port | Meaning |
|---|---|
| 0 | Primary outlet — vapour phase from Vessel, single outlet from Heater/Cooler/Mixer |
| 1 | Secondary outlet — liquid phase from Vessel, second split from Splitter |

dst_port is almost always 0.

---

## Unit Operations

### Heater
Required: `T_out` [K] — outlet temperature.
Optional: `dP` [Pa] — pressure drop (default 0).
Note: CalcMode must be OutletTemperature — this is set automatically by the wrapper. Default DWSIM mode is HeatAdded (heat duty), which will produce no temperature change.

### Cooler
Same as Heater. T_out must be below feed temperature.

### Vessel (Flash Separator)
Optional: `dP` [Pa] — pressure drop (default 0).
Produces: vapour (port 0) and liquid (port 1).
Requires: VLE-capable property package. Will produce no separation if property package has no binary parameters.

### Mixer
Optional: `dP` [Pa] — pressure drop (default 0).
Combines multiple inlet streams into one outlet.
All inlets must carry the same set of compounds.

### Splitter
Required: `split_fractions` — dict mapping outlet stream tag to fraction of total flow.
Fractions must sum to 1.0.
Does not perform phase separation — splits total flow by ratio only.

### Pump
Required: `P_out` [Pa] — outlet pressure.
Optional: `efficiency` [0–1] (default 0.75).
Liquid-only. Do not use for gas streams.

### Compressor
Required: `P_out` [Pa] — outlet pressure.
Optional: `efficiency` [0–1] (default 0.75).
Gas-phase. Cubic EOS (Peng-Robinson, SRK) gives more accurate results than activity models.

### Expander
Required: `P_out` [Pa] — outlet pressure.
Optional: `efficiency` [0–1] (default 0.75).
Gas-phase expansion. Same thermodynamic notes as Compressor.

---

## Property Packages

| Schema name | DWSIM internal name | Best for |
|---|---|---|
| Raoult's Law | Raoult's Law | Ideal mixtures, testing topology |
| NRTL | NRTL | Polar non-ideal VLE, azeotropes — needs binary params |
| UNIQUAC | UNIQUAC | Polar VLE + LLE, size-asymmetric mixtures — needs binary params |
| Peng-Robinson | Peng-Robinson (PR) | Hydrocarbons, light gases, high pressure |
| Soave-Redlich-Kwong | Soave-Redlich-Kwong (SRK) | Light hydrocarbons, high-T gas phase |
| Lee-Kesler-Plöcker | Lee-Kesler-Plöcker | Cryogenic, natural gas processing |

**Critical:** NRTL and UNIQUAC require binary interaction parameters for each compound pair in DWSIM's database. If parameters are absent, DWSIM silently defaults to ideal behaviour — the solver converges but outlet compositions equal feed compositions. This is the most common silent failure in DWSIM.

**Runtime BIP injection (confirmed API):** The CalibrationAgent injects literature parameters directly into DWSIM's property package object before solving, bypassing the built-in database:

```python
# NRTL — access path (both models use identical structure)
pkg  = property_packages["NRTL"]           # or "UNIQUAC"
muni = pkg.GetType().GetProperty("m_uni").GetValue(pkg)
ip   = muni.GetType().GetProperty("InteractionParameters").GetValue(muni)

# NRTL_IPData fields (public, non-InitOnly, set via GetField not GetProperty):
#   A12, A21 [K]  — τ_ij = A_ij / T
#   alpha12        — nonrandomness parameter (NRTL only)
#   comment        — source string
#   B12, B21, C12, C21 — temperature-dependent terms (zero for constant τ)

# Injection requires BOTH orderings; reverse entry swaps A12↔A21
pkg.GetType().GetProperty("AutoEstimateMissingNRTLUNIQUACParameters") \
   .SetValue(pkg, System.Boolean(False))   # must be disabled BEFORE any solve
```

UNIQUAC uses `UNIQUAC_IPData` with the same fields (no alpha12 field). DWSIM's built-in database has 105 NRTL pairs and 88 UNIQUAC pairs (in Portuguese compound names — English names also present).

**Fallback strategy:** When NRTL/UNIQUAC fails silently AND CalibrationAgent cannot find parameters (pair not in RAG corpus), fall back to Raoult's Law to confirm topology is correct.

---

## Unit Conversion Reference

| Quantity | SI unit (required) | Common mistake |
|---|---|---|
| Temperature | Kelvin [K] | Passing °C — 25°C should be 298.15 K |
| Pressure | Pascal [Pa] | Passing bar or kPa — 1 atm = 101325 Pa |
| Molar flow | mol/s | Passing kmol/h |
| Composition | mole fraction [0–1] | Passing mol% (divide by 100) |

---

## Known DWSIM Limitations (current wrapper)

- **No recycle loop support** — circular topologies require tear stream initialisation not yet implemented.
- **No reactive systems** — DWSIM reactors (PFR, CSTR) not yet wired.
- **No electrolyte packages** — NaCl, HCl, NaOH etc. will fail compound addition.
- **Distillation column** — ObjectType exists but rigorous column spec (reflux ratio, stages) not yet implemented in wrapper.
- **Per-unit property packages** — fully supported. Set `"property_package"` on any unit in the flowsheet JSON; the executor applies it via `DWSIMFlowsheet.set_unit_property_package()`. If DWSIM's unit type does not expose a settable `PropertyPackage` attribute, the executor catches the exception and returns an error (routes to HUMAN).
