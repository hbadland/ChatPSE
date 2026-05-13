# Flowsheet Failure Taxonomy

Use these codes exclusively when diagnosing simulation failures.
Each entry defines: what triggers it, what it means physically, where to route it, and what fix to suggest.

---

## Convergence Failures

### SOLVER_FAIL
**Trigger:** `sim.Solved = False` with no other explanation.
**Meaning:** DWSIM's equation-solving loop did not converge within iteration limits.
**Routing:** REFINER
**Fix pattern:** Tighten feed conditions (reduce temperature step), check for recycle loops without tear streams, simplify topology and re-solve incrementally.

### NUMERIC_FAIL
**Trigger:** NaN or Inf in any stream T, P, or flow.
**Meaning:** A division by zero or overflow occurred — usually caused by unphysical initial conditions or a missing property package parameter.
**Routing:** THERMO if composition unchanged from feed; REFINER otherwise.
**Fix pattern:** Check T and P units (must be K and Pa). Verify property package has parameters for all compound pairs.

---

## Conservation Failures

### MASS_BALANCE
**Trigger:** Total molar flow at terminal outlets differs from feed by more than 1%.
**Meaning:** Streams are disconnected, a unit operation is bypassed, or the solver partially converged.
**Routing:** REFINER
**Fix pattern:** Verify all intermediate streams appear in both a source connection and a destination connection. Check no unit op is isolated.

---

## Physical Bounds Failures

### UNPHYSICAL_T
**Trigger:** Any stream T < 100 K or T > 2000 K.
**Meaning:** Almost always a unit conversion error (°C passed instead of K) or solver divergence.
**Routing:** REFINER
**Fix pattern:** Confirm all temperatures in JSON are in Kelvin. 25°C = 298.15 K. 80°C = 353.15 K.

### UNPHYSICAL_P
**Trigger:** Any stream P < 100 Pa or P > 1×10⁸ Pa.
**Meaning:** Unit conversion error (bar or kPa passed instead of Pa) or solver divergence.
**Routing:** REFINER
**Fix pattern:** Confirm all pressures in JSON are in Pascals. 1 atm = 101325 Pa. 1 bar = 100000 Pa.

### ENERGY_UNPHYSICAL
**Trigger:** Heater outlet T < feed T, or Cooler outlet T > feed T.
**Meaning:** T_out spec is wrong — heater is set to cool or cooler is set to heat.
**Routing:** REFINER
**Fix pattern:** Swap T_out value or correct the unit type (Heater vs Cooler).

---

## Phase Split Failures

### ZERO_OUTLET
**Trigger:** A terminal outlet stream has molar flow = 0.
**Meaning:** Either (a) no phase of that type exists at the given conditions (physically correct), or (b) the property package produced no separation.
**Routing:** Check first whether zero flow is physically expected. If not → THERMO.
**Fix pattern:** If flash vessel: check feed T is above bubble point for vapour outlet, below dew point for liquid outlet. If not, the property package may be missing parameters.

### NO_SEPARATION
**Trigger:** Outlet stream compositions are identical (within 1%) to feed composition after a flash vessel.
**Meaning:** The property package produced no VLE. Most common cause: NRTL or UNIQUAC selected but binary interaction parameters absent from DWSIM's database — DWSIM silently uses zero parameters, producing ideal behaviour.
**Routing:** THERMO
**Fix pattern:** Fall back to Raoult's Law to confirm topology works, then investigate parameter availability for NRTL/UNIQUAC. Alternatively use Peng-Robinson if appropriate.

### WRONG_PHASE_DIR
**Trigger:** Heavy component (higher normal boiling point) is concentrated in the vapour outlet; light component concentrated in liquid outlet.
**Meaning:** Phase assignment is inverted — wrong property package or wrong port assignment.
**Routing:** REFINER (default, LLM may escalate to THERMO). Deterministic fallback routes to REFINER. When the LLM Stage 2 is available, it may route to THERMO if it determines the phase inversion is caused by the property package rather than a port assignment error.
**Fix pattern:** Verify vapour outlet uses src_port=0 and liquid outlet uses src_port=1. If ports are correct, the property package is producing physically wrong VLE.

---

## Composition Failures

### COMP_SUM
**Trigger:** Any stream's mole fractions sum to a value outside [0.98, 1.02].
**Meaning:** Solver partial convergence or phase-split arithmetic error.
**Routing:** REFINER (retry with simplified conditions first).
**Fix pattern:** Retry solve. If persistent, simplify to fewer compounds and re-add incrementally.

### PARAM_MISSING
**Trigger:** NRTL or UNIQUAC used AND outlet compositions ≈ feed compositions (NO_SEPARATION pattern) AND solver reports Solved=True.
**Meaning:** DWSIM silently accepted the property package but had no binary interaction parameters — defaulted to ideal behaviour.
**Routing:** CALIBRATION → (fallback) THERMO
**Fix pattern (CALIBRATION):** CalibrationAgent queries the RAG corpus for literature NRTL/UNIQUAC binary interaction parameters (τ₁₂, τ₂₁, α). If ALL compound pairs are found, injects them into `flowsheet["binary_parameters"]` and re-runs the Executor. Zero LLM calls — purely deterministic retrieval. Provenance (source, T range, fit_data type) is recorded in the flowsheet JSON for auditability.
**Fix pattern (THERMO fallback):** If CalibrationAgent cannot find parameters for any compound pair, falls through to ThermoAgent which switches to Raoult's Law or Peng-Robinson. A warning is added to the orchestrator result listing the missing pairs.

---

## Terminal Failures

### INFEASIBLE
**Trigger:** Same failure code persists after 3 or more Refiner iterations with no improvement.
**Meaning:** The process as specified is either thermodynamically infeasible or requires capabilities beyond the current system (recycle convergence, reactive systems, electrolytes).
**Routing:** HUMAN
**Fix pattern:** Report to user with specific diagnosis. Do not retry further.

---

## Routing Summary

| Code | → REFINER | → THERMO | → BASIS | → HUMAN |
|---|---|---|---|---|
| SOLVER_FAIL | ✓ | | | |
| NUMERIC_FAIL | if units wrong | if no params | | |
| MASS_BALANCE | ✓ | | | |
| UNPHYSICAL_T | ✓ | | | |
| UNPHYSICAL_P | ✓ | | | |
| ENERGY_UNPHYSICAL | ✓ | | | |
| ZERO_OUTLET | if conditions | if no sep | | |
| NO_SEPARATION | | ✓ | | |
| WRONG_PHASE_DIR | default (LLM may → THERMO) | LLM only | | |
| COMP_SUM | ✓ | | | |
| PARAM_MISSING | | CALIBRATION→THERMO | | |
| INFEASIBLE | | | | ✓ |

## REPLAN Routing

**Trigger:** REFINER returns `success=False` (topology error unfixable by parameter patching — wrong connections, duplicate ports, missing outlets).

**Action:** Re-invoke PlannerAgent with structured error feedback (`topology_feedback` parameter) and re-run ThermoAgent. Max 2 replans; cycling detection (flowsheet hash) acts as safety net.

**Feedback content:** Critic signals (SOLVER_FAIL, MASS_BALANCE, NUMERIC_FAIL, UNPHYSICAL_T/P) and Critic diagnosis text, formatted as Planner-digestible constraints.

**If REPLAN budget exhausted:** Escalate to HUMAN.
