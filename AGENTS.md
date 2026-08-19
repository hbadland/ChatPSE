# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Commands

```bash
# Run all host-side agent unit tests
PYTHONPATH=. python3.9 agents/test_calibration.py
PYTHONPATH=. python3.9 agents/test_critic.py
PYTHONPATH=. python3.9 agents/test_refiner.py
PYTHONPATH=. python3.9 agents/test_physics_check.py
PYTHONPATH=. python3.9 agents/test_basis.py      # 3 tests require ANTHROPIC_API_KEY

# Run DWSIM integration tests (requires Docker container running)
docker exec priceless_elion sh -c "cd /workspaces/multiAgentFlowsheet && PYTHONPATH=. python3.9 dwsim/test_dwsim_wrapper.py"

# Run end-to-end demo (requires Docker + API key)
PYTHONPATH=. python3.9 examples/demo.py
PYTHONPATH=. python3.9 examples/demo.py --description "separate acetone and water at 80°C"

# Run benchmark pipeline (requires Docker + API key)
PYTHONPATH=. python3.9 agents/benchmark_pipeline.py

# Run ablation study (requires Docker + API key)
PYTHONPATH=. python3.9 agents/ablation.py
```

## Architecture

### Multi-agent loop (agents/orchestrator/__init__.py)

The pipeline follows: **Basis → Planner → Thermo → [Executor → Critic → route] × N**

```
User description
    ↓ BasisAgent          — normalise compound names to exact DWSIM names (2-stage: lookup + LLM)
    ↓ PlannerAgent        — produce flowsheet JSON from normalised description
    ↓ ThermoAgent         — assign property packages per unit
    ↓ loop (max_iterations):
        Executor          — translate flowsheet JSON → DWSIM API calls → solve
        CriticAgent       — detect failures, assign routing code
        routing:
          PASS            — return result
          REFINER         — RefinerAgent patches flowsheet JSON, re-run Executor
          CALIBRATION     — CalibrationAgent injects BIPs, re-run Executor (→ THERMO fallback)
          THERMO          — ThermoAgent switches property package
          BASIS           — BasisAgent re-runs with execution feedback
          HUMAN           — unrecoverable, return to user
```

### Flowsheet JSON (agents/schema.py)

All agents share one serialisable dict. Key fields:

```json
{
  "compounds": ["Ethanol", "Water"],
  "property_package": "NRTL",
  "binary_parameters": [              // injected by CalibrationAgent
    {
      "model": "NRTL",
      "compound_a": "Ethanol", "compound_b": "Water",
      "A12": 586.1, "A21": -195.0, "alpha12": 0.5765,
      "source": "Gmehling et al. 1977",
      "T_min_K": 333, "T_max_K": 373
    }
  ],
  "streams": [...],
  "units": [...],
  "connections": [["FEED","V-01",0,0], ...]
}
```

`schema.validate()` checks composition sums, tag references, acyclicity, and `binary_parameters` structure. All units must be in `SUPPORTED_UNIT_TYPES`. Connections must form a DAG.

### Agent responsibilities

| Agent | File | LLM? | Key method |
|-------|------|-------|-----------|
| BasisAgent | agents/basis.py | Stage 2 only | `.identify(description)` |
| PlannerAgent | agents/planner.py | Yes | `.plan(description, compounds)` |
| ThermoAgent | agents/thermo.py | Yes | `.assign(flowsheet)` |
| Executor | agents/executor.py | No | `.run(flowsheet)` |
| CriticAgent | agents/critic.py | Stage 2 only | `.critique(execution, flowsheet, iteration)` |
| RefinerAgent | agents/refiner.py | Stage 2 only | `.refine(flowsheet, report)` |
| CalibrationAgent | agents/calibration.py | **No** | `.run(flowsheet)` |

### CalibrationAgent (agents/calibration.py)

Intercepts `PARAM_MISSING` failures (NRTL/UNIQUAC selected but no binary interaction parameters → outlet ≈ feed). Zero LLM calls — O(1) dict lookup keyed by `(norm_a, norm_b, model)`.

- Corpus: `rag/sources/binary_parameters.json` — 211 pairs (173 NRTL, 38 UNIQUAC), ChemSep + DECHEMA
- All aliases for each compound are indexed in both orderings at startup
- All-or-nothing: `success=True` only if every compound pair is covered and passes the temperature guard
- Temperature guard: 10–20% outside fit interval → warning note; >20% → hard block (`success=False`)
- On success: populates `flowsheet["binary_parameters"]`, loop continues to Executor
- On failure: ThermoAgent fallback (switches to simpler package)

### DWSIM wrapper (dwsim/dwsim_wrapper.py)

Runs inside Docker only. Key facts:
- All units [K, Pa, mol/s] — no exceptions
- `set_nrtl_parameters()` / `set_uniquac_parameters()` inject BIPs via .NET reflection into `pkg.m_uni.InteractionParameters` — must be called after `set_property_package()` but before `solve()`
- `AutoEstimateMissingNRTLUNIQUACParameters` is disabled inside both injection methods
- NRTL/UNIQUAC injection requires BOTH orderings: forward entry and reverse entry with A12↔A21 swapped (separate objects)
- Splitter ratios: index-set in-place on the existing 3-slot ArrayList — never Clear/Add

### Context system (context/)

Loaded at import time into module-level strings. Injected into LLM prompts:
- `DWSIM_KNOWLEDGE` — unit operation parameter reference, property package selection, BIP injection API
- `FAILURE_TAXONOMY` — all failure codes, routing decisions, fix patterns
- `COMPOUND_DATABASE` — compound aliases and DWSIM exact names (parsed by BasisAgent Stage 1)

### Two-stage agent pattern

BasisAgent, CriticAgent, and RefinerAgent all follow the same pattern:
- **Stage 1**: Deterministic rule-based processing (zero cost, always runs)
- **Stage 2**: LLM call (only when Stage 1 is incomplete or escalates)

### Failure codes (agents/critic.py → _CODE_ROUTING)

| Code | Routes to |
|------|-----------|
| SOLVER_FAIL | REFINER |
| NUMERIC_FAIL | THERMO |
| MASS_BALANCE | REFINER |
| UNPHYSICAL_T/P | REFINER |
| ENERGY_UNPHYSICAL | REFINER |
| ZERO_OUTLET | THERMO |
| NO_SEPARATION | THERMO |
| PARAM_MISSING | CALIBRATION |
| INFEASIBLE | HUMAN |

### Property packages (DWSIM names, case-sensitive)

`"Raoult's Law"`, `"NRTL"`, `"UNIQUAC"`, `"Peng-Robinson"`, `"Soave-Redlich-Kwong"`, `"Lee-Kesler-Plöcker"`

## Repository layout

```
agents/          — all agent code and unit tests
  calibration.py — CalibrationAgent (no LLM, O(1) BIP lookup)
  schema.py      — flowsheet JSON schema + validate()
  executor.py    — DWSIM API calls (runs inside Docker)
  critic.py      — failure detection + routing
  orchestrator/  — pipeline loop
context/         — markdown knowledge files loaded at import
dwsim/           — DWSIM wrapper + integration tests (Docker only)
  dwsim_wrapper.py
  test_dwsim_wrapper.py
rag/
  sources/binary_parameters.json  — 211-pair BIP corpus
```

## Docker

All DWSIM code runs inside the container (`priceless_elion` is the typical container name). Scripts that import `dwsim.dwsim_wrapper` will fail on the host. Agent-level tests (no DWSIM import) run on the host with `PYTHONPATH=.`.
