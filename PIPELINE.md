# ChatPSE — Pipeline Technical Reference

This document provides verified tables for the ChatPSE pipeline extracted directly from the source code. Every quantitative value is cited to the file and line where it is set. Values in Draft 4 of the manuscript that were not confirmed against the code are labelled **[unverified]**.

---

## 1. Pipeline stages

**Entry point:** `benchmark_runner.py:119` passes `max_iterations` (CLI default `--max-iter 6`, `benchmark_runner.py:74`) to `OrchestratorV2.__init__` (code default `max_iterations=10`, `orchestrator_v2.py:333` — overridden by the benchmark CLI). When `USE_LANGGRAPH=1`, `OrchestratorV2.run()` delegates to `GraphPipeline.run()` (`agents/graph_pipeline.py`).

### Stage 0 — Basis (`agents/basis.py → BasisAgent.identify()`)

| Property | Detail |
|---|---|
| Inputs | Raw user description (`str`) |
| Outputs | `BasisResult`: `dwsim_compounds` (list), `normalised_description` (str), `suggested_compositions` (dict), `concentration_hints` (list) |
| Stage 1 (deterministic) | Regex + dictionary lookup against `context/compound_database.md`; returns `stage="ANCHORS_ONLY"` when all compounds uniquely resolved |
| Stage 2 (model-driven) | LLM verifier-completer; skipped when Stage 1 resolves all anchors without mixture ambiguity |
| Max retries | 2 (`basis.py:491`); attempt 0 at `temperature=0.0`, attempt 1 at `temperature=0.3`; exhaustion falls back to Stage 1 anchors with `stage="PARTIAL"` |
| Terminal condition | `basis.success=False` → `BASIS_FAILED` outcome when unsupported compounds detected |

### Stage 1a — Unit extraction (`agents/stage1/unit_extractor.py → UnitExtractor.extract()`)

| Property | Detail |
|---|---|
| Inputs | Condensed or full description (str), `compounds` list, `tier` string |
| Outputs | `SemanticUnits` — ordered list of `SemanticUnit(tag, type, role, reaction, setpoint)` |
| Nature | Model-driven (LLM). Deterministic `_keyword_fallback()` used only on full retry exhaustion. |
| Max retries | 3 (`unit_extractor.py:314`) |
| Temperature schedule | `retry_temperature(attempt)`: 0.0 at attempt 0, 0.3 at attempts 1–2 (`llm.py:183`) |
| Seed | `retry_seed(attempt, description)` — SHA-256-derived, stable for given input |
| Token budget | 16 384 (validation tier); 12 288 (> 10 compounds); 8 192 (> 300 words or > 5 compounds); 4 096 otherwise (`unit_extractor.py:336–344`) |
| Prompt swap | Third attempt may use `_MINIMAL_SYSTEM` prompt when prior attempts returned malformed output |
| Fallback | `_keyword_fallback()` (deterministic) on exhaustion |

### Stage 1b — Stream extraction (`agents/stage1/stream_extractor.py → StreamExtractor.extract()`)

| Property | Detail |
|---|---|
| Inputs | Full description (str), compounds, `unit_tags` (list), `unit_roles` (dict), `concentration_hints`, `suggested_compositions` |
| Outputs | `SemanticTopology` — list of `SemanticStream(tag, src, dst, is_feed, T, P, flow, composition, is_recycle, recycle_target)` |
| Nature | Model-driven (LLM). Post-processing (`_reconcile_unit_refs`, `_dedupe_stream_tags`, `_resolve_qualitative_pressure`) is deterministic. |
| Max retries | 3 (`stream_extractor.py:241`); same temperature/seed schedule as Stage 1a; `max_tokens=8192` fixed |
| Prompt swap | Final attempt uses `_MINIMAL_SYSTEM` if previous error was empty output or unescaped markdown |
| Fallback | `RuntimeError` on exhaustion (no deterministic fallback for stream topology) |

### Stage 1c — Recycle guards (deterministic, orchestrator)

Three sequential guards applied after 1a + 1b, all deterministic:

| Guard | Location | Effect |
|---|---|---|
| Target validity + fuzzy resolution | `orchestrator_v2.py:500–548`, `graph_pipeline.py:1292–1336` | Validates and fuzzy-matches `recycle_target` tags |
| Multi-recycle deduplication | `orchestrator_v2.py:550–575`, `graph_pipeline.py:1338–1355` | Removes duplicate recycle flags on the same stream |
| Phrase guard | `orchestrator_v2.py:576–588`, `graph_pipeline.py:1356–1366` | Clears `is_recycle=True` unless raw description contains one of 6 `_RECYCLE_PHRASES`; prevents hallucinated recycles |

### Stage 1d — Completeness loop (`agents/stage1/completeness.py`) — off by default

| Property | Detail |
|---|---|
| Trigger | `COMPLETENESS_LOOP=1` env var or `completeness` ablation mode |
| Nature | Model-driven LLM critic + deterministic span-verification guard (claimed units must appear verbatim in description) |
| Max outer iterations | `MAX_ITERS=3` (`completeness.py:33`) |
| Early halt | When critic returns no accepted new units |

### Stage 2 — IR construction (`agents/stage2/graph_builder.py → GraphBuilder.build()`)

| Property | Detail |
|---|---|
| Inputs | `SemanticUnits`, `SemanticTopology`, `compounds` list |
| Outputs | `FlowsheetGraph` (unit nodes + stream edges, no operating params yet) |
| Nature | **Deterministic.** Zero LLM calls. |
| Post-processing | Compound canonicalisation, reconciliation, back-fill; then `normalise(graph)` + `validate(graph)` pass 1; all deterministic |
| Reactor seeding | `ConversionReactorNode.reaction` seeded from `SemanticUnit.reaction` at construction time |

### Topology repair — GraphPipeline only (`graph_pipeline.py → _topology_repair_node()`)

Triggered when validation pass 1 returns `INVALID_TOPOLOGY` with a repairable pattern.

| Fix | Nature | Detail |
|---|---|---|
| Fix 1 — vessel outlets | Deterministic | `_repair_vessel_outlets()`: adds missing phase outlet; physical superheat guard (`_SUPERHEAT_MARGIN_K=12.0 K`) |
| Fix 2 — recycle repropagate | Deterministic | `_repropagate_recycles()`: restores `is_recycle` cleared by phrase guard by tracing NetworkX cycle |
| Fix 3 — missing units | LLM route | `_detect_missing_units()`: flags streams referencing absent tags; routes to LLM repair node |

### Stage 3a — Thermodynamic mapping (`agents/stage3/thermo_mapper.py → ThermoMapper.assign()`)

| Property | Detail |
|---|---|
| Inputs | `FlowsheetGraph` (without property package), description str |
| Outputs | `FlowsheetGraph` with `property_package` set; `binary_parameters` injected for NRTL/UNIQUAC cases |
| `PackageSelector` | **Deterministic** rule-based family scoring (see Section 4) |
| `ThermoLLMFallback` | Model-driven; invoked only when ≥ 2 candidate packages remain after deterministic selection |
| LLM max retries | 2 (`thermo_components.py:128`); temperature schedule `retry_temperature(attempt)`; `max_tokens=256` |
| `BIPInjector` | **Deterministic** corpus lookup from `rag/sources/binary_parameters.json` |
| Env overrides | `THERMO_TIEBREAK=deterministic` → skip LLM, take `candidates[0]`; `THERMO_COVERAGE_GUARD=1` → raise `ThermoCoverageGuard` (→ `PARAM_MISSING`) for polar systems with no BIP coverage |

### Stage 3b — Parameter mapping (`agents/stage3/param_mapper.py → ParamMapper.assign()`)

| Property | Detail |
|---|---|
| Inputs | `FlowsheetGraph`, description str |
| Outputs | `FlowsheetGraph` with `T_out`, `P_out`, `temperature_K`, `pressure_Pa`, `conversion`, `reaction` set per unit |
| Priority | (1) Structured per-unit setpoint — **deterministic**; (2) description regex parser — **deterministic**; (3) bubble-point/heuristic estimator — **deterministic**; (4) LLM fallback — model-driven (only when 1–3 fail) |
| LLM max retries | 2; temperature schedule `retry_temperature(attempt)` |

### Stage 3c — Global consistency (`ir/consistency.py → GlobalConsistencyPass.apply()`)

**Deterministic.** Enforces cross-unit T/P monotonicity constraints and applies a backward pass to propagate set-point implications through the graph.

### Stage 3d — Rule store application (`agents/rule_store.py → FailureRuleStore.apply()`)

**Deterministic.** Applies synthesised repair rules from prior benchmark cases accumulated in the store.

After Stage 3: `normalise(graph)` pass 2 + `validate(graph)` pass 2. Failure → `INVALID_JSON` outcome.

### Stage 3→4 bridge — IR serialisation (`ir/to_dwsim.py`)

**Deterministic.** Translates `FlowsheetGraph` to a DWSIM-ready dict. `VARIANT_B=1` with a supplied `reference_file` enables reference-seeded reactor params and recycle INIT stream population (ablation arm only; disabled during all reported benchmark scoring).

### Stage 4 — Execution loop

**Outer loop** (`orchestrator_v2.py:771–911`; `graph_pipeline.py` `_execute_node` / `_repair_node`)

| Property | Detail |
|---|---|
| Default max iterations | `--max-iter 6` via CLI (`benchmark_runner.py:74`); code default `max_iterations=10` (`orchestrator_v2.py:333`) overridden by CLI |
| Beam extension ceiling | `_BEAM_MAX_ITER=15` (`orchestrator_v2.py:767`): activated when `n_cond_errors > 1` triggers beam search (`orchestrator_v2.py:839–844`) |
| Stagnation detection | SHA-256 hash of `dwsim_json` compared each iteration; stall → `STALLED` (GraphPipeline only) |
| Terminal conditions | `PASS`, `HUMAN` (any `is_terminal` error), `MAX_ITER` (loop exhausted), `STALLED` |

**Stage 4a — Executor** (`agents/executor.py`): **Deterministic** DWSIM invocation.

**Stage 4b — ErrorClassifier** (`agents/stage4/error_classifier.py`): Deterministic routing rules; LLM call for ambiguous failure messages only.

**Stage 4c — RepairAgent** (`agents/stage4/repair_agent.py → RepairAgent.repair()`)

| Property | Detail |
|---|---|
| Deterministic repairs | `DeterministicRepair.apply()` — see Section 5 |
| Beam search activation | When `n_cond_errors > 1`: calls `BeamRepairSearch.search()` |
| Single-error path | `_search_condition_fix()`: physics candidates + optional LLM candidate |
| LLM temperature | `retry_temperature(attempt)` (0.0 first attempt, 0.3 thereafter) |
| Reference-guided refinement | `_reference_guided_refinement` (VARIANT_B only): corrects `T_out` deviating > 10 K from reference matched by composition L1 distance (threshold 0.3); re-solves once |

---

## 2. Inference temperatures and retry limits

### Temperature schedule (`agents/llm.py:171–183`)

```python
def retry_temperature(attempt: int) -> float:
    return 0.0 if attempt == 0 else 0.3
```

Attempt 0 is greedy-deterministic. Attempt ≥ 1 uses 0.3 to break stale outputs. Capped at 0.3 for JSON stability with small models.

### Global retry wrapper (`agents/llm.py`)

| Parameter | Value | Notes |
|---|---|---|
| API retries (rate-limit / timeout) | `max_retries=6, base_delay=15.0 s` (`llm.py:44`) | Exponential backoff: `min(15.0 × 2^attempt, 120.0)` s; covers Google / Anthropic / OpenAI / Groq |
| Ollama retries | `max_retries=3, base_delay=1.0 s` (`llm.py:307`) | Exponential backoff: `min(1.0 × 2^attempt, 30.0)` s |
| Empty-response retry delay | `min(2.0 × 2^attempt, 10.0)` s | Applied within each call before the outer retry counts |
| Ollama wall-clock timeout | `_OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "120"))` s (`llm.py:307`) | |
| Default model | `DEFAULT_MODEL = "gemini-2.5-flash"` (`llm.py:132`) | Benchmark default overridden to `qwen3:30b-a3b` via `--model` CLI arg |
| Context window | `OLLAMA_NUM_CTX` env var, default `16384` (`llm.py:324, 453`) | |

### Per-callsite table

| File | Function | Temperature | Max application retries | Behaviour on exhaustion |
|---|---|---|---|---|
| `orchestrator_v2.py:168` | `_summarise_for_unit_extraction` | `0.0` (hard-coded) | 1 (no retry) | Condensed summary used as-is |
| `orchestrator_v2.py:436` | Second-pass tight summariser | `0.0` (hard-coded) | 1 (no retry) | Only when first summary > 150 words |
| `basis.py:491–499` | `BasisAgent._stage2_llm` | `retry_temperature(attempt)` | 2 | Fall back to Stage 1 anchors (`stage="PARTIAL"`) |
| `unit_extractor.py:314` | `UnitExtractor.extract` | `retry_temperature(attempt)` + `retry_seed` | 3 | `_keyword_fallback()` (deterministic) |
| `stream_extractor.py:241` | `StreamExtractor.extract` | `retry_temperature(attempt)` + `retry_seed` | 3 | `RuntimeError` |
| `completeness.py:100` | `_critic_call` | `retry_temperature(attempt)` | Governed by `MAX_ITERS=3` outer loop | Outer loop halts |
| `thermo_components.py:128` | `ThermoLLMFallback.select` | `retry_temperature(attempt)` | 2 | Fall back to `candidates[0]` |
| `param_mapper.py:102` | `ParamMapper._assign_unit` (LLM fallback) | `retry_temperature(attempt)` | 2 | Unit left with deterministic estimate |

---

## 3. Supported IR unit classes and ports

All entries from `ir/graph.py`. "Phase" values are the `PortSpec.phase` field; `enforce_phase=True` raises `ValueError` at `add_stream()` time when source outlet phase is violated.

The IR type string (e.g. `"Heater"`) is the `UNIT_TYPE` class attribute. The mapping to DWSIM object type lives in `ir/to_dwsim.py` (not reproduced here).

| Class | `UNIT_TYPE` | Inlet ports | Outlet ports | Required params | Notable constraints |
|---|---|---|---|---|---|
| `HeaterNode` | `"Heater"` | 1 (any) | 1 (any) | `T_out` [K, 50–2000] | — |
| `CoolerNode` | `"Cooler"` | 1 (any) | 1 (any) | `T_out` [K, 50–2000] | — |
| `SeparatorNode` | `"Vessel"` | 1 (mixed) | 2: port 0 = vapour, port 1 = liquid | none listed in `REQUIRED_PARAMS` | Outlet phase enforced on connection |
| `MixerNode` | `"Mixer"` | 1 required + up to 15 optional (any); `max_inlets()=16` | 1 (any) | — | — |
| `SplitterNode` | `"Splitter"` | 1 (any) | 2 (any) | `split_fractions` (dict; values must sum to 1.0) | — |
| `PumpNode` | `"Pump"` | 1 (liquid) | 1 (liquid) | `P_out` [Pa, 100–1×10⁸] | Liquid inlet enforced |
| `CompressorNode` | `"Compressor"` | 1 (vapour) | 1 (vapour) | `P_out` [Pa, 100–1×10⁸] | Vapour inlet enforced |
| `ExpanderNode` | `"Expander"` | 1 (vapour) | 1 (vapour) | `P_out` [Pa, 100–1×10⁸] | Vapour inlet enforced |
| `ConversionReactorNode` | `"ConversionReactor"` | 1 (any) | 1 (any) | `temperature_K` [50–3000 K], `pressure_Pa` [100–1×10⁸ Pa], `conversion` [0–1], `reaction` (stoichiometry str) | — |
| `ColumnNode` | `"Column"` | 1 (any) | 2: port 0 = distillate, port 1 = bottoms | `light_key`, `heavy_key`, `light_key_frac_bottoms`, `heavy_key_frac_distillate`, `reflux_ratio` (> 0), `condenser_pressure_Pa`, `boiler_pressure_Pa` | Reboiler energy stream attached at DWSIM-mapping time; not an IR port |
| `DecanterNode` | `"Decanter"` | 1 (mixed) | 3: port 0 = vapour (optional), port 1 = liquid-1 (required), port 2 = liquid-2 (required) | — | — |

**Note:** Optional parameters (efficiency, pressure drop, etc.) are sourced from `rag/sources/unit_specs.json` defaults via `DEFAULT_FILL` repair strategy; they are not listed in `REQUIRED_PARAMS` on the node classes.

---

## 4. Thermodynamic shortlist rules

**Decision logic:** `rag/retriever.py → ThermoRetriever.select()`.
**Package metadata:** `rag/sources/thermo_models.json` (used for prompt context only; `"priority"` field is not read by `select()`).

### Compound class definitions (`retriever.py:192–215`)

| Class label | Members (lowercase) |
|---|---|
| `ALCOHOLS` | methanol, ethanol, 1-propanol, 2-propanol, n-propanol, isopropanol, 1-butanol, n-butanol, isobutanol, 2-butanol, 1-pentanol, ethylene glycol, glycerol |
| `KETONES` | acetone, methyl ethyl ketone, mek, cyclohexanone, methyl isobutyl ketone, mibk, acetophenone |
| `ESTERS` | ethyl acetate, methyl acetate, butyl acetate, isopropyl acetate |
| `ETHERS` | diethyl ether, mtbe, tetrahydrofuran, thf, 1,4-dioxane, diisopropyl ether |
| `CHLORINATED` | chloroform, dichloromethane, dcm, carbon tetrachloride, 1,2-dichloroethane, chlorobenzene |
| `AROMATICS` | benzene, toluene, o/m/p-xylene, ethylbenzene, styrene, naphthalene |
| `ALKANES` | methane–octane (C1–C8), cyclohexane, methylcyclohexane |
| `LIGHT_GASES` | methane, ethane, propane, n-butane, isobutane, nitrogen, oxygen, CO₂, H₂S, hydrogen, argon |
| `POLAR_OTHER` | acetic acid, formic acid, acetonitrile, DMSO, ammonia, H₂S, water |
| `WATER` | water |

### Known azeotrope pairs (`retriever.py:217–232`)

18 pairs encoded as `frozenset` objects. Key pairs: ethanol/water, methanol/water, n-propanol/water, isopropanol/water, n-butanol/water, ethyl acetate/ethanol, ethyl acetate/water, acetone/chloroform, acetone/methanol, diethyl ether/water, n-hexane/ethanol, benzene/cyclohexane, THF/water, acetonitrile/water.

### Decision tree (`ThermoRetriever.select()`, `retriever.py:250–403`)

Computed flags (in order):

| Flag | Condition |
|---|---|
| `is_polar` | Any compound in `{ALCOHOLS, KETONES, ESTERS, ETHERS, POLAR_OTHER, WATER}` |
| `has_azeo` | Any compound pair is in `_AZEOTROPES` |
| `_gas_dominated` | `n_gas_like ≥ 2 AND n_gas_like > n_activity_polar` |
| `water_is_steam` | `WATER + LIGHT_GASES + _gas_dominated + (T ≥ 400 K OR steam keyword in description)` |
| `acid_gas_system` | `{H₂S, CO₂, SO₂} ∩ compounds AND _gas_dominated` |
| EOS override | `(water_is_steam OR acid_gas_system)` → forces `is_polar=False, has_azeo=False` → EOS path |
| `is_light_gas` | `LIGHT_GASES in classes AND NOT is_polar` |
| `is_cryogenic` | `T_K < 200.0` |
| `is_high_press` | `P_Pa > 3×10⁵ Pa` (3 bar) |

Steam keywords triggering EOS override (`retriever.py:308–309`): `"steam"`, `"reform"`, `"syngas"`, `"combust"`, `"flue gas"`, `"gasif"`, `"high temperature"`, `"high-temperature"`, `"furnace"`, `"cracker"`, `"cracking"`, `"pyrolysis"`.

Candidate construction order:

| Condition | Packages added (in order) |
|---|---|
| `is_cryogenic AND is_light_gas` | `Lee-Kesler-Plöcker` |
| `is_light_gas OR (is_high_press AND NOT is_polar)` | `Peng-Robinson`, `Soave-Redlich-Kwong` |
| `is_polar OR has_azeo` | If BIP corpus covers all pairs: `NRTL` (if available), `UNIQUAC` (if available); if `is_high_press`: also `Peng-Robinson`; always `Raoult's Law` |
| `NOT is_polar AND NOT is_light_gas` | `Peng-Robinson`, `Raoult's Law` |
| Fallback (empty result) | `Peng-Robinson`, `NRTL`, `Raoult's Law` |

BIP coverage gate (`retriever.py:345–373`): `BIPRetriever.has_full_coverage()` used to gate NRTL/UNIQUAC when `is_polar OR has_azeo`. When coverage missing and `THERMO_COVERAGE_GUARD=1`: `ThermoCoverageGuard` raised → `PARAM_MISSING` outcome. When guard off: `thermo_coverage="MISSING"` recorded on `graph.metadata`.

Deduplication: already-tried packages (`exclude` set from prior repair iterations) removed before LLM tiebreak. LLM selects from remaining candidates when > 1 present (unless `THERMO_TIEBREAK=deterministic`).

---

## 5. Repair classifications and budgets

### `ErrorType` enum (`ir/types.py:16–25`)

| Value | Meaning | Deterministic repair available |
|---|---|---|
| `MISSING_PARAM` | BIPs absent, or required param not set | Yes → `PARAM_INJECT` or `DEFAULT_FILL` |
| `INVALID_TOPOLOGY` | Connectivity / port violation | Yes → `TOPOLOGY_FIX`; complex cases → `HUMAN` |
| `CONVERGENCE_FAILURE` | Solver diverged | Partial → `CONDITION_FIX` (physics candidates first) |
| `INVALID_UNIT_CONFIG` | T_out below bubble point, phase mismatch | Partial → `CONDITION_FIX` |
| `UNPHYSICAL_VALUES` | T in °C, P in bar | Yes → `UNIT_CONVERSION` |
| `PHASE_MISMATCH` | Vapour into Pump, liquid into Compressor | Yes → `PORT_REPAIR` |
| `MASS_BALANCE` | Inlet ≠ outlet flow | Partial → `CONDITION_FIX`; severe → `HUMAN` |
| `INFEASIBLE` | Thermodynamically impossible | No → `HUMAN` |

### `RepairStrategy` enum (`ir/types.py:27–36`)

| Value | LLM call | Handler | `SimError.is_deterministic` |
|---|---|---|---|
| `PARAM_INJECT` | No | `DeterministicRepair.inject_bips()` — corpus lookup, both orderings inserted | True |
| `TOPOLOGY_FIX` | No | `DeterministicRepair.fix_topology()` — re-runs `normalise(graph)` | True |
| `THERMO_SWITCH` | No | `DeterministicRepair.switch_package()` — calls `ThermoRetriever.select(exclude=tried_packages)` | True |
| `UNIT_CONVERSION` | No | `DeterministicRepair.fix_unit_conversions()` — T < 100 → +273.15 K; P < 500 → ×10⁵ Pa | True |
| `DEFAULT_FILL` | No | `DeterministicRepair.apply_defaults()` — fills optional params from `rag/sources/unit_specs.json` | True |
| `PORT_REPAIR` | No | `DeterministicRepair.fix_port_violations()` — reassigns `src_port`/`dst_port` | True |
| `CONDITION_FIX` | Fallback | `BeamRepairSearch.search()` or `_search_condition_fix()` | False |
| `HUMAN` | No | Logged; loop terminates with `outcome="HUMAN"` | False |

`ir/types.py:98–105`: `is_deterministic` returns `True` for all strategies except `CONDITION_FIX` and `HUMAN`.

### `DeterministicRepair` catalogue (`ir/repair.py`)

| Method | Strategy | Effect |
|---|---|---|
| `inject_bips()` | `PARAM_INJECT` | O(1) corpus lookup from `binary_parameters.json`; inserts both forward and reverse entries |
| `fix_topology()` | `TOPOLOGY_FIX` | Calls `normalise(graph)` — re-runs port assignment and vessel port heuristics |
| `fix_unit_conversions()` | `UNIT_CONVERSION` | Stream T < 100 K → +273.15; stream/unit P < 500 → ×10⁵ Pa |
| `apply_defaults()` | `DEFAULT_FILL` | Fills missing optional params from `unit_specs.json` defaults |
| `fix_port_violations()` | `PORT_REPAIR` | Reassigns `src_port`/`dst_port` on mismatch streams |
| `switch_package()` | `THERMO_SWITCH` | Calls `ThermoRetriever.select()` with current `tried_packages` as exclusions |

### `BeamRepairSearch` parameters (`agents/stage4/beam_search.py`)

| Parameter | Value | Location |
|---|---|---|
| Beam width | 3 | `beam_search.py:102` |
| Beam depth (steps per search call) | 2 | `beam_search.py:103` |
| Run local optimiser | `True` | `beam_search.py:104` |
| Exploration phase length | `max(1, depth − 1) = 1` step | `beam_search.py:143` |
| Effective beam width | `ExploreExploitScheduler.effective_beam_width(3)` — widens in exploration phase | `agents/stage4/explore_exploit.py` |
| Exploration phase uncertainty floor | `max(uncertainty, 0.6)` | `explore_exploit.py` |
| Diversity threshold (T) | `_T_DIV_THRESH = 15.0 K` | `beam_search.py:70` |
| Diversity threshold (P) | `_P_DIV_THRESH = 0.25` (fractional) | `beam_search.py:71` |
| Beam state score | `ir_errors × 100 + ir_warnings + sim_penalty` | `beam_search.py:90` |
| Beam activation | `n_cond_errors > 1` | `orchestrator_v2.py:839–844` |

### Candidate generation per `CONDITION_FIX` step

| Source | Nature | Filtering |
|---|---|---|
| `_deterministic_candidates()` | Physics-based (bubble-point-derived, directional heuristic, margin model) | Removes already-tried values via `RepairMemory.tried_values(tag, param)` |
| LLM candidate | `llm_agent._llm_candidate()` — one additional candidate per step | Only when `llm_agent is not None` |
| Minimum spacing | T: `_MIN_T_SPACING=10.0 K`; P: `_MIN_P_RATIO=1.25` | `repair_agent.py:48–49` |

### `RepairMemory` limits (`repair_agent.py:52–54`)

| Limit | Value |
|---|---|
| `_MAX_HISTORY_PER_TAG` | 20 records per target tag (oldest trimmed) |
| `_CREDIT_WINDOW` | 10 recent records for `credit_score()` |
| `_OSC_ROUND_DP` | 1 decimal place for oscillation detection |

### Ablation patches (`benchmark/ablation.py`)

The `no_physics` and `no_coupling` modes are applied via `apply_ablation(config)` context manager (`ablation.py:137`), which monkey-patches the live modules:

| Mode | Patch | Implementation |
|---|---|---|
| `no_physics` | `bubble_point_K → _null_bubble_point` (returns `None`) | Scans all loaded modules for references to the original function and replaces them (`ablation.py:154–203`) |
| `no_coupling` | `ParameterCouplingMap.get_coupled_boosts → _no_coupling_boosts` (returns `{}`) | Class-level method replacement (`ablation.py:229–261`) |
| `no_rule_store` | `BIPRetriever → NullRetriever`; `FailureRuleStore` initialised empty | Store not written to during the run (`ablation.py:83, 326`) |
| `greedy` | `beam_width=1`, `disable_physics=False`, `disable_coupling=False` | `CONFIGS["greedy"]` in `ablation.py:56` |

Each mode also sets `beam_width` explicitly; `no_physics` and `no_coupling` both use `beam_width=3` (same as `full_ccs`, `ablation.py:56–59`). Patches are verified at activation time by a probe call logged with `[ABLATION]` prefix.

---

## 6. Reference flowsheet provenance

**Source:** `benchmark/reference_flowsheets/PROVENANCE.md`

### Construction methodology

Reference stream conditions were separately constructed within the same research project; they were not harvested from the system under test. For each case:

1. The correct flowsheet (units, connectivity, operating conditions) was determined by expert judgement from the process description.
2. The flowsheet was constructed directly in DWSIM via the simulator wrapper, bypassing the system's extraction and IR-construction stages (`VARIANT_B` / reference-injection disabled).
3. DWSIM solved the specified flowsheet; the resulting stream conditions (T, P, vapour fraction, composition, molar flow) were recorded as the reference.

Each reference is a DWSIM solution to an expert-specified flowsheet. The comparison isolates flowsheet-construction correctness: because the generated and reference flowsheets are solved within the same DWSIM environment, simulator thermodynamic accuracy cancels in the comparison.

### Validity criteria

| Quantity class | Definition | Scoring role |
|---|---|---|
| Set-point quantities | Operating conditions explicitly stated in the description and imposed on DWSIM (e.g. compressor discharge P, cooler outlet T) | Exact by construction; **gating** (CRITICAL: T ± 5 K, P ± 5%) |
| Computed quantities | Conditions DWSIM computes from the thermodynamic model (flash vapour fraction, phase compositions) | Model-dependent uncertainty; vapour fraction is **non-gating WARNING** (± 0.05) |

**Minimum-match gate:** MAPE reported when ≥ 3 streams match, or ≥ 2 matches covering ≥ 80% of reference streams. A complete 2/2 match is sufficient for 2-stream cases (P2, S1, S2).

### Vapour-fraction sensitivity analysis (`PROVENANCE.md §X.4`)

| Case | Mixture | Reference vf | Δvf per ±2 °C | Verdict |
|---|---|---|---|---|
| SAN_03 | benzene/toluene | 0.450 | 0.294 | Model-sensitive → vf secondary |
| GEN_01 | hexane/heptane | 0.560 | 0.343 | Model-sensitive → vf secondary |
| EASY_01 | acetone/water | 0.675 | 0.038 | Precise to tolerance |
| F1 | pentane/octane | 0.384 | 0.031 | Precise to tolerance |
| F3 | acetone/toluene | 0.405 | 0.092 | Mildly model-sensitive → vf secondary |
| F4 | ethanol/water | 0.500 | 0.163 | Model-sensitive → vf secondary |
| F2 | methanol/water | 0.469 | 0.190 | Model-sensitive → vf secondary |

### Per-case construction notes (selected)

| Case | Process | Package | Key notes |
|---|---|---|---|
| SAN_04 | Propane compression + condensation | Peng-Robinson | Feed T assumed 25 °C; η = 0.75 assumed |
| GEN_03 | n-Heptane compression + condensation | Peng-Robinson | Feed 60 °C stated |
| EASY_04 | Propylene compression + condensation | Peng-Robinson | **Corrected 12 → 15 bar**: 12 bar does not condense propylene at 30 °C |
| EASY_02 | Propane condensation + pumping | Peng-Robinson | **Restructured**: original 2 bar / 0 °C = superheated vapour; no liquid to pump |
| SAN_03 | Benzene/toluene heat + flash | Peng-Robinson | **Corrected 100 → 95 °C**: 100 °C gives all-vapour; vf secondary |
| GEN_01 | Hexane/heptane heat + flash | Peng-Robinson | **Corrected 80 → 85 °C**: 80 °C gives all-liquid |
| EASY_01 | Acetone/water heat + flash | NRTL | **Corrected feed 70/30 → 50/50**: 70/30 has no stable mid-range flash at 1 atm |
| P1 | Two-stage N₂ compression + intercooling | Peng-Robinson | Both intermediate pressures stated; targets stage-collapse fault |
| C1 | Benzene/toluene FUG column | Peng-Robinson | Saturated liquid feed; R = 1.3 × R_min; mass balance closes exactly |
| C3 | Methanol/water FUG column | NRTL | Only column case on activity-model path; R_min = 0.68 |
| M1 | Adiabatic mixer (hexane + heptane) | Peng-Robinson | Two distinct feed compositions; exercises composition-set fix |

### Integrity

- Self-consistency verified: feeding each reference back through scoring reproduces every stream with 0.0 aggregate MAPE on T, P, and vf for all 20 capability cases.
- **Dropped case D1** (water/n-butanol decanter): 3-phase SVLLE validated mechanically, but neither NRTL nor UNIQUAC reproduced literature mutual solubilities within tolerance; excluded pending validated LLE parameters.
- Four cases required thermodynamic correction of internally inconsistent original descriptions (EASY_04, EASY_02, SAN_03, GEN_01, EASY_01); corrections documented with physical justification.
- FUG column mass balance closes exactly; an earlier ~4% imbalance was traced to a wrapper bug (`PROP_MS_2` read as molar rather than mass flow), since corrected.

---

## Gaps and unverified items

| Item | Status |
|---|---|
| DWSIM object type strings | Not on `NodeIR` subclasses; live in `ir/to_dwsim.py` (not read for this document) |
| `"priority"` field in `thermo_models.json` | Present in JSON but not read by `ThermoRetriever.select()`; governs prompt context only |
| `completeness.py` per-call retry budget | Governed by outer `MAX_ITERS=3` loop; no per-call retry counted separately |
| Model digest and Ollama version | Not recoverable from code; see release checklist |
| `ExploreExploitScheduler` exact width schedule | Partially read; full schedule in `agents/stage4/explore_exploit.py` |
