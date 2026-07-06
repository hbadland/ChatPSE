# multiAgentFlowsheet — Contributions Audit (read-only fact extraction)

**Repo:** `/Users/harrybadland/ImperialCollegeLondon/multiAgentFlowsheet`
**Branch:** `master`  ·  **HEAD:** `1369473` ("Split validation tier into complexity bins; add FOSSEE ORC case + run-all script")
**Audit date:** 2026-07-01
**Scope:** No code was modified. Every claim below is cited to `file:line` (or commit hash for git-derived facts). Where numbers conflict across sources, all versions are reported.

> Verdict legend: **TRUE** = fully implemented as described · **PARTIAL** = implemented but with a caveat that changes the claim · **NOT-FOUND** = no supporting code located.

---

## Claim 1 — "Typed intermediate representation enforcing physical constraints at graph assembly, before any simulator call"

**Verdict: TRUE** (with one nuance about *which* constraints are checked at construction vs. at a later validation pass).

**Where it lives**
- Typed IR node hierarchy: `ir/graph.py:43` (`NodeIR` base) → concrete typed subclasses `HeaterNode` (`ir/graph.py:102`), `CoolerNode` (`:118`), `SeparatorNode`/Vessel (`:134`), `MixerNode` (`:150`), `SplitterNode` (`:163`), `PressureChangerNode` (`:183`) → `PumpNode` (`:196`), `CompressorNode` (`:204`), `ExpanderNode` (`:212`), `ConversionReactorNode` (`:220`).
- Each node declares `PORT_SPECS` and `REQUIRED_PARAMS` as class attributes and its own `validate_construction()` (e.g. Heater T_out range `ir/graph.py:110-115`; Splitter split-fraction sum `:172-180`; PressureChanger P_out range `:187-193`; ConversionReactor `:233-247`).
- **Construction-time enforcement** — `FlowsheetGraph.add_unit()` calls `node.validate_construction()` and *raises* `ValueError` on violation (`ir/graph.py:328-341`). `FlowsheetGraph.add_stream()` raises on phase mismatch against the source unit's outlet `PortSpec` (`ir/graph.py:343-372`, raise at `:364`). Module docstring states this explicitly: "invalid states are rejected during construction, not just during validation" (`ir/graph.py:6-7`, `:314-317`).
- **Multi-level validation before simulation** — `ir/validate.py:144` `validate()` runs Level 1 Schema (`_schema_validate`, `:160`), Level 2 Graph/connectivity/ports/DAG (`_graph_validate`, `:222`), Level 3 Physics (T/P ranges, phase, thermo feasibility, mass balance) (`_physics_validate_with_metrics`, `:308`). Docstring: "All checks are deterministic. No LLM calls." (`ir/validate.py:9`).
- **Ordering — validation precedes the simulator call.** In the two orchestrators, `validate(graph)` is called twice and `to_dwsim(graph)` (the DWSIM translation) strictly after:
  - `agents/orchestrator_v2.py`: `validate` #1 at `:625`, `validate` #2 at `:676`, `to_dwsim` at `:732` (and a repair-loop `validate` at `:869` before `to_dwsim` at `:874`).
  - `agents/graph_pipeline.py` (LangGraph variant): `validate` #1 at `:1404`, #2 at `:1561`, `to_dwsim` at `:1602` (repair-loop `validate` `:1778` → `to_dwsim` `:1791`).
- The IR is explicitly "simulator-independent" (`ir/graph.py:308-309`); DWSIM only enters via `ir/to_dwsim.py`.

**Caveats / completeness**
- `add_unit(strict=False)` deliberately *skips* construction-time checks "during repair when params are temporarily incomplete" (`ir/graph.py:328-340`). So the "enforced at assembly" guarantee is bypassable by design on the repair path — the safety net there is the later `validate()` call, not construction.
- Construction-time checks are *range/sum* checks per node (T/P bounds, fraction sums). The richer physical constraints (Heater T_out > feed T, Pump P_out > feed P, phase-into-pump, mass balance, bubble-point feasibility) live in `ir/validate.py` Level 3 (`:347-539`), which is a separate pass, not the constructor. The claim is accurate if "graph assembly" is read to include the `validate()` step that runs on the assembled graph; it is an overstatement if read as "all physical constraints enforced inside the constructor."
- Two `SeparatorNode` outlet-count rules exist in two places (construction defers to `add_stream`, `ir/graph.py:146-147`; hard "exactly 2 outlets" check in `ir/validate.py:275-279`).

---

## Claim 2 — "Detection and mitigation of silent DWSIM convergence failure (solved=True masking zero separation due to missing BIPs)"

**Verdict: TRUE.**

**Detection (the exact failure mode named in the claim)**
- `agents/critic.py:564-574` — `PARAM_MISSING` is raised *only* when `property_package ∈ {NRTL, UNIQUAC}` **AND** `result.solved` is True **AND** a `NO_SEPARATION` signal is present. Evidence string: "`{pp}` used but outlet ≈ feed — binary interaction parameters likely absent from DWSIM database." This is precisely "solved=True masking zero separation due to missing BIPs."
- `NO_SEPARATION` itself is computed deterministically by comparing terminal outlet compositions against the flow-weighted feed composition within 1% (`agents/critic.py:530-562`).
- Independent post-hoc benchmark detectors of the same mode:
  - `benchmark/physics_eval.py:1154` `_check_separation_achieved` — CRITICAL check on actual DWSIM outlet compositions; docstring calls out the three failure modes that "all pass DWSIM's solved=True check, pass mass balance, and produce two non-zero outlet streams" (`benchmark/physics_eval.py:1160-1169`).
  - `_check_bip_injected` (`benchmark/physics_eval.py:673-694`) — CRITICAL: "NRTL/UNIQUAC without BIPs silently produces outlet ≈ feed."
  - `_check_vle_bubble_point` (`benchmark/physics_eval.py:1004`) cross-validates DWSIM VF against an independent Raoult's-Law bubble point.
- Related silent-failure guards: `SPURIOUS_SEPARATION` (near-unity relative volatility yet large composition spread — "AutoEstimate artefact") `agents/critic.py:680-750`; `WRONG_PHASE_DIR` (vapour/liquid outlets inverted) `:613-677`.

**Mitigation**
- Routing: `PARAM_MISSING → CALIBRATION` (`agents/critic.py:82`, `_CODE_ROUTING`). `CalibrationAgent` injects BIPs from a corpus (zero-LLM, per CLAUDE.md `agents/calibration.py`), with `THERMO` (property-package switch) as the documented fallback if coverage/temperature guard fails.
- `NO_SEPARATION → THERMO` (`agents/critic.py:80`) directly switches package when BIP injection is not applicable.

**Caveats / completeness**
- Detection depends on the DWSIM executor actually returning stream compositions; `NO_SEPARATION` requires `len(terminal_tags) > 1` (`agents/critic.py:557`), so a single-outlet topology won't trigger it.
- The DWSIM wrapper disables `AutoEstimateMissingNRTLUNIQUACParameters` inside the injection methods (per CLAUDE.md) — this is what makes the "no BIP ⇒ outlet≈feed" signal reliable rather than DWSIM silently guessing parameters. That behavior is asserted in CLAUDE.md; I read the wrapper header (`dwsim/dwsim_wrapper.py:1-45`) but did not line-verify the AutoEstimate-disable inside `set_nrtl_parameters()` in this pass.

---

## Claim 3 — "Benchmark methodology evaluating physical correctness independently of solver convergence"

**Verdict: TRUE / PARTIAL** — physical correctness is scored on a **separate axis** from convergence/outcome, but most physics checks still require that DWSIM execution *produced output* (they read `final_execution.stream_results`); a subset degrade to IR-only or skip.

**Where it lives**
- `benchmark/physics_eval.py` implements a check evaluator with per-check `severity` (`CRITICAL`/`WARNING`/`INFO`) and `source` (`execution`/`IR`/`pipeline`/`none`) tagging (`physics_eval.py:1-66`). Convergence is *just one check* among ~20: `"convergence"` reads `pipeline_result.outcome`/`converged` (`physics_eval.py:234-243`, `source="pipeline"`).
- **Separate scoring axes** in `benchmark/metrics.py`:
  - `success` = pipeline outcome == PASS (`metrics.py:36`, `:289`).
  - `critical_physics_pass_rate` = CRITICAL physics checks passed / run (`metrics.py:135-138`), explicitly documented so "a pass/fail decision on thermodynamic correctness is not diluted by structural/presence checks" (`metrics.py:113-119`).
  - `physics_check_pass_rate` (all severities) `metrics.py:129-132`.
  - `AggregateMetrics` reports `success_rate`, `physics_pass_rate`, and `critical_physics_pass_rate` as distinct fields (`metrics.py:432-436`, printed separately `:487-490`).
- **Physics-only evaluation for cases whose reference is invalid** — commit `9c228cb` ("Benchmark: physics-only eval for excluded-invalid references"). A reference flagged `excluded-invalid-reference` still gets converged + physics evaluated, with reference-MAPE skipped (`benchmark/runner.py:512-566`; fields `reference_excluded`, `reference_excluded_reason` in `metrics.py:108-111`).
- **Ground-truth comparison as a third, separate axis** (validation tier): `run_reference_comparison` returns MAPE_T / MAPE_P / VF error at strict ±5 K / ±5% / ±0.05 thresholds (`physics_eval.py:1330-1343`), aggregated in `_ref_match_aggregate` (`metrics.py:525-553`).

**Caveats / completeness**
- Many CRITICAL physics checks read from `final_execution.stream_results` (e.g. `mass_balance` `physics_eval.py:810`, `separation_achieved` `:1154`, `two_phase_outlet` `:584`). If execution produced no data they return `INFO/passed=True/source="none"` (a "skip"), which *inflates* pass rates when there is nothing to check. So "independent of solver convergence" holds in the sense that a *converged-but-wrong* result is caught (solved=True but separation_achieved fails); it does **not** mean physics is scored when the solver produced nothing.
- Some checks have a documented IR-only fallback (`source="IR"`, WARNING) reflecting "LLM intent, not simulated outcome" (e.g. temp/pressure direction `physics_eval.py:334-403`, `:406-467`).
- The file header contains a self-audit table listing FP/FN risk per check and two historically fixed bugs (`temp_*` hardcoded-298.15 K bug; `two_phase_outlet` all-streams bug) — `physics_eval.py:31-65`, `:334-351`, `:584-603`. Useful for the "how tested" narrative.

---

## Claim 4 — "Constrained coupled repair search modelling inter-unit parameter dependencies explicitly" (CoupledSettler)

**Verdict: TRUE.**

**Where it lives**
- `ir/coupling.py` — two classes:
  - `ParameterCouplingMap.get_coupled_boosts()` (`ir/coupling.py:49-132`): 5 directed, graph-topology-aware coupling rules (Heater.T_out→downstream Vessel `:80-86`; Heater→upstream Pump/Compressor `:88-94`; Pump.P_out→upstream Cooler `:97-103`; Cooler.T_out→downstream Pump `:106-112`; Compressor.P_out→downstream conditioning `:115-121`; Expander.P_out→downstream `:124-130`). Returns additive priority boosts, zero LLM.
  - `CoupledSettler.settle()` (`ir/coupling.py:137-312`): after a P_out change on Pump/Compressor/Expander, recomputes bubble point at the new pressure (`bubble_point_K`, `:181`) and deterministically re-targets coupled upstream Cooler/Heater (`:198-241`) and downstream Cooler/Heater (`:250-300`) using the learned `MarginModel` (`:185`, `get_margin(...)`). Multi-pass with `_MAX_SETTLE_PASSES=3` convergence (`:38`, `:189-308`). Never mutates the input graph (`:187` copy) and never touches the just-fixed param (docstring `:149`).
- **Wiring (it is actually used, not dead code):**
  - `agents/stage4/beam_search.py:47` imports both classes; `_coupling = ParameterCouplingMap()` (`:65`), `_settler = CoupledSettler()` (`:66`).
  - In the beam loop: coupling boosts applied at `beam_search.py:160`; per-candidate `_settler.settle(cand_g, target_error.target.tag, param_name)` after the consistency pass and before cache-validation (`beam_search.py:236-237`).
  - Ablatable: `no_coupling` mode monkeypatches `get_coupled_boosts` to return `{}` (`benchmark/ablation.py:224-248`); diagnostics track `coupled_settler_resolved` (`benchmark/diagnostics.py:210`, `:642`).

**Caveats / completeness**
- `CoupledSettler.settle()` only fires for `fixed_param == "P_out"` on Pump/Compressor/Expander (`ir/coupling.py:175-176`); a T_out fix does not trigger settling (the T-side coupling is expressed only as priority boosts in `ParameterCouplingMap`, not joint settling). So "coupled repair" is asymmetric: pressure changes settle temperatures, but not vice-versa.
- Margins come from `ir/margin_model.py` (`MarginModel`, learned via trimmed mean, sliding window, hard bounds — per project memory `2026-05-20`); the settler falls back to fixed defaults (15 K / 20 K) when the model is cold (`ir/coupling.py:201-204`, `:228-231`).
- "Constrained … search" = beam search over candidates with IR-validation ranking (the settler is the *coupling* component inside that search); the beam/search machinery itself is `agents/stage4/beam_search.py` + `agents/stage4/repair_agent.py`.

---

## Claim 5 — "Cross-run rule synthesis via FailureRuleStore, distilling successful LLM-assisted repairs into deterministic zero-LLM rules"

**Verdict: TRUE** (with the caveat that what is distilled is the *applied fix value* per pattern, and by default the store is **cleared between benchmark runs**, so "cross-run" learning is opt-in).

**Where it lives**
- `agents/rule_store.py` — `FailureRuleStore` (`:121`) keyed by `(unit_type, error_code, downstream_type)`; `record_fix()` (`:134-166`) accumulates applied fix values + compound classes; `active_rules()` returns patterns with `count ≥ RULE_THRESHOLD` (`RULE_THRESHOLD = 3`, `:29`, `:170-172`); `apply_to_graph()` (`:182-240`) patches the IR with the **median** fix (`median_fix()`, `:80-84`) when unit type + downstream type + compound-class generalisation match and the current value differs by >5 K / >5% Pa. JSON `save()`/`load()` persistence to `results/rule_store.json` (`:32`, `:244-261`).
- Compound-class generalisation so a rule transfers across similar systems, not just exact compounds (`classify_compounds` `:52-59`, `matches_compounds` `:86-97`).
- **Wiring (actually used):**
  - Both orchestrators construct/load it and apply rules in Stage 3 before repair: `agents/orchestrator_v2.py:326-338` (init/load), `:658-661` (`apply_to_graph`), `:861-866` (save after repair); helper `_record_repairs_in_store` records `CONDITION_FIX` repairs into the store (`:910-938`, `store.record_fix` at `:938`).
  - Same in the LangGraph pipeline: `agents/graph_pipeline.py:73` import, `:935-941` load, `:1545-1547` apply, `:1769-1772` save.
- **Ablation:** `no_rule_store` mode replaces the store with a fresh empty `FailureRuleStore()` so prior-case knowledge has no effect (`benchmark/ablation.py:278`, `:311`), and swaps the RAG retriever for a null stub.

**Caveats / completeness**
- **"Successful LLM-assisted repairs" is a loose fit.** The store records the *value that was applied* for a `(unit, error_code, downstream)` pattern via `record_fix` (called from `_record_repairs_in_store`, `orchestrator_v2.py:910-938`). It does not verify that the repair ultimately *converged in DWSIM* before recording — it records the fix that was written to the IR (`CONDITION_FIX`). So it distills "repairs that were applied and recurred ≥3×," not provably-successful ones. Worth softening the wording in the thesis or adding a success-gate.
- By default `benchmark_runner.py` **deletes `results/rule_store.json` before each run** to keep runs independent (`benchmark_runner.py:197-231`, flag `--no-rule-store`/mode names `:63`, `:87`, and `rule_store_cleared_between_runs` reporting `:293-307`). Genuine cross-run accumulation only happens in explicit multi-run mode (`benchmark_runner.py:336-339`). The mechanism is real; whether it's "on" depends on run configuration.
- "Zero-LLM" is accurate: `apply_to_graph` is pure dict/graph manipulation, no LLM import.

---

## Claim 6 — "Fully open-source pipeline — no proprietary LLM or simulator license required"

**Verdict: PARTIAL — needs qualification.** The *default benchmark entrypoints* run on open-weight Qwen via Ollama and DWSIM (open-source), so a fully-open configuration exists and is the documented default. But the codebase still *ships and defaults some paths to* proprietary LLM providers, and one library-level default model is proprietary. Nothing forces proprietary use, but "fully open-source pipeline" is only true of a specific configuration.

### Simulator
- **DWSIM** — open-source process simulator (GPL/LGPL family). Loaded via .NET `clr.AddReference` on `DWSIM.Automation.dll` etc. from `/usr/local/lib/dwsim/` (`dwsim/dwsim_wrapper.py:38-45`), through **pythonnet** on **coreclr** (`dwsim/dwsim_wrapper.py:24-36`). No commercial simulator (Aspen/gPROMS) anywhere. ✅ Open-source. (Note: DWSIM runs inside a Docker container per CLAUDE.md; the license is not vendored/quoted in-repo — there is no root `README.md` and no `LICENSE` file surfaced in this audit.)

### LLM stack — `agents/llm.py` is provider-agnostic (`:1-27`)
Providers supported and their license status:
| Provider | Trigger prefix | Open? | Reference |
|---|---|---|---|
| **Ollama** (Qwen3, Llama, Mistral, Phi, DeepSeek, Gemma — local, open-weight) | `qwen/llama/mistral/phi/deepseek/gemma` | ✅ open-weight, self-hosted | `agents/llm.py:129`, `:277-298` |
| Groq (hosted Llama/Qwen/Gemma/Mixtral) | `llama-3/qwen-qwq/gemma2-/mixtral-8` | ⚠️ open-weight models but **proprietary hosted API + key** (`GROQ_API_KEY`) | `agents/llm.py:128`, `:254-268` |
| **Google Gemini** | `gemini` | ❌ proprietary (`GOOGLE_API_KEY`) | `agents/llm.py:122`, `:210-221` |
| **Anthropic Claude** | `claude` | ❌ proprietary (`ANTHROPIC_API_KEY`) | `agents/llm.py:123`, `:224-235` |
| **OpenAI GPT/o1/o3** | `gpt/o1/o3` | ❌ proprietary (`OPENAI_API_KEY`) | `agents/llm.py:124`, `:238-251` |

**Default model — conflicting defaults, this is the key flag for Claim 6:**
- `agents/llm.py:131` `DEFAULT_MODEL = "gemini-2.5-flash"` → **proprietary**. `agents/graph_pipeline.py:897` uses `DEFAULT_MODEL` (so the LangGraph pipeline defaults to Gemini unless overridden). `CriticAgent` also defaults to `DEFAULT_MODEL` (`agents/critic.py:227`).
- **But the benchmark runners default to open-weight Qwen via Ollama:** `benchmark_runner.py:46` `--model default="qwen3:30b-a3b"`; `benchmark/runner.py:293` `model="qwen3:14b"`; stage-1 diagnostic default `qwen3:30b-a3b` (commit `23d9a8b`). Project memory confirms the migration to local Qwen3 via Ollama (OpenAI-compatible, no API key).
- Net: an all-open configuration (Qwen3/Ollama + DWSIM) is real and is what the benchmark harness runs. The *pipeline is not hardwired open* — `DEFAULT_MODEL` and the still-present anthropic/google-genai/openai SDKs mean a fresh run of `demo.py`/`graph_pipeline` with no `--model` override hits Gemini.

### Python dependencies (`requirements.txt`)
```
dwsimopt · openai · anthropic · google-genai · pythonnet>=3.0.1 ·
chromadb · torch==2.1.0 · sentence-transformers ·
langchain · langchain-community · langgraph
```
All are open-source *packages* (the `openai`/`anthropic`/`google-genai` **SDKs** are OSS even though the **services** they call are proprietary). Flags:
- `anthropic`, `openai`, `google-genai` — SDKs for proprietary services. Only imported lazily inside their provider branch (`agents/llm.py:212`, `:226`, `:240`), so they're not required at runtime for an Ollama-only run, but they're declared dependencies.
- **`chromadb`, `torch==2.1.0`, `sentence-transformers` are declared but NOT imported anywhere** in the Python source (grep across repo: no `import chromadb` / `import torch` / `sentence_transformers` / `SentenceTransformer`). The RAG retriever is explicitly non-embedding: "All lookups are deterministic (no embeddings, no vector search)" (`rag/retriever.py:9-10`). → These three heavy deps appear vestigial; safe to flag as removable and as *not* part of the actual pipeline.
- `langchain` / `langgraph` — actually used (`agents/graph_pipeline.py`, `agents/stage1/topology_chain.py`). Open-source.

**Bottom line for Claim 6:** Reword to something like *"can run fully open-source (Qwen3-14B/30B via Ollama + DWSIM), the configuration used for all benchmark results; no proprietary model or simulator license is required for reproduction"* — and note the code retains optional proprietary-provider support and a proprietary library-level `DEFAULT_MODEL`.

---

## Other genuinely novel components found (not in your list)

Ranked by how paper-worthy they look. All are implemented and (except where noted) wired into the pipeline.

1. **Learned `MarginModel` for physically-coupled safety margins** — `ir/margin_model.py`. Sliding-window (MAX_OBS=20) trimmed-mean (TRIM_FRAC=0.10) estimator of the K/× margin a Heater/Cooler needs relative to a bubble point, with hard bounds; singleton `get_global_margin_model()`; JSON persistence. Consumed by `CoupledSettler` (`ir/coupling.py:185`, `:201`, `:228`). Novel: margins *learned from repair experience* rather than fixed constants. (Details per project memory 2026-05-20; not line-verified this pass.)

2. **Explore/exploit scheduler for the repair search** — `agents/stage4/explore_exploit.py` (`ExploreExploitScheduler`). Raw-count state machine with hysteresis (STAGNATION_WINDOW / RECOVERY_WINDOW = 2) deciding when the beam search widens (explore) vs. narrows (exploit). Ablatable and tracked in metrics (`explore_steps`/`exploit_steps`, `metrics.py:53-54`).

3. **Beam search with trajectory credit assignment + diversity pruning** — `agents/stage4/beam_search.py`. `BeamState.trajectory`, back-propagated credit at end, `_diverse_beam_prune` (L1 param-distance floor), oscillation escape. This is the "search" substrate Claim 4 sits inside and is itself non-trivial.

4. **GlobalConsistencyPass** — `ir/consistency.py`. Deterministic, zero-LLM T/P propagation (forward + `_backward_propagate`) that fills/repairs coupled Heater→Vessel and Pump→Cooler conditions before execution. Runs inside every beam candidate (`beam_search.py:235`). (Per project memory; header not re-read this pass.)

5. **Independent thermodynamic estimator** — `ir/thermo_estimation.py`: Antoine coefficients (~30 compounds), iterative Raoult's-Law bubble point, pressure-dependent boiling point. Used both for repair targets *and* as an **independent cross-check of DWSIM** in `_check_vle_bubble_point` (`physics_eval.py:1004-1147`). The "independent oracle" framing is a genuine methodological contribution for Claim 3.

6. **Zero-LLM `CalibrationAgent` + curated BIP corpus** — `agents/calibration.py` + `rag/sources/binary_parameters.json`. O(1) dict lookup, all-or-nothing coverage, temperature-guard on the fit interval. This is the concrete mitigation for Claim 2.

7. **Deterministic, non-embedding RAG retriever** — `rag/retriever.py`. Three JSON corpora (BIPs, thermo models, unit specs) with deterministic keyed lookup, replacing prompt-embedded knowledge. Notable precisely *because* it avoids a vector DB (contradicting the `chromadb`/`torch` deps).

8. **Typed error taxonomy driving deterministic-vs-LLM routing** — `ir/types.py` (`ErrorType`, `RepairStrategy`, `SimError.is_deterministic` `:96-105`). "No free-form strings in routing logic — all branching is on enum values" (`ir/types.py:80-83`). Enables the two-stage (deterministic-first, LLM-only-on-ambiguity) agent pattern seen in `agents/critic.py` (`_is_unambiguous` `:795-811`, deterministic fallback `:827-848`).

9. **CCS-specific search-quality metrics** — `benchmark/metrics.py`: error-recurrence classification into oscillation/coupling/propagation (`_compute_recurrence` `:186-243`), propagation lag (`:246-261`), error-reduction-per-candidate (`:264-266`). These quantify *repair-search behaviour*, distinct from outcome accuracy — good thesis material on evaluation.

10. **Ablation harness** — `benchmark/ablation.py` with modes `no_physics`, `no_rule_store`, `no_coupling`, `full_ccs` (`benchmark_runner.py:63`), each with runtime *verification* that the component was actually disabled (`ablation.py:213-248`). Lets you attribute gains to each of Claims 3/4/5 individually.

11. **StateCache / coordinate-descent polish / StructuralHeuristics** — `ir/state_cache.py` (MD5-hashed validation memoization, used at `beam_search.py:238`), `ir/local_optimiser.py` (`coordinate_descent` post-beam polish), `ir/structural_heuristics.py` (4 topology smell rules). Supporting machinery per project memory 2026-05-20.

12. **Ground-truth validation tier with reference flowsheets + MAPE** — `benchmark/reference_flowsheets/*.json` (VAL_01–VAL_10, FOS_01 FOSSEE ORC, template), strict ±5 K / ±5% / ±0.05 stream matching (`physics_eval.py:1330-1343`). Includes handling of *known-invalid references* (physics-only eval, commit `9c228cb`). Note the open reference-reactor bug tracked in your memory ([[project_ref_reactor_bug]]).

---

## Numerical / factual conflicts to resolve before citing

1. **BIP corpus size.**
   - CLAUDE.md: "211 pairs (173 NRTL, 38 UNIQUAC)".
   - Project memory: "211 pairs (173 NRTL, 38 UNIQUAC)".
   - **Actual file now:** `rag/sources/binary_parameters.json` = **260 records (222 NRTL, 38 UNIQUAC)** (counted this session). → The corpus grew; docs are stale. Use 260/222/38, or re-count at submission time.

2. **Headline benchmark accuracy — multiple, non-comparable numbers.**
   - Project memory (Haiku, real executor, 50 cases): **26/50 = 52%** outcome accuracy.
   - Commits `cd88358` / `428a4b8` ("v2 stable"): **95.6% original benchmark, 93.3% hard benchmark, 10% validation**.
   - Commit `c7394fb` ("Phase 1", LangGraph): **44/45, CRITICAL physics 95.6%**.
   These measure different things (outcome accuracy vs. CRITICAL-physics pass rate vs. validation-tier pass) on different suites/models. They are not in conflict *if labelled*, but must not be quoted as one "accuracy" number. Recommend a table with (suite, model, metric, value, commit).

3. **Default model.** `DEFAULT_MODEL="gemini-2.5-flash"` (`agents/llm.py:131`, proprietary) vs. benchmark defaults `qwen3:14b` / `qwen3:30b-a3b`. See Claim 6.

4. **`n_sim_calls == n_iterations` assumption** — `metrics.py:287` hardcodes one executor call per iteration ("v2"). If any iteration skips execution, sim-call counts in the paper would be overstated. Verify against logs before citing sim-call efficiency.

5. **Two parallel orchestrators exist** — `agents/orchestrator_v2.py` and the LangGraph `agents/graph_pipeline.py`. They duplicate the validate→to_dwsim ordering and rule-store wiring. Be explicit about which one produced your reported results (benchmark runner path).

---

## Quick verdict table

| # | Claim | Verdict | Primary evidence |
|---|-------|---------|------------------|
| 1 | Typed IR, physical constraints at assembly pre-simulator | **TRUE** (nuance: some constraints at construction, richer ones in `validate()`; both pre-`to_dwsim`) | `ir/graph.py:328-372`, `ir/validate.py:144-539`, `orchestrator_v2.py:625/676/732` |
| 2 | Detect+mitigate silent DWSIM solved=True / missing-BIP no-separation | **TRUE** | `agents/critic.py:530-574`, `physics_eval.py:673-694,1154`, routing `critic.py:82` |
| 3 | Physical correctness scored independently of convergence | **TRUE/PARTIAL** (separate axis; most checks need execution output) | `metrics.py:135-138,432-436`, `physics_eval.py:1-66`, commit `9c228cb` |
| 4 | Coupled repair search, explicit inter-unit dependencies (CoupledSettler) | **TRUE** (P_out→T settling only; wired into beam) | `ir/coupling.py:49-312`, `beam_search.py:65-66,160,236` |
| 5 | Cross-run rule synthesis (FailureRuleStore) → deterministic rules | **TRUE** (records applied fixes not verified-converged; cross-run off by default) | `agents/rule_store.py:121-261`, `orchestrator_v2.py:658-938`, `benchmark_runner.py:197-231` |
| 6 | Fully open-source, no proprietary LLM/simulator license | **PARTIAL** (open config exists & is benchmark default; proprietary SDKs + `DEFAULT_MODEL=gemini` remain) | `agents/llm.py:122-131`, `requirements.txt`, `benchmark_runner.py:46`, `dwsim_wrapper.py:38-45` |
