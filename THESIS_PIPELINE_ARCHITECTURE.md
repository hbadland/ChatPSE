# multiAgentFlowsheet — Pipeline Architecture (methodology-grade description)

**Repo:** `/Users/harrybadland/ImperialCollegeLondon/multiAgentFlowsheet`
**Branch:** `master`  ·  **HEAD:** `1369473`
**Audit date:** 2026-07-01 · **READ-ONLY** (no code modified)
**Primary source of truth for this document:** `agents/graph_pipeline.py` (the LangGraph `StateGraph` build), cross-checked against `agents/orchestrator_v2.py`, `agents/llm.py`, and the stage-agent constructors. Comments/docstrings were cross-checked against actual node behaviour; **mismatches are flagged in §8**.

> Everything below is cited to `file:line`. Where a claim in a comment/docstring contradicts the code, both are reported.

---

## 1. Two coexisting pipelines and which one runs

There are **two** orchestration implementations of the *same* agent sequence:

1. **`OrchestratorV2`** — hand-written sequential Python loop (`agents/orchestrator_v2.py`).
2. **`GraphPipeline`** — a LangGraph `StateGraph` wrapping the identical agents (`agents/graph_pipeline.py`).

**Selection is by environment variable.** `OrchestratorV2.run()` transparently delegates to `GraphPipeline` when `USE_LANGGRAPH ∈ {1,true,yes}` (`agents/orchestrator_v2.py:361-370`):

```
if os.environ.get("USE_LANGGRAPH") in ("1","true","yes"):
    self._graph_pipeline = GraphPipeline(
        model=self._model, max_iterations=self._max_iter, rule_store=self._rule_store)
    return self._graph_pipeline.run(description, reference_file=..., tier=...)
```

- The delegation **does not pass `recursion_limit`**, so `GraphPipeline` uses its own default of **100** (see §7).
- The validation-tier launch command uses `USE_LANGGRAPH=1` explicitly: `benchmark/run_validation_tiers.py:12` (`PYTHONPATH=. USE_LANGGRAPH=1 OLLAMA_BASE_URL=http://localhost:11434/v1 …`).
- `GraphPipeline` requires `langgraph` installed; it raises `ImportError` otherwise (`agents/graph_pipeline.py:902-904`). Import guard: `agents/graph_pipeline.py:28-44`.

**Note for the thesis:** the two pipelines are intended to be equivalent (`agents/graph_pipeline.py:4-10`, `:887-889` "Identical results to OrchestratorV2"), but the reproduction is measured at **44/45**, not identical — commit `c7394fb` ("Phase 1 verified: LangGraph scaffold reproduces v2 baseline (44/45, CRITICAL physics 95.6%, EASY_03 repair-node TypeError fixed)"). Report as "reproduces v2 to 44/45," not "identical."

The rest of this document describes **`GraphPipeline`** (the `StateGraph`), since that is what the user asked for and what the validation tiers run.

---

## 2. Model assignment per stage

### 2.1 There is exactly ONE model, threaded to every LLM-capable agent
`GraphPipeline.__init__` constructs all agents with the same `model` argument (`agents/graph_pipeline.py:917-933`):

| Agent (constructor) | Line | Takes `model`? | LLM usage |
|---|---|---|---|
| `BasisAgent(model=model)` | `:917` | yes | Two-stage: Stage 1 deterministic lookup, **Stage 2 LLM** only if needed |
| `UnitExtractor(model=model)` | `:918` | yes | **LLM** (Stage-1 topology extraction) |
| `StreamExtractor(model=model)` | `:919` | yes | **LLM** (Stage-1 stream/condition extraction) |
| `GraphBuilder()` | `:920` | **no** | Deterministic (IR assembly) |
| `ThermoMapper(model=model, retriever=…)` | `:928` | yes | Hard rules + **LLM** for package assignment |
| `ParamMapper(model=model, retriever=…)` | `:929` | yes | Deterministic-first (regex + physical estimator); **LLM fallback** for one unknown param |
| `GlobalConsistencyPass()` | `:930` | **no** | Deterministic T/P propagation, zero LLM |
| `Executor()` | `:931` | **no** | DWSIM solve (no LLM) |
| `ErrorClassifier(model=model)` | `:932` | yes | Stage 1 deterministic (`_deterministic_classify`), **Stage 2 LLM** for ambiguous (`agents/stage4/error_classifier.py:7-8,96-99,174-177,216`) |
| `RepairAgent(model=model, retriever=…)` | `:933` | yes | Beam search (deterministic candidate generation + IR-validation ranking); **optional LLM** candidate |

Also LLM-invoked with the same `self._model`: the optional description **summariser** in `_topology_node` (`agents/graph_pipeline.py:1181`, `:1200`).

There is **no per-stage model differentiation** — the same model string is used everywhere. If your methodology needs "which model does the Planner vs the Critic use," the answer is: *the same one*.

### 2.2 Which concrete model — conflicting defaults
- **Library default:** `DEFAULT_MODEL = "gemini-2.5-flash"` (`agents/llm.py:131`). `GraphPipeline`'s constructor defaults to this (`agents/graph_pipeline.py:897`). So a bare `GraphPipeline()` uses **Gemini** (proprietary).
- **Benchmark entrypoints override to open-weight Qwen3 (local via Ollama):**
  - `benchmark_runner.py:46` → `--model` default **`qwen3:30b-a3b`**.
  - `benchmark/runner.py:8`, `:293` → **`qwen3:14b`**.
  - `stage1_diag.py`, `partition_val01.py:70`, `agents/variant_b_diag.py` pass a model explicitly; `agents/test_variant_b.py:158` hardcodes `qwen3:14b`.

So **the reported/validation runs use Qwen3 (14B or 30B-A3B) served locally by Ollama**; the Gemini default only applies if nothing overrides it.

### 2.3 How the model is invoked (Ollama / local)
Provider is inferred from the model-name prefix in `agents/llm.py:137-153`:
- prefix `qwen` → **Ollama** provider (`_OLLAMA` set, `agents/llm.py:129`).
- Ollama is called through the **OpenAI-compatible** client at `OLLAMA_BASE_URL` (default `http://localhost:11434/v1`) with `api_key` = `$OLLAMA_API_KEY` or literal `"ollama"` (`agents/llm.py:133`, `:277-298`).
- Qwen3 specifics: a `"/no_think"` token is prepended to every call (`agents/llm.py:285-286`), `<think>…</think>` blocks are stripped from the reply (`_strip_thinking`, `:115-117`, `:298`), a **hard 120 s wall-clock timeout** wraps each call (`_OLLAMA_TIMEOUT`, `:274`, `_call_with_timeout` `:83-110`), and `num_ctx=16384` is passed via `extra_body` (`:294`). Retries: 3 attempts, base delay 1 s (`:296-297`).
- Retry temperature schedule: attempt 0 → 0.0 (deterministic), later attempts → 0.3 (`retry_temperature`, `agents/llm.py:170-182`).

(Gemini/Anthropic/OpenAI/Groq branches also exist in `agents/llm.py:210-268`; not used on the Qwen benchmark path.)

---

## 3. LangGraph shared-state schema

Defined as a `TypedDict` **`PipelineState`** (`agents/graph_pipeline.py:101-156`). Every field, its type, and its role, in declaration order:

| Field | Type | Group | Meaning (per code) |
|---|---|---|---|
| `description` | `str` | Inputs | Raw user process description (`:103`) |
| `tier` | `str` | Inputs | Benchmark tier (`"standard"` / validation) (`:104`) |
| `reference_file` | `Optional[str]` | Inputs | Path to reference flowsheet JSON, or None (`:105`) |
| `max_iterations` | `int` | Inputs | Stage-4 loop ceiling from constructor (`:106`) |
| `t_start` | `float` | Inputs | `time.time()` at run start (`:107`) |
| `variant_b_active` | `bool` | Variant B | True iff `VARIANT_B=1` **and** a `reference_file` exists (`:110`) |
| `topology_source` | `Optional[str]` | Variant B | `"reference-exact"` \| `"reference-inferred-connections"` (`:111`) |
| `reference_unit_params` | `dict` | Variant B | `tag → {T_out,P_out,…}` from reference (`:112`) |
| `variant_b_inferred_feed` | `bool` | Variant B | Feed stream was synthesised, not given (`:113`) |
| `basis_result` | `Optional[Any]` | Stage 0 | `BasisAgent` result object (`:116`) |
| `norm_desc` | `str` | Stage 0 | Normalised description (`:117`) |
| `compounds` | `list[str]` | Stage 0 | Exact DWSIM compound names (`:118`) |
| `sem_units` | `Optional[Any]` (`SemanticUnits`) | Stage 1 | Extracted units (`:121`) |
| `sem_topo` | `Optional[Any]` (`SemanticTopology`) | Stage 1 | Extracted streams/connectivity (`:122`) |
| `recycle_origin` | `dict` | Stage 1 | `tag → {is_recycle, recycle_target, dropped_by}` captured **before** recycle guards mutate streams; used by Fix 2 (`:123-126`) |
| `ir_graph` | `Optional[Any]` (`FlowsheetGraph`) | Stage 2 | The typed IR graph (`:129`) |
| `ir_report` | `Optional[Any]` (`ValidationReport`) | Stage 2 | Latest validation report (`:130`) |
| `missing_units` | `list[dict]` | Stage 2 | `[{stream, missing_tag, role}]` — streams referencing an absent unit tag (Fix 3) (`:133`) |
| `dwsim_json` | `Optional[dict]` | Stage 3 | Serialised DWSIM flowsheet from `to_dwsim` (`:136`) |
| `reference_data` | `Optional[dict]` | Stage 3 | Loaded reference dict (`:137`) |
| `tried_packages` | `list[str]` | Stage 3 | Property packages already attempted (`:138`) |
| `iteration` | `int` | Stage 4 | Current repair-loop index (`:141`) |
| `eff_max_iter` | `int` | Stage 4 | Effective loop ceiling (may be raised to 15 by beam extension) (`:142`) |
| `beam_extended` | `bool` | Stage 4 | Whether the beam-search iteration extension already fired (`:143`) |
| `repair_memory` | `Optional[Any]` (`RepairMemory`) | Stage 4 | Cross-iteration repair trial memory (`:144`) |
| `sim_hints` | `Optional[Any]` (`SimulationHints`) | Stage 4 | Directional hints parsed from execution (`:145`) |
| `execution` | `Optional[Any]` (`ExecutionResult`) | Stage 4 | Latest DWSIM result (`:146`) |
| `errors` | `list[Any]` (`ClassifiedError`/`SimError`) | Stage 4 | Classified errors for the repair node (`:147`) |
| `prev_hash` | `Optional[str]` | Stage 4 | MD5 of previous `dwsim_json` (loop-detection logging) (`:148`) |
| `warnings` | `Annotated[list[str], operator.add]` | Accumulator | **Reducer = list concat** — appended across nodes (`:151`) |
| `iterations_log` | `Annotated[list[Any], operator.add]` | Accumulator | **Reducer = list concat** — `IterationRecord`s appended (`:152`) |
| `outcome` | `str` | Routing/output | Terminal/branch signal; **initialised to `"MAX_ITER"`** (`:155`, `:1038`) |

**Reducer semantics (important for the methodology):** only `warnings` and `iterations_log` use the `operator.add` reducer (LangGraph appends each node's returned list). **All other fields are last-write-wins** — a node overwrites them by returning the key. `outcome` defaults to `"MAX_ITER"`, so any path that terminates without explicitly setting `outcome` is reported as MAX_ITER (`:1038`).

The initial state dict is built in `run()` at `agents/graph_pipeline.py:1006-1039`.

---

## 4. Nodes (graph vertices)

Registered in `_build_graph()` (`agents/graph_pipeline.py:948-983`). Node name → node function → what it does:

| Node | Function | Line | Behaviour (LLM?) |
|---|---|---|---|
| `basis` | `_basis_node` | `:951`, def `:1143` | `BasisAgent.identify` — normalise compound names; sets `compounds`, `norm_desc`. On failure → `outcome="BASIS_FAILED"`. (Stage-2 LLM.) Asserts Variant B never reaches it (`:1146-1149`). |
| `topology` | `_topology_node` | `:952`, def `:1162` | Optional summarisation (>200 words, non-validation tier) then **`UnitExtractor.extract` + `StreamExtractor.extract`** (LLM), then **3 recycle guards** (target-validity/fuzzy `:1253-1281`, multi-recycle dedup `:1283-1308`, phrase guard `:1310-1323`). On LLM ValueError/Timeout → `outcome="PLAN_FAILED"` (`:1325-1331`). *TopologyChain is intentionally NOT used (see §8).* |
| `reference_topology` | `_reference_topology_node` | `:953`, def `:1083` | **Variant B only** — build `SemanticUnits`/`SemanticTopology` from the reference JSON, bypassing all LLM extraction; connectivity inferred if the reference's `connections` array is empty (`:1117-1125`). |
| `build` | `_build_node` | `:954`, def `:1340` | `GraphBuilder.build` → `FlowsheetGraph`; deterministic compound-name canonicalisation (`canonicalize_list`/`_compound`/`_reaction`), compound reconciliation, composition back-fill, then `normalise(graph)` (`:1394-1396`). No LLM. |
| `validate` | `_validate_node` | `:955`, def `:1400` | `validate(graph)` (schema+graph+physics, no LLM). Classifies invalidity into repairable topology patterns (vessel-outlet Fix 1, cycle Fix 2, missing-unit Fix 3) → `outcome="INVALID_TOPOLOGY"`, else `"INVALID_IR"`, else valid (`:1407-1433`). |
| `topology_repair` | `_topology_repair_node` | `:956`, def `:1435` | **Deterministic, no LLM.** Fix 1 add missing Vessel phase outlet (with superheat guard), Fix 2 repropagate dropped `is_recycle`, Fix 3 detect missing units. Re-validates → `TOPO_OK` / `TOPO_INFEASIBLE` / `TOPO_REPAIR_LLM` (`:1454-1508`). |
| `thermo` | `_thermo_node` | `:957`, def `:1510` | `ThermoMapper.assign` (LLM) → `ParamMapper.assign` (deterministic+LLM fallback) → `GlobalConsistencyPass.apply` → `FailureRuleStore.apply_to_graph` → `normalise` → `validate` #2 → reference load + reactor-param injection (validation tier) → **`to_dwsim`** (`:1529-1611`). Failure → `PLAN_FAILED`/`INVALID_JSON`. |
| `execute` | `_execute_node` | `:958`, def `:1613` | MAX_ITER guard; `Executor.run(dwsim_json)` (DWSIM, no LLM); `SimulationHints.from_execution`; solved+no-critic-failures → `PASS` (+ optional reference-guided refinement); `ErrorClassifier.classify` (Stage-1 det + Stage-2 LLM); terminal error → `HUMAN`; **beam-search iteration extension**; else `outcome="CONTINUE"` (`:1614-1729`). |
| `repair` | `_repair_node` | `:959`, def `:1731` | `RepairAgent.repair` (beam search; deterministic + optional LLM), record repairs into `FailureRuleStore` + `save`, `normalise` + `validate`, re-inject reactor params (validation tier), `to_dwsim`, **`iteration += 1`**, `outcome="CONTINUE"`. On exception forces `iteration = eff_max_iter` to exit next loop (`:1807-1821`). |

---

## 5. Edges, conditional routing, and the repair loop

All edges from `_build_graph()` (`agents/graph_pipeline.py:963-982`). Routing functions are module-level and stateless (`:160-198`).

### 5.1 Conditional entry point
`g.set_conditional_entry_point(_route_entry, {...})` (`:963-965`):
- `_route_entry` (`:160-163`): `"reference_topology"` if `variant_b_active` else `"basis"`.

### 5.2 Static edges
- `reference_topology → build` (`:966`)
- `build → validate` (`:970`)
- `repair → execute` (`:981`) — **this is the loop back-edge**

### 5.3 Conditional edges
| From | Router (line) | Mapping | Decision logic |
|---|---|---|---|
| `basis` | `_route_basis` (`:166-167`) | `{topology, END}` (`:968`) | `END` if `outcome=="BASIS_FAILED"` else `topology` |
| `topology` | `_route_stage1` (`:170-171`) | `{build, END}` (`:969`) | `END` if `outcome=="PLAN_FAILED"` else `build` |
| `validate` | `_route_validate` (`:174-180`) | `{thermo, topology_repair, END}` (`:972-975`) | `INVALID_IR→END`; `INVALID_TOPOLOGY→topology_repair`; else `thermo` |
| `topology_repair` | `_route_topology_repair` (`:183-190`) | `{thermo, repair, END}` (`:977-978`) | `TOPO_OK→thermo`; `TOPO_INFEASIBLE→END`; else (incl. `TOPO_REPAIR_LLM`) `→repair` |
| `thermo` | `_route_thermo` (`:193-194`) | `{execute, END}` (`:979`) | `END` if `outcome ∈ {PLAN_FAILED, INVALID_JSON}` else `execute` |
| `execute` | `_route_execute` (`:197-198`) | `{repair, END}` (`:980`) | `END` if `outcome ∈ {PASS, HUMAN, MAX_ITER}` else (`CONTINUE`) `→repair` |

### 5.4 The repair loop (Stage 4)
```
        thermo ──► execute ──(CONTINUE)──► repair ──► execute ──► …
                     │                                   ▲
                     ├─(PASS│HUMAN│MAX_ITER)──► END      │
                     └───────────────────────────────────┘  (repair always returns to execute, :981)
```
- `execute` and `repair` alternate. `execute` decides PASS/HUMAN/MAX_ITER (exit) vs CONTINUE (→ repair); `repair` unconditionally returns to `execute` (`:981`) after incrementing `iteration`.

### 5.5 Full stage sequence (happy path + branches)
```
[entry] ─(normal)→ basis → topology → build → validate → thermo → execute ⇄ repair → … → END(PASS)
        └(VARIANT_B)→ reference_topology → build → …(as above, LLM extraction skipped)

Branch exits to END:
  basis: BASIS_FAILED
  topology: PLAN_FAILED
  validate: INVALID_IR
  topology_repair: TOPO_INFEASIBLE
  thermo: PLAN_FAILED | INVALID_JSON
  execute: PASS | HUMAN | MAX_ITER

Deterministic topology-repair detour:
  validate ─(INVALID_TOPOLOGY)→ topology_repair ─┬─(TOPO_OK)→ thermo
                                                 ├─(TOPO_INFEASIBLE)→ END
                                                 └─(TOPO_REPAIR_LLM)→ repair → execute ⇄ …
```

---

## 6. Loop-termination / MAX_ITER conditions (exact)

- **Loop ceiling.** `eff_max_iter` starts at `max_iterations` (constructor default **10**, `agents/graph_pipeline.py:898`, seeded into state at `:1029`). `_execute_node` returns `outcome="MAX_ITER"` when `iteration >= eff_max_iter` (`:1618-1619`).
- **Beam-search extension.** On the first iteration where the classified errors contain **more than one** `CONDITION_FIX` strategy and `beam_extended` is False, the loop ceiling is raised to `_BEAM_MAX_ITER = 15` (`:1704-1718`). So the effective ceiling is **10, or 15 once a multi-parameter (coupled) repair is detected**.
- **Iteration increment** happens only in `_repair_node` (`iteration + 1`, `:1801`). `_execute_node` does not increment.
- **Repair exception → forced exit.** If `_repair_node` throws, it sets `iteration = eff_max_iter` so the next `_execute_node` immediately returns MAX_ITER (`:1814-1818`).
- **Missing flowsheet → HUMAN.** If `execute` is reached with `dwsim_json is None` (e.g. a `topology_repair → repair` hand-off before `thermo` built the flowsheet), it returns `outcome="HUMAN"` (`:1624-1627`).
- **Default outcome.** State initialises `outcome="MAX_ITER"` (`:1038`); any silent termination is therefore reported as MAX_ITER.
- **PASS condition.** `execute` returns PASS only when `execution.solved` **and** `_no_critic_failures(execution)` (`:1650`).

---

## 7. LangGraph recursion limit — confirmed **100**

- **Constructor default:** `recursion_limit: int = 100` (`agents/graph_pipeline.py:900`), stored as `self._recursion_limit` (`:913`).
- **Applied at invoke:** `self._app.invoke(initial, config={"recursion_limit": self._recursion_limit})` (`:1041-1042`).
- **Not overridden on the benchmark path:** the `OrchestratorV2 → GraphPipeline` delegation constructs `GraphPipeline(model=…, max_iterations=…, rule_store=…)` **without** a `recursion_limit` kwarg (`agents/orchestrator_v2.py:363-368`), so the default **100** is what actually runs. No call site anywhere passes a different value (grep of `recursion_limit` across repo: only the definition, storage, invoke, and the error-handler log strings — `agents/graph_pipeline.py:900,908,913,1042,1049-1055`; plus an unrelated log-parsing string in `benchmark/aggregate_per_run.py:188-189`).
- **Provenance:** set by commit **`96504e3`** ("Raise LangGraph recursion_limit to 100; terminate cleanly as MAX_ITER").
- **Rationale (per code comment `:908-912`):** LangGraph counts every node visit (super-step) against the limit; the repair loop self-terminates at `eff_max_iter ≤ 15` ⇒ "~36 super-steps incl. setup," and the **LangGraph default of 25 would crash first** with `GraphRecursionError`, so it is raised to 100.
- **Backstop behaviour:** if the ceiling is ever hit, `run()` catches `GraphRecursionError` and returns a clean `outcome="MAX_ITER"` with a `[recursion]` warning rather than crashing (`:1043-1057`).

**Confirmed: the recursion limit is currently 100 and is the effective value on the validation/benchmark path.**

---

## 8. Comment/docstring vs. actual-behaviour mismatches (flagged)

1. **Module docstring says "Phase 1 … No new logic … reproduces orchestrator_v2 results identically"** (`agents/graph_pipeline.py:4-10`, `:887-889`). **Actual code has substantially more than a Phase-1 pass-through:** a deterministic `topology_repair` node with Fix 1/2/3 (`:1435-1508`), a Variant-B `reference_topology` entry path (`:1083-1141`), and recycle guards. These arrived in later commits (e.g. `f854b12`, `dd30f60` "Phase 2 … topology-repair node"). **The "Phase 1 / no new logic" framing is stale** — don't cite it as the current architecture.

2. **"Identical results to OrchestratorV2"** (`:888-889`) vs. measured **44/45** reproduction (commit `c7394fb`). Report the measured figure.

3. **Docstring line 7-8 "TopologyChain (the 4-call LangChain path) is deliberately excluded"** — this **matches** the code: `_topology_node` always uses `UnitExtractor`+`StreamExtractor` (`:1225-1239`), and `TopologyChain` is imported only to log availability, never instantiated (`:53-60`, `:922-926`). (Consistent — noted so you can state it positively.)

4. **`_build_graph` comment "validate → topology_repair (repairable INVALID_TOPOLOGY) | thermo | END"** (`:971`) — matches `_route_validate` and the edge map (`:972-975`). Consistent.

5. **Node vs. router outcome-string coupling is implicit, not enforced.** `_topology_repair_node` emits `"TOPO_REPAIR_LLM"` (`:1504`) but `_route_topology_repair` routes it via a catch-all `else → "repair"` (`:189-190`); the conditional-edge mapping only lists `{thermo, repair, END}` (`:977-978`). Works correctly, but the LLM-handoff outcome is handled by fall-through, not an explicit case — worth noting as a robustness caveat (a future new outcome string would silently route to `repair`).

6. **`outcome` default `"MAX_ITER"`** (`:1038`) means the `basis` success path (which returns no `outcome` key, `:1156-1160`) leaves `outcome=="MAX_ITER"` in state until a later node overwrites it. `_route_basis` only checks for `"BASIS_FAILED"`, so this is benign — but the state's `outcome` field is transiently "MAX_ITER" mid-run, which can mislead logging that reads it before completion.

---

## 9. Cross-references to the companion audit

- Model/open-source stack, provider routing, and the `DEFAULT_MODEL=gemini` vs `qwen3` conflict are analysed in `THESIS_CONTRIBUTIONS_AUDIT.md` (Claim 6).
- The IR typing/validation invoked by the `build`/`validate`/`thermo` nodes (`ir/graph.py`, `ir/validate.py`) is covered there (Claim 1).
- `GlobalConsistencyPass`, `FailureRuleStore`, `CoupledSettler`, and the beam search inside `RepairAgent` (all invoked from `thermo`/`repair` nodes) are covered there (Claims 4, 5, and "other novel components").

## 10. Open conflicts / things to verify before final write-up

- **`max_iterations` default = 10** everywhere I found (`agents/graph_pipeline.py:898`, `agents/orchestrator_v2.py:325`, `partition_val01.py:70`, `stage1_diag.py:176`), but individual diagnostic scripts pass their own (`agents/variant_b_diag.py` uses `args.max_iter`). State the value used for your reported runs.
- **Two model defaults** (`qwen3:30b-a3b` in `benchmark_runner.py:46` vs `qwen3:14b` in `benchmark/runner.py:293`) — different benchmark entrypoints default to different Qwen sizes. Confirm which produced each results table.
- **Node-count / super-step estimate** ("~36 super-steps," `:911`) is a code comment, not measured; if you cite it, mark it as the author's estimate.
