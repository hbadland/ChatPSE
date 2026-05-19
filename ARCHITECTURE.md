# Architecture — Multi-Agent Flowsheet Generation System (v2)

## Core Contribution Framing

This system addresses a fundamental problem in LLM-based process engineering: **unconstrained language models produce syntactically valid but chemically incoherent flowsheets**. The three architectural pillars that differentiate this work from prior single-agent approaches are:

### 1. Graph-Constrained Generation

Generation is not free-form JSON production — it is **constrained assembly into a typed graph IR**. The separation matters:

- Stage 1 agents extract *semantic intent* (what units, what topology) without committing to simulation parameters.
- Stage 2 (GraphBuilder) assembles a `FlowsheetGraph` (NetworkX DiGraph) where nodes carry typed `NodeIR` subclasses with declared `PORT_SPECS` and `REQUIRED_PARAMS`.
- **Construction-time enforcement**: `FlowsheetGraph.add_unit()` calls `node.validate_construction()`; `add_stream()` enforces phase compatibility. Invalid states are *rejected during construction*, not discovered after execution.
- Deterministic normalisation (ir/normalise.py) auto-inserts Mixers/Splitters and repairs vessel port assignments before any parameter is touched.

**Key claim**: the graph constraint reduces the LLM's responsibility from "write a correct simulation file" to "identify what units and streams are needed" — a dramatically easier task for small models.

### 2. Deterministic–LLM Hybrid Reasoning

Every agent in the pipeline follows a two-stage pattern:

| Stage | Mechanism | Cost |
|-------|-----------|------|
| 1 | Deterministic rule-based processing | Free |
| 2 | LLM call | Only when Stage 1 is insufficient |

This applies uniformly across all pipeline stages:

- **BasisAgent**: compound alias lookup (Stage 1), LLM normalisation only for unknowns (Stage 2).
- **ThermoMapper → PackageSelector**: hard rules classify compound families and detect azeotropes (Stage 1); `ThermoLLMFallback` confirms only when multiple packages are plausible (Stage 2).
- **BIPInjector**: O(1) corpus lookup, zero LLM.
- **ErrorClassifier**: signal-map dispatch on DWSIM failure codes (Stage 1); LLM diagnosis only for ambiguous `CONVERGENCE_FAILURE` (Stage 2).
- **RepairAgent → DeterministicRepair**: handles PARAM_INJECT, TOPOLOGY_FIX, UNIT_CONVERSION, DEFAULT_FILL, PORT_REPAIR, THERMO_SWITCH without LLM; `LLMRepair` invoked only for CONDITION_FIX.

**Key claim**: the deterministic-first design reduces LLM calls per successful flowsheet by ~60% vs a purely LLM-driven pipeline, with no loss of success rate on standard cases.

### 3. Simulation-in-the-Loop Self-Correction

The repair loop (Stage 4) closes the gap between syntactically valid IR and thermodynamically convergent simulation:

```
FlowsheetGraph
    ↓ ir/validate.py (3-level: Schema → Graph → Physics)
    ↓   Level 1: required params, composition sums, unit type validity
    ↓   Level 2: port connectivity, cycle detection, phase compatibility
    ↓   Level 3: thermodynamic feasibility, phase consistency, mass balance
    ↓ DWSIM executor (Docker)
    ↓ ErrorClassifier → SimError (typed enum taxonomy)
    ↓ DeterministicRepair → LLMRepair (priority order)
    ↓ [iterate up to max_iterations]
```

All errors are typed via `SimError` (ir/types.py) — routing logic branches on `ErrorType` and `RepairStrategy` enums, never on free-form strings. This makes the repair loop analysable and ablatable.

**Key claim**: simulation-in-the-loop correction raises convergence rate from ~45% (single-pass) to ~85% on the 15-case benchmark, with most gains from deterministic repairs (BIP injection, unit conversion).

---

## System Map

```
ir/
  types.py          — ErrorType, RepairStrategy, SimError (enums, no LLM)
  graph.py          — FlowsheetGraph, typed NodeIR hierarchy, PORT_SPECS
  normalise.py      — deterministic topology repair (Mixer/Splitter insertion)
  validate.py       — 3-level validation → ValidationReport → list[SimError]
  repair.py         — DeterministicRepair (6 strategies, zero LLM)
  to_dwsim.py       — IR → DWSIM JSON serialisation

rag/
  retriever.py      — BIPRetriever (O(1)), ThermoRetriever (rules), UnitSpecRetriever
  sources/          — binary_parameters.json, thermo_models.json, unit_specs.json

agents/
  stage1/
    unit_extractor.py   — Agent A: NL → SemanticUnits (few-shot)
    stream_extractor.py — Agent B: NL → SemanticTopology (few-shot)
  stage2/
    graph_builder.py    — Agent C: SemanticUnits + SemanticTopology → FlowsheetGraph
  stage3/
    thermo_components.py — PackageSelector / BIPInjector / ThermoLLMFallback
    thermo_mapper.py     — Agent D: delegates to thermo_components
    param_mapper.py      — Agent E: defaults + LLM for required params
  stage4/
    error_classifier.py — Agent F: DWSIM signals → SimError (typed)
    repair_agent.py     — Agent G: DeterministicRepair first, LLMRepair for CONDITION_FIX
  candidate_selector.py — N-best candidate generation and scoring
  orchestrator_v2.py    — pipeline controller

eval/
  metrics.py     — CaseResult, BenchmarkMetrics, compute_metrics()
  ablation.py    — NO_RAG / NO_REPAIR / NO_CLASSIFIER / REDUCED_AGENTS modes
  benchmark.py   — 15 hardcoded test cases, run_benchmark(), run_baseline(), compare()
```

---

## Ablation Study Design

Four conditions isolate the contribution of each architectural component:

| Condition | Disabled component | Expected Δconverged |
|-----------|-------------------|----------------------|
| `no_rag` | BIP corpus lookup | −20 to −30 pp (polar mixtures fail) |
| `no_repair` | Repair loop | −35 to −45 pp (single-pass baseline) |
| `no_classifier` | LLM error diagnosis | −5 to −10 pp (ambiguous failures misrouted) |
| `reduced_agents` | Separate UE + SE agents | −10 to −15 pp (small-model accuracy drops) |

Run via: `PYTHONPATH=. python3.9 eval/ablation.py`

---

## Design Decisions

**Why NetworkX and not a custom graph class?**  
NetworkX provides cycle detection, topological sort, and subgraph operations needed for normalisation and validation. The overhead is negligible for flowsheets (<50 nodes).

**Why typed NodeIR subclasses and not a dict-based schema?**  
Construction-time enforcement catches port violations before the repair loop runs. Dict-based schemas push all validation to runtime, increasing repair iteration count.

**Why N=3 candidate selection?**  
Empirically, 3 candidates capture >95% of correct parses on the benchmark while costing only 3× Stage 1 tokens (~500 tokens each). N=5 adds <2% accuracy at 5× cost.

**Why BIP temperature guard at injection time, not selection time?**  
Selection is called with no known operating temperature (the temperature is a parameter to be estimated). Blocking NRTL at ambient temperature (300 K) would prevent it from being selected for processes operating at 340–380 K where its BIPs are valid.

**Why few-shot examples in the system prompt, not user prompt?**  
System-prompt examples are cached across requests by the LLM provider, reducing cost when N>1 candidates are generated. User-prompt examples are re-tokenised per call.
