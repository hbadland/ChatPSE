# Multi-Agent LLM Systems: A Literature Review
## Agent Structure, Prompt Design, Context Management, Task Delegation, and LLM Call Efficiency

*Compiled May 2026 — sources from 2024–2026*

---

## 1. Agent Structure and Decomposition

### How many agents do top systems use?

There is no single optimal number. Empirically documented ranges:

| Domain | System | Agents | Notes |
|--------|--------|--------|-------|
| Software engineering | MASAI, SWE-agent, OpenHands | 4–6 | Each with a bounded single objective |
| Scientific simulation | Sketch2Simulation, Text-to-Simulation | 4–9 | Layered interpretation→synthesis→evaluation |
| Reasoning/QA | MAR, MARS | 5–7 | Actor + critic ensemble + judge |
| Business process | CrewAI sequential | 2–4 | Simpler pipelines |

The consistent theme across high-performing systems is that each agent has a **single, bounded objective**. MASAI's authors explicitly attribute their SWE-bench results to "avoiding unnecessarily long trajectories" through strict sub-agent scoping.

### Canonical agent role patterns

**Planner–Executor–Critic–Refiner** is the most common decomposition in engineering and simulation pipelines:

- **Planner**: converts natural language to a structured plan or topology; operates at the highest level of abstraction; output is always a typed intermediate representation
- **Executor**: applies the plan to an external environment (simulator, compiler, tool); deterministic where possible; no LLM calls in ideal case
- **Critic**: evaluates execution output against ground truth or domain rules; Stage 1 deterministic checks first, LLM only for ambiguous cases
- **Refiner**: applies targeted patches based on Critic diagnosis; receives minimal context (diagnosis + compact flowsheet state), not full history

**Actor–Critic–Ensemble** patterns (MAR, MARS):

- MAR uses 4 critic personas designed orthogonally: Skeptic (evidence-focused), Logician (specification-focused), Creative (exploration-focused), Verifier — preventing shared confirmation bias that afflicts single-agent Reflexion
- MARS assigns independent reviewers working in parallel with no reviewer-to-reviewer communication, then a meta-reviewer synthesises; this achieves ~50% token reduction vs Multi-Agent Debate while matching or exceeding its accuracy

### Hierarchical vs flat architectures

Empirical comparisons consistently favour hierarchical over flat:

- A financial document processing study found hierarchical (manager delegates to workers) achieved F1 = 0.921 at 1.4× baseline cost; fully reflexive achieved F1 = 0.943 at 2.3× cost — hierarchical is near-Pareto dominant
- MARS (hierarchical star topology) reduces tokens ~50% while matching Multi-Agent Debate (fully-connected) on GPQA (36.33% vs 31.00%) and MMLU
- Fully-connected architectures (every agent communicates with every other) scale poorly: performance degrades roughly as O(n²) communication paths, and LLM attention over irrelevant agent messages compounds the problem

**When flat architectures work**: for simple 2–3 agent pipelines where the overhead of a manager is not justified, or when all agents are genuinely peer-level (e.g. parallel ensemble reviewers in MARS).

### Single-responsibility principle: empirical support

A clinical NLP workload study (npj Health Systems, 2026) provided the clearest quantitative evidence:

- Each task delegated to its own dedicated worker agent
- **5 tasks**: multi-agent = 90.6% accuracy; single agent = 73.1% (+17 points)
- **80 tasks**: multi-agent = 65.3%; single agent = 16.6% (+48 points)
- Token usage: 65-fold reduction in the multi-agent setting
- The mechanism stated explicitly: "each worker receives only tokens relevant for a single decision so attention is not diluted across irrelevant material"

The accuracy gap grows with scale — the multi-agent advantage is not just ergonomic, it is functionally essential at high task counts.

---

## 2. Prompt Design for Small Models

### System prompt length

Across prompt length studies on LLaMA and Mistral variants (2025):

- Short prompts (<50% of baseline length) consistently degrade results across all 9 tested domains
- Simple tasks: 50–100 words optimal; moderate complexity: 150–300 words
- Longer prompts with domain background and explicit constraints consistently improve structured generation
- **Critical exception**: irrelevant length is actively harmful — 80% irrelevant context degrades Gemini 2.0 Flash from 85% to 72% accuracy; this is worse than slightly too-short prompts

The practical rule: **prompts should be exactly as long as the task requires, and no longer**. Padding, boilerplate, and generic disclaimers all impose cognitive cost.

### Chain-of-thought for small models: does it work?

Results are more negative than the field generally assumes:

- Llama-3-8B, Llama-3-70B, and Mistral-7B all achieved **0% accuracy on multi-hop reasoning tasks** even in clean conditions (no distractors) — chain-of-thought alone cannot rescue intrinsically multi-hop tasks in small models
- Self-critique and meta-cognitive interventions (asking the model to review its own output) often **harm** small model performance
- DBT-structured prompts (decompose-before-think): +7% on StrategyQA for 8B models, +16.2% for 14B models — structured decomposition instructions help more than free-form CoT
- RAG-based grounding reduces errors by 7.6% and is more reliable than self-critique for small models (Self-RAG study)

The practical implication: **do not rely on small models to self-diagnose complex reasoning failures**. External structure (tools, deterministic validators, explicit decomposition instructions) is more reliable than asking the model to think harder.

### Few-shot examples

- HiPlan uses 2–4 demonstrations in step-hint prompts at temperature=0.0
- Text-to-Simulation uses CoT + few-shot in the Parameter Configuration Agent, which operates at temperature=0.9
- Sketch2Simulation enforces strict JSON schemas at the interpretation layer before synthesis agents ever receive structured data
- Self-RAG fine-tunes the model to emit special reflection tokens rather than using in-context few-shot — avoids prompt-space overhead for models you control the weights of

**Observed pattern**: the most effective few-shot examples in structured-output tasks are schema-complete worked examples — showing exact input context plus exact valid JSON output. Abstract structural examples ("fill in the values") are less effective than concrete worked ones.

### JSON schema enforcement: the penalty is severe

The most surprising and consistently replicated finding in this domain:

- Requiring JSON output **reduced GSM8K accuracy by 27.3 percentage points** in models without constrained decoding
- This is not a model quality issue — it is a formatting-reasoning conflict: the model allocates reasoning capacity to maintaining valid JSON structure, leaving less for the reasoning task itself
- The fix: either (a) separate the reasoning step from the formatting step — reason first in free text, format in a second call; or (b) use constrained decoding so the model does not need to reason about JSON validity at all

**Temperature differentiation by function** (explicitly documented in Text-to-Simulation):

| Agent function | Temperature |
|----------------|-------------|
| Task understanding, evaluation, verification | 0.1 |
| Topology generation, parameter search, creative generation | 0.9 |

This matches the intuition: stable agents doing interpretation/routing should be deterministic; generative agents doing synthesis should explore.

### Role prompting

- Role-playing prompts with specific expertise reduce hallucination by grounding outputs in a claimed knowledge domain
- ORPP (EMNLP 2025): automatic prompt optimisation for role-playing agents; Gemini-1.5-Flash-8B achieved a 21.62-point accuracy increase just from switching to an optimised JSON prompt format
- The effect is larger for smaller models — their outputs are more sensitive to framing

---

## 3. Context Window Management

### The core problem

Agent trajectories grow unboundedly. Each tool call appends its input and output to the context. In a multi-agent pipeline, the same tokens appear with different preceding contexts, blocking naive KV cache reuse. Without active management, context cost dominates total inference cost after the first 5–10 iterations.

Empirical data: Text-to-Simulation (GPT-4o, 3 MCTS child nodes) consumed **994.7K tokens per run** and took 913 seconds. Expanding to 5 nodes: 1,325.6K tokens. Token cost scales super-linearly with exploration breadth.

### AgentDiet — trajectory compression (FSE 2026, arxiv 2509.23586)

The most systematically evaluated context compression approach for agents:

- **Trigger**: step exceeds 500 tokens AND compression would save >500 tokens
- **Compression model**: cheap model (GPT-4o mini, 12× cheaper than main model)
- **Three waste categories removed**: useless (cache directories, verbose output), redundant (duplicated tool responses), expired (data no longer needed by current step)
- **Analysis window**: examines steps s-a-b through s to reduce step s-a (delay=2, window=1)
- **Token reduction**: 39.9–59.7% input tokens
- **Cost reduction**: 21.1–35.9% total
- **Task performance impact**: −1% to +2% (neutral to slightly positive)
- **Key finding**: on harder benchmarks with longer trajectories, performance *improved* from context reduction — irrelevant context was actively harmful

Overhead from the compression calls: 5.2–14.8%. Net cost saving is still strongly positive.

### State-based workflow design (Sketch2Simulation)

An architectural alternative to compression: **each agent receives only its validated upstream state**, not the full trajectory history. The state is a typed, validated JSON intermediate representation:

```
s_{k+1} = A_k(s_k)
```

Each agent takes the current state in and produces the next state out. No agent sees raw prior conversation history. This eliminates the need for compression entirely by design, but requires careful IR (intermediate representation) design upfront.

### KV cache management (KVFlow, arxiv 2507.07400)

For systems where multiple agents share an inference server:

- Models the execution schedule as an Agent Step Graph; assigns "steps-to-execution" values
- Eviction policy: prioritise KV nodes for agents running soonest
- Speedup: **1.83× for single workflows with large prompts; 2.19× for concurrent workflows** vs SGLang baseline

### LLMLingua (Microsoft Research) — extractive compression

- Up to **20× compression** with only 1.5% performance loss on reasoning tasks
- Extractive reranker compression at 4.5× ratio: +7.89 F1 on 2WikiMultihopQA (better than uncompressed — long contexts with irrelevant material hurt more than compression)
- Abstractive compression at same ratio: −4.69 F1 (worse — abstractive summaries lose critical detail)
- Rule: extractive compression is safe; abstractive compression is risky

### Practical ceiling

Model performance degrades measurably beyond ~4K tokens in production use for many open-source models. Even if the model technically supports longer contexts, attention quality degrades before hitting the hard limit.

---

## 4. Task Granularity — How Much to Ask Each LLM

### Quantitative cognitive load evidence

ToolLoad-Bench (arxiv 2601.20412) derived an exponential decay model:

```
Accuracy ≈ exp(−(k · CL_total + b))
```

Where:
- `k` = load sensitivity parameter — how gracefully the model handles complexity increases
- `b` = baseline capability
- `CL_total` = total cognitive load of the task

| Model | Overall Accuracy | Notes |
|-------|-----------------|-------|
| xLAM2-32B (fine-tuned specialist) | 78.8% | k = 0.034 — very low load sensitivity |
| GPT-4o | 68% | |
| Claude 3.7 Sonnet | 64.8% | |
| Qwen3-235B | 58% | |
| Qwen3-8B | 38.6% | k > 0.1 — high load sensitivity |
| Llama 3.3-70B | 17% | Catastrophic on complex tool tasks |

Fine-tuned specialists win via **dual advantage**: both lower baseline difficulty (`b`) and lower load sensitivity (`k`). For a fixed model, the only lever is reducing `CL_total` — i.e. giving each call a smaller task.

### Multi-hop reasoning: zero tolerance for bundled complexity

Multi-hop reasoning without decomposition (arxiv 2509.19517):
- Llama-3-8B, Llama-3-70B, Mistral-7B: **0% on any multi-hop task** regardless of context quality
- Gemini-2.0-Flash: 85% → 72% as irrelevant context grows to 80% of total

The implication for agent design: any task requiring more than one conceptual "step" must be decomposed into separate atomic LLM calls. You cannot ask a small model to perform compound reasoning in a single call and expect reliable results.

### Instance bundling causes performance collapse

When multiple instances are processed in a single call (arxiv 2603.22608):
- All models: slight degradation at 20–100 instances, then performance collapse on larger counts
- Instance count has a stronger negative effect than raw context length
- Single-task isolation prevents attention dilution across instances

### MASAI explicit step limits

MASAI enforces hard caps per sub-agent:
- Test Template Generator: max 3 retries
- Fault Localizer: max 25 ReAct steps
- Edit Localizer: max 25 steps

These are not soft guidelines — they are hard limits that trigger agent termination and escalation. Without them, **FM-1.3 (step repetition, 15.7% of failures) and FM-1.5 (unaware of termination, 12.4%)** compound indefinitely.

---

## 5. LLM Call Counts — Efficiency and Reliability

### Documented call budgets

| System | Calls per run | Notes |
|--------|--------------|-------|
| Text-to-Simulation (3 MCTS nodes) | ~15–30 estimated | 994.7K tokens, 913s total |
| Sketch2Simulation (9 agents) | 10–20 clean run | Higher with execution failures |
| MAR (6–7 agents) | 300–400 API calls | 3× cost of single-agent Reflexion |
| MASAI (SWE-bench) | ~50–100 steps | Bounded per sub-agent |
| Compiled AI (post-compilation) | 0 | Single generation call; all execution deterministic |

MAR achieves the best downstream accuracy (+3–6 points over Reflexion) but at 3× cost. For production systems with tight budgets, MARS (parallel independent reviewers) achieves similar gains at ~50% of MAR's token cost.

### Deterministic stages: the zero-cost baseline

The highest-leverage architectural pattern across all reviewed systems is maximising the fraction of work that requires zero LLM calls:

**Compiled AI** (arxiv 2604.05150): generates business logic once (20–50 lines of code), then:
- All subsequent execution: 100% deterministic, zero additional LLM calls
- BFCL task completion: 96% at zero execution tokens
- Speed: 4.5ms vs 2,004ms (450× faster)
- Cost: 40× lower at 1M transactions/month

**Sketch2Simulation**: A3 normalization agent is purely rule-based — zero LLM cost for topology cleanup. Represents ~10% of pipeline steps at zero cost.

**Practical rule**: any stage that can be expressed as a deterministic function (validation, schema enforcement, topology wiring for standard patterns, unit conversion) should never use an LLM call. The LLM should be reserved for disambiguation, generation, and diagnosis of novel cases.

### Model routing by task complexity

Amazon cascade study: 60–70% of LLM calls in a typical agent pipeline are routine tasks (classification, extraction, formatting) that small models handle as well as large ones.

Systems that exploit this:

- **BudgetMLAgent**: cheap model for routine agentic calls, escalates on failure → **94% cost reduction** ($0.931 → $0.054 per task) while maintaining success rate
- **xRouter** (Salesforce, arxiv 2510.08439): RL-trained Qwen2.5-7B router decides to answer locally or invoke large model; near GPT-5 accuracy on Olympiad Bench at ~1/8 the cost
- **AgentCollab** (arxiv 2603.26034): small model as default controller, large model invoked only when trajectory hits difficult segments via difficulty-aware cumulative escalation

### Retry strategies

Temperature scheduling on retry is consistently beneficial but rarely documented in papers:

- MASAI: 3 retries for Test Template Generator; no temperature variation documented
- Text-to-Simulation: temp=0.1 for stable agents, temp=0.9 for generative agents — but no retry loop variation
- The general principle from practitioner literature: temperature=0 on attempt 0 (deterministic, reliable for structured JSON); raise to 0.3 on retry 1 (break internally-consistent-but-wrong outputs that reproduce identically at temp=0)

Retry without temperature variation guarantees the same wrong output: if a model at temperature=0 produces invalid JSON once, it will produce exactly the same invalid JSON on every retry. Temperature escalation on retry is the only reliable way to break this.

### Cycling detection

A documented production incident: agent A asked agent B for clarification, B asked A back → infinite loop ran for **11 days**, cost escalated from $127/week to $47,000/month. No exaggeration — this was a real deployed system.

Required safeguards:
1. Message queues with deduplication
2. Circuit breakers (detect identical output hash across N iterations)
3. Explicit max-step counters per agent
4. Hard cost controls at the orchestrator level
5. MAST taxonomy: FM-1.3 (step repetition) + FM-1.5 (unaware of termination) = 28.1% of all failures combined — the two most critical failure modes for long-running systems

### Parallel execution

M1-Parallel (arxiv 2507.08944): 5 parallel teams with early termination → **2.2× speedup** over sequential with no accuracy loss. The key design: teams work independently and the first acceptable result terminates the ensemble.

---

## 6. Structured Output and JSON Reliability

### Constrained decoding: the reliability floor

**XGrammar** (arxiv 2411.15100):
- Persistent parsing stack + context-independent pre-checks
- **100× faster than prior grammar libraries**
- Default backend for vLLM, SGLang, TensorRT-LLM as of early 2026
- <40 microseconds per token overhead in JSON generation
- XGrammar slightly outperforms Llguidance in repeated-schema scenarios (due to caching)

**JSONSchemaBench** (arxiv 2501.10868, 10,000 real-world schemas):
- Best framework supports twice as many schemas as worst
- Constrained decoding provides ~3% improvement on reasoning tasks vs unconstrained
- Outlines: ~93% success on JSON schemas; XGrammar/LM-Format-Enforcer: 60–93% depending on schema complexity
- Over-constraining (blocking valid outputs) is more frequent than under-constraining — test with your specific schemas

**Practical accuracy numbers for 7B–14B models**:

| Setup | JSON correctness |
|-------|-----------------|
| Unconstrained, no fine-tuning | ~79% syntactic, ~56% semantic |
| Prompt-only schema enforcement | ~79% syntactic (same — prompt doesn't help syntax) |
| Constrained decoding (XGrammar) | ~96–99% syntactic |
| Fine-tuned + constrained decoding | ~99.5% syntactic + semantic |

The gap between prompt-only and constrained decoding is approximately **17–20 percentage points** on syntactic correctness alone, independent of model size or reasoning quality.

### Function calling vs raw JSON generation

AgentArch benchmark (arxiv 2509.10769):
- **Function calling: 0% hallucination rate** in multi-agent settings
- **ReAct + raw JSON: up to 36% hallucination rate** in the same settings
- Function calling substantially outperforms ReAct across most models tested

This is the strongest single argument for using structured tool/function-calling interfaces over raw text prompting when reliability matters.

### Separating reasoning from formatting

The 27.3-point GSM8K accuracy drop from JSON formatting requirements points to a clean architectural solution: **two-call split for reasoning + formatting**:

1. Call 1: free-text reasoning (no JSON constraint, full model capacity for reasoning)
2. Call 2: format the reasoning result into JSON (trivial task, low model capacity required, constrained decoding appropriate)

This adds one LLM call but recovers the full reasoning capacity of the model.

---

## 7. Failure Handling and Recovery

### MAST taxonomy: where failures actually come from

The MAST study (arxiv 2503.13657) annotated 1,642 traces from multiple production multi-agent systems:

**FC1 — System Design (43.8% of all failures)**:
| Code | Failure Mode | Frequency |
|------|-------------|-----------|
| FM-1.1 | Disobey task specification | 11.8% |
| FM-1.3 | Step repetition | 15.7% — most common single failure |
| FM-1.5 | Unaware of termination conditions | 12.4% |
| FM-1.4 | Loss of conversation history | 2.8% |
| FM-1.2 | Disobey role specification | 1.5% |

**FC2 — Inter-Agent Misalignment (32.15% — unique to multi-agent)**:
| Code | Failure Mode | Frequency |
|------|-------------|-----------|
| FM-2.6 | Reasoning-action mismatch | 13.2% — second most common |
| FM-2.3 | Task derailment | 7.4% |
| FM-2.2 | Fail to ask clarification | 6.8% |
| FM-2.5 | Ignored other agent input | 1.9% |
| FM-2.4 | Information withholding | 0.85% |

**FC3 — Task Verification (24.5%)**:
| Code | Failure Mode | Frequency |
|------|-------------|-----------|
| FM-3.3 | Incorrect verification | 9.1% |
| FM-3.2 | No/incomplete verification | 8.2% |
| FM-3.1 | Premature termination | 6.2% |

**Critical insight**: **44% of all failures are system design issues**, not model capability failures. Clearer role specifications, explicit termination conditions, and step counters would theoretically eliminate nearly half of all failures without any model improvement.

Adding high-level objective verification at the orchestrator level = **+15.6% improvement** in documented cases.

### Reflexion and its limits

Reflexion (Shinn et al.): agent generates natural language reflection on failure → stored as episodic memory for next attempt.

Results:
- ALFWorld: 56–64% with Reflexion vs 50–59% ReAct
- Limitation: same model acts, evaluates, and reflects → **confirmation bias** — the model that made the error is being asked to diagnose it

MAR's fix: diverse critic personas eliminate shared bias. Three orthogonal critics (Skeptic, Logician, Creative) are less likely to share the same misconception than a single self-reflective agent. This is the key architectural insight behind multi-critic systems.

### Self-RAG vs self-critique for small models

- Self-RAG (7B model): reduces errors by 7.6%, outperforms ChatGPT on OpenDomain QA
- Mechanism: retrieval-grounded verification is more reliable than pure self-critique
- Meta-cognitive interventions (asking the model to check its own work) without RAG **harm** small model performance more often than they help

For small models, **external ground truth** (simulation feedback, schema validation, tool output verification) is categorically more reliable than self-critique.

### Recovery pattern hierarchy

Effective systems order recovery from cheapest to most expensive:
1. **Deterministic fix** (free): unit conversion, composition normalisation, port swapping
2. **Schema retry with temperature escalation**: same prompt, higher temperature, max 2–3 attempts
3. **Targeted LLM patch** with diagnosis context
4. **Agent-level replan**: full topology rebuild (expensive, last resort)
5. **Human escalation**: thermodynamically infeasible cases

Skipping levels — jumping to full replan on the first failure — is both expensive and usually wrong. Most failures are addressable at levels 1–3.

---

## 8. Domain-Specific Multi-Agent Systems (Science/Engineering)

### From Text to Simulation (arxiv 2601.06776)

The most directly analogous published system to this pipeline. LangGraph + LangChain + GPT-4o primary, tested against Claude Sonnet 4, AutoGen, CrewAI, MetaGPT.

**Agent structure** (4 sequential agents):

| Agent | Task | Temperature |
|-------|------|-------------|
| Task Understanding | NL → structured requirements | 0.1 |
| Topology Generation | Requirements → directed graph G=(V,E) | 0.9 |
| Parameter Configuration | CoT + few-shot → operating conditions | 0.9 |
| Evaluation Analysis | Score across 5 dimensions | 0.1 |

Plus an **auxiliary thermodynamic pre-validation workflow** as a gate before MCTS expansion.

**Enhanced MCTS over process configurations**:
- 3 children per expansion node
- Dual-layer value function: immediate fitness + potential fitness, dynamically weighted α(t)
- Dynamic revisit for underperforming nodes
- Each MCTS node = a complete process configuration, not an individual unit operation

**Results**:
- Simulation convergence rate: **80.3%** (vs 23.4% GPT-4o single-agent baseline; 100% expert manual)
- Overall design score: 73.88/100 (vs 86.16 expert)
- Design time: 913s vs 8,301s manual (89% reduction)
- Token budget: 994.7K tokens per run at 3 MCTS children

**What this means for small-model pipelines**: the MCTS approach (exploring multiple configurations and selecting the best) is explicitly designed for cases where a single forward pass is unreliable. With small models, this approach is more valuable, not less — but the token cost at 3+ explorations may be prohibitive unless each evaluation is cheap.

### Sketch2Simulation (arxiv 2603.24629)

9-agent system for visual engineering diagrams → Aspen HYSYS. The most architecturally detailed published system.

**Three-layer architecture**:

*Interpretation layer* (3 agents, all Gemini 3 Flash):
- A1 Descriptor: visual diagram → text description
- A1.1 Validation: checks A1 output for completeness
- A2 Extractor: text → typed JSON intermediate representation

*Normalization layer* (1 agent, **zero LLM**):
- A3: rule-based topology cleanup and validation — no model calls

*Synthesis layer* (4 agents):

| Agent | Model | Task |
|-------|-------|------|
| B1 Basis | Qwen2.5-Coder-7B + RAG | Component names → HYSYS database lookup |
| B2 Instantiation | Qwen2.5-Coder-7B | Create unit operations in simulator |
| B3 Configuration | Qwen3-Coder-30B | Stream connections (more complex task → larger model) |
| B4 Execution | Qwen3-Coder-30B + rules | Analyze execution trace → targeted corrections → retry |

**Key design decisions**:
1. State-based workflow: `s_{k+1} = A_k(s_k)` — each agent gets only its validated upstream state
2. Typed JSON IR enforced at A2 before any synthesis agent touches the data
3. Model is matched to task complexity: 7B for simple tasks, 30B for stream connectivity
4. A3 normalization is purely rule-based — zero LLM cost for this stage
5. B4 uses both LLM and rules: LLM analyzes trace, rules enforce fixes deterministically

**F1 scores by process complexity**:

| Process | Component F1 | Stream F1 | Connection F1 |
|---------|-------------|-----------|---------------|
| Desalting / Merox | 1.00 | 1.00 | 1.00 |
| Distillation | 0.97 | 0.93 | 0.97 |
| Aromatic complex | 0.98 | 0.96 | 0.98 |

### ChemCrow (Nature Machine Intelligence 2024)

- GPT-4 + **18 expert-designed chemistry tools** (ReAct-style)
- Tools: name conversion, reaction planning, safety checks, literature search, web search, Python execution, quantum chemistry interfaces
- Successfully planned and executed synthesis of insect repellent and 3 organocatalysts
- Expert evaluation: ChemCrow outperforms GPT-4 alone "by a large margin"
- Key insight: **LLM as pure reasoning engine; all domain knowledge lives in tools, not model weights**

This is the critical architectural lesson from chemistry agents: don't try to bake domain knowledge into prompts. Build it into tools that return ground truth. The model's job is reasoning over tool outputs, not remembering chemistry.

### How scientific agents differ from general agents

Seven properties distinguish science/engineering agents from coding or QA agents:

1. **Domain knowledge in tools, not weights**: chemistry databases, simulation APIs, property calculators return ground truth that the model doesn't need to hallucinate
2. **External environment as critic**: simulation convergence/failure is an objective signal, not a soft LLM evaluation — this eliminates the confirmation bias problem entirely
3. **Long evaluation cycles**: a simulation may take 30–300 seconds per run; MCTS exploration must balance breadth against per-evaluation cost
4. **MCTS or iterative search is essential**: parameter spaces are vast; a single greedy forward pass will miss the valid region
5. **Temperature differentiation is pronounced**: interpretation tasks want 0.1; topology and parameter generation want 0.9 — the gap is wider than in general agents
6. **Typed intermediate representations are non-negotiable**: free-text passing between agents fails in simulation pipelines; every handoff must be typed and validated
7. **Rule-based preprocessing is disproportionately valuable**: unit conversion, composition normalisation, port assignment, topology validation — these are zero-LLM-cost stages that collectively prevent the most common failure modes

---

## 9. Benchmark Numbers

### GAIA benchmark — small models with/without tools

(arxiv 2601.11327, Qwen3 models with Agentic-Reasoning framework)

| Model | No tools | Best agentic | Best mode |
|-------|----------|-------------|-----------|
| Qwen3-4B | 9.70% | 18.18% | Planner thinking |
| Qwen3-8B | 6.06% | 16.36% | Full thinking |
| Qwen3-14B | 7.27% | 19.39% | Planner thinking |
| Qwen3-32B | 9.70% | 25.45% | No thinking |
| Claude Opus 4.5 (frontier) | — | 77.5% | — |

**Key finding**: tool augmentation gives larger gains than thinking mode for small models. Full thinking with tool orchestration actually **degrades** 4B performance (13.33% → 9.09%) — thinking tokens and tool use compete for context capacity.

Planner-only thinking (model thinks during task decomposition but not during individual tool calls) consistently outperforms full thinking for ≤14B models.

### ALFWorld — household task completion

| Model | Method | Success rate |
|-------|--------|-------------|
| LLaMA-3.3-70B | ReAct | 50% |
| Mixtral-8x22B | ReAct | 59% |
| LLaMA-3.3-70B | HiPlan | **94%** |
| Mixtral-8x22B | HiPlan | 82% |

HiPlan's retrieval of relevant milestone demonstrations (offline library) gives LLaMA-3.3-70B an 88% relative improvement over ReAct. The offline library = zero additional inference cost at task time.

### SWE-bench (software engineering)

- MASAI: 28.33% on SWE-bench Lite (state of art when published, 2024)
- SWE-agent + GPT-4-turbo (single agent): ~12.5% on full SWE-bench

Multi-agent decomposition vs single agent: roughly 2.3× improvement at similar or lower cost per issue (<$2 for MASAI vs higher for monolithic approaches).

### Where small models fail vs succeed

**Fail**:
- Multi-hop reasoning without decomposition: 0% (Llama-3-8B, Mistral-7B)
- Complex tool orchestration with full thinking: 4B degrades with thinking enabled
- High cognitive load tool tasks: Llama 3.3-70B scores only 17% on ToolLoad-Bench
- Mixed heterogeneous task batches: performance collapses beyond ~10 concurrent tasks

**Succeed**:
- Atomic single-task calls with tools: Qwen3-4B with tools (18%) outperforms Qwen3-32B without tools (9%) on GAIA
- Tool-augmented factual retrieval: Self-RAG 7B outperforms ChatGPT on Open-Domain QA
- Structured code generation with strict schema: Qwen2.5-Coder-7B achieves F1=1.00 on simple processes in Sketch2Simulation
- Fine-tuned tool calling: fine-tuned 7B achieves 77.55% on ToolBench vs ChatGPT-CoT at 26.00%

---

## 10. Cross-Cutting Design Principles

Synthesised from all reviewed systems and studies:

### 1. Single responsibility per LLM call

Every system achieving top results in small-model settings has atomic task boundaries. Multi-task bundling causes performance collapse in 7B–14B models due to attention dilution and competing formatting/reasoning demands.

### 2. Separate reasoning from formatting

Requiring JSON output costs 27 points on GSM8K without constrained decoding. The reliable pattern: reason first in free text, format in a second call, or use XGrammar/constrained decoding to handle formatting outside the model's attention.

### 3. Planner-only thinking for small models

Full thinking degrades tool orchestration in ≤14B models. Thinking tokens and tool call context compete for the same capacity. Restrict thinking to the planning/decomposition stage; disable it for individual tool calls and structured generation.

### 4. Compress trajectories early and continuously

AgentDiet's 39–60% token reduction with near-zero accuracy loss is the highest-leverage single intervention for long-running agents. Compress before the context window degrades attention quality, not after.

### 5. Explicit termination conditions everywhere

28.1% of all failures (FM-1.3 + FM-1.5 combined) stem from agents that don't know when to stop. Hard step limits and explicit DONE criteria in every agent's system prompt are not optional.

### 6. Function calling over ReAct for reliability

Function calling = 0% hallucination rate in multi-agent settings vs up to 36% for ReAct (AgentArch benchmark). Where the inference infrastructure supports it, structured function calling is categorically more reliable.

### 7. Temperature differentiation by function

Stable/interpretive agents: 0.1. Generative/exploratory agents: 0.9. Temperature variation on retry (0.0 first attempt, 0.3+ on retries) breaks identical-wrong outputs that a deterministic model reproduces indefinitely.

### 8. Hierarchical over flat for cost-efficiency

Hierarchical architecture at 1.4× baseline cost matches reflexive at 2.3× cost at near-equal F1. The advantage grows with pipeline complexity and task count.

### 9. Maximise the deterministic fraction

Zero-LLM stages (validation, schema enforcement, standard topology wiring, unit conversion) should handle everything they can. The LLM should see only genuinely ambiguous inputs. Sketch2Simulation's A3 normalization and the Text-to-Simulation thermodynamic pre-check exemplify this.

### 10. Route by task complexity, not by agent turn

60–70% of LLM calls in typical pipelines are routine. Routing these to smaller/cheaper models (BudgetMLAgent: 94% cost reduction; xRouter: near-GPT-5 accuracy at 1/8 cost) is the fastest path to production-viable economics.

### 11. External validation beats self-critique for small models

For small models, external ground truth (simulation convergence, schema validation, test execution) is categorically more reliable than asking the model to evaluate its own output. Build hard validators, not self-reflection loops.

---

## References

| Paper | URL | Notes |
|-------|-----|-------|
| From Text to Simulation | arxiv.org/abs/2601.06776 | Most analogous system to this pipeline |
| Sketch2Simulation | arxiv.org/abs/2603.24629 | Most architecturally detailed science agent |
| AgentDiet | arxiv.org/abs/2509.23586 | Context compression for agents |
| MAST failure taxonomy | arxiv.org/abs/2503.13657 | 1,642 annotated failure traces |
| HiPlan | arxiv.org/abs/2508.19076 | Milestone library retrieval |
| MAR | arxiv.org/abs/2512.20845 | Multi-critic reflection |
| MARS | arxiv.org/abs/2509.20502 | Parallel independent reviewers |
| MASAI | arxiv.org/abs/2406.11638 | SWE-bench decomposition |
| MetaGPT | arxiv.org/abs/2308.00352 | SOP-driven multi-agent |
| XGrammar | arxiv.org/abs/2411.15100 | Constrained decoding |
| JSONSchemaBench | arxiv.org/abs/2501.10868 | Schema compliance benchmark |
| AgentArch | arxiv.org/abs/2509.10769 | Function calling vs ReAct |
| ToolLoad-Bench / Cognitive Load | arxiv.org/abs/2601.20412 | Cognitive load framework |
| Multi-hop reasoning limits | arxiv.org/abs/2509.19517 | 0% small model multi-hop |
| KVFlow | arxiv.org/abs/2507.07400 | Multi-agent KV caching |
| Compiled AI | arxiv.org/abs/2604.05150 | Deterministic post-generation execution |
| xRouter | arxiv.org/abs/2510.08439 | RL-based model routing |
| AgentCollab | arxiv.org/abs/2603.26034 | Difficulty-aware escalation |
| Can Small Agents Beat Large? (GAIA) | arxiv.org/abs/2601.11327 | Small model agentic benchmarks |
| Prompt length effects | arxiv.org/abs/2502.14255 | Prompt length vs accuracy study |
| Acon context compression | arxiv.org/abs/2510.00615 | 26–54% memory reduction |
| AFlow | ICLR 2025 | Automated workflow generation via MCTS |
| Self-RAG | selfrag.github.io | Small model retrieval-augmented generation |
| ChemCrow | arxiv.org/abs/2304.05376 | Chemistry tool-augmented agent |
| npj Health Systems clinical study | nature.com/articles/s44401-026-00077-0 | Multi-agent vs single-agent scaling |
| npj AI physics simulation | npj AI 2025 | Plan-act-reflect-revise for physics |
| Nature Comms Eng distillation agent | nature.com/articles/s44172-025-00583-3 | Single reasoning agent + tools |
| M1-Parallel | arxiv.org/abs/2507.08944 | Parallel team early termination |
| vLLM structured decoding | blog.vllm.ai | Constrained decoding backend overview |
| BFCL Leaderboard | gorilla.cs.berkeley.edu | Function calling leaderboard |
