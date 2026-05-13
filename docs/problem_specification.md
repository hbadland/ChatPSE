# Problem Specification: Multi-Agent Flowsheet Synthesis

*This document formalises the problem solved by the system and the guarantees provided by its design. It is intended for inclusion in the NeurIPS paper Methods section (Section 3).*

---

## 1. Problem Statement

### 1.1 Input Space

Let $\mathcal{D}$ denote the space of natural language strings. The system receives a single input $d \in \mathcal{D}$: a *process description* written in free-form English. A process description specifies some combination of:

- the chemical species present (by any name: IUPAC, trivial, abbreviation, or trade name),
- the physical conditions (temperature, pressure, molar flow, composition),
- the intended unit operations and their sequence,
- and the separation or transformation objective.

No structured input is required. The system must infer all missing information.

### 1.2 Output Space

The system produces a *DWSIM flowsheet* $F$ defined as a 5-tuple:

$$F = (C,\ P,\ \mathcal{S},\ \mathcal{U},\ \mathcal{E})$$

where:

- $C = \{c_1, \ldots, c_n\}$ is the **compound set** — exact DWSIM compound names (case-sensitive strings matching DWSIM's internal database).
- $P \in \mathcal{P}$ is the **property package** — one of the supported thermodynamic models: Raoult's Law, NRTL, UNIQUAC, Peng–Robinson (PR), Soave–Redlich–Kwong (SRK), or Lee–Kesler–Plöcker (LKP).
- $\mathcal{S} = \{s_1, \ldots, s_m\}$ is the **stream set**. Each stream $s_i$ carries conditions $(T_i, P_i, \dot{n}_i, \mathbf{x}_i)$ where $T_i \in \mathbb{R}_{>0}$ [K], $P_i \in \mathbb{R}_{>0}$ [Pa], $\dot{n}_i \in \mathbb{R}_{\geq 0}$ [mol/s], and $\mathbf{x}_i \in \Delta^{|C|-1}$ is a mole-fraction vector on the unit simplex.
- $\mathcal{U} = \{u_1, \ldots, u_k\}$ is the **unit operation set**. Each $u_j$ has a type $\tau_j \in$ {Heater, Cooler, Vessel, Mixer, Splitter, Pump, Compressor, Expander} and type-specific parameters (e.g.\ $T_\text{out}$ [K] for Heater/Cooler, $P_\text{out}$ [Pa] for Pump/Compressor/Expander, split fractions for Splitter).
- $\mathcal{E} \subseteq (\mathcal{S} \cup \mathcal{U}) \times (\mathcal{S} \cup \mathcal{U}) \times \mathbb{N} \times \mathbb{N}$ is the **connection set** — a directed graph of (source, destination, source-port, destination-port) tuples. Connections must alternate between streams and unit operations; direct unit-to-unit connections are forbidden.

The flowsheet $F$ is *well-formed* if:
1. Every stream appears in at least one connection.
2. Every unit operation appears in at least one connection.
3. No unit operation is directly connected to another unit operation.
4. At least one stream has no incoming connection (a *feed stream*); feed streams must specify all of $(T, P, \dot{n}, \mathbf{x})$.
5. Mole fractions on each feed stream sum to 1: $\sum_{c \in C} x_{i,c} = 1 \pm 0.02$.

### 1.3 Physical Validity Criterion

A well-formed flowsheet $F$ is *physically valid* — independently of any agent's assessment — if the DWSIM simulator, when run on $F$, produces an ExecutionResult $R$ satisfying all of the following predicates:

**V1 (Convergence)**
$$\text{Solved}(R) = \top$$

**V2 (Mass balance)**
$$\frac{|\sum_{s \in \mathcal{S}_\text{feed}} \dot{n}_s - \sum_{s \in \mathcal{S}_\text{term}} \dot{n}_s|}{\sum_{s \in \mathcal{S}_\text{feed}} \dot{n}_s} \leq 0.01$$

where $\mathcal{S}_\text{feed}$ are feed streams and $\mathcal{S}_\text{term}$ are terminal outlet streams (no outgoing connections).

**V3 (Temperature bounds)**
$$\forall s \in \mathcal{S}:\ 100\ \text{K} \leq T_s \leq 2000\ \text{K}$$

**V4 (Pressure bounds)**
$$\forall s \in \mathcal{S}:\ 100\ \text{Pa} \leq P_s \leq 10^8\ \text{Pa}$$

**V5 (Separation, conditional)**  
If $|\mathcal{S}_\text{term}| > 1$ (i.e.\ the process produces multiple outlet streams), then for each terminal outlet $s \in \mathcal{S}_\text{term}$:
$$\max_{c \in C} |x_{s,c} - \bar{x}_{\text{feed},c}| > 0.01$$

where $\bar{x}_{\text{feed},c}$ is the flow-weighted average feed mole fraction of compound $c$:
$$\bar{x}_{\text{feed},c} = \frac{\sum_{s \in \mathcal{S}_\text{feed}} \dot{n}_s \cdot x_{s,c}}{\sum_{s \in \mathcal{S}_\text{feed}} \dot{n}_s}$$

**V6 (Composition well-defined)**
$$\forall s \in \mathcal{S} \setminus \mathcal{S}_\text{feed}:\ |\mathbf{x}_s| > 0 \quad \text{(composition dict non-empty)}$$

The problem solved by this system is:

> **Given** $d \in \mathcal{D}$, **find** $F$ such that $F$ is well-formed and physically valid.

---

## 2. Failure Taxonomy Formalisation

Each failure code defines a decidable predicate over the ExecutionResult $R$ and flowsheet $F$. This makes the routing function verifiable rather than heuristic.

Let $\mathcal{S}_\text{term}(F)$ denote terminal outlet streams of $F$, $\mathcal{S}_\text{feed}(F)$ feed streams, and $\bar{\mathbf{x}}_\text{feed}$ the flow-weighted average feed composition as defined above.

| Code | Formal Predicate |
|------|-----------------|
| **SOLVER\_FAIL** | $\neg\,\text{Solved}(R)$ |
| **NUMERIC\_FAIL** | $\exists s \in \mathcal{S}:\ T_s \in \{\pm\infty, \text{NaN}\}\ \vee\ P_s \in \{\pm\infty, \text{NaN}\}\ \vee\ \dot{n}_s \in \{\pm\infty, \text{NaN}\}\ \vee\ (\mathbf{x}_s = \varnothing \wedge s \notin \mathcal{S}_\text{feed})$ |
| **UNPHYSICAL\_T** | $\exists s \in \mathcal{S}:\ T_s < 100\ \text{K}\ \vee\ T_s > 2000\ \text{K}$ |
| **UNPHYSICAL\_P** | $\exists s \in \mathcal{S}:\ P_s < 100\ \text{Pa}\ \vee\ P_s > 10^8\ \text{Pa}$ |
| **ENERGY\_UNPHYSICAL** | $\exists u \in \mathcal{U}_\text{Heater}:\ T_\text{out}(u) < T_\text{in}(u) - 1\ \vee\ \exists u \in \mathcal{U}_\text{Cooler}:\ T_\text{out}(u) > T_\text{in}(u) + 1$ |
| **MASS\_BALANCE** | $\displaystyle\frac{|\Sigma_\text{feed} - \Sigma_\text{term}|}{\Sigma_\text{feed}} > 0.01$ where $\Sigma_\text{feed} = \sum_{s \in \mathcal{S}_\text{feed}} \dot{n}_s$, $\Sigma_\text{term} = \sum_{s \in \mathcal{S}_\text{term}} \dot{n}_s$ |
| **ZERO\_OUTLET** | $\exists s \in \mathcal{S}_\text{term}:\ \dot{n}_s = 0$ |
| **NO\_SEPARATION** | $\text{Solved}(R) \wedge |\mathcal{S}_\text{term}| > 1 \wedge \exists s \in \mathcal{S}_\text{term}:\ \dot{n}_s > 0 \wedge \mathbf{x}_s \neq \varnothing \wedge \max_{c \in C}|x_{s,c} - \bar{x}_{\text{feed},c}| < 0.01$ |
| **PARAM\_MISSING** | $P \in \{\text{NRTL, UNIQUAC}\} \wedge \text{Solved}(R) \wedge \text{NO\_SEPARATION}(R, F)$ |
| **COMP\_SUM** | $\exists s \in \mathcal{S}:\ \mathbf{x}_s \neq \varnothing \wedge |\sum_{c} x_{s,c} - 1| > 0.02$ |
| **WRONG\_PHASE\_DIR** | $\exists u \in \mathcal{U}_\text{Vessel}:\ x_{\text{vap}, c_\ell} > x_{\text{liq}, c_\ell} + 0.05\ \wedge\ x_{\text{vap}, c_h} < x_{\text{liq}, c_h} - 0.05$ where $c_\ell$ is the most volatile compound (lowest NBP) and $c_h$ the least volatile (highest NBP) at the vessel outlets |
| **INFEASIBLE** | SOLVER\_FAIL or any CRITICAL predicate holds after $\geq 3$ consecutive iterations with no improvement |

### Routing function

The routing function $\rho: 2^{\text{Codes}} \to \{\text{PASS, REFINER, THERMO, BASIS, HUMAN}\}$ is defined by a priority ordering (highest to lowest):

$$\rho(\mathcal{K}) = \begin{cases}
\text{HUMAN}   & \text{INFEASIBLE} \in \mathcal{K} \\
\text{THERMO}  & \text{NUMERIC\_FAIL} \in \mathcal{K}\ \vee\ \text{PARAM\_MISSING} \in \mathcal{K}\ \vee\ \text{NO\_SEPARATION} \in \mathcal{K}\ \vee\ \text{ZERO\_OUTLET} \in \mathcal{K} \\
\text{REFINER} & \text{SOLVER\_FAIL} \in \mathcal{K}\ \vee\ \text{MASS\_BALANCE} \in \mathcal{K}\ \vee\ \text{ENERGY\_UNPHYSICAL} \in \mathcal{K}\ \vee\ \text{UNPHYSICAL\_T} \in \mathcal{K}\ \vee\ \text{UNPHYSICAL\_P} \in \mathcal{K}\ \vee\ \text{COMP\_SUM} \in \mathcal{K} \\
\text{REFINER} & \text{WRONG\_PHASE\_DIR} \in \mathcal{K} \quad \text{(LLM may override to THERMO)} \\
\text{PASS}    & \mathcal{K} = \varnothing
\end{cases}$$

Note: The LLM Critic (Stage 2) may override the deterministic routing for WRONG\_PHASE\_DIR and for cases where multiple competing codes are present.

---

## 3. Agent Functions

Each agent is a typed function. Probabilistic agents (those that invoke an LLM) are denoted with a tilde.

### 3.1 Basis Agent

$$\text{Basis}: \mathcal{D} \times \mathcal{L}^* \to \mathcal{C}^* \times \mathcal{D}$$

**Inputs:** process description $d$; optional execution feedback list $\ell \in \mathcal{L}^*$ (DWSIM error strings from a previous BASIS routing).

**Outputs:** normalised compound list $C^* \subseteq \mathcal{C}_\text{DWSIM}$ (exact DWSIM names); normalised description $d^* \in \mathcal{D}$ (with compound mentions replaced by DWSIM names).

**Two-stage design:**
- *Stage 1 (deterministic):* Regex scan against a compiled alias database. Cost: $O(|\mathcal{A}| \cdot |d|)$ where $\mathcal{A}$ is the alias set. No LLM call if all anchors are verified non-mixture entries.
- *Stage 2 (LLM):* Verifier-completer that confirms Stage 1 anchors, finds missed compounds, and extracts composition hints. Triggered when Stage 1 returns ambiguous, mixture, or empty anchors, or when feedback is non-empty.

### 3.2 Planner Agent

$$\widetilde{\text{Plan}}: \mathcal{D} \times \mathcal{C}^* \to \mathcal{F}$$

**Inputs:** normalised description $d^*$; compound list $C^*$; optional suggested compositions.

**Output:** a well-formed flowsheet $F \in \mathcal{F}$ (JSON schema-validated).

The Planner always invokes the LLM. Schema validation is deterministic post-processing.

### 3.3 Thermo Agent

$$\widetilde{\text{Thermo}}: \mathcal{F} \to \mathcal{F}$$

**Input/Output:** flowsheet $F$; returns $F'$ with updated property package $P'$ and optional per-unit overrides.

The Thermo agent always invokes the LLM to reason about compound polarity, pressure regime, and presence of azeotropes.

### 3.4 Executor

$$\text{Exec}: \mathcal{F} \to \mathcal{R}$$

**Input:** validated flowsheet $F$.

**Output:** ExecutionResult $R$ containing per-stream $(T, P, \dot{n}, \mathbf{x})$, solver status, and error strings.

The Executor is deterministic: it translates $F$ to DWSIM API calls via pythonnet/CLR and runs the headless solver.

### 3.5 Critic Agent

$$\widetilde{\text{Critic}}: \mathcal{R} \times \mathcal{F} \times \mathbb{N} \to (\rho, \Sigma)$$

**Inputs:** execution result $R$; flowsheet $F$; iteration count $t$.

**Outputs:** routing decision $\rho \in \{\text{PASS, REFINER, THERMO, BASIS, HUMAN}\}$; signal set $\Sigma$ (typed failure signals with code, severity, location, evidence).

**Two-stage design:**
- *Stage 1 (deterministic):* Evaluates all formal predicates from Section 2. If $\Sigma = \varnothing$, returns PASS immediately (no LLM call).
- *Stage 2 (LLM):* Interprets $\Sigma$ in context of $R$ and $F$ to produce a structured diagnosis and routing decision. Only called when $\Sigma \neq \varnothing$.

### 3.6 Refiner Agent

$$\widetilde{\text{Refine}}: \mathcal{F} \times \Sigma \to \mathcal{F}$$

**Inputs:** current flowsheet $F$; failure signals $\Sigma$.

**Output:** updated flowsheet $F'$ with fixes applied.

**Two-stage design:**
- *Stage 1 (deterministic):* Rule-based fixes keyed on failure codes (e.g.\ SOLVER\_FAIL with NRTL → try Raoult's Law if physics check passes; UNPHYSICAL\_T → convert °C to K). Returns early if all failure codes are handled and the result passes schema validation and $\text{physics\_validate}(F') = \varnothing$.
- *Stage 2 (LLM):* Called when Stage 1 cannot resolve all codes, or when the resulting flowsheet has physics errors. Produces a JSON diff applied to $F$.

### 3.7 Pipeline Composition

The full pipeline for iteration $t = 0$ is:

$$F_0 = \text{Thermo}(\text{Plan}(\text{Basis}(d)))$$

Subsequent iterations follow the feedback loop:

$$F_{t+1} = \begin{cases}
\text{Refine}(F_t, \Sigma_t)      & \rho_t = \text{REFINER} \\
\text{Thermo}(F_t)               & \rho_t = \text{THERMO} \\
\text{Thermo}(\text{Plan}(\text{Basis}(d, \ell_t))) & \rho_t = \text{BASIS} \\
F_t                              & \rho_t \in \{\text{PASS, HUMAN}\}
\end{cases}$$

where $(\rho_t, \Sigma_t) = \text{Critic}(\text{Exec}(F_t),\ F_t,\ t)$ and $\ell_t$ are the error strings from $\text{Exec}(F_t)$.

---

## 4. Termination Guarantee

**Theorem.** *The pipeline always terminates in finite time with outcome in* $\{\text{PASS, HUMAN, MAX\_ITER, BASIS\_FAILED, PLAN\_FAILED}\}$.

**Proof.**

Let $H_t$ denote the set of MD5 hashes of flowsheets executed up to and including iteration $t$. Define:

- $N_\text{iter} \in \mathbb{N}$ (`max_iterations`, default 4): maximum number of Executor→Critic→Refiner cycles.
- $N_\text{basis} \in \mathbb{N}$ (`max_basis_reruns`, default 1): maximum number of BASIS routing events.

We show termination by case analysis on the loop body.

**Case 1: $\rho_t = \text{PASS}$ or $\rho_t = \text{HUMAN}$.**  
The loop breaks immediately. $\square$

**Case 2: $\rho_t \in \{\text{REFINER, THERMO}\}$.**  
The loop continues to iteration $t+1$. Since $t$ is bounded by $N_\text{iter}$, the loop exits with MAX\_ITER after at most $N_\text{iter}$ steps.

**Case 3: $\rho_t = \text{BASIS}$.**  
A BASIS re-run increments a counter `basis_reruns`. This counter is bounded by $N_\text{basis}$; on reaching the bound the loop breaks with HUMAN. $\square$

**Case 4: Cycling.**  
Before each execution, the hash $h_t = \text{MD5}(F_t)$ is computed. If $h_t \in H_{t-1}$, the loop breaks immediately with HUMAN — no execution occurs. Since $|H_t| \leq t+1$ and $t \leq N_\text{iter}$, this detection fires in at most $N_\text{iter}$ steps.

If $h_t \notin H_{t-1}$, it is added: $H_t = H_{t-1} \cup \{h_t\}$. The Refiner may produce $F_{t+1} = F_t$ (no progress), which is detected on the next iteration. Thus $H_t$ grows monotonically and is bounded by $N_\text{iter}$.

**Combined bound.** The total number of loop iterations before termination is at most:

$$T_\text{max} = N_\text{iter} + 1$$

since the loop counter runs from 0 to $N_\text{iter} - 1$ and cycling detection may fire at any step. Each iteration is bounded in time by the finite cost of one LLM call (bounded by the API timeout) plus one DWSIM solve (bounded by the solver iteration limit). Therefore the pipeline terminates in finite time. $\blacksquare$

**Corollary.** The infeasibility threshold $\theta_\text{infeasible} = N_\text{iter} - 1$ is set by the Orchestrator at construction time, ensuring INFEASIBLE routing always fires on the final available iteration rather than requiring $N_\text{iter} \geq 4$ unconditionally.

---

## 5. Scope

### 5.1 In Scope

The system is designed for and evaluated on processes satisfying the following constraints:

| Property | Supported range |
|----------|----------------|
| **Topology** | Open-loop (feed-forward, acyclic); no recycle streams |
| **Phase behaviour** | Vapour–liquid equilibrium (VLE); vapour–liquid–liquid (VLLE) for NRTL/UNIQUAC |
| **Unit operations** | Heater, Cooler, Flash Vessel, Mixer, Splitter, Pump, Compressor, Expander |
| **Compounds** | Any compound in DWSIM's built-in database (~1,800 entries); pure components and non-reactive mixtures |
| **Property packages** | Raoult's Law (ideal VLE), NRTL, UNIQUAC (non-ideal VLE), Peng–Robinson, SRK, Lee–Kesler–Plöcker (EOS for gases and high-pressure liquids) |
| **Pressure range** | 100 Pa – 10⁸ Pa (near-vacuum to 1,000 bar) |
| **Temperature range** | 100 K – 2,000 K |
| **Process description language** | English; any compound naming convention (IUPAC, trivial, abbreviation, trade name) |

### 5.2 Out of Scope

The following classes of processes are explicitly unsupported, with the primary technical reason:

| Class | Reason |
|-------|--------|
| **Reactive systems** (reactors, PFR, CSTR) | DWSIM reactor objects require stoichiometry specification; the Planner's output schema does not include reaction data |
| **Electrolyte systems** (NaCl, NaOH, HCl, H₂SO₄, etc.) | DWSIM's electrolyte property packages require activity models with ion-specific parameters not in the general compound database; compound names like "brine", "caustic", or "lye" are caught by the Basis agent's unsupported-compound check |
| **Recycle loops** | Tear-stream iteration (Wegstein, direct substitution) is not implemented in the Executor's flowsheet construction; acyclicity of $\mathcal{E}$ is not currently enforced but recycles cause DWSIM solver divergence |
| **Rigorous distillation columns** | Tray-by-tray column specification (reflux ratio, number of stages, feed tray) is not yet wired in the wrapper; the DistillationColumn object type exists in DWSIM but is not exposed in the unit operation schema |
| **Solid–liquid systems** (crystallisation, filtration) | No solid phase model in the supported property packages |
| **Polymer systems** | DWSIM's PC-SAFT and similar polymer packages are not in the property package mapping |
| **Multi-phase (three-phase) flash with liquid–liquid split** | Partially supported via NRTL/UNIQUAC but not validated; the Critic's NO\_SEPARATION check does not distinguish VLLE from VLE failure |

### 5.3 Limitations and Assumptions

1. **DWSIM property database completeness.** The system assumes that all confirmed compound names are present in the running DWSIM instance's compound database. Compounds added to `compound_database.md` but absent from the DWSIM installation will cause `AddCompound()` to fail silently, detected by the Executor's post-add check.

2. **Binary interaction parameter availability.** NRTL and UNIQUAC models require binary interaction parameters ($\tau_{ij}$, $\alpha_{ij}$) for each compound pair. DWSIM's built-in database covers common pairs (water/alcohols, etc.) but not all. Missing parameters cause DWSIM to default to ideal behaviour — detected by the NO\_SEPARATION / PARAM\_MISSING predicates.

3. **Property ID stability.** The DWSIM automation API property identifiers (e.g.\ `PROP_MS_0` for temperature, `PROP_SEP_0` for vessel pressure drop) are discovered empirically and may change between DWSIM versions. The system is validated against DWSIM 9.0.4.

4. **LLM non-determinism.** The Planner, Thermo, Critic (Stage 2), and Refiner (Stage 2) agents invoke LLMs that are stochastic. Two runs on the same description may produce different flowsheets. The termination guarantee (Section 4) holds regardless of LLM output because the outer loop is bounded by finite counters independent of LLM behaviour.
