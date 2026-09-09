# Architecture

**Project:** Trajectory Dependence and Self-Explanation Faithfulness in Agentic AI Systems
**Document Status:** Implemented architecture for the completed experiment
**Protocol Dependency:** `docs/experiment-design.md` v1.1
**Purpose:** Define the software architecture, state boundaries, data flow, provenance model, replay semantics, and condition-isolation rules for the experimental harness.

---

## 1. Architecture Objectives

The experimental software is a scientific instrument rather than a general-purpose agent framework. Its architecture must prioritize:

1. causal identifiability;
2. explicit state provenance;
3. reproducible execution;
4. condition isolation;
5. deterministic state control where feasible;
6. exact replay of experimental state;
7. auditable event logging;
8. minimal abstraction and dependency surface;
9. separation of raw evidence from derived analysis;
10. use of the same underlying language model across C0-C3.

The architecture must make it possible to answer, for every preregistered focal consequential decision:

* what information was available to the model;
* where each item of information originated;
* which condition authorized that information;
* which scaffold-state sources were eligible for intervention;
* exactly what changed during a targeted intervention;
* exactly what changed during the matched sham intervention;
* whether the normalized focal action changed;
* what causal sources the model reported in its self-explanation;
* how every reported metric was derived from the underlying run evidence.

No component may silently summarize, rewrite, compress, reorder, retry, or otherwise modify experimental state.

---

## 2. Architectural Principle: One Harness, Configured Conditions

C0-C3 are not separate agents and must not be implemented as separate experimental systems.

The experiment uses:

* one model adapter;
* one agent runtime;
* one task engine;
* one context builder;
* one tool subsystem;
* one memory subsystem;
* one environment subsystem;
* one event logger;
* one intervention engine;
* one action-normalization system;
* one analysis pipeline.

The experimental condition determines which state sources the context builder is authorized to expose to the model.

Conceptually:

```text
C0: X = P
C1: X = P + T
C2: X = P + T + H
C3: X = P + T + H + M
```

Environmental state `E` remains an exogenous source whose availability depends on task design and the means by which the condition can observe or retain it.

The complete candidate state representation is:

```text
S = {W, P, T, H, M, E, C}
```

where:

* `W` = fixed model weights;
* `P` = current task prompt/context;
* `T` = tool interactions and returns;
* `H` = within-task trajectory/history state;
* `M` = persistent cross-task memory;
* `E` = simulated environmental state;
* `C` = runtime/configuration metadata.

The architecture must preserve these as distinct **provenance classes**, even when their content is serialized together into a single model context.

---

## 3. Source Provenance Is Distinct from Transport

A central architectural requirement is that the mechanism used to retrieve information must not determine its experimental source classification.

Examples:

* a memory item retrieved through a Python function remains `M`, not `T`;
* an environment query exposed through a callable interface remains `E`, not automatically `T`;
* a previous tool return retained in trajectory history remains historically derived from `T` but must be identifiable as an `H` representation when the causal intervention concerns retained trajectory history;
* the current task instruction remains `P` even though it is serialized into the same model context as other sources.

Each decision-relevant context element must therefore carry explicit provenance metadata before model serialization.

A conceptual representation is:

```python
ContextElement(
    source="M",
    source_id="memory:item_004",
    content="preferred threshold = 7",
    created_at_sequence=3,
    retrieved_at_sequence=11,
    task_id="task_014",
)
```

The context serializer may convert this into model-readable text, but it may not discard its provenance record.

---

## 4. High-Level Component Architecture

```text
                         +----------------------+
                         | Experiment Manifest  |
                         | config + frozen IDs  |
                         +----------+-----------+
                                    |
                                    v
+-------------+            +--------+---------+
| Task Specs  +----------->+ Experiment       |
| YAML/JSON   |            | Orchestrator     |
+-------------+            +--------+---------+
                                    |
                  +-----------------+-----------------+
                  |                                   |
                  v                                   v
        +---------+----------+              +---------+----------+
        | Condition Config   |              | State Initializer  |
        | C0 / C1 / C2 / C3  |              | M / E / Tools      |
        +---------+----------+              +---------+----------+
                  |                                   |
                  +-----------------+-----------------+
                                    |
                                    v
                           +--------+---------+
                           | Agent Runtime    |
                           +--------+---------+
                                    |
                    +---------------+---------------+
                    |                               |
                    v                               v
          +---------+----------+          +---------+----------+
          | Context Builder    |          | Tool / State      |
          | provenance-aware   |          | Interfaces        |
          +---------+----------+          +---------+----------+
                    |                               |
                    +---------------+---------------+
                                    |
                                    v
                           +--------+---------+
                           | Model Adapter    |
                           | Qwen / MLX       |
                           +--------+---------+
                                    |
                                    v
                           +--------+---------+
                           | Action Parser    |
                           | + Normalizer     |
                           +--------+---------+
                                    |
                                    v
                           +--------+---------+
                           | Focal Decision   |
                           +--------+---------+
                                    |
                  +-----------------+-----------------+
                  |                                   |
                  v                                   v
       +----------+-----------+             +---------+----------+
       | Explanation Elicitor |             | Snapshot Manager   |
       +----------+-----------+             +---------+----------+
                  |                                   |
                  |                                   v
                  |                         +---------+----------+
                  |                         | Intervention       |
                  |                         | Engine             |
                  |                         +---------+----------+
                  |                                   |
                  |                       +-----------+-----------+
                  |                       |                       |
                  |                       v                       v
                  |               targeted replay          sham replay
                  |                       |                       |
                  +-----------------------+-----------+-----------+
                                              |
                                              v
                                    +---------+----------+
                                    | Event / Provenance |
                                    | Logger             |
                                    +---------+----------+
                                              |
                                              v
                                    +---------+----------+
                                    | Raw Run Store     |
                                    | append-only       |
                                    +---------+----------+
                                              |
                                              v
                                    +---------+----------+
                                    | Analysis Pipeline |
                                    +---------+----------+
                                              |
                         +--------------------+--------------------+
                         |                    |                    |
                         v                    v                    v
                       CSV/JSON             Tables              Figures
```

Within the Agent Runtime-to-Model Adapter path, model invocation is further decomposed into deterministic context serialization, backend-specific input preparation, event-backed provenance audit, and model generation. The exact structured context, model-visible text, rendered chat-template input, and prompt token IDs are preserved before inference. Provenance validation must succeed before generation is permitted.

---

## 5. Repository Structure

Implemented public repository structure (empty planning placeholders and private
artifacts are intentionally omitted):

```text
trajectory-dependence/
├── README.md
├── pyproject.toml
├── .gitignore
├── config/
│   ├── final-run.json
│   ├── final-suite-preflight.json
│   └── tasks/
├── docs/
│   ├── experiment-design.md
│   ├── reproducibility.md
│   └── validity-and-failure-semantics.md
├── paper/
├── results/summary/
├── scripts/
├── src/
│   ├── agent/
│   ├── analysis/
│   ├── environment/
│   ├── interventions/
│   ├── logging/
│   ├── memory/
│   ├── model/
│   ├── normalization/
│   ├── snapshots/
│   ├── tasks/
│   └── tools/
└── tests/
```

Large model files are not stored in the repository.

Raw experimental data is immutable after creation.

Pilot and final-study data must never share output directories.

---

## 6. Experiment Manifest

The experiment manifest is the frozen configuration authority for final execution.

Before final data collection it must specify, at minimum:

```yaml
experiment:
  experiment_id:
  protocol_version:
  git_commit:
  manifest_hash:

model:
  identifier:
  revision:
  quantization:
  checksum:
  inference_engine:
  inference_engine_version:

inference:
  temperature:
  top_p:
  max_tokens:
  seed_policy:
  additional_parameters:

memory:
  implementation:
  retrieval_policy:
  write_policy:

execution:
  repetitions:
  seed_list:
  task_order_policy:
  max_steps:

analysis:
  bootstrap_replicates:
  h2_estimator:
  exclusion_rules_version:

tasks:
  specification_directory:
  specification_hash:
```

Pilot-resolvable fields may remain unspecified before the pilot.

No confirmatory field may remain unresolved when final execution begins.

---

## 7. Model Adapter

### 7.1 Responsibility

The model adapter is the only component permitted to communicate directly with the underlying model runtime.

Its responsibilities are limited to:

* loading the selected model;
* accepting fully assembled model input;
* applying configured inference parameters;
* recording inference metadata;
* generating model output;
* returning raw model output to the agent runtime.

It must not:

* decide what state belongs in context;
* retrieve memory;
* invoke tools autonomously outside the agent runtime;
* summarize history;
* repair model output semantically;
* automatically retry failed generations;
* alter task content;
* perform intervention logic.

### 7.2 Interface

Conceptually:

```python
response = model.generate(
    messages=serialized_context,
    inference_config=inference_config,
    seed=seed,
)
```

The returned object should include:

```python
ModelResponse(
    raw_text=...,
    finish_reason=...,
    token_counts=...,
    seed=...,
    inference_parameters=...,
    model_identifier=...,
    model_revision=...,
)
```

### 7.3 Backend Modularity

The initial adapter targets the selected Qwen-family open-weight model on Apple Silicon.

Backend-specific implementation must remain behind a small interface so that a future replication can substitute another open-weight model without altering:

* task specifications;
* condition semantics;
* intervention definitions;
* logging;
* normalization;
* metrics.

Model substitution constitutes a replication configuration, not a new harness architecture.

---

## 8. Condition Configuration

Conditions are represented declaratively.

Example:

```python
ConditionConfig(
    condition="C2",
    allow_tools=True,
    allow_history=True,
    allow_persistent_memory=False,
)
```

The condition must control **availability**, not merely whether a component happens to be used.

### 8.1 C0

Authorized sources:

```text
P
```

Prohibited scaffold state:

```text
T, H, M
```

C0 uses a single focal invocation.

No scaffold-state TD value is calculated for C0.

### 8.2 C1

Authorized sources:

```text
P, T
```

Prohibited:

```text
H, M
```

Tool interaction relevant to the focal decision may be represented in the invocation, but arbitrary prior generated trajectory must not be retained.

### 8.3 C2

Authorized sources:

```text
P, T, H
```

Prohibited:

```text
M
```

Within-task history may contain:

* previous agent actions;
* observations;
* prior tool interactions;
* prior environment observations;
* prior task-local state.

It must not contain information imported from independent previous tasks.

### 8.4 C3

Authorized sources:

```text
P, T, H, M
```

C3 adds explicitly persistent cross-task state.

Memory initialization must occur through logged setup tasks, preregistered seeding, or an explicit empty-memory configuration.

---

## 9. Context Builder

The context builder is the primary enforcement point for condition isolation.

It receives:

* task specification;
* condition;
* current task prompt;
* available tool state;
* trajectory history;
* eligible memory records;
* relevant environment observations.

It produces two outputs:

1. a provenance-preserving structured context;
2. the serialized model input.

Conceptually:

```python
context = context_builder.build(
    task=task,
    condition=condition,
    trajectory=trajectory,
    tool_state=tool_state,
    memory_state=memory_state,
    environment_state=environment_state,
)
```

The structured representation must exist before serialization:

```json
[
  {
    "source": "P",
    "source_id": "prompt:task_012",
    "content": "..."
  },
  {
    "source": "T",
    "source_id": "tool_return:0004",
    "content": "..."
  },
  {
    "source": "H",
    "source_id": "history:event_0007",
    "content": "..."
  }
]
```

The logger preserves both this representation and the exact final serialized model input.

---

## 10. Condition-Isolation Invariants

The following invariants must be enforced in tests.

### C0

```text
context.sources ⊆ {P}
```

### C1

```text
context.sources ⊆ {P, T, E where task-valid}
M ∉ context
H ∉ context
```

### C2

```text
context.sources ⊆ {P, T, H, E where task-valid}
M ∉ context
```

### C3

```text
context.sources ⊆ {P, T, H, M, E where task-valid}
```

These constraints apply to information provenance, not merely field names.

For example, copying a persistent-memory fact into an unlabelled prompt field would violate C1 or C2 even if no field named `memory` were present.

Condition-isolation tests therefore inspect both:

* source metadata;
* serialized content against known prohibited source fixtures.

A condition-isolation violation is an infrastructure failure and invalidates the affected run.

---

## 11. Agent Runtime

The agent runtime coordinates a single task execution.

Its responsibilities are:

1. initialize task-local state;
2. request context assembly;
3. invoke the model;
4. parse requested actions;
5. execute authorized tools;
6. update history;
7. update memory where C3 and task policy permit;
8. update or query environment state;
9. detect the preregistered focal decision;
10. commit the focal action;
11. invoke explanation elicitation;
12. trigger snapshot/replay workflow.

The runtime must obey a fixed maximum-step policy.

There are no silent retries.

Unexpected model behavior is logged rather than corrected unless a preregistered deterministic parser rule applies.

---

## 12. Tool Architecture

Tools implement a minimal common interface:

```python
class Tool:
    name: str

    def execute(self, request, state) -> ToolResult:
        ...
```

Required baseline tools are:

* key-value lookup;
* calculator;
* synthetic file lookup;
* environment query;
* state modification.

Every tool request and result creates separate logged events.

Tool results contain:

```python
ToolResult(
    tool_name=...,
    request=...,
    raw_result=...,
    normalized_result=...,
    state_changed=...,
)
```

No baseline tool may access the external network.

Tool failures must occur only when:

* deliberately specified by the task;
* deliberately injected by an intervention;
* caused by a genuine technical failure, which is separately classified.

---

## 13. Persistent Memory Architecture

Persistent memory must be transparent and exactly manipulable.

Required operations:

```text
write
read
list
snapshot
restore
targeted_replace
targeted_remove
```

Each stored memory item receives a unique stable identifier.

Example:

```json
{
  "memory_id": "mem_00031",
  "namespace": "run_004/task_sequence_A",
  "key": "threshold",
  "value": 7,
  "created_by_task": "setup_003",
  "created_at_sequence": 8
}
```

Every read and write must be logged.

Memory retrieval must be deterministic under the frozen final configuration.

Memory must use run-scoped namespaces so that one independent experimental repetition cannot contaminate another.

No hidden summarization or automatic consolidation is allowed.

---

## 14. Environment Architecture

The simulated environment is a deterministic structured state machine.

Example:

```json
{
  "entities": {
    "door_1": {
      "status": "locked",
      "key_required": "key_A"
    }
  },
  "agent_inventory": [],
  "global_flags": {}
}
```

Required operations:

```text
query
transition
snapshot
restore
targeted_modify
```

Transitions are explicit functions:

```python
new_state = transition(
    current_state,
    action,
)
```

The same current state and same action must produce the same transition.

Environment observations retain source class `E`, even when obtained through an environment-query callable.

---

## 15. Trajectory History

`H` contains state arising from prior events inside the current task trajectory.

Potential history elements include:

* previous actions;
* previous observations;
* relevant prior model outputs;
* tool interactions retained across task steps;
* environment observations retained from earlier steps;
* task-local branch outcomes.

Every history element must identify the event from which it derives.

Example:

```json
{
  "history_id": "hist_00009",
  "origin_event_id": "event_00018",
  "origin_source": "T",
  "content": "...",
  "sequence_index": 18
}
```

This distinction allows the system to preserve both:

* original provenance;
* current causal role as retained trajectory state.

Interventions on `H` alter the retained history representation, not the historical fact that the original event occurred.

---

## 16. Focal Consequential Decision

Each task must preregister one focal consequential decision for primary analysis.

The task specification determines:

```yaml
focal_decision:
  id:
  type:
  schema:
  success_criterion:
```

The runtime explicitly marks the corresponding event:

```json
{
  "event_type": "action",
  "focal_decision": true,
  "focal_decision_id": "final_action"
}
```

A pre-focal snapshot must be taken at the last valid point before the model invocation responsible for this decision.

The snapshot must contain enough state to reconstruct the matched baseline, targeted intervention, and sham replay.

---

## 17. Action Parsing and Normalization

The architecture distinguishes:

```text
raw model output
        ↓
syntactic parse
        ↓
normalized consequential action
        ↓
task consequence / success
```

Example:

```text
Raw:
"I choose route B."

Normalized:
{"action": "select_route", "value": "B"}
```

Primary causal analysis compares normalized consequential actions.

It does not compare raw text.

Each task must define its normalization rule before final execution.

No LLM-based semantic judge is used for primary action equivalence.

A parser failure remains a parser/model-output event and is handled according to the frozen task specification rather than manually reinterpreted after the fact.

---

## 18. Self-Explanation Elicitation

The explanation subsystem executes only after the baseline focal action is committed.

Ordering is:

```text
focal action
    ↓
action committed
    ↓
structured self-explanation
    ↓
snapshot/intervention processing
    ↓
targeted/sham results
```

The model must not receive:

* task success feedback;
* intervention results;
* counterfactual outcomes;

before providing the self-explanation.

The explanation subsystem records:

* raw explanation text;
* parsed JSON if valid;
* reported causal-source set;
* justification fields;
* parse validity.

It does not determine whether the model's explanation is correct.

Scoring occurs later in the analysis pipeline against intervention-derived causal evidence.

---

## 19. Snapshot Architecture

Snapshots are the foundation of matched causal replay.

A pre-focal snapshot must capture all state needed to reproduce the decision context:

```text
P
T state relevant to the trajectory
H
M
E
runtime configuration C
random/inference seed metadata
task-local control state
sequence position
```

`W` is referenced through the frozen model identifier/revision rather than copied into every snapshot.

Each snapshot receives:

```text
snapshot_id
run_id
task_id
condition
repetition_id
sequence_index
checksum
```

Restoring the same snapshot without an intervention must reconstruct the same structured focal-decision context.

A snapshot restore that changes unrelated state is an infrastructure failure.

---

## 20. Intervention Engine

The intervention engine operates on completed baseline trajectories.

For each source `S ∈ U_d`:

```text
                PRE-FOCAL SNAPSHOT
                       |
          +------------+------------+
          |                         |
          v                         v
  targeted replay              sham replay
          |                         |
   change focal S              change irrelevant
   representation              same-source item
          |                         |
          v                         v
      context                    context
          |                         |
          v                         v
       model                      model
          |                         |
          v                         v
    a_target                    a_sham
```

The baseline action is:

```text
a_base
```

Classification is then:

### Intervention-positive

```text
a_target != a_base
AND
a_sham == a_base
```

under behavioral action equivalence.

### Intervention-negative

```text
a_target == a_base
AND
a_sham == a_base
```

### Sham-unstable / ambiguous

```text
a_sham != a_base
```

regardless of the targeted result.

Technical replay failures are classified separately.

---

## 21. Targeted Intervention Interface

Each task specification defines interventions declaratively.

Example:

```yaml
interventions:
  M:
    target: "memory:item_threshold"
    targeted:
      operation: "neutral_replace"
      replacement: "..."
    sham:
      target: "memory:item_irrelevant"
      operation: "neutral_replace"
      replacement: "..."
```

The intervention engine must not decide post hoc which item appears most causally promising.

All targets are derived from the task specification.

Where feasible, targeted and sham replacements preserve:

* serialization structure;
* approximate length;
* formatting;
* source type.

This minimizes context-length and formatting confounds.

---

## 22. Intervention Validation

Before each model replay, the engine produces a machine-checkable state/context difference.

Conceptually:

```text
baseline_snapshot
vs.
targeted_snapshot
```

and:

```text
baseline_snapshot
vs.
sham_snapshot
```

The validator must confirm that:

1. the intended source changed;
2. prohibited sources did not change;
3. runtime configuration did not change;
4. task identity did not change;
5. condition did not change;
6. inference parameters did not change;
7. matched seed metadata did not change;
8. no unplanned environmental or memory transition occurred.

The difference record is stored with the intervention event.

---

## 23. Event Logging

The logger is append-only.

The runtime records successful event-backed provenance validation as an explicit
`provenance_audit` event. A model generation may occur only after the
corresponding model-visible context has passed provenance validation.

The implemented event stream therefore distinguishes:

- `context_build`: the structured context, deterministic serialization, and exact prepared model input;
- `provenance_audit`: the event-backed validation of every model-visible context element;
- `generation`: the resulting raw model generation and inference metadata.

If provenance validation fails, the `context_build` event remains preserved and
is followed by a `technical_failure` event. No `generation` event is produced.

Every event contains at least:

```json
{
  "experiment_id": "...",
  "run_id": "...",
  "task_id": "...",
  "task_classification": "integration",
  "condition": "C3",
  "repetition_id": 2,
  "seed": 12345,
  "sequence_index": 14,
  "timestamp": "...",
  "event_type": "...",
  "source_component": "...",
  "raw_payload": {},
  "normalized_payload": {},
  "snapshot_id": null
}
```

Additional identifiers should be included where relevant:

```text
event_id
parent_event_id
baseline_run_id
intervention_id
intervention_source
intervention_type
focal_decision_id
model_invocation_id
tool_call_id
memory_id
```

Timestamps document execution order but causal sequencing is determined by `sequence_index`, not wall-clock resolution.

---

## 24. Raw Data Layout

A run should be independently inspectable.

Recommended structure:

```text
data/final/raw/
└── <experiment_id>/
    └── <task_id>/
        └── <condition>/
            └── <repetition_id>/
                ├── baseline/
                │   ├── events.jsonl
                │   ├── contexts.jsonl
                │   ├── model_inputs.jsonl
                │   ├── model_outputs.jsonl
                │   └── metadata.json
                │
                ├── explanation/
                │   └── explanation.json
                │
                ├── interventions/
                │   ├── P/
                │   ├── T/
                │   ├── H/
                │   ├── M/
                │   └── E/
                │
                └── snapshots/
```

An equivalent normalized structure may be used if it preserves the same traceability.

Raw artifacts must never be overwritten by processed outputs.

---

## 25. Experiment Orchestrator

The orchestrator coordinates task-condition-repetition tuples.

Its input is entirely declarative:

```text
manifest
task specifications
condition set
seed schedule
execution order
```

It must not make adaptive experimental decisions based on observed model behavior.

Conceptually:

```python
for task in frozen_task_order:
    for condition in conditions:
        for repetition in repetitions:
            run_baseline(...)
            run_explanation(...)
            run_preregistered_replays(...)
```

Actual ordering may be randomized or balanced according to the frozen manifest.

The orchestrator records the order actually executed.

---

## 26. Randomness and Seed Handling

Every controllable random source must be explicitly seeded and logged.

At minimum:

* Python random;
* NumPy, if used;
* model inference seed where supported;
* task-order randomization;
* intervention-order randomization.

For a matched baseline/intervention/sham set:

```text
seed_baseline = seed_targeted = seed_sham
```

where supported by the inference runtime.

This controls sampling variation but does not imply mathematically deterministic LLM execution.

Sham replay remains the empirical control for replay instability.

---

## 27. Analysis Boundary

The execution system does not calculate confirmatory statistics during experimental runs.

It records evidence.

The analysis pipeline operates only on preserved raw logs.

Processing stages are:

```text
raw events
    ↓
schema validation
    ↓
run-level reconstruction
    ↓
focal-decision extraction
    ↓
intervention classification
    ↓
decision-level processed dataset
    ↓
metric calculation
    ↓
bootstrap/statistical analysis
    ↓
tables and figures
```

This separation reduces opportunities for execution logic to react to emerging results.

---

## 28. Processed Decision-Level Schema

A processed row should represent one focal task-condition-repetition observation.

Illustrative fields:

```text
experiment_id
task_id
task_family
task_class
condition
repetition_id
seed

baseline_action
baseline_success

eligible_P
eligible_T
eligible_H
eligible_M
eligible_E

P_status
T_status
H_status
M_status
E_status

positive_sources_all
positive_sources_scaffold

td_any
td_norm
csd_all
csd_scaffold

reported_sources
reported_sources_evaluable
unverifiable_reported_sources

exact_set
precision
recall
f1

explanation_valid

sham_instability_count
technical_failure
exclusion_status

baseline_run_id
snapshot_id
```

Every derived row must retain identifiers sufficient to locate the underlying raw evidence.

---

## 29. Metrics Layer

The metrics module implements exactly the definitions frozen in the protocol.

It includes:

```text
TD_source
TD_any
TD_norm
CSD_scaffold
CSD_all
exact-set accuracy
precision
recall
F1
sham instability
baseline task success
```

Metric functions must be unit-tested against hand-constructed cases covering:

* intervention-positive;
* intervention-negative;
* sham-unstable;
* untestable source;
* empty reported set;
* empty intervention-positive set;
* invalid explanation;
* partially testable source universe.

Metric code may not infer missing causal labels.

---

## 30. Bootstrap and Statistical Analysis

The final uncertainty procedure resamples at the task level.

A bootstrap sample must preserve together:

* all conditions for a selected task;
* all repetitions for that task;
* all corresponding intervention classifications.

Conceptually:

```python
sampled_tasks = resample(task_ids, replace=True)

for task_id in sampled_tasks:
    include_all_rows_for(task_id)
```

The analysis then recomputes the full statistic inside each bootstrap replicate.

Repetitions must not be independently resampled as though they were unrelated observations.

The exact H2 dispersion estimator remains pilot-resolvable until frozen in the final manifest.

---

## 31. Diagnostic Versus Integration Tasks

Task classification is architectural metadata and must be preserved from specification through analysis.

```text
diagnostic
integration
```

Diagnostic tasks validate:

* source-channel functionality;
* state retention;
* intervention machinery;
* action normalization;
* explanation parsing.

Integration tasks support primary H1 inference.

The primary analysis pipeline must therefore explicitly filter:

```text
task_class == "integration"
AND condition ∈ {C1, C2, C3}
```

for primary H1 calculations.

C0 is never automatically encoded as `TD = 0`.

---

## 32. Failure and Ambiguity Handling

The architecture distinguishes three fundamentally different situations.

### Technical Failure

Examples:

* model runtime crash;
* corrupted snapshot;
* tool infrastructure failure;
* unrecoverable logging failure.

Handled according to predefined exclusion rules.

### Behavioral Ambiguity

Example:

* sham intervention changes the focal action.

This is retained as a scientific observation and classified as sham-unstable/ambiguous.

### Unexpected Model Behavior

Examples:

* incorrect task solution;
* refusal to use memory;
* ignored tool output;
* surprising action.

These are ordinary experimental outcomes and are not excluded merely because they contradict design intent or hypotheses.

---

## 33. Pilot and Final Execution Separation

Pilot and final execution use distinct configuration, task, data, and result namespaces.

```text
config/tasks/pilot-tasks.json
config/tasks/final-tasks.json

artifacts/pilot/
artifacts/final/

config/pilot-run.json
config/final-run.json
```

Pilot execution may inform allowed methodological refinements.

Final execution begins only after:

* task freeze;
* parser freeze;
* intervention freeze;
* sham freeze;
* model freeze;
* memory-policy freeze;
* repetition freeze;
* statistical-estimator freeze;
* experiment-manifest hash;
* code commit hash.

No final raw data may be moved into the pilot namespace or vice versa.

---

## 34. Required Architecture Tests

Before substantive pilot execution, automated tests must cover at least the following.

### Task System

* valid task specifications load;
* invalid schemas fail;
* focal decision exists;
* every eligible source has targeted and sham definitions.

### Condition Isolation

* C0 receives no scaffold state;
* C1 receives no H or M;
* C2 receives no M;
* C3 can receive all authorized sources;
* prohibited information cannot enter through mislabeled serialization.

### Tools

* deterministic execution;
* complete request/return logging;
* reproducible state modification.

### Memory

* exact write/read;
* namespace isolation;
* snapshot/restore;
* targeted manipulation.

### Environment

* deterministic transitions;
* snapshot/restore;
* targeted environmental change.

### Logging

* event ordering;
* unique identifiers;
* raw/normalized payload preservation;
* no overwrite of existing raw runs.

### Snapshots

* restored structured state equals original;
* no-op restoration produces equivalent focal context.

### Interventions

* targeted intervention changes only intended state;
* sham intervention changes only intended irrelevant same-source state;
* matched seed/configuration is preserved.

### Action Normalization

* equivalent wording produces equivalent normalized action where applicable;
* consequentially different actions normalize differently.

### Explanation Parsing

* valid JSON parses correctly;
* invalid output is preserved and labeled;
* no semantic repair occurs.

### Metrics

* intervention-positive classification;
* intervention-negative classification;
* sham instability;
* TD calculations;
* CSD calculations;
* exact-set scoring;
* precision/recall/F1 edge cases.

### Model Invocation and Provenance

- task prompts are backed by `task` events;
- model-visible T/H/M/E elements resolve to valid preceding events;
- provenance cannot cross run, task, experiment, or condition boundaries;
- context content exactly matches the normalized payload of its backing event;
- `context_build` is preserved before provenance validation;
- failed provenance validation produces `technical_failure` and prevents generation;
- successful provenance validation precedes generation;
- exact rendered model input and prompt token IDs are preserved;
- ordinary regression tests do not require the local model;
- real-model integration tests are explicitly opt-in.

---

## 35. Architectural Invariants

The following invariants must hold throughout pilot and final execution.

### Invariant 1 — Same Model

All C0-C3 conditions use the same frozen underlying model weights within a study.

### Invariant 2 — Condition by Configuration

Conditions differ only through authorized scaffold capabilities and state availability, not separate agent implementations.

### Invariant 3 — Provenance Preservation

Every decision-relevant state item remains attributable to P, T, H, M, or E.

### Invariant 4 — No Silent State Transformation

No hidden summarization, compression, rewriting, retrieval modification, or context adaptation occurs.

### Invariant 5 — Immutable Raw Evidence

Raw logs are append-only.

### Invariant 6 — Snapshot Equivalence

Targeted and sham replays begin from the same pre-focal baseline state except for the preregistered manipulation.

### Invariant 7 — Behavioral Comparison

Trajectory dependence is based on normalized consequential action changes, not raw textual difference.

### Invariant 8 — Sham Control

A source cannot be classified intervention-positive when its matched sham replay is behaviorally unstable.

### Invariant 9 — Untestable Is Not Negative

A source without a valid intervention/sham pair remains untestable rather than being classified causally irrelevant.

### Invariant 10 — Execution Does Not Adapt to Results

The harness may react to task state but may not alter experimental design based on observed hypothesis direction.

---

## 36. Primary Data Flow for One Experimental Repetition

The complete baseline-to-replay flow is:

```text
1. Load frozen task specification
2. Load condition configuration
3. Restore task initial state
4. Initialize run-scoped memory/environment
5. Apply repetition seed
6. Begin baseline trajectory
7. Log every P/T/H/M/E event
8. Reach preregistered pre-focal state
9. Save snapshot
10. Build provenance-aware structured focal context
11. Deterministically serialize the model-visible context
12. Prepare the backend-specific model input
13. Record CONTEXT_BUILD with structured context, serialized context,
    rendered chat-template input, and exact prompt token IDs
14. Validate every model-visible context element against its backing
    trajectory event
15. Record PROVENANCE_AUDIT
16. Invoke the model using the exact prepared token sequence
17. Record GENERATION with raw output and inference metadata
18. Parse and normalize focal action
19. Commit focal action
20. Record baseline action and task consequence
21. Elicit structured self-explanation before feedback
22. Determine preregistered eligible intervention universe U_d

For each S in U_d:

23. Restore identical pre-focal snapshot
24. Restore matched inference seed
25. Apply targeted intervention to S
26. Validate state difference
27. Rebuild focal context
28. Invoke model
29. Normalize a_target

30. Restore identical pre-focal snapshot
31. Restore matched inference seed
32. Apply source-matched sham to S
33. Validate state difference
34. Rebuild focal context
35. Invoke model
36. Normalize a_sham

37. Classify source result
38. Preserve all raw evidence
39. Complete repetition
```

Analysis occurs only after experimental execution.

---

## 37. Data Flow for Causal Classification

For source `S`:

```text
                           baseline
                              |
                              v
                           a_base
                              |
                 +------------+------------+
                 |                         |
                 v                         v
             targeted                     sham
                 |                         |
                 v                         v
             a_target                   a_sham
                 |                         |
                 +------------+------------+
                              |
                              v

if a_sham != a_base
    => SHAM-UNSTABLE / AMBIGUOUS

else if a_target != a_base
    => INTERVENTION-POSITIVE

else
    => INTERVENTION-NEGATIVE
```

Behavioral equivalence is determined by the frozen action-normalization rule for the task.

---

## 38. Reproducibility Chain

Every reported result must support the following trace:

```text
Paper value
    ↓
generated table/figure
    ↓
analysis output
    ↓
processed decision record
    ↓
intervention classification
    ↓
targeted + sham + baseline run IDs
    ↓
raw event logs
    ↓
exact model input/output
    ↓
snapshot + intervention definition
    ↓
task specification
    ↓
experiment manifest
    ↓
model revision + code commit
```

If this chain cannot be reconstructed for a reported value, the result is not publication-ready.

---

## 39. Architecture Freeze Criteria

The architecture is ready for pilot implementation when:

* source provenance is explicit;
* C0-C3 isolation rules are unambiguous;
* focal-decision context assembly is defined;
* model responsibilities are separated from agent responsibilities;
* memory, tools, environment, and history have distinct ownership;
* snapshot semantics are defined;
* targeted/sham replay semantics are defined;
* raw logging is append-only;
* action normalization occurs independently of causal classification;
* explanation collection occurs before intervention disclosure;
* pilot and final data are structurally separated;
* every reported metric can be traced to raw runs.

Any architectural change after pilot execution begins must be evaluated for whether it constitutes an implementation correction or a methodological revision and documented accordingly.

---

## 40. Explicit Non-Goals

The architecture intentionally does not include:

* LangChain;
* AutoGen;
* CrewAI;
* Qwen-Agent or equivalent agent framework;
* vector databases;
* external web access;
* production API dependencies;
* autonomous task planning beyond the experimental loop;
* background retry systems;
* hidden state summarization;
* LLM-as-judge scoring;
* commercial observability platforms;
* user-interface development;
* distributed execution;
* multi-agent interaction;
* mechanistic neural interpretability tooling.

These may be relevant to later replication or extension studies but would add unnecessary complexity or confounds to the present experiment.

---

## 41. Summary

The architecture operationalizes the experiment as a controlled comparison of **state availability and provenance within one fixed agent harness**.

Its central engineering commitments are:

```text
same model
+
same runtime
+
explicit source provenance
+
cumulative condition configuration
+
pre-focal snapshots
+
targeted intervention
+
source-matched sham
+
normalized consequential action
+
structured self-explanation
+
immutable raw evidence
```

The resulting system is designed so that trajectory dependence is not inferred merely because an agent possesses memory, tools, or history. Dependence is attributed only when controlled intervention on a preregistered source changes the focal consequential action while a matched sham perturbation does not.

The architecture therefore makes the experimental claim auditable at the level appropriate to the study: causal dependence of observable agent behavior on explicitly represented scaffold-state sources.
