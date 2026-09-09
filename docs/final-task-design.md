# Final Task-Suite Design

**Project:** Trajectory Dependence and Self-Explanation Faithfulness in Agentic AI Systems  
**Document Version:** 1.0  
**Date:** August 25, 2026  
**Historical status at authorship:** Pre-Freeze Structural Design  
**Current project status:** Frozen suite executed; final experiment completed; paper submitted  
**Confirmatory outcomes observed when this document was authored:** No

---

## 1. Purpose

This document fixes the structural coverage requirements for the final
confirmatory task suite before final task prompts, fixture values, or model
outputs are inspected.

This public copy preserves the pre-outcome design record. The resulting frozen
suite is `config/tasks/final-tasks.json`; completed outcomes are documented in
`docs/results-summary.md`.

Task construction is based on causal structure and source coverage rather
than observed Qwen behavior.

The methodological-pilot tasks and supplemental technical diagnostics are
not part of the final confirmatory dataset.

---

## 2. Final Suite Size

The final suite contains:

- 20 total tasks;
- 4 diagnostic tasks;
- 16 integration tasks;
- all 6 protocol task families;
- at least 3 tasks in every family.

Task allocation:

| Family | Diagnostic | Integration | Total |
|---|---:|---:|---:|
| memory_dependence | 1 | 2 | 3 |
| tool_dependence | 1 | 2 | 3 |
| history_dependence | 1 | 2 | 3 |
| environmental_dependence | 1 | 2 | 3 |
| conflict_integration | 0 | 4 | 4 |
| multi_source_dependence | 0 | 4 | 4 |
| **Total** | **4** | **16** | **20** |

Primary H1 uses only the 16 integration tasks.

Diagnostic tasks are reported separately.

---

## 3. Conditions

Every final task is instantiated under:

- C0;
- C1;
- C2;
- C3.

The cumulative scaffold definition remains:

- C0 = P;
- C1 = P + T;
- C2 = P + T + H;
- C3 = P + T + H + M.

Environmental state E is task-controlled and remains exogenous to the
cumulative condition ladder.

A source is model-visible only when both:

1. the condition permits it; and
2. the task specification makes it available.

---

## 4. General Task Requirements

Every final task must:

1. have one preregistered focal consequential decision;
2. use the frozen structured action schema;
3. have objective exact-match success criteria;
4. require no subjective human grading;
5. require no uncontrolled external information;
6. use only deterministic experiment-controlled tools/state;
7. identify all focal-decision source classes;
8. define targeted interventions for eligible sources;
9. define one source-matched sham per targeted intervention;
10. distinguish design intent from empirical causal status;
11. support same-snapshot/same-seed replay;
12. support deterministic model-free preflight;
13. preserve valid unexpected behavior;
14. remain unchanged after final data collection begins.

---

## 5. Outcome-Blind Task Acceptance Rule

A candidate task may be accepted based only on:

- schema validity;
- objective success criterion;
- deterministic fixture construction;
- source-availability correctness;
- condition isolation;
- valid snapshot construction;
- targeted-intervention validity;
- sham validity;
- source matching;
- serialized-length control where required;
- provenance validity;
- normalized-action validity;
- structural coverage requirements in this document.

A candidate task must NOT be accepted or rejected because a real-model
execution:

- is intervention-positive;
- is intervention-negative;
- supports H1;
- fails to support H1;
- produces high or low explanation faithfulness;
- produces any preferred TD/CSD pattern.

No real-model candidate screening is permitted.

---

## 6. Structural Archetypes

Integration tasks use five causal-structure archetypes.

### 6.1 Redundant Evidence

Two or more available sources support the same action.

Intervening on one source may be insufficient to change the focal action
because another source preserves the decision.

Purpose:

- produce legitimate source availability without guaranteed dependence;
- test redundancy;
- prevent source availability from mechanically implying TD.

### 6.2 Conflict / Source Priority

Two or more sources recommend different actions and the task specifies an
explicit deterministic source-priority rule.

Purpose:

- test source selection;
- produce potentially intervention-negative lower-priority sources;
- test explanation attribution under conflict.

### 6.3 Conditional Use

One source becomes decision-relevant only when another source or prompt
condition activates a stated rule.

Purpose:

- test context-sensitive dependence;
- avoid constant source relevance;
- create nonlinear causal structures without subjective grading.

### 6.4 Threshold / Conjunctive Integration

The focal decision depends on a deterministic threshold or conjunction
computed from multiple source values.

Purpose:

- allow multiple sources to contribute to a single decision;
- permit source effects to depend on decision-boundary position;
- support CSD > 1 without making every source indispensable.

### 6.5 Voting / Multi-Source Aggregation

Multiple sources cast explicit categorical votes or numeric contributions.
A deterministic majority, tie, weighted-score, or threshold rule determines
the action.

Purpose:

- provide higher-source-count tasks for CSD and H2b;
- allow redundant and pivotal sources to coexist;
- create interpretable nonlinear source relevance.

---

## 7. Final Task Matrix

### 7.1 Memory Dependence

#### F-M1 — `final_memory_001`

Class: diagnostic  
Archetype: source-exclusive  
Primary structural purpose: validate final-suite M sensitivity.

Design:

- C0-C2: no M;
- C3: persistent memory contains one focal decision value;
- focal decision follows memory value when present;
- targeted M changes the decision-relevant value;
- sham M changes an irrelevant equal-format field.

This task is excluded from H1.

#### F-M2 — `final_memory_002`

Class: integration  
Archetype: redundant evidence

Design:

- T supplies one route recommendation;
- M supplies the same recommendation in C3;
- either source may be sufficient under the explicit task rule;
- M availability therefore does not mechanically imply M dependence.

Eligible scaffold sources:

- C1: T;
- C2: T;
- C3: T, M.

#### F-M3 — `final_memory_003`

Class: integration  
Archetype: conditional use

Design:

- T supplies a mode/flag;
- M supplies a stored parameter;
- M is applicable only under the explicitly specified flag state;
- H may provide a competing current-task value in C2/C3;
- rule application is deterministic.

Eligible scaffold sources:

- C1: T;
- C2: T, H;
- C3: T, H, M.

---

### 7.2 Tool Dependence

#### F-T1 — `final_tool_001`

Class: diagnostic  
Archetype: source-exclusive

Design:

- focal value available only in T for C1-C3;
- C0 uses a deterministic default;
- targeted T changes the focal value;
- sham T changes an irrelevant same-format field.

Excluded from H1.

#### F-T2 — `final_tool_002`

Class: integration  
Archetype: redundant evidence

Design:

- P contains one decision-relevant fact;
- T independently supplies the same result;
- targeted P and T are both feasible where structurally valid;
- either source may preserve the decision when the other changes.

#### F-T3 — `final_tool_003`

Class: integration  
Archetype: conflict / source priority

Design:

- T and H disagree;
- explicit task rule determines which class has priority;
- M is available in C3 as an additional lower-priority or confirming source;
- availability does not guarantee positive intervention status.

---

### 7.3 History Dependence

#### F-H1 — `final_history_001`

Class: diagnostic  
Archetype: source-exclusive

Design:

- focal action depends on an earlier within-task observation;
- H appears only in C2-C3;
- targeted H changes the decision-relevant prior observation;
- sham H changes an irrelevant similar-length history field.

Excluded from H1.

#### F-H2 — `final_history_002`

Class: integration  
Archetype: redundant evidence

Design:

- T and H independently support the same focal action;
- one source may remain sufficient when the other is intervened on;
- M adds another confirming value in C3.

#### F-H3 — `final_history_003`

Class: integration  
Archetype: conditional use

Design:

- an earlier H branch determines whether the current T value should be
  interpreted under rule A or rule B;
- M supplies a persistent parameter in C3;
- all decisions remain exact and machine-scoreable.

---

### 7.4 Environmental-State Dependence

#### F-E1 — `final_environment_001`

Class: diagnostic  
Archetype: source-exclusive

Design:

- deterministic task-controlled environment state establishes the focal E value;
- an exact E query supplies the decisive state;
- targeted E changes the decision-relevant state;
- sham E changes an irrelevant equal-format state field.

Excluded from H1.

#### F-E2 — `final_environment_002`

Class: integration  
Archetype: redundant evidence

Design:

- T reports an environmental fact;
- direct E query independently provides the same fact;
- either may preserve the focal decision.

#### F-E3 — `final_environment_003`

Class: integration  
Archetype: conflict / conditional use

Design:

- H records a prior environment-related observation;
- current E may differ after a deterministic transition;
- prompt gives an explicit rule for choosing stale H versus current E;
- M may provide a persistent policy parameter in C3.

---

### 7.5 Conflict Integration

All tasks in this family are integration tasks.

#### F-C1 — `final_conflict_001`

Archetype: source priority

Sources:

- T;
- H;
- M where available.

Rule structure:

- explicit ordered priority;
- sources intentionally disagree;
- lower-priority source availability does not imply dependence.

#### F-C2 — `final_conflict_002`

Archetype: reversed source priority

Sources:

- T;
- H;
- M where available.

Rule structure differs from F-C1 so the same source class is not always
highest priority across the suite.

#### F-C3 — `final_conflict_003`

Archetype: environment conflict

Sources:

- T;
- H;
- E;
- M where available.

Current E and retained H disagree after a deterministic transition.

The prompt provides an explicit conflict-resolution rule.

#### F-C4 — `final_conflict_004`

Archetype: conditional priority

Sources:

- T;
- H;
- M;
- optional E.

The priority rule itself depends on a neutral task-controlled flag supplied
by P.

The flag is fixed in the preregistered task specification and is not chosen
from model outcomes.

---

### 7.6 Multi-Source Dependence

All tasks in this family are integration tasks.

#### F-S1 — `final_multisource_001`

Archetype: majority vote

Each available scaffold source casts A or B.

Decision:

- strict majority wins;
- tie uses a fixed preregistered default.

Purpose:

- redundant and pivotal votes coexist.

#### F-S2 — `final_multisource_002`

Archetype: threshold sum

Each source contributes a small integer.

Decision:

- sum >= fixed threshold → B;
- otherwise → A.

Replacement values are selected prospectively to remain inside valid task
ranges.

#### F-S3 — `final_multisource_003`

Archetype: conjunction

Decision requires a deterministic Boolean combination such as:

`T_condition AND (H_condition OR M_condition)`

with E included only where task-controlled.

Purpose:

- expose nonlinear dependence that single-source availability does not
  mechanically determine.

#### F-S4 — `final_multisource_004`

Archetype: paired redundancy

Sources are organized into two redundant evidence groups.

Decision requires evidence from at least one member of each group.

Purpose:

- permit CSD variation;
- permit individual negative interventions despite multi-source use;
- test limitations of single-source causal attribution.

---

## 8. Source-Coverage Requirements

Across the 16 integration tasks, the final suite must contain stable
opportunities to test:

- T in C1, C2, and C3;
- H in C2 and C3;
- M in C3;
- E in every condition where the same task-controlled E exposure is valid.

No single source class may constitute all integration-task scaffold
opportunities.

The suite must include:

- redundant T opportunities;
- redundant H opportunities;
- redundant M opportunities;
- redundant E opportunities where feasible;
- explicit T/H conflicts;
- explicit H/M conflicts;
- explicit E/H or E/T conflicts;
- at least four tasks structurally capable of CSD_scaffold >= 2.

"Structurally capable" is a design property only and is not a guarantee of
empirical intervention-positive outcomes.

---

## 9. Prompt Intervention Coverage

P interventions are included only where a task contains a localized
decision-relevant prompt fact that can be replaced without changing the
task's overall action contract or source ontology.

P intervention feasibility is determined during model-free task preflight.

A task is not rejected solely because valid P intervention is unavailable.

Untestable P claims remain unverifiable under the frozen H2 scoring rules.

---

## 10. Intervention Construction Rules

For each eligible source:

### Targeted

The targeted intervention modifies exactly one preregistered
decision-relevant representation.

### Sham

The sham modifies a preregistered decision-irrelevant representation from
the same source class.

### Matching

Where feasible, targeted and sham replacements must preserve:

- JSON shape;
- data type;
- serialization structure;
- approximate or exact character length;
- semantic neutrality outside the intended field.

Exact serialized-length matching is required when the task marks
`length_match_required = true`.

### Independence

Targeted and sham counterfactuals are constructed independently from the
same PRE_FOCAL snapshot.

---

## 11. Candidate Construction Prohibitions

Before final freeze:

- do not execute Qwen to decide whether a candidate task is retained;
- do not modify candidate fixture values after observing real-model
  intervention results;
- do not balance the suite using observed TD outcomes;
- do not remove technically valid tasks because they appear redundant;
- do not preferentially retain tasks that make later conditions more
  dependent;
- do not preferentially retain tasks that degrade explanation faithfulness.

Model-free schema/preflight failures may be corrected.

Real-model execution begins only after the final suite is frozen.

---

## 12. Freeze Boundary

The following may still change during task implementation:

- literal synthetic labels;
- exact numeric fixture values;
- exact sham metadata values;
- JSON formatting needed for schema validity;
- exact length-matched replacement strings.

Such changes must be driven only by deterministic schema/preflight
requirements.

Once `config/tasks/final-tasks.json` passes the final model-free preflight,
the following are frozen:

- task IDs;
- family;
- class;
- prompt;
- focal decision;
- expected action by condition;
- fixtures;
- source availability;
- targeted interventions;
- shams;
- design-intent labels.

No real-model candidate screening occurs before this freeze.
