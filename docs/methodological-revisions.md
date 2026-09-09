# Pilot Revision Record

**Project:** Trajectory Dependence and Self-Explanation Faithfulness in Agentic AI Systems  
**Document Version:** 1.0  
**Date:** August 25, 2026  
**Status:** Post-Pilot Revision Lock  
**Current project status:** Final confirmatory experiment completed; conference paper submitted  
**Pilot Experiment:** `trajectory_dependence_pilot`  
**Pilot Task Suite:** `trajectory_dependence_pilot` v0.1.1  
**Pilot Execution Plan SHA256:** `6ebc659dce986fd42335509fc3325f9f301177dc70c321d2fc3063ad3f136e63`

---

## 1. Purpose

This document records all methodological and implementation changes made after inspection of the methodological pilot and before final confirmatory data collection.

The methodological pilot is preserved unchanged and is not part of the final confirmatory dataset.

Changes recorded here are permitted only when they address:

- implementation defects;
- incomplete instrumentation;
- measurement sensitivity;
- pilot-resolvable analysis choices;
- reproducibility requirements; or
- final task-suite coverage.

No change may be made for the purpose of obtaining a desired H1 or H2 result.

This record is intentionally retained in its historical, post-pilot form. It
documents changes made before the final confirmatory run and is not a status
description of the current repository.

---

## 2. Pilot Disposition

**Disposition:** PASS WITH CONTROLLED POST-PILOT REVISIONS

The pilot established that the primary experimental apparatus is operational.

### 2.1 Pilot execution

- planned runs: 24;
- completed runs: 24;
- behavior-valid runs: 24;
- successful task runs: 24;
- valid self-explanations: 24;
- invalid self-explanations: 0.

### 2.2 Intervention results

- targeted/sham intervention pairs: 25;
- intervention-positive: 17;
- intervention-negative: 8;
- sham-unstable/ambiguous: 0;
- behaviorally unclassifiable: 0.

### 2.3 Diagnostic channel validation

The pilot diagnostic tasks successfully demonstrated intervention sensitivity for:

- T — tool state;
- H — within-task history;
- M — persistent-memory read state;
- E — environmental state.

All diagnostic targeted interventions were intervention-positive and all corresponding sham interventions were stable.

### 2.4 Observability validation

The post-pilot artifact and observability audits confirmed:

- complete run-level artifact recovery;
- complete model-generation recovery;
- event sequence integrity;
- focal action event-chain integrity;
- explanation-after-action ordering;
- snapshot identity consistency;
- targeted/sham replay integrity;
- inference-configuration consistency;
- runtime-metadata consistency;
- no technical-failure events;
- preservation of raw pilot evidence.

---

## 3. Revision R1 — Runtime Persistent-Memory Write Path

**Date:** August 25, 2026  
**Affected Component:** `src/agent/scaffold.py`  
**Category:** Implementation / instrumentation correction  
**Pilot Outcome Direction Known:** Yes

### Original Specification

The event vocabulary and runtime supported `MEMORY_WRITE`, and the persistent-memory store exposed an exact `write()` operation.

However, `ScaffoldController` exposed only `read_memory()` and therefore no production agent-scaffold execution path emitted `MEMORY_WRITE`.

The methodological pilot exercised persistent-memory reads but did not exercise an end-to-end runtime memory write followed by a later read.

### Revised Specification

Add:

`ScaffoldController.write_memory(...)`

The method:

1. is permitted only under C3;
2. creates one exact persistent-memory record;
3. records `created_by_task`;
4. records the sequence position corresponding to the `MEMORY_WRITE` event;
5. emits `MEMORY_WRITE`;
6. does not create a model-visible M context element;
7. requires a subsequent `MEMORY_READ` to expose M to the model.

### Provenance Invariant

A model-visible M context element remains causally anchored to:

`MEMORY_READ`

and never directly to:

`MEMORY_WRITE`

Therefore the revision does not alter the definition or scoring of source M.

### Technical Reason

The methodological pilot exposed an instrumentation gap between the already-existing memory-store write operation and the production scaffold event pathway.

The correction completes the state lifecycle already specified by the protocol.

### Relationship to Pilot Outcomes

The direction of pilot H1/H2 observations was already known when this correction was made.

The correction was not motivated by intervention direction or self-explanation performance and does not alter any pilot result.

---

## 4. Revision R2 — Supplemental Dynamic-State Diagnostics

**Date:** August 25, 2026  
**Affected Components:** Supplemental diagnostic execution layer  
**Category:** Post-pilot technical validation  
**Pilot Outcome Direction Known:** Yes

### Original Specification

The pilot protocol required validation of:

- persistent-memory write/read behavior; and
- environment state transitions.

The six-task methodological pilot directly exercised:

- persistent-memory reads; and
- environmental queries.

It did not exercise both complete dynamic state-production chains end to end.

### Revised Specification

Two supplemental technical diagnostics were executed after the methodological pilot:

#### R2-A — Persistent-memory lifecycle

`MEMORY_WRITE`
→ persistent state
→ `MEMORY_READ`
→ M context element
→ provenance audit
→ focal action
→ self-explanation

#### R2-B — Environment lifecycle

`ENVIRONMENT_TRANSITION`
→ modified deterministic environment
→ `ENVIRONMENT_QUERY`
→ E context element
→ provenance audit
→ focal action
→ self-explanation

### Dataset Status

Both supplemental diagnostics are explicitly marked:

- `study_role = supplemental_technical_diagnostic`;
- `post_pilot = true`;
- `included_in_h1 = false`;
- `included_in_h2 = false`.

They are not pilot hypothesis evidence and will not enter final confirmatory estimates.

### Technical Reason

The supplemental runs close two pilot-objective coverage gaps without altering the original methodological pilot.

---

## 5. Revision R3 — Frozen Analysis Semantics

**Date:** August 25, 2026  
**Affected Components:** `src/analysis/`  
**Category:** Pilot-resolvable analysis lock  
**Pilot Outcome Direction Known:** Yes

### Original Specification

Protocol v1.1 defined:

- U_d;
- I_d;
- Z_d;
- R_d;
- R_eval;
- TD_source;
- TD_any;
- TD_norm;
- CSD_scaffold;
- CSD_all;
- exact-set accuracy;
- precision;
- recall;
- F1;
- unverifiable-source claims.

The exact executable edge-case behavior had not yet been frozen in analysis code.

### Revised Specification

The executable analysis layer now freezes the following rules.

#### Intervention universe

`U_d` contains only source classes for which a valid targeted/sham comparison is actually available.

Source availability alone does not place a source in U_d.

#### Intervention-positive set

`I_d` consists only of intervention-positive source classes.

Ambiguous and behaviorally unclassifiable comparisons do not become positive or negative evidence.

#### Explanation evaluation

`R_eval(d) = R_d ∩ U_d`

Untestable reported sources are recorded as unverifiable rather than automatically treated as false positives.

`Other` is retained descriptively but excluded from causal-source set scoring unless a future preregistration explicitly maps it.

#### Empty-set rules

- empty R_eval → precision undefined;
- empty I_d → recall undefined;
- F1 is undefined when its required components are undefined;
- exact-set comparison remains defined for empty sets.

#### TD rules

C0 has no scaffold-state TD value.

TD_any and TD_norm require stable eligible scaffold comparisons.

Diagnostic tasks do not contribute to primary H1.

### Technical Reason

Encoding these definitions before final task construction prevents later discretionary interpretation of final outputs.

---

## 6. Revision R4 — Unavailable-Source Claim Diagnostic

**Date:** August 25, 2026  
**Affected Component:** Analysis diagnostics  
**Category:** Secondary post-pilot diagnostic  
**Pilot Outcome Direction Known:** Yes

### Pilot Observation

Some valid self-explanations named source classes that were not model-visible in the focal context.

Examples occurred in the methodological pilot for unavailable H or E sources.

### Revised Specification

Add the secondary diagnostic:

`UnavailableSourceClaim = reported source class not present in the model-visible focal context`

Report:

1. number of unavailable-source claims;
2. proportion of standard reported source claims that are unavailable;
3. number and proportion of runs containing at least one unavailable-source claim.

### Relationship to H2

Unavailable-source claims do not change:

- R_eval;
- intervention-positive classification;
- exact-set scoring;
- precision;
- recall;
- F1.

The diagnostic is reported separately.

### Technical Reason

The measure distinguishes two different concepts:

- **unverifiable** — source may have been visible but lacked a valid intervention;
- **unavailable** — source was not present in the model-visible focal context.

No self-explanation prompt change is made in response to this pilot finding.

---

## 7. Revision R5 — H2b Primary Estimator Lock

**Date:** August 25, 2026  
**Affected Component:** Statistical analysis plan  
**Category:** Pilot-resolvable estimator selection  
**Pilot Outcome Direction Known:** Yes

### Original Specification

Protocol v1.1 permitted either:

1. a task-cluster-bootstrap regression slope; or
2. a rank-based association with task-clustered bootstrap uncertainty

for evaluating the relationship between CSD_scaffold and explanation quality.

The exact estimator was explicitly pilot-resolvable.

### Revised Specification

The primary H2b estimator is:

**task-cluster-bootstrap slope from a simple regression of per-decision F1 on CSD_scaffold**

The regression is recomputed independently inside each task-cluster bootstrap replicate.

Primary reported quantity:

`beta_F1,CSD`

with a 95% task-cluster bootstrap confidence interval.

### Secondary H2b Analysis

Also report:

`beta_recall,CSD`

using the same task-cluster bootstrap procedure where recall is defined.

The recall result is secondary because F1 captures both omitted and over-attributed causal-source claims.

### Bootstrap

Use:

**5,000 task-cluster bootstrap replicates**

for final reported H1 and H2 confidence intervals.

Task IDs are sampled with replacement and all conditions/repetitions belonging to a sampled task remain together.

### Interpretation

A negative F1 slope is consistent with the H2b dispersion-penalty prediction.

A confidence interval including zero is inconclusive.

A positive association weakens or contradicts H2b.

No estimator selection will occur after final outcomes are inspected.

---

## 8. Revision R6 — Integration-Task Coverage / TD_any Ceiling Control

**Date:** August 25, 2026  
**Affected Component:** Final integration task suite  
**Category:** Measurement sensitivity / task coverage  
**Pilot Outcome Direction Known:** Yes

### Pilot Observation

The two methodological-pilot integration tasks produced scaffold dependence in C1, causing pilot TD_any to be saturated at the first scaffold condition.

The pilot therefore demonstrated insufficient dynamic range for TD_any as the sole primary H1 indicator within that very small integration-task sample.

### Revised Specification

The original pilot integration tasks remain unchanged as pilot evidence.

The final confirmatory suite will use a larger set of new preregistered integration tasks designed before final execution.

Task construction must increase structural diversity and H1 measurement resolution without selecting tasks based on whether they produce the predicted H1 direction.

Final integration tasks must include multiple preregistered causal structures, including cases in which:

- one scaffold source may be decision-relevant;
- multiple scaffold sources may be redundant;
- multiple scaffold sources may conflict;
- source relevance may depend on a decision boundary;
- source combinations may create ties or threshold effects;
- a scaffold source may be available yet intervention-negative.

### Prohibited Construction Rule

A candidate task must not be accepted or rejected because pilot-like model execution produces a desired C1, C2, or C3 intervention outcome.

Task acceptance must be based on structural validity, deterministic success criteria, valid intervention/sham construction, and source coverage before confirmatory outcomes are viewed.

### H1 Reporting

TD_any remains the preregistered primary H1 measure.

Mandatory robustness reports remain:

- TD_source;
- TD_norm;
- CSD_scaffold;
- C3-C1 risk difference;
- adjacent C2-C1 and C3-C2 differences.

The H1 definition is not changed.

---

## 9. Revision R7 — Wall-Clock Runtime Instrumentation

**Date:** August 25, 2026  
**Affected Components:** Final execution instrumentation  
**Category:** Reproducibility / performance measurement  
**Pilot Outcome Direction Known:** Yes

### Original Specification

Pilot generation logs captured:

- prompt tokens;
- generated tokens;
- prompt throughput;
- generation throughput;
- peak memory.

Inference duration could therefore be estimated from token counts and throughput.

The runner did not log true orchestration wall-clock duration for the complete matched baseline/targeted/sham unit.

### Revised Specification

Before final execution, add monotonic wall-clock timing for:

1. baseline focal inference;
2. self-explanation inference;
3. each targeted replay;
4. each sham replay;
5. complete task-condition-repetition execution;
6. each matched baseline/targeted/sham causal unit where applicable.

Use a monotonic timer.

Persist durations as execution metadata; timing must not alter model-visible context.

### Technical Reason

The pilot protocol explicitly requires runtime characterization. Derived throughput-based duration is not equivalent to complete wall-clock runtime.

---

## 10. Items Explicitly Not Revised

The following remain unchanged after the pilot:

### Experimental conditions

- C0 = P;
- C1 = P + T;
- C2 = P + T + H;
- C3 = P + T + H + M;
- E remains exogenous/task-controlled rather than a fifth cumulative condition.

### Behavioral measurement

- one preregistered focal consequential decision per task;
- strict structured action parsing;
- normalized behavioral comparison;
- no LLM-as-judge for primary behavior comparison.

### Causal intervention semantics

- source-specific targeted intervention;
- source-matched sham;
- same PRE_FOCAL snapshot;
- matched seed;
- intervention-positive iff targeted behavior changes and sham does not;
- intervention-negative iff targeted behavior does not change and sham is stable;
- sham change → ambiguous;
- technical invalidity remains separate from behavioral evidence.

### Self-explanation

The self-explanation prompt and structured response contract remain unchanged.

Self-explanation remains elicited:

1. after the baseline focal action;
2. before outcome feedback;
3. before intervention disclosure.

### Provenance

- P, T, H, M, and E remain distinct source classes;
- transport mechanism does not redefine provenance;
- model-visible M remains anchored to MEMORY_READ;
- model-visible E remains anchored to valid environment observation/transition provenance under the frozen auditor.

### Pilot evidence

No pilot output is removed, relabeled, rerun in place, or incorporated into the final confirmatory dataset.

---

## 11. Frozen Analysis Decisions

The following analysis definitions are now frozen:

1. TD_source;
2. TD_any;
3. TD_norm;
4. CSD_scaffold;
5. CSD_all;
6. R_eval;
7. exact-set accuracy;
8. explanation precision;
9. explanation recall;
10. explanation F1;
11. empty-set handling;
12. sham-instability classification;
13. baseline task-success treatment;
14. diagnostic/integration analysis separation;
15. unavailable-source claim diagnostic;
16. H2b primary F1-on-CSD cluster-bootstrap slope;
17. H2b secondary recall-on-CSD cluster-bootstrap slope;
18. 5,000 task-cluster bootstrap replicates.

These definitions may not be changed after final data collection begins except under a documented implementation-bug deviation procedure.

---

## 12. Items Still Requiring Final Freeze

The following remain unresolved and must be fixed before confirmatory execution:

1. exact final task list;
2. final task versions;
3. final diagnostic/integration classification;
4. final focal decisions;
5. final targeted intervention definitions;
6. final sham definitions;
7. final repetition count;
8. final seed list;
9. final task execution order;
10. final intervention replay order;
11. exact model revision;
12. local model checksum(s);
13. final wall-clock instrumentation implementation;
14. final analysis-script versions;
15. final exclusion criteria representation;
16. final experiment manifest;
17. experiment-manifest SHA256;
18. frozen experiment Git commit hash.

---

## 13. Changes Prohibited From This Point Forward

Unless correcting a documented implementation defect, do not change:

- H1 or H2 wording;
- condition semantics;
- primary metric definitions;
- intervention-positive classification;
- self-explanation scoring;
- causal-source ontology;
- focal action normalization semantics;
- missingness rules;
- pilot outcome labels.

Once the final task suite is frozen, do not change:

- task content;
- task class;
- focal decision;
- success criteria;
- intervention targets;
- sham targets.

Once final data collection begins, no confirmatory analysis rule may be changed in response to observed results.

---

## 14. Next Required Stage

The next stage is:

**Final Task-Suite Construction and Pre-Freeze Validation**

No final confirmatory data may be collected until that stage, model identity resolution, runtime instrumentation, manifest generation, and final Git freeze are complete.
