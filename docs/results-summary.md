# Results Summary

## Completion and integrity

All 400 planned runs completed. Every run produced a valid focal behavior and a
valid structured self-explanation. Of those behaviors, 385 satisfied the task
success criterion and 15 did not. All valid behavior, including unsuccessful
task behavior, remained in analysis.

Across 610 matched targeted/sham intervention pairs, 265 were
intervention-positive and 345 intervention-negative. No pair was sham-unstable
or behaviorally unclassifiable. Diagnostic comparisons were positive for all 45
channel checks: T 15/15, H 10/10, M 5/5, and E 15/15.

## H1: trajectory dependence

The preregistered monotonic prediction was not supported. On integration tasks:

| Condition | N | TD_any | Mean TD_norm | Mean CSD_scaffold | Multi-source positive |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 | 80 | 0.688 | 0.625 | 0.750 | 0.063 |
| C2 | 80 | 0.750 | 0.417 | 0.938 | 0.125 |
| C3 | 80 | 0.625 | 0.302 | 1.000 | 0.313 |

Trajectory dependence was common but not monotonic. Secondary measures are
consistent with dependence becoming distributed across more sources in richer
configurations; the submitted paper labels this interpretation emergent rather
than preregistered.

## H2: self-explanation faithfulness

| Evaluation set | N | Exact-set accuracy | Macro precision | Macro recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| All evaluable | 290 | 0.586 | 0.661 | 1.000 | 0.926 |
| Integration evaluable | 245 | 0.510 | 0.599 | 1.000 | 0.906 |

H2a was supported: explanations did not perfectly recover the complete
intervention-defined causal-source set. The dominant error was over-attribution,
not omission of an observed positive source. H2b was not supported: explanation
F1 did not decline with causal-source dispersion under the preregistered
analysis.

These metrics are restricted to evaluable intervention coverage. Unverifiable
source claims are reported separately and are not automatically scored false.

## Interpretation boundary

The results establish behavioral sensitivity to controlled interventions on
represented state sources. They do not establish complete internal neural
causation, generalize beyond the tested model/task environment, or support
claims about consciousness, sentience, phenomenology, or moral patiency.

See the [machine-readable aggregate](../results/summary/metrics.json),
[validity semantics](validity-and-failure-semantics.md), and
[submitted manuscript](../paper/trajectory-dependence.pdf).
