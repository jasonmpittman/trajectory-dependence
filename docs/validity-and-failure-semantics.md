# Validity and Failure Semantics

The evaluation records several independent outcome dimensions. Treating them as
one pass/fail flag would discard valid evidence and obscure infrastructure
failures.

| Dimension | Meaning | Final observation |
| --- | --- | ---: |
| Run completion | Planned run produced its run bundle | 400/400 |
| Task success | Normalized action matched the task's condition-specific success rule | 385/400 |
| Behavioral validity | Focal action was parseable and classifiable | 400/400 |
| Explanation validity | Structured self-explanation passed schema parsing | 400/400 |
| Technical-failure event | Runtime emitted an explicit technical failure | 0 |
| Intervention stability | Targeted/sham pair was behaviorally classifiable | 610/610 |
| Logging completeness | Strict baseline, explanation, snapshot, replay-chain, config, and sequence checks all passed | 260/400 |

## Task failure is valid behavior

The 15 task-unsuccessful decisions were well-formed model actions. They remain
in the primary behavioral analysis. Excluding them would condition the causal
analysis on correctness and erase a real part of model behavior.

## Invalid behavior is not a technical failure

A malformed or out-of-contract action would be preserved as behaviorally
invalid. It would not be silently retried or automatically labeled as an
infrastructure failure. No final-run focal action was invalid.

## Intervention outcomes

For an eligible source:

- `intervention_positive`: targeted replay changes normalized behavior and sham does not;
- `intervention_negative`: neither targeted nor sham changes normalized behavior;
- `sham_unstable_ambiguous`: sham changes behavior, with or without a targeted change;
- `behaviorally_unclassifiable`: baseline, targeted, or sham action cannot be compared;
- `technical_failure`: the matched unit cannot be evaluated because execution failed;
- `untestable`: the design cannot construct the required source-level comparison.

The final 610 pairs contained 265 positive and 345 negative outcomes, with no
ambiguous or behaviorally unclassifiable pair.

## Replay-chain audit finding

The generated observability audit reports 140 runs with
`replay_chains_valid = false`; these are the same 140 runs that prevent the
stricter `logging_complete` flag. The corresponding sampled records retain:

- a valid baseline focal chain;
- a valid explanation chain after the action;
- contiguous baseline and replay sequence indices;
- a valid pre-action snapshot identity;
- matching replay inference configuration;
- consistent runtime metadata and token counts;
- no technical-failure event.

The public repository does not include raw submitted-run event streams, so this
curated release does not claim to re-adjudicate the 140 flags. It reports them
as a replay-provenance/observability limitation, separate from completion,
behavioral validity, explanation validity, and task success.

## Context-length control

Targeted replays had an exact prompt-token-count match rate of 1.000. Sham
replays had an aggregate exact match rate of 0.803 because all 120 E-source sham
replays differed by one prompt token even though their model-visible character
length was unchanged. The other T/H/M/P sham comparisons were exact token-count
matches.

This does not prove the E shams were behaviorally confounded; no sham changed a
normalized action. It is nevertheless a control imperfection and is disclosed.

## Timing

Exact orchestration wall-clock time was not logged. Matched-unit runtime values
in the observability report are estimates derived from logged token counts and
throughput, not direct elapsed-time measurements.
