# Reproducibility

## Reproduction levels

This repository supports three distinct levels of reproduction:

1. **Model-free validation:** import the harness, run 496 deterministic tests,
   and reconstruct the exact 400-run execution plan and its hash.
2. **Full experimental rerun:** provide a compatible local model checkpoint and
   regenerate the 400 baselines, 400 explanations, and 1,220 intervention
   replays.
3. **Analysis reproduction:** run the analysis and observability writers over
   regenerated artifacts. Raw outputs from the submitted run are deliberately
   not included in this curated work sample.

## Environment

The completed experiment recorded Python 3.13.15, `mlx` 0.32.0, and `mlx-lm`
0.31.3 on Apple Silicon with 128 GB unified memory. The core harness is standard
library Python and its model-free tests do not require MLX or a GPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

For local model execution on Apple Silicon:

```bash
python -m pip install -e '.[dev,model]'
export TRAJECTORY_MODEL_PATH=/absolute/path/to/local/model
```

The model adapter refuses remote tokenizer code and loads only from the
configured local directory. The absolute checkpoint path is never persisted in
runtime metadata.

## Deterministic validation

Run the test suite:

```bash
python -m pytest -q
```

Validate the exact final execution plan without loading a model:

```bash
python scripts/run_pilot.py --preflight --config config/final-run.json
```

Expected preflight invariants:

```text
planned_runs: 400
checked_runs: 400
checked_intervention_pairs: 610
estimated_model_generations: 2020
plan_sha256: 1c9ac4c5a8fd51f214fb239bbdd3d890414bb129a27ad1d02d63b4c02345b3c9
```

The planner deterministically combines 20 tasks, four conditions, five
repetitions, and the frozen seed policy. Preflight validates task schemas,
source availability, intervention targets, sham matching, run/output identity,
and expected generation counts without creating output directories.

## Full execution

First validate local inference:

```bash
python scripts/model_smoke.py
python scripts/fingerprint_model.py
```

Then execute:

```bash
python scripts/run_pilot.py --execute --config config/final-run.json
```

The internal `Pilot*` classes and script name are legacy generic runner
vocabulary. The selected config identifies `trajectory_dependence_final`, the
20-task final suite, five repetitions, and `artifacts/final` output.

The runner refuses to overwrite an existing experiment root. Preserve the
generated raw bundle unchanged before analysis.

## Analysis and observability

```bash
python scripts/analyze_pilot.py \
  --pilot-root artifacts/final/trajectory_dependence_final \
  --processed-dir data/processed/final \
  --report results/final-report.md

python scripts/audit_pilot.py \
  --pilot-root artifacts/final/trajectory_dependence_final \
  --processed-dir data/processed/final \
  --report results/final-observability-report.md
```

Writers refuse to replace existing derived outputs unless `--overwrite` is
explicitly supplied. Raw evidence is never modified by the analyzers.

## What is and is not fixed

The public configuration fixes the task suite, condition matrix, repetition
count, seed algorithm/base seed, action/explanation token limits, model
identifier, quantization label, and output structure. The final run record did
not populate an exact model revision or model checksum; those fields remain
`null` to preserve the actual record rather than invent provenance.

Consequently, the code and execution plan are directly reproducible, while an
independent bit-for-bit model-output replication is limited by the missing
checkpoint revision/checksum and normal hardware/runtime variation.
