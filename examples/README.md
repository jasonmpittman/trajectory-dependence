# Examples

Run the public example's model-free preflight:

```bash
python scripts/run_pilot.py --preflight --config config/example.json
```

This validates the six-task methodological suite with one repetition under all
four conditions. It does not load a model or create artifact directories.

For the exact completed 400-run plan, use:

```bash
python scripts/run_pilot.py --preflight --config config/final-run.json
```
