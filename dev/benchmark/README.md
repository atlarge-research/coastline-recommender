# Benchmark Service

Predictor benchmark suite comparing throughput and latency across all performance predictors.

## Run

From the repo root (the `benchmark` package resolves with `dev/` on the path):

```bash
PYTHONPATH=dev uv run python -m benchmark.main

# Kavier-only mode
PYTHONPATH=dev uv run python -m benchmark.main --kavier-only

# Exclude 128-GPU configurations
PYTHONPATH=dev uv run python -m benchmark.main --exclude-128gpu
```

Results are written to `benchmark/results/` (e.g. `<timestamp>-results.csv`).
