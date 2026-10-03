# Predictor benchmark

Compares the throughput and step-latency accuracy, and the prediction time, of the performance predictors.

## Run

From the repo root (the `benchmark` package resolves with `dev/` on the path):

```bash
PYTHONPATH=dev uv run python -m benchmark.main

# Kavier-only mode
PYTHONPATH=dev uv run python -m benchmark.main --kavier-only

# Exclude 128-GPU configurations
PYTHONPATH=dev uv run python -m benchmark.main --exclude-128gpu
```

Results are written to `dev/benchmark/results/` (by default `<timestamp>-results.csv`).
