# Benchmark Service

Predictor benchmark suite comparing throughput and latency across all performance predictors.

## Run

```bash
python -m benchmark.main

# Kavier-only mode
python -m benchmark.main --kavier-only

# Exclude 128-GPU configurations
python -m benchmark.main --exclude-128gpu
```

Results are written to `benchmark/results/` (e.g. `<timestamp>-results.csv`).
