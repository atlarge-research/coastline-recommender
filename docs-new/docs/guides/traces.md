# Traces

Have a fine-tuning trace — one job per row? Coastline recommends a configuration for every job,
then draws the cluster timeline those recommendations produce.

## 1. Recommend per job

```bash
coastline recommend-trace --input trace.csv --output recommended.csv --method kavier
```

Each row gains a recommended layout and an `estimated_duration_<method>` column.

Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--goal` | `min_gpu` | `min_gpu` or `performance` |
| `--method` | `kavier` | Duration estimator: `kavier`, `tabpfn`, `xgb` |
| `--feasibility` | `autoconf` | `rules` runs without the AutoConf extra |
| `--lookup` | – | Measured-runs CSV (or `default`) for cache-backed estimates |
| `--cluster-gpus` | `16` | Total cluster GPUs |
| `--node-gpus` | `8` | GPUs per node |

## 2. Plot the timeline

```bash
coastline plot-trace --input recommended.csv --output timeline.pdf
```

Jobs are FIFO-scheduled onto the cluster; the figure shows GPUs in use and jobs queued over time.
Requires the `[plot]` extra. To do both steps in one go, add `--visual` to `recommend-trace`.

!!! tip "Bring your own trace"
    Any CSV works as long as it has the workload columns (model, method, GPU, tokens per sample,
    batch size — [common spellings accepted](batch.md#input-columns)) plus a submission time
    column for the timeline.
