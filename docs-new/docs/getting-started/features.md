# Features

What Coastline does, at a glance. Each links to the guide or reference that covers it.

## Recommend

- **Single workload** — one call, ranked list of configurations back.
  [Recommend a workload](../guides/workload.md)
- **Batch** — a DataFrame or CSV of workloads, one recommendation per row; a bad row is marked
  infeasible instead of failing the batch. [Batch recommendations](../guides/batch.md)
- **Traces** — annotate every job in a fine-tuning trace with a recommended layout and estimated
  duration, then plot the cluster timeline. [Traces](../guides/traces.md)

## Predict

- **Throughput + runtime** — analytical Kavier physics by default (~6% median error), or trained
  ML models down to ~2%. [Predictors](../concepts/predictors.md)
- **Power + energy** — every recommendation carries predicted watts and tokens-per-watt.
- **Feasibility** — an OOM-aware classifier drops configurations that would crash, before ranking.

## Rank

- **Goals** — `performance`, `balanced`, `energy`, `min_gpu`; or explicit α/β weights for the
  throughput↔energy trade-off. [Ranking strategies](../concepts/strategies.md)
- **Safeguards** — `max_slowdown` guarantees you're never recommended a config slower than
  N× the fastest feasible one.

## Extend

- **Tune on your own runs** — `coastline tune` fits a predictor to your measured data; it's picked
  up automatically. [Tune a predictor](../guides/tune.md)
- **Pluggable backends** — swap the throughput predictor (`kavier`, `cache`, `intelligent`, any
  named ML model) and the energy path (`kavier_power`, `opendc`) per run.

## Surfaces

- **Python** — `import coastline`; single, batch, and CSV APIs. [SDK reference](../reference/sdk.md)
- **CLI** — one `coastline` command, six subcommands. [CLI reference](../reference/cli.md)
- **Dashboard** — a FastAPI web UI with a wizard and REST API. [Dashboard](../guides/dashboard.md)
