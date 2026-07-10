# Python SDK

`import coastline` exposes three call shapes over one engine. The import is light — heavy
backends load only when a predictor needs them.

| Call | Input | Returns |
|---|---|---|
| `coastline(...)` → `.recommend(workload)` | one workload | `list[Recommendation]`, best first |
| `coastline.recommend(batch, ...)` | DataFrame / list of dicts | `DataFrame` |
| `coastline.recommend_csv(config, in, out)` | CSV file | – (writes CSV) |

## The advisor

The module itself is callable — it returns a configured advisor:

```python
import coastline

advisor = coastline(
    predictor="kavier",          # throughput: kavier | intelligent | cache | <ml name>
    energy="kavier_power",       # kavier_power | opendc
    feasibility="autoconf",      # autoconf | rules | none
)
```

!!! note
    The Python constructor defaults to `predictor="kavier"`; YAML-driven runs (CLI, dashboard)
    default to `intelligent`. An unknown predictor name raises `ValueError` listing valid options.

### `advisor.recommend(workload, **options)`

`workload` is a `WorkloadSpec`, a dict (aliases accepted), or a CSV path.

| Option | Default | Meaning |
|---|---|---|
| `goal` | – | `balanced` \| `performance` \| `energy` \| `min_gpu`; overrides strategy/preset |
| `strategy`, `preset` | `multi_objective`, `balanced` | Explicit [strategy](../concepts/strategies.md) |
| `alpha`, `beta` | – | Manual energy/throughput weights |
| `total_gpus` | `[1, 2, 4, 8, 16]` | Cluster sizes to sweep |
| `batch_sizes` | `[4, 8, 16, 32, 64]` | Batch sizes to sweep |
| `top_k` | `5` | Recommendations to return |
| `max_gpus` | `16` | Upper bound on total GPUs |

## Data shapes

**Workload fields** (`WorkloadSpec`):

| Field | Type | Required |
|---|---|---|
| `llm_model` | str | yes |
| `fine_tuning_method` | str | yes |
| `gpu_model` | str | yes |
| `tokens_per_sample` | int > 0 | yes |
| `batch_size` | int > 0 | yes |
| `gpus_per_node`, `number_of_nodes` | int ≥ 1 | no |
| `torch_dtype`, `enable_roce` | str, bool | no |

**`Recommendation`:**

| Field | Meaning |
|---|---|
| `total_gpus`, `gpus_per_node`, `number_of_nodes` | The recommended layout |
| `predicted_throughput` | tokens/sec |
| `predicted_runtime_seconds` | seconds |
| `strategy` | Strategy that produced it |
| `metadata` | `batch_size`, `predicted_power_watts`, `tokens_per_watt`, feasibility info |

## Batch APIs

See [Batch recommendations](../guides/batch.md) for both surfaces, column aliases, and output
columns.

```python
out_df = coastline.recommend(df, goal="balanced", predictor="kavier", top_k=1)
coastline.recommend_csv("config/batch_config.yaml", "in.csv", "out.csv")
```

## Runnable tour

`docs/usage.py` in the repository exercises all three call shapes end-to-end — run it with
`make demo`.
