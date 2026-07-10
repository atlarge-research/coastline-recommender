# Batch recommendations

Two batch surfaces: an in-process DataFrame API, and a config-driven CSV pipeline.

## DataFrame in, DataFrame out

```python
import coastline, pandas as pd

df = pd.DataFrame([
    {"model": "mistral-7b-v0.1", "method": "lora", "gpu_model": "NVIDIA-A100-SXM4-80GB",
     "tokens_per_sample": 2048, "batch_size": 8},
    {"model": "granite-3.3-8b", "method": "lora", "gpu_model": "L40S",
     "tokens_per_sample": 1024, "batch_size": 16},
])

out = coastline.recommend(df, goal="balanced", predictor="kavier", top_k=1)
```

Each input row comes back with the ranked pick appended: `rank`, `total_gpus`, `gpus_per_node`,
`number_of_nodes`, `batch_size`, `throughput_tok_s`, `runtime_s`, `energy_wh`, `energy_kwh`,
`tokens_per_watt`, `power_w`, `feasible`, `error`, `rationale`.

- Per-row columns (e.g. a `goal` column) **override** the call's keyword arguments.
- A bad row yields `feasible=False` plus an `error` — the rest of the batch still runs.

## CSV in, CSV out

Production batch runs are config-driven — safeguards and predictors are declared in YAML, so runs
are reproducible:

```bash
coastline recommend --config config/batch_config.yaml \
                    --input workloads.csv --output recommendations.csv
```

or from Python:

```python
coastline.recommend_csv("config/batch_config.yaml", "workloads.csv", "recommendations.csv")
```

The output CSV echoes each input row plus: `recommended_total_gpus`, `recommended_gpus_per_node`,
`recommended_number_of_nodes`, `recommended_batch_size`, `predicted_throughput`,
`predicted_runtime_seconds`, `predicted_power_watts`, `tokens_per_watt`, `feasible`, `rationale`.

## Input columns

Common spellings are accepted everywhere (first name is canonical):

| Field | Accepted headers |
|---|---|
| model | `model`, `llm_model`, `model_name` |
| method | `method`, `fine_tuning_method`, `peft` |
| GPU | `gpu_model`, `gpu` |
| sequence length | `tokens_per_sample`, `seq_len`, `tokens`, `max_tokens` |
| batch size | `batch_size`, `batch` |

Anything else? Remap arbitrary headers with an `input.columns` block in the config — see
[Configuration](../reference/configuration.md).
