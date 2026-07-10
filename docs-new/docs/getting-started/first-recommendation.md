# First recommendation

Two minutes, either surface.

## Python

```python
import coastline

advisor = coastline(predictor="kavier")
results = advisor.recommend(
    {"model": "mistral-7b-v0.1", "method": "lora",
     "gpu_model": "NVIDIA-A100-SXM4-80GB", "tokens_per_sample": 1024, "batch_size": 32},
    goal="performance", max_gpus=8,
)

best = results[0]
print(best.total_gpus, best.predicted_throughput, best.predicted_runtime_seconds)
```

The workload is a plain dict. Five fields matter:

| Field | Aliases | Example |
|---|---|---|
| `model` | `llm_model`, `model_name` | `mistral-7b-v0.1` |
| `method` | `fine_tuning_method`, `peft` | `lora`, `full`, `gptq-lora` |
| `gpu_model` | `gpu` | `NVIDIA-A100-SXM4-80GB` |
| `tokens_per_sample` | `seq_len`, `tokens` | `1024` |
| `batch_size` | `batch` | `32` |

`goal` states your intent — it picks the ranking strategy for you:

- `performance` — fastest configs first
- `balanced` — weigh throughput and energy equally (default)
- `energy` — most efficient configs first
- `min_gpu` — fewest GPUs that will work

## CLI

The guided REPL asks for the same fields interactively:

```bash
coastline interactive
```

Or run a config file and get JSON back:

```bash
coastline run --config config/coastline_functionality/config.yaml
```

## Next

See what else Coastline can do in [Features](features.md), or go deeper with the
[guides](../guides/workload.md).
