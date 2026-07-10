# Configuration

YAML is the interface for reproducible runs: `coastline run`, `coastline recommend`, and
`recommend_csv()` all read the same schema. Sample configs live in `config/`.

```yaml
workload:                         # the fine-tuning job (coastline run)
  llm_model: "mistral-7b-v0.1"
  fine_tuning_method: "lora"      # full | lora | gptq-lora
  tokens_per_sample: 2048
  batch_size: 8

system:
  default_gpu: "NVIDIA-A100-SXM4-80GB"   # fallback when grid.gpu_models is empty

strategy:
  name: "multi_objective"         # multi_objective | min_gpu
  preset: "balanced"              # balanced | performance | energy  (or alpha/beta)
  max_slowdown: 2.0               # optional: drop configs slower than 2× the fastest

predictors:
  performance: "intelligent"      # intelligent | kavier | cache | <ml model name>
  energy: "kavier_power"          # kavier_power | opendc
  feasibility: "autoconf"         # autoconf | rules | none

grid:                             # the search space
  gpu_models: ["NVIDIA-A100-SXM4-80GB"]
  batch_sizes: [4, 8, 16, 32, 64]
  total_gpus: [1, 2, 4, 8, 16]
  top_k: 3
```

## `strategy`

| Key | Default | Meaning |
|---|---|---|
| `name` | `min_gpu` | `multi_objective` (weighted ranking) or `min_gpu` (fewest feasible GPUs) |
| `preset` | `balanced` | `balanced` \| `performance` \| `energy`, plus `-frontier` variants |
| `alpha`, `beta` | – | Explicit energy/throughput weights; mutually exclusive with `preset` |
| `max_slowdown` | off | Runtime guard: never recommend > N× slower than the fastest feasible |

## `predictors`

| Key | Default | Meaning |
|---|---|---|
| `performance` | `intelligent` | Throughput predictor — see [Predictors](../concepts/predictors.md) |
| `energy` | `kavier_power` | Power predictor (`opendc` needs `OPENDC_BIN_PATH`) |
| `feasibility` | `autoconf` | `autoconf` \| `rules` \| `none` |
| `lookup` | bundled | Measured-runs CSV for the cache path, or `default` |
| `opendc_calibration_factor` | `1.0` | Scaling factor for OpenDC power |
| `autoconf_model_version` | `3.1.0` | AutoConf classifier version |

## `grid`

| Key | Default | Meaning |
|---|---|---|
| `gpu_models` | – | GPUs to consider (the CLI uses the first entry) |
| `batch_sizes` | `[4, 8, 16, 32, 64]` | Batch sizes to sweep |
| `total_gpus` | `[1, 2, 4, 8, 16]` | Cluster sizes to sweep; node layout is derived |
| `top_k` | `3` | Ranked configurations to return (`min_gpu` returns 1) |

## Batch input remap

`coastline recommend` accepts arbitrary CSV headers via an optional remap
(common spellings are [already accepted](../guides/batch.md#input-columns)):

```yaml
input:
  columns:
    my_model_column: llm_model
    my_gpu_column: gpu_model
```

!!! note "Legacy configs"
    Old configs with an `orchestrator:` block are translated automatically — but only when no
    modern `predictors:` block is present.
