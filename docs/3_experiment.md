# Setting up an experiment

In this section, you will learn how to set up an experiment. In this section you will find:

1. [The structure of the config folder](#1-the-structure-of-the-config-folder) — where each config lives.
2. [config.yaml](#config-yaml) — the entry point: the everyday config the CLI and the API load.
3. [default.yaml](#default-yaml) — the fallback when no config is given.
4. [experiment.yaml](#experiment-yaml) — the preferred policy config when unspecified.
5. [infrastructure.yaml](#infrastructure-yaml) — the cluster caps the API enforces.
6. [batch_config.yaml](#batch-config-yaml) — the config for batch experiments.
7. [demo.yaml](#demo-yaml) — the annotated reference in the user playground.

## 1. The structure of the config folder

Coastline is shipped with a `config` folder, which contains two subfolders, one necessary for coastline_functionality (please edit with care) and a user playground.

```text
config/
├── batch_config.yaml            # batch CSV experiments (coastline recommend-job)
├── coastline_functionality/     # configs the code loads — edit with care
│   ├── config.yaml              # everyday config the CLI and API load
│   ├── default.yaml             # fallback when no config is given
│   ├── experiment.yaml          # preferred policy config when unspecified
│   ├── infrastructure.yaml      # cluster GPU caps the API enforces
│   └── run_database.csv         # measured-runs lookup DB (cache/intelligent)
└── user_playground/
    └── demo.yaml                # annotated reference: every option + allowed values
```

Run an experiment by pointing `coastline recommend-job --config` at a config:

```console
coastline recommend-job --config config/coastline_functionality/config.yaml
```

The recommendation prints as JSON:

```json
{
  "timestamp": "2026-07-11T20:17:07.150886",
  "configuration": {
    "total_gpus": 4,
    "gpus_per_node": 4,
    "workers": 1
  },
  "strategy": "multi_objective_balanced",
  "performance": {
    "throughput_tokens_per_sec": 15091.68
  },
  "energy": {
    "power_watts": 223.04,
    "efficiency_tokens_per_watt": 67.66
  },
  "metadata": {
    "predicted_power_watts": 223.04,
    "combined_score": 0.82,
    "rank": 1,
    "selection_policy": "balanced",
    "tokens_per_watt": 67.66,
    "throughput_score": 0.84,
    "power_score": 0.8,
    "feasibility": {
      "Rule-Based Classifier error": "",
      "Predictive Model Classifier error": null
    },
    "batch_size": 64,
    "workflow": "grid_feasibility_simulate_policy",
    "preset": "balanced",
    "alpha": 0.5,
    "beta": 0.5
  }
}
```

Write an artifact instead with `--output-dir`.

## 2. config.yaml { #config-yaml }

The everyday config the CLI and the API load. The config declares four blocks: `workload` (the fine-tuning job), `strategy` (the [recommendation policy](4_recommendation_policies.md)), `predictors` (the [simulation models](5_simulation_models.md)), and `grid` (the search space). The grid sweeps `batch_sizes` × `total_gpus`; the workload fields seed the layout.

```yaml
workload:
  llm_model: "mistral-7b-v0.1"
  fine_tuning_method: "lora"
  tokens_per_sample: 2048
  batch_size: 8

strategy:
  name: "multi_objective"
  preset: "balanced"

predictors:
  performance: "intelligent"
  energy: "kavier_power"
  feasibility: "autoconf"

grid:
  gpu_models: [ "NVIDIA-A100-SXM4-80GB" ]
  batch_sizes: [ 4, 8, 16, 32, 64 ]
  total_gpus: [ 1, 2, 4, 8, 16 ]
  top_k: 3
```

Below, specs for all of the config fields, individually:

### 2.1 workload { #config-workload }

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `llm_model` | String | yes | LLM to fine-tune. Must exist in the Kavier LLM spec library or in trained ML artifacts. |
| `fine_tuning_method` | String | yes | PEFT method: `full`, `lora`, or `gptq-lora` (exactly these three). |
| `tokens_per_sample` | Int | yes | Sequence length. Any int > 0. |
| `batch_size` | Int | yes | Seed per-device batch size. The grid overrides the seed during the sweep. |
| `gpus_per_node` | Int | no | Seed node layout. The grid derives the real layout from `total_gpus`. |
| `number_of_nodes` | Int | no | Seed node count. Same derivation as `gpus_per_node`. |
| `torch_dtype` | String | no | Optional hint for the ML features: `bfloat16` or `float16`. |
| `enable_roce` | Bool | no | Optional interconnect hint for the ML features. |

### 2.2 strategy { #config-strategy }

The `strategy:` block declares the [recommendation policy](4_recommendation_policies.md).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | String | no | `multi_objective` (default) or `min_gpu`. |
| `preset` | String | no | Weight preset of the multi-objective policy: `balanced` (default), `performance`, or `energy`. |
| `alpha`, `beta` | Float | no | Explicit weights in [0, 1]; setting the weights overrides `preset`. |
| `max_slowdown` | Float | no | Safeguard: reject any configuration slower than N× the fastest feasible configuration. Omit for no cap. |

### 2.3 predictors { #config-predictors }

The `predictors:` block selects one [simulation model](5_simulation_models.md) per axis.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `performance` | String | no | `intelligent` (default), `kavier`, `cache`, or a trained model name (`tabpfn`, `catboost`, ...). |
| `energy` | String | no | `kavier_power` (default). |
| `feasibility` | String | no | `autoconf` (default, OOM-aware; see the [feasibility checker](6_feasibility_checker.md)) or `rules` (divisibility only). |
| `lookup` | String | no | Measured-runs DB for `cache`/`intelligent`: a CSV path, or `default` for the bundled `run_database.csv`. |

### 2.4 grid { #config-grid }

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `gpu_models` | List<String> | no | GPU models to sweep; the first entry is the primary GPU. Falls back to `system.default_gpu` when empty. |
| `batch_sizes` | List<Int> | no | Per-device batch sizes to sweep. |
| `total_gpus` | List<Int> | no | GPU counts to sweep. |
| `top_k` | Int | no | Number of ranked recommendations to return. |

## 3. default.yaml { #default-yaml }

The fallback config when the CLI runs without `--config`, and the last candidate in the API config lookup. Same schema as [config.yaml](#config-yaml). The policy is `min_gpu` on purpose: the distinct policy distinguishes the fallback from `experiment.yaml`, and the test suite relies on the distinction.

## 4. experiment.yaml { #experiment-yaml }

The preferred policy config when a run specifies no policy. Same schema as [config.yaml](#config-yaml), extended with a descriptive `system` block (`available_gpu_models`, `max_gpus`, `gpu_memory`, `constraints`) and a legacy `orchestration` block. The CLI parses neither extension; both document the cluster for the API.

## 5. infrastructure.yaml { #infrastructure-yaml }

Cluster caps the API loads and enforces. Edit with care: wrong values change what the API allows.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `total_gpus` | Int | yes | Cluster-wide GPU budget. |
| `max_nodes` | Int | yes | Maximum number of nodes. |
| `max_gpus_per_node` | Int | yes | Maximum GPUs per node. |
| `gpu_models` | List<String> | yes | GPU types physically present in the cluster; the UI picker lists these. |


## 6. batch_config.yaml { #batch-config-yaml }

The config for batch experiments: one recommendation per row of a workload CSV. Same `strategy`/`predictors`/`grid` schema as [config.yaml](#config-yaml), plus an optional `input.columns` remap for arbitrary CSV headers.

```text
# workloads.csv
model_name,method,gpu_model,tokens_per_sample,batch_size
mistral-7b-v0.1,lora,NVIDIA-A100-SXM4-80GB,1024,16
granite-3.3-8b,full,NVIDIA-A100-SXM4-80GB,4096,4
```

```console
coastline recommend-job --config config/batch_config.yaml --input workloads.csv --output recommendations.csv
```

Each row gains the recommended configuration, the predictions, and a rationale:

```text
# recommendations.csv (excerpt)
model_name,...,recommended_total_gpus,recommended_batch_size,predicted_throughput,predicted_power_watts,feasible,rationale
mistral-7b-v0.1,...,8,32,37577.5,220.8,True,"8 GPUs (8×1, batch 32) picked for the best throughput-vs-energy balance, 4% faster than the runner-up (8 GPUs, batch 16)."
```

### 6.1 The input workload CSV { #input-csv }

One workload per row. Coastline accepts the canonical column names and the listed alternate spellings; remap any other header under `input.columns` in the batch config.

| Column | Alternate spellings | Type | Required | Description |
|--------|--------------------|------|----------|-------------|
| `llm_model` | `model_name` | String | yes | LLM to fine-tune. |
| `fine_tuning_method` | `method`, `peft` | String | yes | `full`, `lora`, or `gptq-lora`. |
| `gpu_model` | `gpu` | String | yes | GPU model to seed the sweep. |
| `tokens_per_sample` | `seq_len` | Int | yes | Sequence length. |
| `batch_size` | `batch` | Int | yes | Seed per-device batch size. |
| `number_gpus` | — | Int | no | Seed GPU count. |
| `number_nodes` | — | Int | no | Seed node count. |

A ready-to-run sample ships in `config/coastline_functionality/sample_workloads.csv`:

```text
model_name,method,gpu_model,tokens_per_sample,batch_size
mistral-7b-v0.1,lora,NVIDIA-A100-SXM4-80GB,1024,16
granite-3.3-8b,full,NVIDIA-A100-SXM4-80GB,4096,4
granite-3.1-2b,lora,NVIDIA-A100-SXM4-80GB,1024,16
```

```console
coastline recommend-job --config config/batch_config.yaml --input config/coastline_functionality/sample_workloads.csv --output recommendations.csv
```

## 7. demo.yaml { #demo-yaml }

An annotated reference in `config/user_playground/`: every option with the allowed values, safe to edit. Nothing in the playground is loaded by default, so experiments there never affect a real run.
