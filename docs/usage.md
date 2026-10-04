# Usage

The CLI, the dashboard, and the Python API connect to one core engine.

## CLI

The `coastline` command has five subcommands. Each documents its flags with `--help`.

| Subcommand | Use |
|---|---|
| `recommend-job` | Search and rank the best configurations for one job, or for a CSV of jobs. |
| `recommend-trace` | Recommend a configuration for every job in a fine-tuning trace. |
| `simulate` | Predict the throughput and power of one configuration. |
| `explain` | Explain a recommendation: the score breakdown and the feasibility verdict. |
| `utils` | `tune` a data-driven predictor on your own runs, `trace-to-runs`, and `plot-trace`. |

Simulate one configuration, then explain the recommendation for the same job:

```bash
coastline simulate --model mistral-7b-v0.1 --method lora --gpu-model NVIDIA-A100-SXM4-80GB \
  --tokens 2048 --batch-size 8 --gpus-per-node 4 --nodes 1
coastline explain --model mistral-7b-v0.1 --method lora --gpu-model NVIDIA-A100-SXM4-80GB \
  --tokens 2048 --batch-size 8
```

`--predictor` selects the throughput model and `--feasibility` the feasibility check; see
[How a recommendation is made](recommendation.md).

### One job

`recommend-job --config` reads the workload, the policy, the predictors, and the search grid from a configuration
file, and returns the best configuration as JSON:

```yaml
workload:
  llm_model: mistral-7b-v0.1
  fine_tuning_method: lora          # full | lora | qlora | gptq-lora
  tokens_per_sample: 2048           # sequence length
  batch_size: 8                     # per device
strategy:
  name: multi_objective             # or min_gpu, which uses no grid
  preset: performance               # performance (default) | balanced | energy (or energy-saver)
predictors:
  performance: kavier               # kavier | cache | intelligent | a data-driven model
  energy: kavier_power
  feasibility: autoconf             # autoconf | rules | none
grid:
  gpu_models: [NVIDIA-A100-SXM4-80GB]
  batch_sizes: [4, 8, 16, 32, 64]   # batch sizes to search
  total_gpus: [1, 2, 4, 8, 16, 32]  # GPU counts to search
```

```bash
coastline recommend-job --config experiment.yaml --output-dir runs/exp-01
```

The recommendation goes to `runs/exp-01/recommendation.json`; without `--output-dir` it is printed. In a clone of
the repository, `coastline recommend-job` with no flags runs the job declared in
`config/coastline_functionality/experiment.yaml`. `coastline recommend-job --interactive` starts a guided REPL.

### Many jobs

From a clone, a CSV of workloads goes in and a CSV with one recommendation per row comes out:

```bash
coastline recommend-job --config config/batch_config.yaml \
  --input config/coastline_functionality/sample_workloads.csv --output recommendations.csv
coastline recommend-trace --input config/coastline_functionality/sample_trace.csv --output recommended_trace.csv
```

The workloads CSV has the columns `llm_model`, `fine_tuning_method`, `gpu_model`, `tokens_per_sample`, and
`batch_size`, and optionally the job's layout, `gpus_per_node` and `number_of_nodes`. `recommend-trace` reads a
fine-tuning trace with the columns of `sample_trace.csv`, where `metadata.batch_size` is each job's total batch; a
row whose total does not split evenly over its GPUs is kept unchanged under the weighted goals. `--goal` sets the
goal (default `performance`; `min_gpu` keeps each job's total batch), and `--visual` also draws the cluster
timeline (needs the `[plot]` extra).

### Infrastructure

In a clone, `config/coastline_functionality/infrastructure.yaml` declares the available infrastructure: 32
NVIDIA-A100-SXM4-80GB GPUs, distributed equally across 4 compute nodes. Without that file, Coastline assumes 64
GPUs, 8 per node. `--cluster-gpus` overrides the total, and the `INFRASTRUCTURE_CONFIG` and `EXPERIMENT_CONFIG`
environment variables point to other files.

## Dashboard

```bash
coastline-ui
```

The dashboard runs at <http://127.0.0.1:8000>; `COASTLINE_UI_HOST` and `COASTLINE_UI_PORT` change the address. A
user defines a fine-tuning workload, requests a recommendation, and appends the job to the workload queue. In admin
mode, the sysadmin imports a CSV of jobs and runs the queue on the cluster. The playground compares the predictors
on one configuration, through simulation-based or cache-retrieval what-if analysis.

## Python API

A runnable script: `python docs/usage.py`.

```python
--8<-- "docs/usage.py"
```
