# Getting started

In this section, you will learn how to install Coastline and make your first recommendation. 
In this section you will find step-by-step guidelines for using Coastline as a:

1. [Programmatic tool](#1-using-coastline-as-a-programmatic-tool) — install Coastline, make your first recommendation, choose an objective.
2. [Command-line tool](#2-using-coastline-as-a-command-line-tool) — run the fallback config, a declared experiment, or a batch of workloads.
3. [Dashboard](#3-using-coastline-as-a-dashboard) — the GUI and the in-terminal REPL.


## 1. Using Coastline as a programmatic tool

### 1.1 Install Coastline

Install Coastline from PyPI. The default installation contains the recommendation
policies ([min-GPU](...), [multi-objective](...)), the [Kavier simulator](...) for performance and energy
predictions, [IBM AutoConf](...) for feasibility checking, and the [dashboard](...).

```console
python3.13 -m venv .venv    # Coastline requires Python >=3.11
source .venv/bin/activate
pip install coastline-recommender
```

!!! warning
    The package name is `coastline-recommender`; the import name is `coastline`.
    `pip install coastline` installs an unrelated package that also imports as `coastline`.

!!! tip
    Need data-driven simulation models, want to train your own, or want to visualize the cluster under a workload? Install
    the optional extras:

    ```console
    pip install "coastline-recommender[ml]"     # data-driven predictors (TabPFN, XGBoost, ...)
    pip install "coastline-recommender[plot]"   # trace plotting
    ```

For development, work from a checkout (uv-native):

```console
git clone https://github.com/atlarge-research/coastline-recommender
cd coastline-recommender
uv sync
```

### 1.2 Make your first recommendation

Describe the workload through the programmatic interface.

```python
import coastline

my_coastline = coastline(throughput_estim="kavier")

workload = {"llm_model": "mistral-7b-v0.1",
            "fine_tuning_method": "lora",
            "gpu_model": "NVIDIA-A100-SXM4-80GB",
            "tokens_per_sample": 1024,
            "batch_size": 32}

results = my_coastline(workload, total_gpus=[1, 2, 4, 8], preset="balanced")
print(results[0])
```
The output is:
```console
gpus_per_node=2 number_of_nodes=1 total_gpus=2 strategy='multi_objective_balanced' predicted_throughput=7710.76 predicted_runtime_seconds=None metadata={'predicted_power_watts': 223.04, 'combined_score': 0.72, 'rank': 1, 'selection_policy': 'balanced', 'tokens_per_watt': 34.57, 'throughput_score': 0.59, 'power_score': 0.85, 'feasibility': {'Rule-Based Classifier error': '', 'Predictive Model Classifier error': None}, 'batch_size': 64, 'workflow': 'grid_feasibility_simulate_policy', 'preset': 'balanced', 'alpha': 0.5, 'beta': 0.5}
```

Read the recommendation as: 
```
run the job on 2 GPUs 
on one node 
with per-device batch size 64.
This configuration is estimated to have a 
throughput of 7,711  tokens/s 
and a power draw of 223 W.
```

### 1.3 Choose an objective

The `preset` selects the throughput-energy weighting of the [multi-objective recommendation policy](4_recommendation_policies.md):

- `performance` — fastest configuration.
- `energy` — most energy-efficient configuration.
- `balanced` — the default trade-off.

## 2. Using Coastline as a command-line tool

Install the same `coastline-recommender` package ([step 1.1](#11-install-coastline)); the installation provides the `coastline` command.
Learn the batch workflow in [setting up an experiment](3_experiment.md).

The simplest command runs the bundled fallback config and prints the recommendation as JSON:

```console
coastline run
```

The tool will output, for the default configuration:

```json
{
  "timestamp": "2026-07-11T20:17:04.455517",
  "configuration": {
    "total_gpus": 4,
    "gpus_per_node": 4,
    "workers": 1
  },
  "strategy": "min_gpu",
  "performance": {
    "throughput_tokens_per_sec": 1835.89
  },
  "energy": {
    "power_watts": 220.85,
    "efficiency_tokens_per_watt": 8.31
  },
  "metadata": {
    "predicted_power_watts": 220.85,
    "combined_score": 0.25,
    "rank": 1,
    "selection_policy": "min_gpu",
    "tokens_per_watt": 8.31,
    "throughput_score": 0.55,
    "power_score": 0.98,
    "feasibility": {
      "Rule-Based Classifier error": "",
      "Predictive Model Classifier error": null
    },
    "batch_size": 32,
    "workflow": "grid_feasibility_simulate_policy"
  }
}
```

Or, Coastline can run a declared experiment (see [setting up an experiment](3_experiment.md)):

```console
coastline run --config config/coastline_functionality/config.yaml
```
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

Or, the most complex: a batch experiment, one recommendation per row of a workload CSV:

```text
# workloads.csv
model_name,method,gpu_model,tokens_per_sample,batch_size
mistral-7b-v0.1,lora,NVIDIA-A100-SXM4-80GB,1024,16
granite-3.3-8b,full,NVIDIA-A100-SXM4-80GB,4096,4
```

```console
coastline recommend --config config/batch_config.yaml --input workloads.csv --output recommendations.csv
```

```text
# recommendations.csv (excerpt)
model_name,...,recommended_total_gpus,recommended_batch_size,predicted_throughput,feasible,rationale
mistral-7b-v0.1,...,8,32,37577.5,True,"8 GPUs (8×1, batch 32) picked for the best throughput-vs-energy balance, ..."
```

See [batch_config.yaml](3_experiment.md#batch-config-yaml) for the batch config schema.

## 3. Using Coastline as a dashboard

### 3.1 Coastline GUI

Start the dashboard, then open http://127.0.0.1:8000 in the browser:

```console
coastline-ui
```

If the setup went correctly, you should be able to see the following interface:

![The Coastline dashboard](media/web_interface.png)


### 3.2 Coastline REPL (in-terminal interactive interface)

The guided REPL asks for the workload interactively:

```console
coastline interactive
```

If the setup went correctly, you should be able to see the following interface:

![The Coastline REPL](media/terminal_interface.png)




Next, learn how to [set up an experiment](3_experiment.md) or read about the [recommendation policies](4_recommendation_policies.md).
