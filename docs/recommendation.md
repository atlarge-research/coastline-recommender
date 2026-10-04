# How a recommendation is made

The recommendation process begins with the user's input, infrastructure, and LLM specification, and selects a
recommendation policy from the set of supported policies. For the multi-objective policy, Coastline runs a
four-stage pipeline: multi-dimensional exploration, feasibility filtration, simulation of the alternatives, and
ranking the recommendations. The min-GPU policy follows its own algorithm; see [Min-GPU](#min-gpu).

## Input

A workload sets the LLM to fine-tune (`llm_model`), the fine-tuning method (`fine_tuning_method`: `full`, `lora`,
`qlora`, or `gptq-lora`), the GPU model (`gpu_model`), the number of tokens per sample (`tokens_per_sample`), and the
batch size (`batch_size`). The batch size is per device. LLM and GPU names follow the Kavier catalog, e.g.,
`mistral-7b-v0.1` and `NVIDIA-A100-SXM4-80GB`. A workload may also set the job's own layout, `gpus_per_node` and
`number_of_nodes`; only min-GPU reads it.

## 1. Grid

Coastline conducts a two-dimensional grid search over the infrastructure topology: the cartesian product between
the total number of GPUs and the batch size. Configurations larger than the available infrastructure are left out.

## 2. Feasibility

Every configuration is checked for feasibility, and only the valid configurations continue. Feasibility precedes
simulation, since an infeasible configuration needs neither a prediction nor a recommendation.

| `feasibility` | Check |
|---|---|
| `autoconf` (default) | IBM AutoConf determines whether a configuration may fail, e.g., due to lack of memory. Computationally light rules run first. |
| `rules` | Structural sanity guards only, with no out-of-memory check. |
| `none` | No check. |

If AutoConf cannot be loaded, Coastline stops with an error. `COASTLINE_ALLOW_RULES_FALLBACK=1` falls back to
`rules`.

The opt-in empirical guard also rejects a configuration when its per-device batch size times tokens per sample
exceeds the device memory limit determined empirically in the thesis, about 60,000 tokens on A100 SXM4. Enable it
with `empirical_oom_guard=True` in `coastline(...)`, or with `empirical_oom_guard: true` under `predictors` in a
configuration file.

## 3. Prediction

For each feasible configuration, a performance model predicts the training throughput (tokens per second), and
Kavier predicts the power (`energy: kavier_power`).

| `predictor` | Model |
|---|---|
| `kavier` | Kavier, the analytical (physics-driven) model. |
| `cache` | Exact-match retrieval from a database of measured runs. |
| `intelligent` | Cache lookup first; on a cache miss, simulation with the `fallback` model (`kavier` by default). |
| `random_forest`, `xgboost`, `lightgbm`, `catboost`, `bayesian_ridge`, `svr`, `knn`, `gaussian_process`, `deep_learning`, `tabpfn` | Data-driven models; they need the `[ml]` extra. |

The Python API, `simulate`, and `explain` use `kavier` by default; the default configuration file uses
`intelligent`. `cache` and `intelligent` read measured runs from the CSV set by `lookup`, else from
`$DATA_DIR/profiling-dataset/raw_trace.csv`, else from a small bundled sample.

The data-driven models are in `src/coastline/sdk/predictors/performance/data_driven/portfolio/`. The PyPI wheel leaves out `tabpfn`, `random_forest`, `gaussian_process`, `svr`, `knn`, and `custom/`.
Use a clone of the repository for those, or tune `tabpfn` or `xgboost` on your own measured runs with
`coastline utils tune`. TabPFN is orders of magnitude slower than most other models.

## 4. Ranking

The goal selects the recommendation policy. Every entry point uses `performance` when no goal is given.

| `goal` | Policy |
|---|---|
| `performance` (default) | Multi-objective, prioritizing fast execution. |
| `balanced` | Multi-objective, with an equal trade-off between performance and energy. |
| `energy` | Multi-objective, prioritizing low power draw. |
| `min_gpu` | Min-GPU: the smallest feasible GPU allocation for the job's total batch. |

In a configuration file, `strategy.name` (`multi_objective` or `min_gpu`) and `strategy.preset` (`performance`,
`balanced`, or `energy`) select the same policies. Goals and presets also take other names, the same on every entry
point: `energy-saver` for `energy`, `runtime` or `throughput` for `performance`, and `min-gpu` or `fewest` for
`min_gpu`. Letter case does not matter, and `-`, `_` and a space between words read alike.

The multi-objective policy scores every feasible configuration and ranks them according to the user-selected
trade-off between performance and sustainability. A configuration n scores

    S_n = alpha * s_r,n + beta * s_e,n

where s_r,n is its runtime score and s_e,n its energy score, each min-max normalized to [0, 1] over the feasible
configurations, with 1 the best. The runtime score uses 1 / throughput, since every configuration processes the
same work; the energy score uses the cluster power, the per-GPU power times the GPU count. alpha weights
performance (runtime) and beta weights energy:

| Preset | alpha (runtime) | beta (energy) |
|---|---|---|
| `performance` | 0.8 | 0.2 |
| `balanced` | 0.5 | 0.5 |
| `energy` (`energy-saver`) | 0.2 | 0.8 |

`strategy.alpha` and `strategy.beta` in a configuration file, or `alpha=` and `beta=` in `coastline(...).recommend`,
set the weights directly and override the preset. They are divided by their sum.

The `top_k` highest-scoring configurations are returned. When several configurations reach the same top score,
Coastline prefers, in order, the highest predicted throughput, the fewest GPUs, and the smallest batch size. The
optional `max_slowdown` of `coastline.recommend` keeps only the configurations within that factor of the fastest
feasible one.

## Min-GPU

The min-GPU policy follows the min-GPU recommender of IBM AutoConf. It returns the smallest GPU allocation that passes
the feasibility check and keeps the job's total batch: the per-device batch size times the job's GPUs,
`gpus_per_node` x `number_of_nodes`, or 1 GPU when the workload sets no layout. It does not search the grid. For
g = 1, 2, 4, ... while g is at most the available GPUs, it takes the workload with g GPUs in total, the total batch
split evenly over them, and the fewest nodes that hold g GPUs. A g that does not divide the total batch, leaves
less than one sample per device, or needs more nodes than are available is skipped. Each candidate is checked with
the configured check (`autoconf` or `rules`, plus the empirical guard when it is on). The first feasible
configuration is the recommendation. Min-GPU returns one configuration unless `top_k` is set; with `top_k` above 1,
the first `top_k` feasible ones are returned in that order.

For example, take full fine-tuning of `granite-3.1-8b-instruct` with 7621 tokens per sample on
`NVIDIA-A100-SXM4-80GB`, run at 4 samples per device on 8 GPUs, a total batch of 32. AutoConf rejects 1 GPU at 32
and 2 GPUs at 16 per device and admits 4 GPUs at 8, so min-GPU recommends 4 GPUs at 8 per device. The verdicts
depend on the workload: a smaller job may fit on 1 GPU at the same total batch.

No simulation decides the choice. Coastline simulates only the returned configurations, to report their
throughput, runtime, and power; a configuration the predictor cannot predict is left out. If no GPU count is
feasible, or no returned configuration can be predicted, Coastline stops with an error. `max_slowdown` and the
grid's batch sizes and GPU counts do not apply to min-GPU.
