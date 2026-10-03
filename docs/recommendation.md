# How a recommendation is made

The recommendation process begins with the user's input, infrastructure, and LLM specification, and selects a
recommendation policy from the set of supported policies. Coastline then runs a four-stage pipeline:
multi-dimensional exploration, feasibility filtration, simulation of the alternatives, and ranking the
recommendations.

## Input

A workload sets the LLM to fine-tune (`llm_model`), the fine-tuning method (`fine_tuning_method`: `full`, `lora`,
`qlora`, or `gptq-lora`), the GPU model (`gpu_model`), the number of tokens per sample (`tokens_per_sample`), and the
batch size (`batch_size`). The batch size is per device. LLM and GPU names follow the Kavier catalog, e.g.,
`mistral-7b-v0.1` and `NVIDIA-A100-SXM4-80GB`.

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

The data-driven models are in `src/coastline/sdk/predictors/performance/data_driven/portfolio/`, the large ones
via Git LFS. The PyPI wheel leaves out `tabpfn`, `random_forest`, `gaussian_process`, `svr`, `knn`, and `custom/`.
Use a clone of the repository for those, or tune `tabpfn` or `xgboost` on your own measured runs with
`coastline utils tune`. TabPFN is orders of magnitude slower than most other models.

## 4. Ranking

The goal selects the recommendation policy.

| `goal` | Policy |
|---|---|
| `balanced` | Multi-objective, with an equal trade-off between performance and energy. |
| `performance` | Multi-objective, prioritizing fast execution. |
| `energy` | Multi-objective, prioritizing low power draw. |
| `min_gpu` | Min-GPU: the smallest feasible GPU allocation. |

In a configuration file, `strategy.name` (`multi_objective` or `min_gpu`) and `strategy.preset` (`balanced`,
`performance`, or `energy`) select the same policies.

The multi-objective policy scores every feasible configuration and ranks them according to the user-selected
trade-off between performance and sustainability. The `top_k` highest-scoring configurations are returned. When
several configurations reach the same top score, Coastline prefers, in order, the highest predicted throughput,
the fewest GPUs, and the smallest batch size. The optional `max_slowdown` of `coastline.recommend` keeps only the
configurations within that factor of the fastest feasible one.
