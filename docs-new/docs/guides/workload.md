# Recommend a workload

The single-workload API: configure an advisor once, call it per workload.

```python
import coastline

advisor = coastline(predictor="kavier")     # or "intelligent", "cache", "tabpfn", …

results = advisor.recommend(
    {"model": "mistral-7b-v0.1", "method": "lora",
     "gpu_model": "NVIDIA-A100-SXM4-80GB", "tokens_per_sample": 2048, "batch_size": 8},
    goal="balanced",
    top_k=5,
    max_gpus=16,
)
```

`results` is a list of `Recommendation` objects, best first:

```python
best = results[0]
best.total_gpus                      # e.g. 4
best.gpus_per_node                   # e.g. 4
best.number_of_nodes                 # e.g. 1
best.predicted_throughput            # tokens/sec
best.predicted_runtime_seconds
best.metadata["batch_size"]
best.metadata["predicted_power_watts"]
best.metadata["tokens_per_watt"]
```

## Express your intent

`goal` is the one-knob shortcut. For finer control, replace it with an explicit strategy:

```python
# preset weighting
advisor.recommend(workload, strategy="multi_objective", preset="energy")

# manual weights: alpha = energy, beta = throughput
advisor.recommend(workload, alpha=0.3, beta=0.7)

# fewest GPUs that fit
advisor.recommend(workload, goal="min_gpu")
```

See [Ranking strategies](../concepts/strategies.md) for how presets and weights rank candidates.

## Shape the search space

By default Coastline sweeps `total_gpus=[1, 2, 4, 8, 16]` × `batch_sizes=[4, 8, 16, 32, 64]`.
Override either grid:

```python
advisor.recommend(workload, total_gpus=[1, 2, 4], batch_sizes=[8, 16], top_k=3)
```

!!! tip
    `advisor(workload)` is shorthand for `advisor.recommend(workload)` — the advisor is callable.
