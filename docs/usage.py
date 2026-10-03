"""Show the Coastline Python API on one fine-tuning workload and on a batch.

`import coastline` has three entry points over one engine:

    coastline(predictor=...)       a configured recommender; returns Recommendation objects, best first
    coastline.recommend(batch)     a DataFrame, a list of dicts, or a dict in; a DataFrame out
    coastline.recommend_csv(...)   a CSV of workloads and a config file in; a CSV of recommendations out

A workload sets llm_model, fine_tuning_method, gpu_model, tokens_per_sample, and batch_size
(per device). The goal is balanced, performance, energy, or min_gpu. Here, the three entry points
search the same grid and pick the same configuration. The `coastline` CLI and the `coastline-ui`
dashboard call the same engine.

Run with: python docs/usage.py
"""

import tempfile
from pathlib import Path

import pandas as pd
import yaml

import coastline

job = {
    "llm_model": "mistral-7b-v0.1",
    "fine_tuning_method": "lora",
    "gpu_model": "NVIDIA-A100-SXM4-80GB",
    "tokens_per_sample": 1024,
    "batch_size": 16,
}
batch_sizes = [4, 8, 16, 32]  # per-device batch sizes to search
gpu_counts = [1, 2, 4, 8]  # GPU counts to search

# 1) One workload. Kavier predicts throughput and power; AutoConf checks feasibility (the default).
recommender = coastline(predictor="kavier")
ranked = recommender.recommend(job, goal="balanced", batch_sizes=batch_sizes, total_gpus=gpu_counts)
for rank, rec in enumerate(ranked[:3], start=1):
    print(f"{rank}. {rec.total_gpus} GPUs, batch {rec.metadata['batch_size']}, {rec.predicted_throughput:.0f} tokens/s")

# 2) A batch of workloads: one row per workload, with the chosen configuration and its predictions.
# runtime_s and energy_wh cover dataset_size samples for the given number of epochs.
other = {**job, "llm_model": "granite-3.3-8b", "fine_tuning_method": "full", "tokens_per_sample": 4096, "batch_size": 4}
jobs = pd.DataFrame([job, other])
for goal in ("balanced", "performance", "energy", "min_gpu"):
    picks = coastline.recommend(
        jobs, goal=goal, predictor="kavier", batch_sizes=batch_sizes, max_gpus=8, dataset_size=50_000, epochs=1
    )
    print(f"\n{goal}:")
    print(picks[["llm_model", "total_gpus", "batch_size", "throughput_tok_s", "energy_wh"]].to_string(index=False))

# 3) CSV in, CSV out. The config file sets the policy, the predictors, and the search grid.
tmp = Path(tempfile.mkdtemp())
config = {
    "strategy": {"name": "multi_objective", "preset": "balanced"},
    "predictors": {"performance": "kavier", "energy": "kavier_power", "feasibility": "autoconf"},
    "grid": {"batch_sizes": batch_sizes, "total_gpus": gpu_counts},
}
(tmp / "config.yaml").write_text(yaml.safe_dump(config))
jobs.to_csv(tmp / "jobs.csv", index=False)
coastline.recommend_csv(tmp / "config.yaml", tmp / "jobs.csv", tmp / "recommendations.csv")
print(f"\nrecommend_csv wrote {tmp / 'recommendations.csv'}:")
print(pd.read_csv(tmp / "recommendations.csv")[["llm_model", "recommended_total_gpus", "predicted_throughput"]])
