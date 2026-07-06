"""Coastline — usage examples for the Python API.

`import coastline` is the in-process facade. Two ways in, one engine:

    coastline.recommend(batch)                     -> a pandas DataFrame   (bulk / CSV workflows)
    coastline(predictor=...).recommend(workload)   -> list[Recommendation] (one workload, typed)

Both take the same `goal` ("balanced" | "performance" | "energy" | "min_gpu") and `predictor`
words, and the numbers match the `coastline` CLI and the dashboard — all three call this engine.

Run with:  uv run python docs/usage.py
"""

import tempfile
from pathlib import Path

import coastline

GPU = "NVIDIA-A100-SXM4-80GB"
WORKLOAD = {"model": "mistral-7b-v0.1", "method": "lora", "gpu_model": GPU, "tokens_per_sample": 1024, "batch_size": 32}

# feasibility="rules" keeps these examples runnable without the optional AutoConf OOM checker.
advisor = coastline(predictor="kavier", feasibility="rules")

# --------------------------------------------------------------------------- #
# 1) One workload -> a ranked shortlist of configs (typed Recommendation objects)
# --------------------------------------------------------------------------- #
# More GPUs isn't always better: past a point communication overhead eats the gains, so the
# fastest pick is usually NOT the largest GPU count in the budget.
best = advisor.recommend(WORKLOAD, goal="performance", max_gpus=16)[0]
print("1) fastest config for mistral-7b LoRA on A100-80GB:")
print(f"   -> {best.total_gpus} GPUs, batch {best.metadata['batch_size']}, {best.predicted_throughput:,.0f} tok/s")

# --------------------------------------------------------------------------- #
# 2) The throughput <-> energy trade-off: same workload, three objectives
# --------------------------------------------------------------------------- #
print("\n2) same workload, different goal:")
for goal in ("performance", "balanced", "energy"):
    r = advisor.recommend(WORKLOAD, goal=goal, max_gpus=16)[0]
    total_power = r.metadata["predicted_power_watts"] * r.total_gpus
    print(f"   {goal:12s} -> {r.total_gpus} GPUs, {r.predicted_throughput:,.0f} tok/s, {total_power:,.0f} W")

# --------------------------------------------------------------------------- #
# 3) A batch of workloads -> a DataFrame (rows in, config + predictions out)
# --------------------------------------------------------------------------- #
frame = coastline.recommend(
    [
        {**WORKLOAD, "model": "granite-3.1-2b"},
        {**WORKLOAD, "model": "llama3.2-3b", "tokens_per_sample": 2048},
    ],
    goal="balanced",
    predictor="kavier",
    feasibility="rules",
)
print("\n3) batch -> DataFrame columns:", list(frame.columns))
print(frame[["model", "total_gpus", "throughput_tok_s", "energy_wh"]].to_string(index=False))

# --------------------------------------------------------------------------- #
# 4) A CSV of workloads -> a CSV of recommendations (the production batch path)
# --------------------------------------------------------------------------- #
docs = Path(__file__).resolve().parent
config = docs.parent / "config" / "batch_config.yaml"
out = Path(tempfile.mkdtemp()) / "recommendations.csv"
coastline.recommend_csv(config, docs / "workloads.csv", out)
print(f"\n4) recommend_csv wrote {sum(1 for _ in out.open()) - 1} recommendation rows to {out.name}")
