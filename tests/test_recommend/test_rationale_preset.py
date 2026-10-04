"""The rationale names the preset the strategy ranked with.

The pipeline stores that preset on each pick: the config's preset, the default goal's preset
(performance) when the config sets none, or 'custom' for given weights. The rationale reads it
before the preset the caller passed, so a config without a preset reads as a performance pick.
"""

from __future__ import annotations

import csv
from typing import Any

import pandas as pd
import yaml

import coastline
from coastline.sdk.models.recommendation import Recommendation
from coastline.sdk.recommend import engine

GPU = "NVIDIA-A100-SXM4-80GB"
JOB = {
    "llm_model": "mistral-7b-v0.1",
    "fine_tuning_method": "lora",
    "gpu_model": GPU,
    "tokens_per_sample": 1024,
    "batch_size": 16,
}
GENERIC = "the best throughput/energy trade-off"


def _rec(metadata: dict[str, Any]) -> Recommendation:
    return Recommendation(
        gpus_per_node=8,
        number_of_nodes=1,
        total_gpus=8,
        strategy="multi_objective",
        predicted_throughput=1000.0,
        metadata={"batch_size": 32, **metadata},
    )


def test_the_picks_preset_comes_before_the_callers():
    why = engine.recommendation_rationale(
        [_rec({"preset": "performance"})], {"preset": None, "strategy_name": "multi_objective"}
    )
    assert why == "8 GPUs (8x1, batch 32) picked for the highest throughput."


def test_custom_weights_get_the_generic_phrase_whatever_preset_the_caller_passed():
    # Given weights win over a preset, so the pipeline stores 'custom'.
    why = engine.recommendation_rationale(
        [_rec({"preset": "custom"})], {"preset": "performance", "strategy_name": "multi_objective"}
    )
    assert why.endswith(f"picked for {GENERIC}.")


def test_without_a_preset_on_the_pick_the_callers_preset_is_used():
    why = engine.recommendation_rationale([_rec({})], {"preset": "energy", "strategy_name": "multi_objective"})
    assert why.endswith("picked for the lowest energy.")


def _recommend_csv(tmp_path, strategy: dict) -> dict[str, str]:
    config = {
        "strategy": strategy,
        "predictors": {"performance": "kavier", "energy": "kavier_power", "feasibility": "rules"},
        "grid": {"batch_sizes": [8, 16, 32], "total_gpus": [1, 2, 4, 8]},
    }
    config_path = tmp_path / f"config_{len(list(tmp_path.iterdir()))}.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    jobs = tmp_path / "jobs.csv"
    pd.DataFrame([JOB]).to_csv(jobs, index=False)
    out = tmp_path / "out.csv"
    coastline.recommend_csv(config_path, jobs, out, cluster_gpus=8)
    return next(csv.DictReader(out.open(newline="", encoding="utf-8")))


def test_recommend_csv_without_a_preset_explains_a_performance_pick(tmp_path):
    default = _recommend_csv(tmp_path, {"name": "multi_objective"})
    performance = _recommend_csv(tmp_path, {"name": "multi_objective", "preset": "performance"})

    assert default["feasible"] == "True", default["error"]
    assert default["recommended_total_gpus"] == performance["recommended_total_gpus"]
    assert "picked for the highest throughput" in default["rationale"]
    assert GENERIC not in default["rationale"]


def test_recommend_csv_with_given_weights_keeps_the_generic_phrase(tmp_path):
    row = _recommend_csv(tmp_path, {"name": "multi_objective", "preset": "performance", "alpha": 0.5, "beta": 0.5})
    assert f"picked for {GENERIC}" in row["rationale"]


def test_min_gpu_ignores_a_preset_left_in_the_config():
    """min_gpu picks by GPU count, so a leftover strategy.preset does not set its phrase."""
    line = engine.recommendation_rationale(
        [_rec({"preset": "energy"})], {"strategy_name": "min_gpu", "preset": "energy"}
    )
    assert "the fewest GPUs that fit" in line
    assert "lowest energy" not in line
