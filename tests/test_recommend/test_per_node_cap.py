"""The per-node GPU cap reaches the grid through ``coastline.recommend`` and the engine answers.

A cluster of 4-GPU nodes is never offered an 8-GPU node. Without a cap the engine uses 8 GPUs per
node. The runs use Kavier and the rules checker and check layout bounds only.
"""

from __future__ import annotations

import pytest

import coastline
from coastline.sdk.constants import DEFAULT_GPUS_PER_NODE
from coastline.sdk.recommend import engine

_ROW = {
    "llm_model": "mistral-7b-v0.1",
    "fine_tuning_method": "lora",
    "gpu_model": "NVIDIA-A100-SXM4-80GB",
    "tokens_per_sample": 1024,
    "batch_size": 16,
}


def _answers(**extra) -> dict:
    return {**engine.defaults(engine.resolve_options()), **_ROW, "max_gpus": 16, **extra}


def test_the_context_takes_the_per_node_cap() -> None:
    assert engine.build_context(_answers(max_gpus_per_node=4)).constraints.gpus_per_node == 4


def test_without_a_cap_the_context_keeps_the_default_node_width() -> None:
    assert engine.build_context(_answers()).constraints.gpus_per_node == DEFAULT_GPUS_PER_NODE


def test_the_cap_never_exceeds_the_gpu_budget() -> None:
    assert engine.build_context(_answers(max_gpus=2, max_gpus_per_node=4)).constraints.gpus_per_node == 2


def test_recommend_never_offers_a_node_wider_than_the_cap() -> None:
    df = coastline.recommend(
        [_ROW], predictor="kavier", feasibility="rules", goal="performance", max_gpus=16, max_gpus_per_node=4, top_k=10
    )

    assert df["feasible"].all()
    assert (df["gpus_per_node"] <= 4).all()
    assert (df["gpus_per_node"] * df["number_of_nodes"] == df["total_gpus"]).all()
    assert (df["total_gpus"] <= 16).all()


@pytest.mark.parametrize("cap", [0, -4])
def test_a_cap_below_one_gpu_fails_the_call(cap) -> None:
    with pytest.raises(ValueError, match="max_gpus_per_node"):
        coastline.recommend([_ROW], predictor="kavier", feasibility="rules", max_gpus_per_node=cap)
