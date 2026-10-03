"""Input handling of the ``Coastline`` facade (Kavier path, rules feasibility).

Covers preset spelling, GPU budgets that are not a power of two, and the top_k bound.
"""

from __future__ import annotations

import pytest

from coastline import Coastline

_WL = {
    "llm_model": "mistral-7b-v0.1",
    "fine_tuning_method": "lora",
    "gpu_model": "NVIDIA-A100-SXM4-80GB",
    "tokens_per_sample": 1024,
    "batch_size": 32,
}


@pytest.fixture
def rec() -> Coastline:
    return Coastline("kavier", feasibility="rules")


# Preset spelling
@pytest.mark.parametrize("spelled,canonical", [("Performance", "performance"), ("ENERGY", "energy")])
def test_preset_letter_case_does_not_change_the_pick(rec, spelled, canonical):
    kw = dict(total_gpus=[1, 2, 4, 8], batch_sizes=[16, 32], top_k=1)
    got = rec(_WL, preset=spelled, **kw)[0]
    want = rec(_WL, preset=canonical, **kw)[0]
    assert got.strategy == f"multi_objective_{canonical}"
    assert (got.total_gpus, got.metadata["batch_size"]) == (want.total_gpus, want.metadata["batch_size"])
    assert (got.metadata["alpha"], got.metadata["beta"]) == (want.metadata["alpha"], want.metadata["beta"])


def test_rationale_names_the_goal_of_a_mixed_case_preset():
    # recommend_csv passes the preset to the rationale as written in the config, so
    # 'Performance' must get the same phrase as 'performance'.
    from coastline.sdk.recommend import _goals

    assert _goals.rationale_phrase("Performance") == _goals.rationale_phrase("performance")
    assert _goals.rationale_phrase(None) is None


@pytest.mark.parametrize("preset", ["perfomance", "low_energy"])
def test_unknown_preset_raises(rec, preset):
    with pytest.raises(ValueError, match="unknown preset"):
        rec(_WL, preset=preset, total_gpus=[1, 2], batch_sizes=[16], top_k=1)


# GPU budgets that are not a power of two
def test_an_exact_twelve_gpu_budget_is_reachable(rec):
    # 12 GPUs at up to 8 per node is 6 x 2. Rounding up to 8 x 2 = 16 would exceed the
    # 12-GPU budget and leave the grid empty.
    out = rec(_WL, total_gpus=[12], batch_sizes=[32], max_gpus=12)
    assert [(r.total_gpus, r.gpus_per_node, r.number_of_nodes) for r in out] == [(12, 6, 2)]


def test_twelve_and_sixteen_gpus_are_two_distinct_recommendations(rec):
    out = rec(_WL, total_gpus=[12, 16], batch_sizes=[32], top_k=5)
    layouts = [(r.total_gpus, r.gpus_per_node, r.number_of_nodes, r.metadata["batch_size"]) for r in out]
    assert sorted(layouts) == [(12, 6, 2, 32), (16, 8, 2, 32)]


def test_a_repeated_grid_entry_is_recommended_once(rec):
    out = rec(_WL, total_gpus=[8, 8, 4], batch_sizes=[32, 32], top_k=10, max_gpus=8)
    layouts = [(r.gpus_per_node, r.number_of_nodes, r.metadata["batch_size"]) for r in out]
    assert sorted(layouts) == [(4, 1, 32), (8, 1, 32)]


# top_k bound
@pytest.mark.parametrize("top_k", [0, -3])
def test_top_k_below_one_raises(rec, top_k):
    with pytest.raises(ValueError, match="top_k"):
        rec(_WL, total_gpus=[1, 2], batch_sizes=[16], top_k=top_k)
