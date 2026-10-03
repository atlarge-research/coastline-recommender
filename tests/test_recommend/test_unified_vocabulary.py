"""Both recommend entry points share one goal and predictor vocabulary.

``coastline.recommend(batch, ...)``, which returns a DataFrame, and ``Coastline(...).recommend(wl, ...)``,
which returns objects, take the same ``goal`` and ``predictor`` names, accept the WorkloadSpec field
names as keys and reject the same typos.
"""

from __future__ import annotations

import pandas as pd
import pytest

import coastline
from coastline.sdk.recommend.facade import Coastline

# A known workload, keyed by WorkloadSpec field names.
_WL = {
    "llm_model": "mistral-7b-v0.1",
    "fine_tuning_method": "lora",
    "gpu_model": "NVIDIA-A100-SXM4-80GB",
    "tokens_per_sample": 1024,
    "batch_size": 32,
}


def _facade() -> Coastline:
    return Coastline(predictor="kavier", feasibility="rules")


def test_predictor_is_one_normalized_knob():
    # Positional or keyword, in any letter case, the predictor resolves to "kavier".
    assert (
        Coastline(predictor="kavier").predictor
        == Coastline(predictor="Kavier").predictor
        == Coastline("kavier").predictor
        == "kavier"
    )


@pytest.mark.parametrize(
    "goal,strategy,preset",
    [
        ("balanced", "multi_objective", "balanced"),
        ("performance", "multi_objective", "performance"),
        ("energy", "multi_objective", "energy"),
        ("min_gpu", "min_gpu", None),
    ],
)
def test_goal_is_pure_sugar_for_explicit_strategy_preset(goal, strategy, preset):
    # The (strategy, preset) pairs are written out here. On the same grid, goal=g picks what
    # its pair picks.
    c = _facade()
    by_goal = [(r.total_gpus, r.metadata["batch_size"]) for r in c.recommend(_WL, goal=goal, max_gpus=16)]
    kw = {"strategy": strategy, "max_gpus": 16}
    if preset is not None:
        kw["preset"] = preset
    by_explicit = [(r.total_gpus, r.metadata["batch_size"]) for r in c.recommend(_WL, **kw)]
    assert by_goal and by_goal == by_explicit


def test_both_surfaces_accept_the_workloadspec_field_names():
    # Input keyed by WorkloadSpec field names gives a feasible pick on both entry points.
    frame = coastline.recommend([dict(_WL)], goal="balanced", predictor="kavier", feasibility="rules")
    objs = _facade().recommend(dict(_WL), goal="balanced")
    assert bool(frame.iloc[0]["feasible"]) and objs


@pytest.mark.parametrize(
    "bad,marker",
    [({"goal": "no-such-goal"}, "unknown goal"), ({"predictor": "gpt5"}, "unknown predictor")],
)
def test_batch_isolates_unknown_goal_or_predictor_as_a_failed_row(bad, marker):
    # The batch API fails the row: one feasible=False row, with an error naming the problem
    # and no config.
    frame = coastline.recommend([dict(_WL)], feasibility="rules", **bad)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert not bool(row["feasible"])
    assert marker in str(row["error"])
    assert pd.isna(row["total_gpus"])


@pytest.mark.parametrize("bad", [{"goal": "no-such-goal"}, {"predictor": "gpt5"}])
def test_facade_raises_on_unknown_goal_or_predictor(bad):
    # The facade raises: the predictor is checked at construction, the goal at call time.
    with pytest.raises(ValueError):
        if "predictor" in bad:
            Coastline(**bad)
        else:
            _facade().recommend(_WL, **bad)
