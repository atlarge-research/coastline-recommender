"""Phase-5 unification: both recommend surfaces share one goal/predictor vocabulary.

``coastline.recommend(batch, ...) -> DataFrame`` and ``Coastline(...).recommend(wl, ...) -> objects``
now take the same ``goal`` and ``predictor`` words, accept the same workload-alias keys, and reject
the same typos.
"""

from __future__ import annotations

import pandas as pd
import pytest

import coastline
from coastline.sdk.recommend.facade import Coastline

# A minimal in-library workload in the shared alias spelling that BOTH surfaces now accept.
_WL = {
    "model": "mistral-7b-v0.1",
    "method": "lora",
    "gpu_model": "NVIDIA-A100-SXM4-80GB",
    "tokens_per_sample": 1024,
    "batch_size": 32,
}


def _facade() -> Coastline:
    return Coastline(predictor="kavier", feasibility="rules")


def test_predictor_and_throughput_estim_are_the_same_knob():
    # Both spellings must configure the identical estimator. Oracle: the stored key — if the alias
    # silently diverged (e.g. throughput_estim ignored), the three would not all be "kavier".
    assert (
        Coastline(predictor="kavier").throughput_estim
        == Coastline(throughput_estim="Kavier").throughput_estim
        == Coastline("kavier").throughput_estim
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
    # (strategy, preset) is hand-derived from the goal spec here, NOT read from the resolver under
    # test. On the same grid, goal=g must pick exactly what passing that explicit pair picks. If
    # goal_to_strategy_preset mapped a goal to the wrong pair, by_goal would diverge from this
    # independent hand-written reference and the test goes red.
    c = _facade()
    by_goal = [(r.total_gpus, r.metadata["batch_size"]) for r in c.recommend(_WL, goal=goal, max_gpus=16)]
    kw = {"strategy": strategy, "max_gpus": 16}
    if preset is not None:
        kw["preset"] = preset
    by_explicit = [(r.total_gpus, r.metadata["batch_size"]) for r in c.recommend(_WL, **kw)]
    assert by_goal and by_goal == by_explicit


def test_both_surfaces_accept_the_same_workload_alias_keys():
    # The friendly spelling {model, method, gpu_model, ...} must work on BOTH surfaces (the facade
    # dict path used to reject it, requiring llm_model/fine_tuning_method). Oracle: a feasible pick
    # comes back from alias-keyed input on each surface.
    frame = coastline.recommend([dict(_WL)], goal="balanced", predictor="kavier", feasibility="rules")
    objs = _facade().recommend(dict(_WL), goal="balanced")
    assert bool(frame.iloc[0]["feasible"]) and objs


@pytest.mark.parametrize(
    "bad,marker",
    [({"goal": "no-such-goal"}, "unknown goal"), ({"predictor": "gpt5"}, "unknown predictor")],
)
def test_batch_isolates_unknown_goal_or_predictor_as_a_failed_row(bad, marker):
    # The batch surface's contract is per-row isolation: a bad goal/predictor does not crash the
    # call and is never silently defaulted — it yields one feasible=False row carrying the error.
    # Oracle: count 1, the error names the problem, and no config is fabricated for the failed row.
    frame = coastline.recommend([dict(_WL)], feasibility="rules", **bad)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert not bool(row["feasible"])
    assert marker in str(row["error"])
    assert pd.isna(row["total_gpus"])


@pytest.mark.parametrize("bad", [{"goal": "no-such-goal"}, {"predictor": "gpt5"}])
def test_facade_raises_loudly_on_unknown_goal_or_predictor(bad):
    # The single-workload facade has no per-row batch to isolate into, so it fails loudly: the
    # predictor is validated at construction, the goal at call time — both raise, not silently default.
    with pytest.raises(ValueError):
        if "predictor" in bad:
            Coastline(**bad)
        else:
            _facade().recommend(_WL, **bad)
