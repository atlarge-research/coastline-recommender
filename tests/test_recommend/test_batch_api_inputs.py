"""Input handling of the batch API ``coastline.recommend`` (Kavier path, rules feasibility).

Covers predictor names, fractional epochs, the recommended batch column, the predictor's
failure reason, and the max_slowdown and top_k bounds. Expected values follow from
total_tokens = dataset_size x epochs x tokens_per_sample and runtime_s = total_tokens / throughput.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

import coastline
from coastline.sdk.policies import PolicyFactory
from coastline.sdk.recommend import batch_api as api
from coastline.sdk.recommend import engine

_ROW = {
    "llm_model": "mistral-7b-v0.1",
    "fine_tuning_method": "lora",
    "gpu_model": "NVIDIA-A100-SXM4-80GB",
    "tokens_per_sample": 1024,
    "batch_size": 16,
}


def _recommend(batch, **kw):
    kw.setdefault("predictor", "kavier")
    kw.setdefault("feasibility", "rules")
    kw.setdefault("max_gpus", 8)
    return coastline.recommend(batch, **kw)


# Predictor name: the normalised key is the one that runs
def test_answers_carry_the_normalised_predictor_key():
    base = engine.defaults(engine.resolve_options())
    answers, _ = api._answers_for(dict(_ROW), {"predictor": "XGBoost", "goal": "balanced"}, base)
    assert answers["predictor"] == "xgboost"


def test_a_mixed_case_predictor_runs_that_predictor(monkeypatch):
    # 'Kavier' must build KavierPredictor, as 'kavier' does. Record the class that gets built.
    built = []
    original = PolicyFactory.throughput_predictor

    def spy(predictor_config):
        predictor = original(predictor_config)
        built.append(type(predictor).__name__)
        return predictor

    monkeypatch.setattr(PolicyFactory, "throughput_predictor", staticmethod(spy))
    df = _recommend(_ROW, predictor="Kavier")
    assert bool(df.iloc[0]["feasible"])
    assert built == ["KavierPredictor"]


# Fractional epochs
@pytest.mark.parametrize("epochs", [0.5, 2.5, "1.5"])
def test_fractional_epochs_scale_the_runtime(epochs):
    df = _recommend({**_ROW, "dataset_size": 1_000_000, "epochs": epochs}, top_k=1)
    row = df.iloc[0]
    assert bool(row["feasible"])
    total_tokens = 1_000_000 * float(epochs) * 1024
    assert row["runtime_s"] is not None
    assert row["runtime_s"] * row["throughput_tok_s"] == pytest.approx(total_tokens)
    assert row["energy_wh"] == pytest.approx(row["power_w"] * row["total_gpus"] * row["runtime_s"] / 3600.0)


def test_epochs_kwarg_accepts_a_fraction():
    whole = _recommend(_ROW, dataset_size=100_000, epochs=1).iloc[0]
    half = _recommend(_ROW, dataset_size=100_000, epochs=0.5).iloc[0]
    assert half["runtime_s"] == pytest.approx(whole["runtime_s"] / 2)


@pytest.mark.parametrize("field,value", [("epochs", 0), ("epochs", -1), ("dataset_size", 0), ("dataset_size", -5)])
def test_non_positive_epochs_or_dataset_size_fail_the_row(field, value):
    # A zero or negative token count has no meaningful runtime, so the row fails.
    df = _recommend([{**_ROW, field: value}, dict(_ROW)])
    assert df["feasible"].tolist() == [False, True]
    assert field in str(df.iloc[0]["error"])


def test_an_infinite_epochs_value_fails_the_row():
    df = _recommend({**_ROW, "epochs": math.inf})
    assert not bool(df.iloc[0]["feasible"])
    assert "epochs" in str(df.iloc[0]["error"])


def test_a_non_positive_epochs_kwarg_fails_every_row():
    df = _recommend([dict(_ROW), {**_ROW, "tokens_per_sample": 2048}], epochs=0)
    assert not df["feasible"].any()
    assert all("epochs" in str(e) for e in df["error"])


# Recommended batch in its own column
def test_recommended_batch_size_column_holds_the_recommendation():
    rows = [{**_ROW, "batch_size": 8}, {**_ROW, "batch_size": 16}, {**_ROW, "gpu_model": "H200", "batch_size": 4}]
    df = _recommend(rows, top_k=2)
    assert df["feasible"].tolist() == [True, True, True, True, False]
    # The batch API searches the seed batch and its neighbours (half and double), so each
    # recommendation comes from its own input's set; the failed row has no recommendation.
    seeds_to_grid = {0: {4, 8, 16}, 1: {4, 8, 16}, 2: {8, 16, 32}, 3: {8, 16, 32}}
    for position, grid in seeds_to_grid.items():
        assert int(df.iloc[position]["recommended_batch_size"]) in grid
    assert pd.isna(df.iloc[4]["recommended_batch_size"])


def test_recommended_batch_size_matches_the_ranked_recommendation():
    df = _recommend(_ROW, top_k=3)
    recs = coastline(predictor="kavier", feasibility="rules").recommend(
        _ROW, total_gpus=[1, 2, 4, 8], batch_sizes=[8, 16, 32], top_k=3, max_gpus=8
    )
    assert df["recommended_batch_size"].tolist() == [r.metadata["batch_size"] for r in recs]


def test_empty_batch_schema_includes_the_recommended_batch_column():
    assert "recommended_batch_size" in _recommend([]).columns


# A predictor failure carries the predictor's reason
def test_predictor_failure_reports_the_predictor_reason():
    df = _recommend({**_ROW, "fine_tuning_method": "not-a-method"})
    row = df.iloc[0]
    assert not bool(row["feasible"])
    error = str(row["error"])
    assert "no feasible candidates" not in error
    assert "kavier" in error
    assert "not-a-method" in error


# max_slowdown and top_k bounds
@pytest.mark.parametrize("max_slowdown", [0, 0.5, -1, float("inf")])
def test_a_max_slowdown_that_cannot_hold_fails_the_row(max_slowdown):
    # k < 1 can never be met and a non-finite k is no bound, so the row fails with an error
    # that names max_slowdown.
    df = _recommend(_ROW, top_k=20, max_slowdown=max_slowdown)
    assert len(df) == 1
    assert not bool(df.iloc[0]["feasible"])
    assert "max_slowdown" in str(df.iloc[0]["error"])


def test_a_per_row_max_slowdown_is_checked_per_row():
    df = _recommend([{**_ROW, "max_slowdown": 0.2}, {**_ROW, "max_slowdown": 1.0}])
    assert df["feasible"].tolist() == [False, True]
    assert "max_slowdown" in str(df.iloc[0]["error"])


def test_a_max_slowdown_of_nan_in_a_frame_means_no_guard():
    # A NaN cell is a missing value in a DataFrame, so that row runs without the guard.
    frame = pd.DataFrame([{**_ROW, "max_slowdown": float("nan")}, {**_ROW, "max_slowdown": 1.0}])
    df = _recommend(frame, top_k=20)
    assert df["feasible"].all()
    assert len(df) > 2


def test_a_max_slowdown_kwarg_of_nan_fails_the_row():
    df = _recommend(_ROW, max_slowdown=float("nan"))
    assert not bool(df.iloc[0]["feasible"])
    assert "max_slowdown" in str(df.iloc[0]["error"])


@pytest.mark.parametrize("top_k", [0, -1, "0"])
def test_top_k_below_one_raises(top_k):
    with pytest.raises(ValueError, match="top_k"):
        _recommend(_ROW, top_k=top_k)


def test_top_k_given_as_text_is_still_read_as_a_number():
    # The grid reads top_k with int(), so a numeric string is accepted.
    assert len(_recommend(_ROW, top_k="2")) == 2


def test_a_cache_miss_says_no_measured_run_matches():
    """A job with no matching run in the lookup database fails with an error that says so."""
    df = coastline.recommend(
        [
            {
                "llm_model": "mistral-7b-v0.1",
                "fine_tuning_method": "lora",
                "gpu_model": "NVIDIA-A100-SXM4-80GB",
                "tokens_per_sample": 1024,
                "batch_size": 16,
            }
        ],
        predictor="cache",
        lookup="default",
        feasibility="rules",
    )
    assert not df["feasible"].iloc[0]
    assert "no measured run in the lookup database matches" in df["error"].iloc[0]
