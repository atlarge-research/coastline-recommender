"""Tests for the public batch API ``coastline.recommend``.

Every test uses the analytical ``kavier`` predictor, so no trained ML model is unpickled
(unpickling xgboost or AutoGluon models in the test process can segfault). Expected values
follow from:
  * total_tokens = dataset_size * epochs * tokens_per_sample (defaults 50_000 and 1)
  * runtime_s = total_tokens / throughput
  * energy_wh = power_per_gpu * total_gpus * runtime_s / 3600, energy_kwh = energy_wh / 1000
  * tokens_per_watt = throughput / power_per_gpu
  * gpus_per_node = min(8, total_gpus), number_of_nodes = ceil(total_gpus / 8)
  * the goal presets weight power/throughput 0.8/0.2 (energy) and 0.2/0.8 (performance)
"""

import math

import pandas as pd
import pytest

import coastline
from coastline.sdk.recommend import batch_api as api

# A minimal workload row; the rest of the schema falls back to engine defaults.
_ROW = {
    "llm_model": "mistral-7b-v0.1",
    "fine_tuning_method": "lora",
    "gpu_model": "NVIDIA-A100-SXM4-80GB",
    "tokens_per_sample": 1024,
    "batch_size": 16,
}

# Columns every successful output row carries (the input echoed plus the recommendation).
_RECO_COLUMNS = {
    "rank",
    "total_gpus",
    "gpus_per_node",
    "number_of_nodes",
    "batch_size",
    "throughput_tok_s",
    "runtime_s",
    "energy_wh",
    "energy_kwh",
    "feasible",
}

# engine.defaults() values that enter the token count.
_DATASET_SIZE = 50_000
_EPOCHS = 1


def test_batch_returns_one_ranked_row_per_input_with_derived_runtime():
    # Two rows that differ only in tokens_per_sample. top_k defaults to 1, so each input gives
    # one rank-1 row.
    batch = pd.DataFrame([_ROW, {**_ROW, "tokens_per_sample": 2048}])
    df = coastline.recommend(batch, predictor="kavier", max_gpus=8)
    assert len(df) == 2  # one recommendation per input row
    assert _RECO_COLUMNS <= set(df.columns)
    assert list(df["rank"]) == [1, 1]
    assert bool(df["feasible"].all())

    for i, tokens in enumerate((1024, 2048)):
        row = df.iloc[i]
        total_tokens = _DATASET_SIZE * _EPOCHS * tokens  # 51_200_000 and 102_400_000
        # runtime_s * throughput gives back the token count.
        assert row["runtime_s"] * row["throughput_tok_s"] == pytest.approx(total_tokens)
    # Twice the tokens takes longer. The GPU pick may differ, so only the order is checked.
    assert df.iloc[1]["runtime_s"] > df.iloc[0]["runtime_s"]


def test_energy_and_efficiency_columns_are_mutually_consistent():
    # Each energy column follows from the other output columns. power_w is per GPU, so energy
    # scales with total_gpus.
    df = coastline.recommend(_ROW, predictor="kavier", top_k=3, max_gpus=8)
    for _, row in df.iterrows():
        # energy_wh = power_per_gpu * total_gpus * runtime_s / 3600
        assert row["energy_wh"] == pytest.approx(row["power_w"] * row["total_gpus"] * row["runtime_s"] / 3600.0)
        # kWh = Wh / 1000
        assert row["energy_kwh"] == pytest.approx(row["energy_wh"] / 1000.0)
        # tokens_per_watt = throughput / power_per_gpu
        assert row["tokens_per_watt"] == pytest.approx(row["throughput_tok_s"] / row["power_w"])


def test_node_layout_packs_8_per_node_16_gpus_is_two_nodes():
    # Up to 8 GPUs per node, so 16 GPUs take 2 nodes. goal=performance makes the ranking reach
    # 16 GPUs.
    df = coastline.recommend(_ROW, predictor="kavier", goal="performance", top_k=12, max_gpus=16)
    for _, row in df.iterrows():
        g = int(row["total_gpus"])
        assert row["gpus_per_node"] == min(8, g)  # up to 8 per node
        assert row["number_of_nodes"] == math.ceil(g / 8)  # fewest nodes
        assert row["gpus_per_node"] * row["number_of_nodes"] >= g
    sixteen = df[df["total_gpus"] == 16]
    assert not sixteen.empty  # the max_gpus=16 grid includes a 16-GPU candidate
    assert int(sixteen.iloc[0]["gpus_per_node"]) == 8
    assert int(sixteen.iloc[0]["number_of_nodes"]) == 2  # 16 / 8 = 2 nodes


def test_single_dict_list_and_dataframe_inputs_produce_identical_output():
    # A dict, a list with one dict and a one-row DataFrame give the same output.
    as_dict = coastline.recommend(_ROW, predictor="kavier", max_gpus=8)
    as_list = coastline.recommend([_ROW], predictor="kavier", max_gpus=8)
    as_df = coastline.recommend(pd.DataFrame([_ROW]), predictor="kavier", max_gpus=8)
    assert len(as_dict) == len(as_list) == len(as_df) == 1
    pd.testing.assert_frame_equal(as_dict, as_list)
    pd.testing.assert_frame_equal(as_dict, as_df)


def test_omitted_optional_method_matches_explicit_default_method():
    # Leaving out fine_tuning_method uses the engine default (lora) and gives the same
    # recommendation as passing lora.
    core = {
        "llm_model": "mistral-7b-v0.1",
        "gpu_model": "NVIDIA-A100-SXM4-80GB",
        "tokens_per_sample": 1024,
        "batch_size": 16,
    }
    omitted = coastline.recommend(core, predictor="kavier", max_gpus=8)
    explicit = coastline.recommend({**core, "fine_tuning_method": "lora"}, predictor="kavier", max_gpus=8)
    assert bool(omitted.iloc[0]["feasible"])
    # The recommendation columns match; the echoed input columns differ.
    for col in ("total_gpus", "throughput_tok_s", "energy_wh", "runtime_s"):
        assert omitted.iloc[0][col] == explicit.iloc[0][col]


def test_missing_required_core_field_is_feasible_false_not_defaulted():
    # A row without the required fields comes back feasible=False with 'missing required
    # field'. The engine defaults (A100 / 1024 / 32) are not used for required fields.
    only_model = coastline.recommend({"llm_model": "mistral-7b-v0.1"}, predictor="kavier", max_gpus=8)
    assert len(only_model) == 1
    assert not bool(only_model.iloc[0]["feasible"])
    err = only_model.iloc[0]["error"]
    assert isinstance(err, str) and "missing required field" in err
    # no recommendation column is filled
    assert only_model.iloc[0]["total_gpus"] is None
    assert only_model.iloc[0]["throughput_tok_s"] is None

    # The same model with all required fields is feasible.
    complete = coastline.recommend(
        {
            "llm_model": "mistral-7b-v0.1",
            "gpu_model": "NVIDIA-A100-SXM4-80GB",
            "tokens_per_sample": 1024,
            "batch_size": 16,
        },
        predictor="kavier",
        max_gpus=8,
    )
    assert bool(complete.iloc[0]["feasible"]) and complete.iloc[0]["throughput_tok_s"] > 0


@pytest.mark.parametrize("field", ["gpu_model", "tokens_per_sample", "batch_size"])
def test_missing_required_field_names_the_specific_absent_column(field):
    # The error names the missing field (the first one, if several are missing).
    full = {
        "llm_model": "mistral-7b-v0.1",
        "gpu_model": "NVIDIA-A100-SXM4-80GB",
        "tokens_per_sample": 1024,
        "batch_size": 16,
    }
    row = {k: v for k, v in full.items() if k != field}
    df = coastline.recommend(row, predictor="kavier", max_gpus=8)
    assert not bool(df.iloc[0]["feasible"])
    assert df.iloc[0]["error"] == f"missing required field: {field}"


def test_blank_required_cell_counts_as_missing_not_defaulted():
    # A blank or NaN cell is dropped by _normalise and counts as missing.
    import numpy as np

    batch = pd.DataFrame(
        [
            {
                "llm_model": "mistral-7b-v0.1",
                "gpu_model": "NVIDIA-A100-SXM4-80GB",
                "tokens_per_sample": np.nan,
                "batch_size": 16,
            }
        ]
    )
    df = coastline.recommend(batch, predictor="kavier", max_gpus=8)
    assert not bool(df.iloc[0]["feasible"])
    assert df.iloc[0]["error"] == "missing required field: tokens_per_sample"


def test_field_name_vocabulary_is_the_accepted_input_spelling():
    # Input keys are the WorkloadSpec field names. The row written out in full gives the same
    # recommendation as _ROW, which uses the same names.
    canonical = coastline.recommend(_ROW, predictor="kavier", max_gpus=8)
    explicit = coastline.recommend(
        {
            "llm_model": "mistral-7b-v0.1",
            "fine_tuning_method": "lora",
            "gpu_model": "NVIDIA-A100-SXM4-80GB",
            "tokens_per_sample": 1024,
            "batch_size": 16,
        },
        predictor="kavier",
        max_gpus=8,
    )
    assert bool(explicit.iloc[0]["feasible"])
    for col in ("total_gpus", "throughput_tok_s", "energy_wh"):
        assert explicit.iloc[0][col] == canonical.iloc[0][col]


def test_top_k_yields_k_contiguous_ranks_over_distinct_configs():
    # top_k=3 gives ranks 1, 2 and 3 over three different (total_gpus, batch_size) configs.
    df = coastline.recommend(_ROW, predictor="kavier", top_k=3, max_gpus=8)
    assert len(df) == 3
    assert list(df["rank"]) == [1, 2, 3]
    configs = set(zip(df["total_gpus"], df["batch_size"]))
    assert len(configs) == 3


def test_performance_goal_beats_energy_goal_on_speed_but_draws_more_power():
    # A per-row goal column overrides the goal argument. performance weights power/throughput
    # 0.2/0.8 and energy 0.8/0.2, so the performance pick is at least as fast and the energy
    # pick draws no more total power (power_per_gpu * gpus).
    batch = pd.DataFrame([{**_ROW, "goal": "performance"}, {**_ROW, "goal": "energy"}])
    df = coastline.recommend(batch, predictor="kavier", max_gpus=8)
    perf, energy = df.iloc[0], df.iloc[1]
    assert perf["throughput_tok_s"] >= energy["throughput_tok_s"]
    perf_power_cost = perf["power_w"] * perf["total_gpus"]
    energy_power_cost = energy["power_w"] * energy["total_gpus"]
    assert energy_power_cost <= perf_power_cost


def test_max_slowdown_1x_keeps_only_the_single_fastest_config():
    loose = coastline.recommend(_ROW, predictor="kavier", top_k=5, max_gpus=8)
    tight = coastline.recommend(_ROW, predictor="kavier", top_k=5, max_gpus=8, max_slowdown=1.0)
    # max_slowdown=1 drops every config slower than the fastest.
    assert len(loose) == 5  # no guard: the full top 5
    assert len(tight) == 1  # the fastest feasible config only
    # The balanced top 5 need not contain the fastest config, so compare with its maximum.
    assert tight.iloc[0]["throughput_tok_s"] >= loose["throughput_tok_s"].max()


def test_per_row_predictor_column_overrides_the_kwarg_predictor():
    # A per-row predictor column overrides the predictor argument. The row names an unknown
    # predictor, so it fails with "unknown predictor" although the argument is valid.
    control = coastline.recommend(_ROW, predictor="kavier", max_gpus=8)
    overridden = coastline.recommend({**_ROW, "predictor": "no-such-model"}, predictor="kavier", max_gpus=8)
    assert bool(control.iloc[0]["feasible"])  # kavier from the argument
    assert not bool(overridden.iloc[0]["feasible"])  # the row's unknown predictor is used
    assert "unknown predictor" in str(overridden.iloc[0]["error"])


def test_short_goal_aliases_resolve_to_engine_labels():
    # Short goal names map to the engine's labels.
    assert api._resolve_goal("balanced") == "Multi-objective balanced"
    assert api._resolve_goal("performance") == "Multi-objective lowest runtime"
    assert api._resolve_goal("runtime") == "Multi-objective lowest runtime"
    assert api._resolve_goal("energy") == "Multi-objective energy-saver"
    assert api._resolve_goal("min_gpu") == "Fewest GPUs that fit"
    assert api._resolve_goal("min-gpu") == "Fewest GPUs that fit"
    # A full engine label passes through unchanged.
    assert api._resolve_goal("Multi-objective balanced") == "Multi-objective balanced"
    with pytest.raises(ValueError):
        api._resolve_goal("nonsense")


def test_one_bad_row_is_isolated_not_fatal():
    # A bad GPU in the middle row fails only that row, with an error; the other rows still
    # get recommendations.
    batch = pd.DataFrame(
        [
            _ROW,
            {**_ROW, "gpu_model": "NOT-A-REAL-GPU"},
            {**_ROW, "tokens_per_sample": 2048},
        ]
    )
    df = coastline.recommend(batch, predictor="kavier", max_gpus=8)
    assert len(df) == 3
    assert bool(df.iloc[0]["feasible"]) and df.iloc[0]["throughput_tok_s"] > 0
    assert not bool(df.iloc[1]["feasible"])
    assert isinstance(df.iloc[1]["error"], str) and df.iloc[1]["error"]
    assert bool(df.iloc[2]["feasible"]) and df.iloc[2]["throughput_tok_s"] > 0


def test_rationale_present_on_rank1_only_and_describes_the_top_pick():
    df = coastline.recommend(_ROW, predictor="kavier", top_k=3, max_gpus=8)
    assert "rationale" in df.columns
    rationale = df.iloc[0]["rationale"]
    assert isinstance(rationale, str) and rationale
    # The rationale describes the rank-1 config, so it names its GPU count.
    top_gpus = int(df.iloc[0]["total_gpus"])
    assert f"{top_gpus} GPU" in rationale
    # Runner-up rows have no rationale.
    assert df.iloc[1]["rationale"] is None and df.iloc[2]["rationale"] is None


def test_empty_batch_returns_empty_frame_with_schema():
    df = coastline.recommend([], predictor="kavier")
    assert len(df) == 0
    assert {"rank", "total_gpus", "feasible", "rationale"}.issubset(df.columns)


def test_malformed_batch_raises_clear_typeerror():
    with pytest.raises(TypeError, match="DataFrame"):
        coastline.recommend(42, predictor="kavier")
    with pytest.raises(TypeError, match="dict"):
        coastline.recommend([1, 2], predictor="kavier")


def test_max_gpus_caps_the_grid_to_the_budget():
    # max_gpus=2 limits the GPU counts to {1, 2}.
    df = coastline.recommend(_ROW, predictor="kavier", top_k=5, max_gpus=2)
    assert len(df) >= 1
    assert set(df["total_gpus"]).issubset({1, 2})
    assert (df["gpus_per_node"] <= 2).all()


def test_recommend_is_deterministic():
    a = coastline.recommend(_ROW, predictor="kavier", top_k=3, max_gpus=8)
    b = coastline.recommend(_ROW, predictor="kavier", top_k=3, max_gpus=8)
    # The same input gives the same recommendations.
    pd.testing.assert_frame_equal(a, b)


if __name__ == "__main__":  # pragma: no cover - manual run convenience
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_a_zero_max_gpus_fails_the_row_with_its_name():
    out = coastline.recommend([{**_ROW, "max_gpus": 0}], predictor="kavier", feasibility="rules")
    assert not bool(out.iloc[0]["feasible"])
    assert out.iloc[0]["error"] == "max_gpus must be >= 1, got 0"
