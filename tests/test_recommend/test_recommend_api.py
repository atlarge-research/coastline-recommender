"""Tests for the public batch API ``coastline.recommend`` (Kavier path, hermetic).

Every test pins the analytical ``kavier`` predictor so the engine never unpickles a
trained-ML artifact (the host xgboost/AutoGluon-unpickle segfault) — the same pattern
as ``coastline_recommender/tests/test_ui.py``.

Oracles here are hand-derived from the batch/engine contract:
  * total_tokens = dataset_size * epochs * tokens_per_sample  (defaults 50_000 * 1)
  * runtime_s    = total_tokens / throughput
  * energy_wh    = power_per_gpu * total_gpus * runtime_s / 3600 ; energy_kwh = energy_wh/1000
  * tokens_per_watt = throughput / power_per_gpu
  * node layout: gpus_per_node = min(8, total_gpus) ; number_of_nodes = ceil(total_gpus/8)
  * goal presets rank a weighted power/throughput score (energy 0.8/0.2, perf 0.2/0.8)
"""

import math

import pandas as pd
import pytest

import coastline
from coastline.sdk.recommend import batch_api as api

# A minimal workload row; the rest of the schema falls back to engine defaults.
_ROW = {
    "model": "mistral-7b-v0.1",
    "method": "lora",
    "gpu_model": "NVIDIA-A100-SXM4-80GB",
    "tokens_per_sample": 1024,
    "batch_size": 16,
}

# Columns every successful output row must carry (input echoed + recommendation).
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

# Engine non-interactive defaults for the token-count derivation (engine.defaults()).
_DATASET_SIZE = 50_000
_EPOCHS = 1


def test_batch_returns_one_ranked_row_per_input_with_derived_runtime():
    # Two rows differing only in tokens_per_sample; top_k defaults to 1 so each input
    # yields exactly one rank-1 row. runtime is an INDEPENDENT derivation from the token
    # count, not a snapshot: total_tokens = dataset_size(50_000) * epochs(1) * tokens.
    batch = pd.DataFrame([_ROW, {**_ROW, "tokens_per_sample": 2048}])
    df = coastline.recommend(batch, predictor="kavier", max_gpus=8)
    assert len(df) == 2  # exactly one recommendation per input row
    assert _RECO_COLUMNS <= set(df.columns)
    assert list(df["rank"]) == [1, 1]
    assert bool(df["feasible"].all())

    for i, tokens in enumerate((1024, 2048)):
        row = df.iloc[i]
        total_tokens = _DATASET_SIZE * _EPOCHS * tokens  # 51_200_000 ; 102_400_000
        # runtime_s * throughput must reconstruct the hand-computed token count.
        assert row["runtime_s"] * row["throughput_tok_s"] == pytest.approx(total_tokens)
    # 2x the tokens over the same workload => ~2x the runtime (row1 uses 2x row0's tokens
    # but not necessarily the same GPU pick, so allow the ratio a wide window).
    assert df.iloc[1]["runtime_s"] > df.iloc[0]["runtime_s"]


def test_energy_and_efficiency_columns_are_mutually_consistent():
    # The reported energy/efficiency columns must obey their defining formulas relative to
    # the OTHER reported columns (a cross-check that would catch e.g. a kWh /1000 bug or a
    # power mismatch). power_w is PER-GPU, so energy scales with total_gpus.
    df = coastline.recommend(_ROW, predictor="kavier", top_k=3, max_gpus=8)
    for _, row in df.iterrows():
        # energy_wh = power_per_gpu * total_gpus * runtime_s / 3600
        assert row["energy_wh"] == pytest.approx(row["power_w"] * row["total_gpus"] * row["runtime_s"] / 3600.0)
        # kWh is Wh/1000 — pins the unit conversion (the classic /1000 vs /3600 mixup).
        assert row["energy_kwh"] == pytest.approx(row["energy_wh"] / 1000.0)
        # tokens_per_watt = throughput / power_per_gpu
        assert row["tokens_per_watt"] == pytest.approx(row["throughput_tok_s"] / row["power_w"])


def test_node_layout_packs_8_per_node_16_gpus_is_two_nodes():
    # Layout is hand-derivable: up to 8 GPUs per node. 16 GPUs @ 8/node => 2 nodes.
    # goal=performance so the ranked set spans several GPU counts including 16.
    df = coastline.recommend(_ROW, predictor="kavier", goal="performance", top_k=12, max_gpus=16)
    for _, row in df.iterrows():
        g = int(row["total_gpus"])
        assert row["gpus_per_node"] == min(8, g)  # pack up to 8/node
        assert row["number_of_nodes"] == math.ceil(g / 8)  # minimal node count
        assert row["gpus_per_node"] * row["number_of_nodes"] >= g
    sixteen = df[df["total_gpus"] == 16]
    assert not sixteen.empty  # the max_gpus=16 grid must include a 16-GPU candidate
    assert int(sixteen.iloc[0]["gpus_per_node"]) == 8
    assert int(sixteen.iloc[0]["number_of_nodes"]) == 2  # 16 / 8 = 2 nodes


def test_single_dict_list_and_dataframe_inputs_produce_identical_output():
    # Three input encodings of the SAME single workload must route through one code path
    # and yield identical recommendations (input-form invariance).
    as_dict = coastline.recommend(_ROW, predictor="kavier", max_gpus=8)
    as_list = coastline.recommend([_ROW], predictor="kavier", max_gpus=8)
    as_df = coastline.recommend(pd.DataFrame([_ROW]), predictor="kavier", max_gpus=8)
    assert len(as_dict) == len(as_list) == len(as_df) == 1
    pd.testing.assert_frame_equal(as_dict, as_list)
    pd.testing.assert_frame_equal(as_dict, as_df)


def test_omitted_optional_method_matches_explicit_default_method():
    # Omitting an OPTIONAL knob (method) must fall back to the engine default (lora), i.e.
    # produce the SAME recommendation as spelling that default out explicitly. This proves
    # the default-fill is exactly the documented default, not some other value.
    core = {
        "model": "mistral-7b-v0.1",
        "gpu_model": "NVIDIA-A100-SXM4-80GB",
        "tokens_per_sample": 1024,
        "batch_size": 16,
    }
    omitted = coastline.recommend(core, predictor="kavier", max_gpus=8)
    explicit = coastline.recommend({**core, "method": "lora"}, predictor="kavier", max_gpus=8)
    assert bool(omitted.iloc[0]["feasible"])
    # Recommendation columns must match; the echoed input columns differ (one has 'method').
    for col in ("total_gpus", "throughput_tok_s", "energy_wh", "runtime_s"):
        assert omitted.iloc[0][col] == explicit.iloc[0][col]


def test_missing_required_core_field_is_feasible_false_not_defaulted():
    # H1 regression: a batch row that OMITS a required core field (here gpu_model and the
    # rest) must come back feasible=False with 'missing required field', NOT a confident
    # recommendation silently built on the engine defaults (A100 / 1024 / 32). Compare
    # against a complete row to prove the difference is the missing field, not the model.
    only_model = coastline.recommend({"model": "mistral-7b-v0.1"}, predictor="kavier", max_gpus=8)
    assert len(only_model) == 1
    assert not bool(only_model.iloc[0]["feasible"])
    err = only_model.iloc[0]["error"]
    assert isinstance(err, str) and "missing required field" in err
    # no recommendation columns were filled from a default
    assert only_model.iloc[0]["total_gpus"] is None
    assert only_model.iloc[0]["throughput_tok_s"] is None

    # A complete row is still feasible — the guard fires only on the ABSENT case.
    complete = coastline.recommend(
        {"model": "mistral-7b-v0.1", "gpu_model": "NVIDIA-A100-SXM4-80GB", "tokens_per_sample": 1024, "batch_size": 16},
        predictor="kavier",
        max_gpus=8,
    )
    assert bool(complete.iloc[0]["feasible"]) and complete.iloc[0]["throughput_tok_s"] > 0


@pytest.mark.parametrize("field", ["gpu_model", "tokens_per_sample", "batch_size"])
def test_missing_required_field_names_the_specific_absent_column(field):
    # Each absent core field is named individually (first-missing wins), so the caller can
    # see exactly what to supply. The expected substring is built from the field name, an
    # oracle independent of the implementation's loop order.
    full = {
        "model": "mistral-7b-v0.1",
        "gpu_model": "NVIDIA-A100-SXM4-80GB",
        "tokens_per_sample": 1024,
        "batch_size": 16,
    }
    row = {k: v for k, v in full.items() if k != field}
    df = coastline.recommend(row, predictor="kavier", max_gpus=8)
    assert not bool(df.iloc[0]["feasible"])
    assert df.iloc[0]["error"] == f"missing required field: {field}"


def test_blank_required_cell_counts_as_missing_not_defaulted():
    # A present-but-blank/NaN core cell is 'absent' (dropped by _normalise), so it must
    # also be refused rather than silently defaulted.
    import numpy as np

    batch = pd.DataFrame(
        [
            {
                "model": "mistral-7b-v0.1",
                "gpu_model": "NVIDIA-A100-SXM4-80GB",
                "tokens_per_sample": np.nan,
                "batch_size": 16,
            }
        ]
    )
    df = coastline.recommend(batch, predictor="kavier", max_gpus=8)
    assert not bool(df.iloc[0]["feasible"])
    assert df.iloc[0]["error"] == "missing required field: tokens_per_sample"


def test_alias_spellings_produce_the_same_recommendation_as_canonical():
    # alias spellings (llm_model / peft / gpu / seq_len / batch) must resolve to the same
    # workload as the canonical column names — identical recommendation, not merely "runs".
    canonical = coastline.recommend(_ROW, predictor="kavier", max_gpus=8)
    aliased = coastline.recommend(
        {"llm_model": "mistral-7b-v0.1", "peft": "lora", "gpu": "NVIDIA-A100-SXM4-80GB", "seq_len": 1024, "batch": 16},
        predictor="kavier",
        max_gpus=8,
    )
    for col in ("total_gpus", "throughput_tok_s", "energy_wh"):
        assert aliased.iloc[0][col] == canonical.iloc[0][col]


def test_top_k_yields_k_contiguous_ranks_over_distinct_configs():
    # top_k=3 => exactly 3 rows, ranks 1,2,3 (contiguous, best-first), and three DISTINCT
    # (total_gpus, batch_size) configurations — the table shows real alternatives, never
    # the same config repeated.
    df = coastline.recommend(_ROW, predictor="kavier", top_k=3, max_gpus=8)
    assert len(df) == 3
    assert list(df["rank"]) == [1, 2, 3]
    configs = set(zip(df["total_gpus"], df["batch_size"]))
    assert len(configs) == 3


def test_performance_goal_beats_energy_goal_on_speed_but_draws_more_power():
    # Per-row ``goal`` column overrides the batch kwarg for that row. The two presets have
    # opposite objectives, so the picks must differ in the derivable direction:
    #   performance (0.2 power / 0.8 throughput) -> highest throughput
    #   energy      (0.8 power / 0.2 throughput) -> lowest total power draw (power_per_gpu*gpus)
    batch = pd.DataFrame([{**_ROW, "goal": "performance"}, {**_ROW, "goal": "energy"}])
    df = coastline.recommend(batch, predictor="kavier", max_gpus=8)
    perf, energy = df.iloc[0], df.iloc[1]
    # performance is at least as fast as energy...
    assert perf["throughput_tok_s"] >= energy["throughput_tok_s"]
    # ...and energy draws no more total power than performance (power_cost = W/gpu * gpus).
    perf_power_cost = perf["power_w"] * perf["total_gpus"]
    energy_power_cost = energy["power_w"] * energy["total_gpus"]
    assert energy_power_cost <= perf_power_cost


def test_max_slowdown_1x_keeps_only_the_single_fastest_config():
    loose = coastline.recommend(_ROW, predictor="kavier", top_k=5, max_gpus=8)
    tight = coastline.recommend(_ROW, predictor="kavier", top_k=5, max_gpus=8, max_slowdown=1.0)
    # k=1x drops every config slower than 1x the fastest -> only the fastest survives.
    assert len(loose) == 5  # unguarded: the full top-5
    assert len(tight) == 1  # guarded to the single fastest feasible config
    # The survivor is the FASTEST feasible config, so it is at least as fast as anything in
    # the unguarded ranked table (the balanced ranking may not surface the fastest itself).
    assert tight.iloc[0]["throughput_tok_s"] >= loose["throughput_tok_s"].max()


def test_per_row_predictor_column_overrides_the_kwarg_predictor():
    # The per-row `predictor` column must override the batch `predictor` kwarg. Env-independent
    # oracle (no cache/ML data needed): a valid kwarg (kavier) yields a feasible pick, but a row
    # pinning an UNKNOWN predictor is rejected per-row with "unknown predictor" — which can only
    # happen if the ROW column, not the kwarg, decided the predictor.
    control = coastline.recommend(_ROW, predictor="kavier", max_gpus=8)
    overridden = coastline.recommend({**_ROW, "predictor": "no-such-model"}, predictor="kavier", max_gpus=8)
    assert bool(control.iloc[0]["feasible"])  # kwarg kavier -> feasible
    assert not bool(overridden.iloc[0]["feasible"])  # the row's unknown predictor won -> rejected
    assert "unknown predictor" in str(overridden.iloc[0]["error"])


def test_short_goal_aliases_resolve_to_engine_labels():
    # Exact alias->label map (the interface contract callers rely on).
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
    # A bad GPU in the middle row must not fail the whole batch: it returns
    # feasible=False with an error, while the good rows still get recommendations.
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
    # The rationale describes the rank-1 config, so it must name its GPU count.
    top_gpus = int(df.iloc[0]["total_gpus"])
    assert f"{top_gpus} GPU" in rationale
    # runner-up rows carry no rationale (it belongs to the chosen config only).
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
    # max_gpus=2 => the GPU-count grid is exactly {1, 2} (GPU_BUDGETS <= 2), so no returned
    # config may exceed 2 GPUs and every value must be one of the two budgeted sizes.
    df = coastline.recommend(_ROW, predictor="kavier", top_k=5, max_gpus=2)
    assert len(df) >= 1
    assert set(df["total_gpus"]).issubset({1, 2})
    assert (df["gpus_per_node"] <= 2).all()


def test_recommend_is_deterministic():
    a = coastline.recommend(_ROW, predictor="kavier", top_k=3, max_gpus=8)
    b = coastline.recommend(_ROW, predictor="kavier", top_k=3, max_gpus=8)
    # Same input -> identical recommendations (a reproducible-pipeline guarantee).
    pd.testing.assert_frame_equal(a, b)


if __name__ == "__main__":  # pragma: no cover - manual run convenience
    raise SystemExit(pytest.main([__file__, "-q"]))
