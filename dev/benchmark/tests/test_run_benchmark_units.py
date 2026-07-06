"""Unit tests for the pure/logic helpers in ``benchmark.run_benchmark``.

These tests exercise the unit-testable seams WITHOUT running the full 12-model
benchmark and WITHOUT loading any real trained ML model (which would unpickle
xgboost/catboost/etc. and risk a native segfault). Strategy:

* The ML predictor classes are instantiated at module import in ``run_benchmark``
  but they lazy-load their pickles only inside ``predict()``; merely importing the
  module is therefore safe and loads no models.
* ``evaluate_ml_predictor`` is tested with a hand-written mock predictor.
* ``evaluate_kavier``'s cache ``pred_lookup`` path is tested by monkeypatching the
  validation-CSV loader and using synthetic workloads whose model names are NOT in
  Kavier's spec library, so the live simulator raises and the code falls back to the
  CSV-derived prediction (deterministic, no model load).
* All other helpers are pure functions over dicts / tiny synthetic frames.
"""

import numpy as np
import pandas as pd
import pytest

import benchmark.run_benchmark as rb
from coastline.sdk.models.recommendation import Prediction

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _synthetic_full_test() -> pd.DataFrame:
    """Two-row curated-style frame with the columns the evaluators read."""
    return pd.DataFrame(
        {
            "model_name": ["fakemodel-A", "fakemodel-B"],
            "method": ["lora", "full"],
            "gpu_model": ["FAKE-GPU", "FAKE-GPU"],
            "tokens_per_sample": [512, 1024],
            "batch_size": [4, 8],
            "number_gpus": [2, 4],
            "number_nodes": [1, 2],
            "torch_dtype": ["bfloat16", np.nan],
            "enable_roce": [1.0, np.nan],
        }
    )


class _MockPredictor:
    """Minimal predictor matching the ``predict(workload, ctx) -> Prediction`` API.

    Records the WorkloadSpec objects it is handed so the test can assert how
    ``evaluate_ml_predictor`` decoded the curated row into a workload.
    """

    def __init__(self, throughput=105.0, runtime=11.0, raise_on=None):
        self.seen = []
        self._throughput = throughput
        self._runtime = runtime
        self._raise_on = raise_on  # index at which predict() should raise

    def predict(self, workload, context):
        self.seen.append(workload)
        if self._raise_on is not None and len(self.seen) - 1 == self._raise_on:
            raise RuntimeError("boom")
        return Prediction(
            gpus_per_node=workload.gpus_per_node or 1,
            number_of_nodes=workload.number_of_nodes or 1,
            total_gpus=(workload.gpus_per_node or 1) * (workload.number_of_nodes or 1),
            predicted_throughput=self._throughput,
            predicted_runtime_seconds=self._runtime,
        )


# ---------------------------------------------------------------------------
# _build_meta — total_gpus_used assembly
# ---------------------------------------------------------------------------


def test_build_meta_computes_total_gpus_used():
    meta = rb._build_meta(_synthetic_full_test())
    assert list(meta.columns) == [
        "batch_size",
        "tokens_per_sample",
        "number_gpus",
        "number_nodes",
        "total_gpus_used",
    ]
    # 2*1 = 2, 4*2 = 8
    assert meta["total_gpus_used"].tolist() == [2.0, 8.0]
    assert meta["total_gpus_used"].dtype == np.float64


def test_build_meta_fills_missing_gpu_counts_with_one():
    df = pd.DataFrame(
        {
            "batch_size": [4],
            "tokens_per_sample": [512],
            "number_gpus": [np.nan],
            "number_nodes": [np.nan],
        }
    )
    meta = rb._build_meta(df)
    # NaN counts default to 1 -> 1*1 = 1
    assert meta["total_gpus_used"].tolist() == [1.0]


# ---------------------------------------------------------------------------
# evaluate_kavier — live-simulator-only, fails loudly (no canned-CSV fallback)
# ---------------------------------------------------------------------------


def test_evaluate_kavier_raises_on_unsimulatable_rows():
    """Unknown model names make the live simulator raise per-row; evaluate_kavier
    must refuse to report a partial/stale MdAPE and raise instead of silently
    substituting canned predictions (the removed CSV-fallback behavior)."""
    full_test = _synthetic_full_test()
    ml_data = {
        "full_test": full_test,
        "y_throughput_test": np.array([100.0, 200.0]),
    }
    with pytest.raises(RuntimeError, match="refusing to report a partial MdAPE"):
        rb.evaluate_kavier(ml_data)


# ---------------------------------------------------------------------------
# evaluate_ml_predictor — row-by-row result assembly via mock predictor
# ---------------------------------------------------------------------------


def test_evaluate_ml_predictor_assembles_predictions_and_meta():
    ml_data = {
        "full_test": _synthetic_full_test(),
        "y_throughput_test": np.array([100.0, 200.0]),
        "y_runtime_test": np.array([10.0, 20.0]),
    }
    mp = _MockPredictor(throughput=105.0, runtime=11.0)
    raw = rb.evaluate_ml_predictor(mp, ml_data)

    np.testing.assert_array_equal(raw["y_pred"], np.array([105.0, 105.0]))
    np.testing.assert_array_equal(raw["y_pred_runtime"], np.array([11.0, 11.0]))
    np.testing.assert_array_equal(raw["y_true"], np.array([100.0, 200.0]))
    np.testing.assert_array_equal(raw["y_true_runtime"], np.array([10.0, 20.0]))
    assert raw["n"] == 2
    assert raw["meta"]["total_gpus_used"].tolist() == [2.0, 8.0]
    assert raw["predict_time_s"] >= 0.0


def test_evaluate_ml_predictor_decodes_workload_fields():
    """torch_dtype / enable_roce / counts are decoded from the curated row."""
    ml_data = {
        "full_test": _synthetic_full_test(),
        "y_throughput_test": np.array([100.0, 200.0]),
        "y_runtime_test": np.array([10.0, 20.0]),
    }
    mp = _MockPredictor()
    rb.evaluate_ml_predictor(mp, ml_data)

    w0, w1 = mp.seen
    assert w0.fine_tuning_method == "lora"
    assert w0.gpu_model == "FAKE-GPU"
    # WorkloadSpec canonicalizes llm_model at ingestion (drops org prefix, lowercases),
    # so "fakemodel-A" -> "fakemodel-a". See WorkloadSpec._canonicalize_llm_model.
    assert w0.llm_model == "fakemodel-a"
    assert w0.batch_size == 4 and w0.tokens_per_sample == 512
    assert w0.gpus_per_node == 2 and w0.number_of_nodes == 1
    assert w0.torch_dtype == "bfloat16"
    assert w0.enable_roce is True
    # Row 1 has NaN dtype/roce -> None
    assert w1.torch_dtype is None
    assert w1.enable_roce is None


def test_evaluate_ml_predictor_handles_predict_exception_as_zero():
    """A predictor that raises on a row contributes throughput 0.0 and NaN runtime."""
    ml_data = {
        "full_test": _synthetic_full_test(),
        "y_throughput_test": np.array([100.0, 200.0]),
        "y_runtime_test": np.array([10.0, 20.0]),
    }
    mp = _MockPredictor(throughput=105.0, runtime=11.0, raise_on=0)
    raw = rb.evaluate_ml_predictor(mp, ml_data)
    assert raw["y_pred"][0] == 0.0
    assert raw["y_pred"][1] == 105.0
    assert np.isnan(raw["y_pred_runtime"][0])
    assert raw["y_pred_runtime"][1] == 11.0


def test_evaluate_ml_predictor_none_throughput_becomes_zero():
    """Prediction with predicted_throughput=None contributes 0.0."""
    ml_data = {
        "full_test": _synthetic_full_test(),
        "y_throughput_test": np.array([100.0, 200.0]),
        "y_runtime_test": np.array([10.0, 20.0]),
    }

    class _NonePred:
        def predict(self, workload, context):
            return Prediction(
                gpus_per_node=1,
                number_of_nodes=1,
                total_gpus=1,
                predicted_throughput=None,
                predicted_runtime_seconds=None,
            )

    raw = rb.evaluate_ml_predictor(_NonePred(), ml_data)
    assert raw["y_pred"].tolist() == [0.0, 0.0]
    assert np.isnan(raw["y_pred_runtime"]).all()


# ---------------------------------------------------------------------------
# _apply_metric_status / _throughput_and_latency_metrics / _result_entry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n,expected_ok",
    [(5, True), (1, True), (0, False), (float("nan"), False), (None, False)],
)
def test_apply_metric_status(n, expected_ok):
    m = {"n": n}
    rb._apply_metric_status(m)
    if expected_ok:
        assert m["status"] == "OK"
    else:
        assert m["status"] == "Error: no valid predictions"


def test_throughput_and_latency_metrics_shapes_and_status():
    raw = {
        "y_true": np.array([100.0, 200.0]),
        "y_pred": np.array([110.0, 190.0]),
        "meta": rb._build_meta(_synthetic_full_test()),
        "n": 2,
        "predict_time_s": 0.01,
    }
    thr_m, lat_m = rb._throughput_and_latency_metrics(raw)
    assert thr_m["status"] == "OK"
    assert lat_m["status"] == "OK"
    assert thr_m["n"] == 2
    # MdAPE = median(|110-100|/100, |190-200|/200) = median(10%, 5%) = 7.5%
    assert thr_m["mdape"] == pytest.approx(7.5, abs=1e-6)
    # Latency is a monotone transform; with matched ordering MdAPE matches.
    assert np.isfinite(lat_m["mdape"])


def test_throughput_and_latency_metrics_all_zero_predictions_is_error():
    raw = {
        "y_true": np.array([100.0, 200.0]),
        "y_pred": np.array([0.0, 0.0]),
        "meta": rb._build_meta(_synthetic_full_test()),
        "n": 0,
        "predict_time_s": 0.0,
    }
    thr_m, lat_m = rb._throughput_and_latency_metrics(raw)
    # compute_metrics masks out non-positive preds -> n == 0 -> status error.
    assert thr_m["status"] == "Error: no valid predictions"
    assert lat_m["status"] == "Error: no valid predictions"


def test_result_entry_structure_and_timing():
    raw = {
        "y_true": np.array([100.0, 200.0]),
        "y_pred": np.array([110.0, 190.0]),
        "meta": rb._build_meta(_synthetic_full_test()),
        "n": 2,
        "predict_time_s": 0.02,
    }
    thr_m, lat_m = rb._throughput_and_latency_metrics(raw)
    entry = rb._result_entry("ML", raw, thr_m, lat_m)
    assert entry["type"] == "ML"
    assert entry["throughput"] is thr_m
    assert entry["latency"] is lat_m
    # ms/100 = predict_time_s / n * 100 * 1000 = 0.02/2 * 100000 = 1000 ms
    assert entry["ms_per_100"] == pytest.approx(1000.0, rel=1e-6)


def test_result_entry_ms_per_100_nan_when_n_zero():
    raw = {
        "y_true": np.array([]),
        "y_pred": np.array([]),
        "meta": rb._build_meta(_synthetic_full_test().iloc[0:0]),
        "n": 0,
        "predict_time_s": None,
    }
    entry = rb._result_entry("ML", raw, {"status": "Error: x"}, {"status": "Error: x"})
    assert np.isnan(entry["ms_per_100"])


# ---------------------------------------------------------------------------
# save_results_csv — result-row assembly to CSV
# ---------------------------------------------------------------------------


def _ok_entry():
    raw = {
        "y_true": np.array([100.0, 200.0]),
        "y_pred": np.array([110.0, 190.0]),
        "meta": rb._build_meta(_synthetic_full_test()),
        "n": 2,
        "predict_time_s": 0.02,
    }
    thr_m, lat_m = rb._throughput_and_latency_metrics(raw)
    return rb._result_entry("ML", raw, thr_m, lat_m)


def test_save_results_csv_writes_one_row_per_metric(tmp_path):
    out = tmp_path / "out.csv"
    results = {"RandomForest": _ok_entry()}
    rb.save_results_csv(results, csv_path=out)

    df = pd.read_csv(out)
    assert len(df) == 2  # throughput + latency
    assert set(df["metric"]) == {"throughput", "latency"}
    assert set(df["id"]) == {"PD1"}
    assert set(df["model"]) == {"RandomForest"}
    assert set(df["status"]) == {"OK"}
    for col in ("n", "mdape", "mape", "ms_per_100", "within_20"):
        assert col in df.columns


def test_save_results_csv_error_rows_omit_metric_columns(tmp_path):
    out = tmp_path / "out.csv"
    results = {
        "Kavier": {
            "type": "physics",
            "throughput": {"status": "Error: boom"},
            "latency": {"status": "Error: boom"},
            "ms_per_100": None,
        }
    }
    rb.save_results_csv(results, csv_path=out)

    df = pd.read_csv(out)
    assert len(df) == 2
    assert set(df["status"]) == {"Error: boom"}
    assert set(df["id"]) == {"PA1"}
    # Error rows carry id/model/type/metric/status only -> numeric cols absent/NaN.
    if "mdape" in df.columns:
        assert df["mdape"].isna().all()


def test_save_results_csv_mixed_ok_and_error(tmp_path):
    out = tmp_path / "mixed.csv"
    results = {
        "RandomForest": _ok_entry(),
        "Kavier": {
            "type": "physics",
            "throughput": {"status": "Error: x"},
            "latency": {"status": "Error: x"},
            "ms_per_100": None,
        },
    }
    rb.save_results_csv(results, csv_path=out)

    df = pd.read_csv(out)
    assert len(df) == 4  # 2 models x 2 metrics
    rf = df[df["model"] == "RandomForest"]
    kv = df[df["model"] == "Kavier"]
    assert (rf["status"] == "OK").all()
    assert (kv["status"] == "Error: x").all()
    assert set(rf["id"]) == {"PD1"}
    assert set(kv["id"]) == {"PA1"}


def test_save_results_csv_relative_path_lands_under_results_dir(tmp_path, monkeypatch):
    """A relative csv_path is resolved under BENCHMARKS_DIR/results (not CWD)."""
    monkeypatch.setattr(rb, "BENCHMARKS_DIR", tmp_path)
    rb.save_results_csv({"RandomForest": _ok_entry()}, csv_path="thesis.csv")
    written = tmp_path / "results" / "thesis.csv"
    assert written.exists()
    df = pd.read_csv(written)
    assert set(df["model"]) == {"RandomForest"}


# ---------------------------------------------------------------------------
# prepare_ml_data — test-split loading (reads curated CSV only; no model load)
# ---------------------------------------------------------------------------


def test_prepare_ml_data_returns_aligned_test_split():
    data = rb.prepare_ml_data()
    expected_keys = {
        "X_cat_test",
        "X_num_test",
        "y_throughput_test",
        "y_runtime_test",
        "full_test",
    }
    assert set(data.keys()) == expected_keys
    n = len(data["full_test"])
    assert n > 0
    # Targets aligned with the full_test frame length.
    assert len(data["y_throughput_test"]) == n
    assert len(data["y_runtime_test"]) == n
    assert len(data["X_cat_test"]) == n
    assert len(data["X_num_test"]) == n
    # full_test exposes the columns _build_meta / evaluators rely on.
    for col in ("batch_size", "tokens_per_sample", "number_gpus", "number_nodes"):
        assert col in data["full_test"].columns


def test_prepare_ml_data_max_gpus_filters_test_rows():
    """max_gpus must drop rows where number_gpus * number_nodes exceeds the cap,
    and is never larger than the unfiltered split."""
    full = rb.prepare_ml_data()
    capped = rb.prepare_ml_data(max_gpus=8)

    tot = pd.to_numeric(capped["full_test"]["number_gpus"], errors="coerce").fillna(1) * pd.to_numeric(
        capped["full_test"]["number_nodes"], errors="coerce"
    ).fillna(1)
    assert bool((tot <= 8).all())
    # All aligned arrays shrink/stay together.
    n = len(capped["full_test"])
    assert len(capped["y_throughput_test"]) == n
    assert len(capped["y_runtime_test"]) == n
    assert n <= len(full["full_test"])


def test_prepare_ml_data_max_gpus_zero_yields_empty_split():
    capped = rb.prepare_ml_data(max_gpus=0)
    assert len(capped["full_test"]) == 0
    assert len(capped["y_throughput_test"]) == 0
