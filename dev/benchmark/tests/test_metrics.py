"""Tests for benchmark.metrics, which computes the thesis's accuracy numbers (MdAPE, MAPE,
within-X) and converts throughput to latency."""

import math

import numpy as np
import pytest

from benchmark.metrics import (
    compute_metrics,
    ms_per_100_predictions,
    throughput_to_latency,
)

# compute_metrics: MdAPE, MAPE and within-X on known inputs


def test_mdape_mape_basic_known_values():
    """Errors of 0, 10, 20, 30 and 40% give MdAPE 20 and MAPE 20."""
    y_true = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    y_pred = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
    m = compute_metrics(y_true, y_pred)
    assert m["n"] == 5
    assert m["mdape"] == pytest.approx(20.0)
    assert m["mape"] == pytest.approx(20.0)


def test_mdape_differs_from_mape_on_skewed_errors():
    """MdAPE equals MAPE on two points and differs from it on a skewed set of three."""
    # Two points, errors 50 and 25%: median and mean are both 37.5.
    m2 = compute_metrics([100.0, 200.0], [50.0, 150.0])
    assert m2["mdape"] == pytest.approx(37.5)
    assert m2["mape"] == pytest.approx(37.5)

    # Three points, errors 10, 20 and 90%: MdAPE 20, MAPE 40.
    y_true = np.array([100.0, 100.0, 100.0])
    y_pred = np.array([110.0, 120.0, 190.0])
    m3 = compute_metrics(y_true, y_pred)
    assert m3["mdape"] == pytest.approx(20.0)
    assert m3["mape"] == pytest.approx(40.0)
    assert m3["mdape"] != m3["mape"]


def test_within_thresholds_known_distribution():
    """With errors of 0, 10, 20, 30 and 40%, within_10, within_20 and within_30 are 20, 40 and 60%."""
    y_true = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    y_pred = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
    m = compute_metrics(y_true, y_pred)
    assert m["within_10"] == pytest.approx(20.0)
    assert m["within_20"] == pytest.approx(40.0)
    assert m["within_30"] == pytest.approx(60.0)


def test_within_threshold_is_strict_inequality():
    """An error of exactly 10% is outside within_10, since the bound is strict."""
    m = compute_metrics([100.0], [110.0])  # exactly 10% error
    assert m["within_10"] == pytest.approx(0.0)
    assert m["within_20"] == pytest.approx(100.0)  # 10 < 20


def test_perfect_prediction_zero_error():
    """A perfect prediction gives zero error and r2 = 1."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    m = compute_metrics(y_true, y_pred)
    assert m["mdape"] == pytest.approx(0.0)
    assert m["mape"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["mae"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)
    assert m["within_10"] == pytest.approx(100.0)


def test_rmse_mae_r2_known_values():
    """Residuals 2, -2, 3 and -4 give MAE 2.75, RMSE sqrt(8.25) and R2 = 1 - 33/500 = 0.934."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 36.0])
    m = compute_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx(2.75)
    assert m["rmse"] == pytest.approx(math.sqrt(8.25))
    assert m["r2"] == pytest.approx(0.934)


# compute_metrics: masking and edge cases


def test_zero_in_y_true_is_masked_out():
    """A row with y_true = 0 is dropped; the remaining errors of 10 and 30% give MdAPE and MAPE 20."""
    y_true = np.array([0.0, 100.0, 100.0])
    y_pred = np.array([50.0, 110.0, 130.0])
    m = compute_metrics(y_true, y_pred)
    assert m["n"] == 2
    assert m["mdape"] == pytest.approx(20.0)
    assert m["mape"] == pytest.approx(20.0)


def test_nonpositive_y_pred_is_masked_out():
    """A row with y_pred <= 0 is dropped."""
    y_true = np.array([100.0, 100.0])
    y_pred = np.array([0.0, 90.0])
    m = compute_metrics(y_true, y_pred)
    assert m["n"] == 1
    assert m["mdape"] == pytest.approx(10.0)


def test_nan_and_inf_inputs_are_masked_out():
    """Rows with NaN or inf in either array are dropped, leaving only the pair (100, 110)."""
    y_true = np.array([100.0, np.nan, 100.0, np.inf])
    y_pred = np.array([110.0, 50.0, np.inf, 50.0])
    m = compute_metrics(y_true, y_pred)
    assert m["n"] == 1
    assert m["mdape"] == pytest.approx(10.0)


def test_all_masked_returns_all_nan():
    """When nothing survives the mask, every metric (including n) is NaN."""
    m = compute_metrics([0.0, -1.0], [1.0, 1.0])
    for key in ("n", "mdape", "mape", "r2", "rmse", "mae", "within_10", "within_20", "within_30"):
        assert math.isnan(m[key]), f"{key} should be NaN, got {m[key]}"


def test_empty_input_returns_all_nan():
    """Empty input gives every key, each set to NaN."""
    m = compute_metrics([], [])
    expected_keys = {"n", "mdape", "mape", "r2", "rmse", "mae", "within_10", "within_20", "within_30"}
    assert set(m.keys()) == expected_keys
    assert all(math.isnan(v) for v in m.values())


def test_single_element_input():
    """One pair gives defined error metrics and a NaN r2, which needs at least two samples."""
    m = compute_metrics([50.0], [55.0])  # 10% error
    assert m["n"] == 1
    assert m["mdape"] == pytest.approx(10.0)
    assert m["mape"] == pytest.approx(10.0)
    assert m["mae"] == pytest.approx(5.0)
    assert m["rmse"] == pytest.approx(5.0)
    assert m["within_20"] == pytest.approx(100.0)
    assert math.isnan(m["r2"])


def test_all_identical_y_true_multi_sample_r2_is_nan():
    """A constant y_true makes R2 undefined (SS_tot = 0), so r2 is NaN; the error metrics are unaffected."""
    y_true = np.array([100.0, 100.0, 100.0])
    y_pred = np.array([110.0, 120.0, 130.0])
    m = compute_metrics(y_true, y_pred)
    assert math.isnan(m["r2"])
    assert m["mdape"] == pytest.approx(20.0)  # median(10,20,30)
    assert m["mae"] == pytest.approx(20.0)


def test_returns_native_python_floats_not_numpy():
    """The metrics are Python floats and n is a Python int, ready for JSON and CSV export."""
    m = compute_metrics([100.0, 100.0], [110.0, 130.0])
    assert isinstance(m["n"], int)
    for key in ("mdape", "mape", "r2", "rmse", "mae", "within_10", "within_20", "within_30"):
        assert isinstance(m[key], float)


def test_accepts_python_lists_and_is_order_independent():
    """Python lists are accepted, and MdAPE and MAPE do not depend on row order."""
    a = compute_metrics([100.0, 200.0, 50.0], [110.0, 180.0, 60.0])
    b = compute_metrics([50.0, 100.0, 200.0], [60.0, 110.0, 180.0])
    assert a["mdape"] == pytest.approx(b["mdape"])
    assert a["mape"] == pytest.approx(b["mape"])
    assert a["n"] == b["n"] == 3


# throughput_to_latency


def test_throughput_to_latency_known_value():
    """latency = batch_size * tokens_per_sample * total_gpus / throughput, here 2 * 512 * 8 / 10 = 819.2 s."""
    lat = throughput_to_latency(
        throughput=np.array([10.0]),
        batch_size=np.array([2.0]),
        tokens_per_sample=np.array([512.0]),
        total_gpus=np.array([8.0]),
    )
    assert lat[0] == pytest.approx(819.2)


def test_throughput_to_latency_round_trip():
    """Converting throughput to latency and back gives the original throughput."""
    thr = np.array([10.0, 20.0, 40.0])
    bs = np.array([2.0, 4.0, 1.0])
    tps = np.array([512.0, 256.0, 1024.0])
    gpus = np.array([8.0, 4.0, 2.0])

    lat = throughput_to_latency(thr, bs, tps, gpus)
    total_tokens = bs * tps * gpus
    recovered_thr = total_tokens / lat
    np.testing.assert_allclose(recovered_thr, thr)


def test_throughput_to_latency_zero_throughput_is_nan():
    """A throughput of 0 gives NaN latency."""
    lat = throughput_to_latency(
        throughput=np.array([0.0, 10.0]),
        batch_size=np.array([1.0, 1.0]),
        tokens_per_sample=np.array([1.0, 1.0]),
        total_gpus=np.array([1.0, 1.0]),
    )
    assert math.isnan(lat[0])
    assert lat[1] == pytest.approx(0.1)  # 1*1*1/10


def test_throughput_to_latency_accepts_scalars():
    lat = throughput_to_latency(10.0, 2.0, 512.0, 8.0)
    assert float(lat) == pytest.approx(819.2)


# ms_per_100_predictions


def test_ms_per_100_predictions_known_value():
    """50 predictions in 2 s is 4000 ms per 100."""
    assert ms_per_100_predictions(2.0, 50) == pytest.approx(4000.0)


def test_ms_per_100_predictions_nonpositive_n_is_nan():
    """A count n <= 0 gives NaN."""
    assert math.isnan(ms_per_100_predictions(2.0, 0))
    assert math.isnan(ms_per_100_predictions(2.0, -3))


def test_ms_per_100_predictions_none_time_is_nan():
    """An elapsed time of None gives NaN."""
    assert math.isnan(ms_per_100_predictions(None, 50))
