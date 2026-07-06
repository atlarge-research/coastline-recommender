"""
Tests for benchmark.metrics -- the single source of truth for the thesis's
headline accuracy numbers (MdAPE / MAPE / within-X and the throughput<->latency
conversion).

All expected values are hand-computed in the docstrings so the tests anchor the
implementation rather than restating it. No external data; fast & deterministic.
"""

import math

import numpy as np
import pytest

from benchmark.metrics import (
    compute_metrics,
    ms_per_100_predictions,
    throughput_to_latency,
)

# ---------------------------------------------------------------------------
# compute_metrics -- MdAPE / MAPE / within-X on known inputs
# ---------------------------------------------------------------------------


def test_mdape_mape_basic_known_values():
    """y_true all 100; preds 100/110/120/130/140 -> pct errors 0/10/20/30/40.

    median(0,10,20,30,40) = 20  (MdAPE)
    mean(0,10,20,30,40)   = 20  (MAPE)
    """
    y_true = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    y_pred = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
    m = compute_metrics(y_true, y_pred)
    assert m["n"] == 5
    assert m["mdape"] == pytest.approx(20.0)
    assert m["mape"] == pytest.approx(20.0)


def test_mdape_differs_from_mape_on_skewed_errors():
    """y_true=[100,200], y_pred=[50,150] -> pct errors [50, 25].

    median([25,50]) = 37.5 ; mean([25,50]) = 37.5  (here equal for 2 pts),
    so use a 3-pt skewed set to separate them:
        errors [10, 20, 90] -> mdape=20, mape=40.
    """
    # 2-point sanity (median == mean for two points)
    m2 = compute_metrics([100.0, 200.0], [50.0, 150.0])
    assert m2["mdape"] == pytest.approx(37.5)
    assert m2["mape"] == pytest.approx(37.5)

    # 3-point skewed: errors 10, 20, 90
    y_true = np.array([100.0, 100.0, 100.0])
    y_pred = np.array([110.0, 120.0, 190.0])
    m3 = compute_metrics(y_true, y_pred)
    assert m3["mdape"] == pytest.approx(20.0)
    assert m3["mape"] == pytest.approx(40.0)
    assert m3["mdape"] != m3["mape"]


def test_within_thresholds_known_distribution():
    """pct errors 0/10/20/30/40 (5 pts). within_X uses STRICT '< X'.

    within_10: only {0}                 -> 1/5 = 20%
    within_20: {0,10}                   -> 2/5 = 40%
    within_30: {0,10,20}                -> 3/5 = 60%
    """
    y_true = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    y_pred = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
    m = compute_metrics(y_true, y_pred)
    assert m["within_10"] == pytest.approx(20.0)
    assert m["within_20"] == pytest.approx(40.0)
    assert m["within_30"] == pytest.approx(60.0)


def test_within_threshold_is_strict_inequality():
    """An error of exactly 10% must NOT be counted in within_10 (strict '<')."""
    m = compute_metrics([100.0], [110.0])  # exactly 10% error
    assert m["within_10"] == pytest.approx(0.0)
    assert m["within_20"] == pytest.approx(100.0)  # 10 < 20


def test_perfect_prediction_zero_error():
    """Identical y_true/y_pred (non-degenerate): all error metrics ~0, r2=1."""
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
    """y_true=[10,20,30,40], y_pred=[12,18,33,36] -> residuals [2,-2,3,-4].

    MAE  = mean(|2|,|2|,|3|,|4|)           = 11/4   = 2.75
    RMSE = sqrt(mean(4,4,9,16))            = sqrt(33/4) = sqrt(8.25)
    R2   = 1 - SS_res/SS_tot
           SS_res = 4+4+9+16 = 33
           mean(y_true)=25 ; SS_tot = 225+25+25+225 = 500
           R2 = 1 - 33/500 = 0.934
    """
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 36.0])
    m = compute_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx(2.75)
    assert m["rmse"] == pytest.approx(math.sqrt(8.25))
    assert m["r2"] == pytest.approx(0.934)


# ---------------------------------------------------------------------------
# compute_metrics -- masking / edge cases
# ---------------------------------------------------------------------------


def test_zero_in_y_true_is_masked_out():
    """y_true=0 is dropped (mask y_true>0), avoiding divide-by-zero.

    Remaining pts: errors [10, 30] -> mdape=20, mape=20, n=2.
    """
    y_true = np.array([0.0, 100.0, 100.0])
    y_pred = np.array([50.0, 110.0, 130.0])
    m = compute_metrics(y_true, y_pred)
    assert m["n"] == 2
    assert m["mdape"] == pytest.approx(20.0)
    assert m["mape"] == pytest.approx(20.0)


def test_nonpositive_y_pred_is_masked_out():
    """y_pred<=0 is dropped (mask requires y_pred>0)."""
    y_true = np.array([100.0, 100.0])
    y_pred = np.array([0.0, 90.0])
    m = compute_metrics(y_true, y_pred)
    assert m["n"] == 1
    assert m["mdape"] == pytest.approx(10.0)


def test_nan_and_inf_inputs_are_masked_out():
    """NaN/inf in either array are dropped (mask requires np.isfinite on both).

    Only the first pair (100, 110) survives -> n=1, error 10.
    """
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
    """Empty arrays -> all-NaN dict with the full set of keys."""
    m = compute_metrics([], [])
    expected_keys = {"n", "mdape", "mape", "r2", "rmse", "mae", "within_10", "within_20", "within_30"}
    assert set(m.keys()) == expected_keys
    assert all(math.isnan(v) for v in m.values())


def test_single_element_input():
    """Single surviving pair: percentage metrics well-defined; r2 is NaN
    (sklearn's r2 is undefined for <2 samples)."""
    m = compute_metrics([50.0], [55.0])  # 10% error
    assert m["n"] == 1
    assert m["mdape"] == pytest.approx(10.0)
    assert m["mape"] == pytest.approx(10.0)
    assert m["mae"] == pytest.approx(5.0)
    assert m["rmse"] == pytest.approx(5.0)
    assert m["within_20"] == pytest.approx(100.0)
    assert math.isnan(m["r2"])


def test_all_identical_y_true_multi_sample_r2_is_nan():
    """EDGE: with >=2 samples but a constant y_true, SS_tot=0 so R2 is
    mathematically undefined. sklearn would return 0.0 here, which is
    misleading -- compute_metrics reports NaN instead.

    The error metrics remain well-defined and unaffected.
    y_true=[100,100,100], y_pred=[110,120,130] -> errors 10/20/30.
    """
    y_true = np.array([100.0, 100.0, 100.0])
    y_pred = np.array([110.0, 120.0, 130.0])
    m = compute_metrics(y_true, y_pred)
    assert math.isnan(m["r2"])
    assert m["mdape"] == pytest.approx(20.0)  # median(10,20,30)
    assert m["mae"] == pytest.approx(20.0)


def test_returns_native_python_floats_not_numpy():
    """Metrics are cast to native float/int (n) for clean JSON/CSV export."""
    m = compute_metrics([100.0, 100.0], [110.0, 130.0])
    assert isinstance(m["n"], int)
    for key in ("mdape", "mape", "r2", "rmse", "mae", "within_10", "within_20", "within_30"):
        assert isinstance(m[key], float)


def test_accepts_python_lists_and_is_order_independent():
    """Lists are accepted (np.asarray); MdAPE/MAPE are order-invariant."""
    a = compute_metrics([100.0, 200.0, 50.0], [110.0, 180.0, 60.0])
    b = compute_metrics([50.0, 100.0, 200.0], [60.0, 110.0, 180.0])
    assert a["mdape"] == pytest.approx(b["mdape"])
    assert a["mape"] == pytest.approx(b["mape"])
    assert a["n"] == b["n"] == 3


# ---------------------------------------------------------------------------
# throughput_to_latency -- conversion + round-trip
# ---------------------------------------------------------------------------


def test_throughput_to_latency_known_value():
    """latency = (batch_size * tokens_per_sample * total_gpus) / throughput.

    bs=2, tps=512, gpus=8, thr=10 -> 2*512*8/10 = 8192/10 = 819.2 s.
    """
    lat = throughput_to_latency(
        throughput=np.array([10.0]),
        batch_size=np.array([2.0]),
        tokens_per_sample=np.array([512.0]),
        total_gpus=np.array([8.0]),
    )
    assert lat[0] == pytest.approx(819.2)


def test_throughput_to_latency_round_trip():
    """Converting throughput->latency and back recovers the throughput:
    thr = (bs * tps * gpus) / latency."""
    thr = np.array([10.0, 20.0, 40.0])
    bs = np.array([2.0, 4.0, 1.0])
    tps = np.array([512.0, 256.0, 1024.0])
    gpus = np.array([8.0, 4.0, 2.0])

    lat = throughput_to_latency(thr, bs, tps, gpus)
    total_tokens = bs * tps * gpus
    recovered_thr = total_tokens / lat
    np.testing.assert_allclose(recovered_thr, thr)


def test_throughput_to_latency_zero_throughput_is_nan():
    """throughput<=0 yields NaN latency (no divide-by-zero blow-up)."""
    lat = throughput_to_latency(
        throughput=np.array([0.0, 10.0]),
        batch_size=np.array([1.0, 1.0]),
        tokens_per_sample=np.array([1.0, 1.0]),
        total_gpus=np.array([1.0, 1.0]),
    )
    assert math.isnan(lat[0])
    assert lat[1] == pytest.approx(0.1)  # 1*1*1/10


def test_throughput_to_latency_accepts_scalars():
    """Scalar (0-d) inputs are accepted via np.asarray and broadcast."""
    lat = throughput_to_latency(10.0, 2.0, 512.0, 8.0)
    assert float(lat) == pytest.approx(819.2)


# ---------------------------------------------------------------------------
# ms_per_100_predictions -- timing helper
# ---------------------------------------------------------------------------


def test_ms_per_100_predictions_known_value():
    """elapsed=2.0s over n=50 preds -> (2/50)*100*1000 ms = 4000 ms."""
    assert ms_per_100_predictions(2.0, 50) == pytest.approx(4000.0)


def test_ms_per_100_predictions_nonpositive_n_is_nan():
    """n<=0 -> NaN (guards divide-by-zero / nonsensical counts)."""
    assert math.isnan(ms_per_100_predictions(2.0, 0))
    assert math.isnan(ms_per_100_predictions(2.0, -3))


def test_ms_per_100_predictions_none_time_is_nan():
    """A None elapsed time (unmeasured) -> NaN."""
    assert math.isnan(ms_per_100_predictions(None, 50))
