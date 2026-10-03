"""Tests for the target transforms and the metric computation.

The models train on log1p targets and report in original space, so
``inverse_transform_targets`` (expm1) has to undo ``transform_targets`` (log1p);
a mismatch would skew every reported number.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from .. import common as C


def _targets_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            C.TARGET_COLUMNS["throughput"]: [50.0, 123.4, 1000.0, 4999.5, 0.5],
            C.TARGET_COLUMNS["runtime_seconds"]: [10.0, 600.0, 3600.0, 9999.0, 1.0],
        }
    )


def test_log1p_expm1_round_trip_recovers_original():
    # inverse_transform_targets(transform_targets(y)) == y.
    y = _targets_frame()
    y_log = C.transform_targets(y)
    recovered = C.inverse_transform_targets(y_log)
    np.testing.assert_allclose(recovered, y.to_numpy(dtype=float), rtol=1e-9, atol=1e-9)


def test_transform_targets_matches_log1p_of_known_values():
    # Inputs are e**k - 1, so log1p gives k (plain log or log2 would not):
    #   log1p(e - 1)    = ln(e)    = 1
    #   log1p(e**2 - 1) = ln(e**2) = 2
    #   log1p(0)        = ln(1)    = 0
    y = pd.DataFrame(
        {
            "a": [0.0, math.e - 1.0, math.e**2 - 1.0],
            "b": [math.e**3 - 1.0, 0.0, math.e - 1.0],
        }
    )
    y_log = C.transform_targets(y)
    expected = np.array([[0.0, 3.0], [1.0, 0.0], [2.0, 1.0]])
    np.testing.assert_allclose(y_log.to_numpy(), expected, atol=1e-12)


def test_transform_targets_preserves_columns_and_index():
    y = _targets_frame()
    y.index = [5, 6, 7, 8, 9]  # a non-default index is kept
    y_log = C.transform_targets(y)
    assert list(y_log.columns) == list(y.columns)
    assert list(y_log.index) == list(y.index)
    # Output is a DataFrame, two target columns wide.
    assert y_log.shape == y.shape


def test_inverse_transform_preserves_2d_dual_output_shape():
    """The inverse works element-wise on (n, 2) predictions (throughput, runtime) and keeps the shape."""
    y = _targets_frame()
    y_log = C.transform_targets(y).to_numpy()
    out = C.inverse_transform_targets(y_log)
    # 5 rows, 2 targets; a flatten would give (10,) and a transpose (2, 5).
    assert out.shape == (5, 2)
    np.testing.assert_allclose(out, y.to_numpy(dtype=float), rtol=1e-9)


def test_inverse_transform_expm1_on_list_input():
    # expm1(x) = e**x - 1, checked without going through log1p:
    #   expm1(0) = 0, expm1(ln 2) = 1, expm1(ln 4) = 3
    out = C.inverse_transform_targets([0.0, math.log(2.0), math.log(4.0)])
    np.testing.assert_allclose(out, [0.0, 1.0, 3.0], rtol=1e-9, atol=1e-12)


def test_zero_throughput_maps_to_log_zero_and_back():
    # log1p(0) = 0 and expm1(0) = 0 exactly.
    y = pd.DataFrame({"a": [0.0], "b": [0.0]})
    y_log = C.transform_targets(y)
    assert float(y_log.iloc[0, 0]) == 0.0
    np.testing.assert_allclose(C.inverse_transform_targets(y_log), [[0.0, 0.0]])


# Target name accessors (used by trainers + downstream predictors)


def test_target_column_names_order_is_throughput_then_runtime():
    names = C.get_target_column_names()
    # Throughput first, runtime second: downstream predictors read column 0 as throughput.
    assert names == ["dataset_tokens_per_second", "train_runtime"]
    assert C.get_primary_target_name() == "dataset_tokens_per_second"
    assert names[0] == C.get_primary_target_name()


# calculate_metrics: percentage metrics, zero-filtering, optional log space


def test_perfect_prediction_metrics_are_zero_error_and_unit_r2():
    # A perfect prediction: every error metric is 0, r2 is 1, and all points are within 20%.
    y = np.array([100.0, 200.0, 400.0])
    m = C.calculate_metrics(y, y.copy())["original_space"]
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["mape"] == pytest.approx(0.0)
    assert m["mdape"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)
    assert m["within_20_pct"] == pytest.approx(100.0)


def test_mdape_is_median_and_mape_is_mean_of_abs_pct_error():
    # true = 100 everywhere, pred = 110, 120, 150: absolute percentage errors 10, 20, 50.
    #   MdAPE = median = 20
    #   MAPE  = mean = 80/3 (about 26.667)
    #   within_20 (<= 0.20): 2 of 3 points, so 200/3
    y_true = np.array([100.0, 100.0, 100.0])
    y_pred = np.array([110.0, 120.0, 150.0])
    m = C.calculate_metrics(y_true, y_pred)["original_space"]
    assert m["mdape"] == pytest.approx(20.0)
    assert m["mape"] == pytest.approx(80.0 / 3.0)
    assert m["within_20_pct"] == pytest.approx(200.0 / 3.0)


def test_percentage_metrics_mask_out_nonpositive_true_values():
    """Percentage metrics use only points with ``y_true > 0``, so a zero truth does not divide by zero.

    The two positive points each have a 10% error (100 to 110, 200 to 220).
    """
    y_true = np.array([0.0, 100.0, 200.0])
    y_pred = np.array([5.0, 110.0, 220.0])
    m = C.calculate_metrics(y_true, y_pred)["original_space"]
    # Without the mask, |0-5|/0 would be inf.
    assert m["mdape"] == pytest.approx(10.0)
    assert m["mape"] == pytest.approx(10.0)
    assert np.isfinite(m["mape"])  # inf if the zero truth were not masked


def test_log_space_block_present_only_when_log_inputs_given():
    # log_space is present only when log arrays are passed.
    y = np.array([10.0, 20.0])
    base = C.calculate_metrics(y, y)
    assert "log_space" not in base
    yl = np.log1p(y)
    with_log = C.calculate_metrics(y, y, yl, yl)
    assert "log_space" in with_log
    # A perfect log-space prediction gives r2 == 1.
    assert with_log["log_space"]["r2"] == pytest.approx(1.0)


def test_metrics_accept_column_vector_shape():
    # calculate_metrics reshapes to (-1,), so an (n, 1) column vector works like a flat array.
    col = np.array([[10.0], [20.0]])
    m = C.calculate_metrics(col, col)["original_space"]
    assert m["mdape"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)
