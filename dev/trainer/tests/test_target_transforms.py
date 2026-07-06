"""Target transform round-trip + metric computation.

The whole ML pipeline trains in log1p space and reports in original space, so
the ``transform_targets`` (log1p) / ``inverse_transform_targets`` (expm1)
inverse round-trip is load-bearing: a mismatch silently corrupts every reported
number. These tests pin that contract and the dual-target shape handling.
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
    # Invariant: inverse ∘ transform is the identity. This is the load-bearing
    # contract (a log/expm1 vs log1p/exp mismatch would corrupt every report).
    y = _targets_frame()
    y_log = C.transform_targets(y)
    recovered = C.inverse_transform_targets(y_log)
    np.testing.assert_allclose(recovered, y.to_numpy(dtype=float), rtol=1e-9, atol=1e-9)


def test_transform_targets_matches_hand_computed_log1p():
    # Independent oracle: log1p(x) = ln(1+x). Construct inputs as e**k - 1 so the
    # expected log value is exactly k — derived by INVERTING (via exp) rather than
    # recomputing the impl's own np.log1p, so a swap to bare log/log2 would fail.
    #   x = e**1 - 1 -> ln(e)   = 1
    #   x = e**2 - 1 -> ln(e**2)= 2
    #   x = 0        -> ln(1)   = 0
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
    y.index = [5, 6, 7, 8, 9]  # non-default index must survive
    y_log = C.transform_targets(y)
    assert list(y_log.columns) == list(y.columns)
    assert list(y_log.index) == list(y.index)
    # Output is a DataFrame, two target columns wide.
    assert y_log.shape == y.shape


def test_inverse_transform_preserves_2d_dual_output_shape():
    """Model predictions arrive as an (n, 2) array (throughput + runtime); the
    inverse must operate element-wise and keep the 2-column shape (not flatten
    or transpose it), then recover the originals position-for-position."""
    y = _targets_frame()
    y_log = C.transform_targets(y).to_numpy()
    out = C.inverse_transform_targets(y_log)
    # Shape contract: 5 rows, 2 targets — a flatten would give (10,), a transpose (2, 5).
    assert out.shape == (5, 2)
    np.testing.assert_allclose(out, y.to_numpy(dtype=float), rtol=1e-9)


def test_inverse_transform_expm1_on_list_input():
    # Independent oracle: expm1(x) = e**x - 1, computed WITHOUT routing through
    # log1p at all (unlike a round-trip test). By hand:
    #   x = 0      -> e**0 - 1 = 0
    #   x = ln(2)  -> e**ln2 - 1 = 2 - 1 = 1
    #   x = ln(4)  -> e**ln4 - 1 = 4 - 1 = 3
    out = C.inverse_transform_targets([0.0, math.log(2.0), math.log(4.0)])
    np.testing.assert_allclose(out, [0.0, 1.0, 3.0], rtol=1e-9, atol=1e-12)


def test_zero_throughput_maps_to_log_zero_and_back():
    # Analytic reference point: log1p(0) = ln(1) = 0; the transform must be exact
    # at the origin, and expm1(0) = 0 restores it.
    y = pd.DataFrame({"a": [0.0], "b": [0.0]})
    y_log = C.transform_targets(y)
    assert float(y_log.iloc[0, 0]) == 0.0
    np.testing.assert_allclose(C.inverse_transform_targets(y_log), [[0.0, 0.0]])


# ---------------------------------------------------------------------------
# Target name accessors (used by trainers + downstream predictors)
# ---------------------------------------------------------------------------


def test_target_column_names_order_is_throughput_then_runtime():
    names = C.get_target_column_names()
    # Contract: throughput column FIRST, runtime SECOND. The inverse transform and
    # downstream predictors index column 0 as throughput, so order is load-bearing.
    assert names == ["dataset_tokens_per_second", "train_runtime"]
    assert C.get_primary_target_name() == "dataset_tokens_per_second"
    assert names[0] == C.get_primary_target_name()


# ---------------------------------------------------------------------------
# calculate_metrics: percentage metrics, zero-filtering, optional log space
# ---------------------------------------------------------------------------


def test_perfect_prediction_metrics_are_zero_error_and_unit_r2():
    # Analytic reference: y predicted exactly -> every error metric is 0, r2 is 1,
    # and 100% of points fall within the 20% band.
    y = np.array([100.0, 200.0, 400.0])
    m = C.calculate_metrics(y, y.copy())["original_space"]
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["mape"] == pytest.approx(0.0)
    assert m["mdape"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)
    assert m["within_20_pct"] == pytest.approx(100.0)


def test_mdape_is_median_and_mape_is_mean_of_abs_pct_error():
    # Hand-derived: true=100 for all, pred={110,120,150} -> abs pct errors {10,20,50}%.
    #   MdAPE = median{10,20,50} = 20
    #   MAPE  = mean{10,20,50}   = 80/3 ≈ 26.667
    #   within_20 (<= 0.20): 10% and 20% qualify, 50% does not -> 2/3 * 100 = 200/3
    y_true = np.array([100.0, 100.0, 100.0])
    y_pred = np.array([110.0, 120.0, 150.0])
    m = C.calculate_metrics(y_true, y_pred)["original_space"]
    assert m["mdape"] == pytest.approx(20.0)
    assert m["mape"] == pytest.approx(80.0 / 3.0)
    assert m["within_20_pct"] == pytest.approx(200.0 / 3.0)


def test_percentage_metrics_mask_out_nonpositive_true_values():
    """Percentage metrics mask on ``y_true > 0`` so a zero ground truth does not
    blow the division up to inf/nan. Only the two positive points count, each with
    a hand-derived 10% error: 100->110 and 200->220."""
    y_true = np.array([0.0, 100.0, 200.0])
    y_pred = np.array([5.0, 110.0, 220.0])
    m = C.calculate_metrics(y_true, y_pred)["original_space"]
    # If the 0-true point were NOT masked, |0-5|/0 -> inf and both stats blow up.
    assert m["mdape"] == pytest.approx(10.0)
    assert m["mape"] == pytest.approx(10.0)
    assert np.isfinite(m["mape"])  # guards against the unmasked division-by-zero bug


def test_log_space_block_present_only_when_log_inputs_given():
    # Contract: log_space key is absent without log arrays, present with them.
    y = np.array([10.0, 20.0])
    base = C.calculate_metrics(y, y)
    assert "log_space" not in base
    yl = np.log1p(y)
    with_log = C.calculate_metrics(y, y, yl, yl)
    assert "log_space" in with_log
    # Perfect log-space prediction -> r2 == 1 (analytic reference point).
    assert with_log["log_space"]["r2"] == pytest.approx(1.0)


def test_metrics_accept_column_vector_shape():
    # calculate_metrics reshapes to (-1,), so an (n, 1) column vector must behave
    # identically to a flat one. Perfect prediction -> MdAPE 0 (analytic reference).
    col = np.array([[10.0], [20.0]])
    m = C.calculate_metrics(col, col)["original_space"]
    assert m["mdape"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)
