"""Tests for the OpenDC ``CalibrationEngine`` sweep.

The sweep runs one OpenDC sim per ``calibrationFactor`` on a ``linspace`` grid,
computes MAPE of each simulated power curve against a single ``actual_power``
curve, and reports the factor with the smallest MAPE. Because the factor is
selected ON the same curve the MAPE is measured on, ``best_mape`` is an
optimistic *in-sample* fit, not a held-out generalization estimate. With one
power series and a single scalar to fit, a clean held-out split is infeasible,
so the contract is that the result clearly LABELS the number as in-sample.

The OpenDC binary is never invoked here: ``_run_single_calibration_sim`` is
mocked so every factor ``f`` yields a flat curve at ``100*f`` W. With a flat
ground truth at ``A`` W and a flat sim at ``S`` W the whole MAPE collapses to a
single closed form we can derive by hand:

    MAPE(f) = mean(|A - S| / A) * 100 = |A - 100*f| / A * 100  (%)

That closed form (a *different* expression than the vectorised implementation in
``mape.py``) is the independent oracle for every value asserted below.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from coastline.sdk.predictors.energy.opendc import calibration as calib_mod
from coastline.sdk.predictors.energy.opendc.calibration import (
    CalibrationEngine,
    CalibrationResult,
    CalibrationSweepResult,
)


def _power_df(start: datetime, n: int, level: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [start + timedelta(seconds=60 * i) for i in range(n)],
            "power_draw": [level for _ in range(n)],
        }
    )


class _InlineExecutor:
    """Synchronous stand-in for ProcessPoolExecutor so a monkeypatched, in-process
    ``_run_single_calibration_sim`` is actually used (no pickling to a subprocess)."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args, **kwargs):
        class _Future:
            def __init__(self, value):
                self._value = value

            def result(self):
                return self._value

        return _Future(fn(*args, **kwargs))


def _run_sweep(monkeypatch, actual_level: float = 100.0, fail_all: bool = False):
    """Run a hermetic sweep over factors linspace(0.5, 1.5, 11).

    Each factor ``f`` yields a flat sim curve at ``100*f`` W; the ground truth is
    flat at ``actual_level`` W. When ``fail_all`` is set every sim reports failure
    (no power_df), exercising the "all sims failed" error contract.
    """
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(minutes=20)
    actual = _power_df(start, 21, actual_level)

    def fake_sim(sim_number, calibration_factor, *args, **kwargs):
        if fail_all:
            return CalibrationResult(
                sim_number=sim_number,
                calibration_factor=calibration_factor,
                power_df=None,
                success=False,
                error_message="boom",
            )
        return CalibrationResult(
            sim_number=sim_number,
            calibration_factor=calibration_factor,
            power_df=_power_df(start, 21, 100.0 * calibration_factor),
            success=True,
        )

    monkeypatch.setattr(calib_mod, "_run_single_calibration_sim", fake_sim)
    # Run synchronously in-process so the monkeypatched fake_sim is honored and the
    # canned futures (plain objects) iterate without real-future bookkeeping.
    monkeypatch.setattr(calib_mod, "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(calib_mod, "as_completed", lambda futures: list(futures))

    engine = CalibrationEngine(max_workers=1)
    return engine.run_calibration_sweep(
        gpu_model="NVIDIA-A100-SXM4-80GB",
        gpus_per_node=1,
        number_of_nodes=1,
        workload_dir=Path("/tmp/unused"),
        actual_power=actual,
        simulation_end_time=end,
        min_factor=0.5,
        max_factor=1.5,
        num_points=11,
        mape_window_minutes=60,
    )


def test_best_factor_tracks_actual_power_not_hardcoded(monkeypatch):
    # Ground truth flat at 90 W. Sim at factor f is flat at 100*f, so factor 0.9
    # reproduces 90 W exactly -> MAPE(0.9) = |90 - 90|/90 * 100 = 0, the unique
    # minimum. A hardcoded / defaulted "1.0" would give MAPE(1.0) =
    # |90 - 100|/90 * 100 = 11.1% and lose. So the selected factor must be 0.9,
    # proving the choice is driven by the data, not fixed at 1.0.
    result = _run_sweep(monkeypatch, actual_level=90.0)
    assert result.best_calibration_factor == pytest.approx(0.9)
    assert result.best_mape == pytest.approx(0.0, abs=1e-6)


def test_best_mape_equals_relative_error_at_best_factor(monkeypatch):
    # Ground truth flat at 104 W; grid sims land on {50,60,...,150} W. Absolute
    # gaps |104 - S|: 54,44,34,24,14,4,6,16,... -> smallest at S=100 (factor 1.0).
    # Because every MAPE divides by the same 104, the min-|gap| factor is also the
    # min-MAPE factor. Reported best_mape = |104 - 100|/104 * 100 = 400/104
    # = 3.84615...% (a nonzero value, not the trivial perfect-match 0).
    result = _run_sweep(monkeypatch, actual_level=104.0)
    assert result.best_calibration_factor == pytest.approx(1.0)
    assert result.best_mape == pytest.approx(3.846153846, abs=1e-6)


def test_per_factor_mape_matches_relative_error(monkeypatch):
    # Ground truth flat at 100 W; each entry's MAPE must equal |100 - 100*f|/100*100
    # = |100 - 100*f| (%). Hand-derived for four factors spanning the grid:
    #   f=0.5 -> |100-50|  = 50 % ;  f=0.8 -> |100-80| = 20 %
    #   f=1.0 -> |100-100| = 0  % ;  f=1.5 -> |100-150| = 50 %
    result = _run_sweep(monkeypatch, actual_level=100.0)
    by_factor = {round(e["calibration_factor"], 3): e["mape"] for e in result.all_results}
    assert by_factor[0.5] == pytest.approx(50.0)
    assert by_factor[0.8] == pytest.approx(20.0)
    assert by_factor[1.0] == pytest.approx(0.0, abs=1e-6)
    assert by_factor[1.5] == pytest.approx(50.0)


def test_grid_covers_linspace_of_factors(monkeypatch):
    # min=0.5, max=1.5, num_points=11 -> a uniform grid of step 0.1, enumerated by
    # hand: 0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5 (exactly 11 factors, no
    # endpoints dropped or duplicated).
    result = _run_sweep(monkeypatch, actual_level=100.0)
    factors = sorted(round(e["calibration_factor"], 3) for e in result.all_results)
    expected = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    assert factors == pytest.approx(expected)
    assert all(e["success"] for e in result.all_results)


def test_best_mape_is_labeled_in_sample(monkeypatch):
    # The reported best_mape was measured on the SAME curve used to pick the factor;
    # the result must make that explicit so callers don't read it as held-out.
    result = _run_sweep(monkeypatch, actual_level=100.0)
    assert isinstance(result, CalibrationSweepResult)
    assert getattr(result, "best_mape_is_in_sample", None) is True, (
        "best_mape is an in-sample fit (factor chosen to minimize it on the same "
        "data) and must be explicitly labeled as such"
    )


def test_sweep_raises_when_all_sims_fail(monkeypatch):
    # Contract: with zero valid MAPE results the sweep cannot report a best factor
    # and must raise rather than return a bogus/NaN winner.
    with pytest.raises(RuntimeError):
        _run_sweep(monkeypatch, actual_level=100.0, fail_all=True)
