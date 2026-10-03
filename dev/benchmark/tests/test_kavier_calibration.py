import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark import kavier_calibration as kc
from benchmark.kavier_calibration import (
    calibration_override,
    evaluate,
    fit_calibration,
    load_train_val_test_rows,
    select_calibration,
)

try:
    V2_CALIBRATION = kc.load_v2_calibration()
except FileNotFoundError as exc:
    pytest.skip(f"Kavier calibration table unavailable: {exc}", allow_module_level=True)

pytest.importorskip(
    "kavier.sdk.training",
    reason="kavier.sdk.training engine not importable (pip install kavier to run these tests)",
)

COASTLINE_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def rows():
    try:
        return load_train_val_test_rows()  # (train, val, test)
    except FileNotFoundError as exc:
        pytest.skip(f"coastline profiling trace unavailable: {exc}")


def test_import_is_lazy_no_calibration_read():
    """Importing benchmark.kavier_calibration reads no calibration file, so pytest can collect
    the suite without a kavier checkout."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in ("common", ".", env.get("PYTHONPATH", "")) if p)
    code = (
        "import benchmark.kavier_calibration as kc; "
        "assert kc.load_v2_calibration.cache_info().currsize == 0, 'calibration read at import time'"
    )
    subprocess.run([sys.executable, "-c", code], cwd=COASTLINE_ROOT, env=env, check=True)


def test_loader_falls_back_to_installed_package(monkeypatch):
    """Without a kavier source checkout, the table loads from the installed kavier package data."""
    monkeypatch.setattr(kc, "_SIBLING_CALIBRATION_PATHS", ())
    kc.load_v2_calibration.cache_clear()
    try:
        cal = kc.load_v2_calibration()
        assert "comm_scale" in cal and "model_scale" in cal
    finally:
        kc.load_v2_calibration.cache_clear()


def test_loader_error_is_actionable(monkeypatch):
    """With neither a source checkout nor an installed package, the loader raises
    FileNotFoundError that says how to install kavier."""

    def _no_package(pkg):
        raise ModuleNotFoundError(pkg)

    monkeypatch.setattr(kc, "_SIBLING_CALIBRATION_PATHS", ())
    monkeypatch.setattr(kc.resources, "files", _no_package)
    kc.load_v2_calibration.cache_clear()
    try:
        with pytest.raises(FileNotFoundError, match="pip install kavier"):
            kc.load_v2_calibration()
    finally:
        kc.load_v2_calibration.cache_clear()


def test_v0_published_reproduces_baseline(rows):
    """The published engine (grad_accum_steps=1, backward_factor=2) with the calibration shipped in
    kavier.sdk.training gives about 5.55% test MdAPE on the four calibrated models."""
    _train, _val, test = rows
    assert evaluate(test, 1, 2.0, V2_CALIBRATION)["mdape"] == pytest.approx(5.55, abs=0.5)


def test_select_not_worse_than_v2_on_val(rows):
    """select_calibration includes v2 as a candidate, so its pick is no worse than v2 on the
    validation split."""
    train, val, _test = rows
    pub_val = evaluate(val, 1, 2.0, V2_CALIBRATION)["mdape"]
    cal_dict, info = select_calibration(train, val, 1, 2.0, lambdas=(0.0, 10.0), maxiter=8)
    sel_val = evaluate(val, 1, 2.0, cal_dict)["mdape"]
    assert sel_val <= pub_val + 1e-6


def test_calibration_override_restores_on_exception():
    """calibration_override restores `_CAL` when the body raises."""
    import kavier.sdk.training.calibration as cal_mod

    original = cal_mod._CAL
    with pytest.raises(RuntimeError):
        with calibration_override({"sentinel": True}):
            raise RuntimeError("boom")
    assert cal_mod._CAL is original


def test_v2_calibration_not_mutated_by_fit(rows):
    """fit_calibration leaves the shared calibration dict unchanged."""
    train, _val, _test = rows
    before_comm = V2_CALIBRATION["comm_scale"]
    before_model = dict(V2_CALIBRATION["model_scale"])
    fit_calibration(train, 4, 2.0, base_cal=V2_CALIBRATION, maxiter=5)
    assert V2_CALIBRATION["comm_scale"] == before_comm
    assert V2_CALIBRATION["model_scale"] == before_model
