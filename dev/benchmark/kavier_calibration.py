"""Refit Kavier calibration scales via Powell (log-space) for the gradient-accumulation ablation."""

from __future__ import annotations

import copy
import json
import os
from contextlib import contextmanager
from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from trainer.common import load_and_preprocess_data, split_data

from benchmark.metrics import compute_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]

# Where the shipped Kavier calibration table may live, in preference order: the sibling kavier
# source checkout first, then the same table as package data of an installed kavier.
_SIBLING_CALIBRATION_PATHS = (
    REPO_ROOT / "kavier" / "src" / "kavier" / "sdk" / "training" / "calibration" / "calibration.json",
)
_CALIBRATION_PACKAGE = "kavier.sdk.training"
_CALIBRATION_RESOURCE = ("calibration", "calibration.json")


@lru_cache(maxsize=1)
def load_v2_calibration() -> dict:
    """Return the shipped Kavier calibration table, loading it on first use.

    Deliberately not read at import time so this module (and pytest collection of
    anything that imports it) works in checkouts without the kavier sibling repo.
    Resolution order: sibling source checkout, then installed kavier.sdk.training package
    data; a clear error is raised only when the table is actually requested.
    """
    for path in _SIBLING_CALIBRATION_PATHS:
        if path.is_file():
            return json.loads(path.read_text())
    try:
        resource = resources.files(_CALIBRATION_PACKAGE).joinpath(*_CALIBRATION_RESOURCE)
        if resource.is_file():
            return json.loads(resource.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        pass
    raise FileNotFoundError(
        "Kavier calibration table not found. Tried the sibling source checkout ("
        + ", ".join(str(p) for p in _SIBLING_CALIBRATION_PATHS)
        + ") and the installed kavier.sdk.training package data "
        "(calibration/calibration.json). Clone kavier next to the "
        "coastline checkout or pip install kavier, then retry."
    )


def __getattr__(name: str):
    """Keep ``V2_CALIBRATION`` importable without eager file I/O (PEP 562 lazy attribute)."""
    if name == "V2_CALIBRATION":
        return load_v2_calibration()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), "V2_CALIBRATION"])


def load_train_val_test_rows() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (full_train, full_val, full_test) using the same deterministic split as `make evaluate`."""
    X_cat, X_num, y_df, _, _ = load_and_preprocess_data()
    y_thr = y_df["dataset_tokens_per_second"].values
    y_rt = y_df["train_runtime"].values
    data_dir = Path(os.environ.get("DATA_DIR", str(REPO_ROOT / "trace-archive")))
    full_df = pd.read_csv(data_dir / "profiling-dataset" / "curated_trace.csv")
    if "is_valid" in full_df.columns:
        full_df = full_df.loc[full_df["is_valid"] == 1.0].copy()
    m = (
        full_df["dataset_tokens_per_second"].notna()
        & (full_df["dataset_tokens_per_second"] > 0)
        & full_df["train_runtime"].notna()
        & (full_df["train_runtime"] > 0)
    )
    full_df = full_df.loc[m].copy()
    (_, _, _, _, full_train), (_, _, _, _, full_val), (_, _, _, _, full_test) = split_data(
        X_cat, X_num, y_thr, y_rt, full_df
    )
    return (full_train.reset_index(drop=True), full_val.reset_index(drop=True), full_test.reset_index(drop=True))


@contextmanager
def calibration_override(cal_dict: dict):
    """Temporarily swap the module-global calibration dict the engine reads at call time."""
    import kavier.sdk.training.calibration as cal

    saved = cal._CAL
    cal._CAL = cal_dict
    try:
        yield
    finally:
        cal._CAL = saved


def _row_args(rows: pd.DataFrame) -> list[dict]:
    """Pre-extract per-row engine kwargs for fast repeated evaluation."""
    args = []
    for _, r in rows.iterrows():
        total = int(r["number_gpus"]) * int(r["number_nodes"])
        args.append(
            dict(
                model_name=str(r["model_name"]),
                gpu_model=str(r["gpu_model"]),
                tokens_per_sample=int(r["tokens_per_sample"]),
                batch_size=int(r["batch_size"]),
                method=str(r["method"]),
                num_gpus=total,
                num_nodes=int(r["number_nodes"]),
            )
        )
    return args


def _predict(row_args: list[dict], grad_accum_steps: int, backward_factor: float, cal_dict: dict) -> np.ndarray:
    from kavier.sdk.training.core.engine import simulate_training_step

    out = np.empty(len(row_args), dtype=np.float64)
    with calibration_override(cal_dict):
        for i, a in enumerate(row_args):
            out[i] = simulate_training_step(**a, grad_accum_steps=grad_accum_steps, backward_factor=backward_factor)[
                "tokens_per_second"
            ]
    return out


def _vary_layout(rows: pd.DataFrame, base_cal: dict) -> list[tuple]:
    """Return (kind, key) pairs for the calibration entries exercised by the train rows."""
    models = sorted(rows["model_name"].astype(str).unique())
    methods = sorted(rows["method"].astype(str).unique())
    gpus = sorted(rows["gpu_model"].astype(str).unique())
    totals = sorted({int(a) * int(b) for a, b in zip(rows["number_gpus"], rows["number_nodes"])})
    mgc_table = base_cal["multi_gpu_correction"]["by_num_gpus"]
    mgc_keys = [str(t) for t in totals if t > 1 and str(t) in mgc_table]
    layout: list[tuple] = []
    if any(t > 1 for t in totals):
        layout.append(("comm_scale", None))
    layout += [("mfu_multiplier", g) for g in gpus]
    layout += [("mgc", k) for k in mgc_keys]
    layout += [("method_scale", m) for m in methods]
    layout += [("model_scale", m) for m in models]
    return layout


def _get(cal_dict: dict, kind: str, key):
    if kind == "comm_scale":
        return cal_dict["comm_scale"]
    if kind == "mfu_multiplier":
        return cal_dict["mfu_multiplier"][key]
    if kind == "mgc":
        return cal_dict["multi_gpu_correction"]["by_num_gpus"][key]
    if kind == "method_scale":
        return cal_dict["method_scale"][key]
    if kind == "model_scale":
        return cal_dict["model_scale"][key]
    raise KeyError(kind)


def _apply(base_cal: dict, layout: list[tuple], values) -> dict:
    c = copy.deepcopy(base_cal)
    for (kind, key), v in zip(layout, values):
        v = float(v)
        if kind == "comm_scale":
            c["comm_scale"] = v
        elif kind == "mfu_multiplier":
            c["mfu_multiplier"][key] = v
        elif kind == "mgc":
            c["multi_gpu_correction"]["by_num_gpus"][key] = v
        elif kind == "method_scale":
            c["method_scale"][key] = v
        elif kind == "model_scale":
            c["model_scale"][key] = v
    return c


def fit_calibration(
    train_rows: pd.DataFrame,
    grad_accum_steps: int,
    backward_factor: float,
    base_cal: dict | None = None,
    maxiter: int = 60,
    lam: float = 0.0,
) -> dict:
    """Refit exercised calibration scales (Powell, log-space) minimizing train MdAPE + lam * L2 toward v2 prior."""
    if base_cal is None:
        base_cal = load_v2_calibration()
    args = _row_args(train_rows)
    y_true = pd.to_numeric(train_rows["dataset_tokens_per_second"], errors="coerce").to_numpy(np.float64)
    layout = _vary_layout(train_rows, base_cal)
    log_x0 = np.log(np.array([_get(base_cal, k, key) for (k, key) in layout], dtype=np.float64))

    def objective(log_x: np.ndarray) -> float:
        c = _apply(base_cal, layout, np.exp(log_x))
        md = compute_metrics(y_true, _predict(args, grad_accum_steps, backward_factor, c))["mdape"]
        if not np.isfinite(md):
            return 1e9
        penalty = lam * float(np.mean((log_x - log_x0) ** 2))
        return md + penalty

    res = minimize(objective, log_x0, method="Powell", options={"maxiter": maxiter, "maxfev": 8000})
    return _apply(base_cal, layout, np.exp(res.x))


def select_calibration(
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    grad_accum_steps: int,
    backward_factor: float,
    base_cal: dict | None = None,
    lambdas: tuple = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0),
    maxiter: int = 60,
) -> tuple[dict, dict]:
    """Fit at each lambda; pick by best validation MdAPE. Returns (calibration, {choice, val_mdape})."""
    if base_cal is None:
        base_cal = load_v2_calibration()
    candidates = [("none(v2)", copy.deepcopy(base_cal))]
    for lam in lambdas:
        candidates.append(
            (f"lam={lam}", fit_calibration(train_rows, grad_accum_steps, backward_factor, base_cal, maxiter, lam))
        )
    best_tag, best_cal, best_val = None, None, np.inf
    for tag, cal_dict in candidates:
        vm = evaluate(val_rows, grad_accum_steps, backward_factor, cal_dict)["mdape"]
        if np.isfinite(vm) and vm < best_val:
            best_tag, best_cal, best_val = tag, cal_dict, vm
    return best_cal, {"choice": best_tag, "val_mdape": float(best_val)}


def evaluate(rows: pd.DataFrame, grad_accum_steps: int, backward_factor: float, cal_dict: dict) -> dict:
    """Compute throughput metrics for rows under a given engine variant and calibration."""
    args = _row_args(rows)
    y_true = pd.to_numeric(rows["dataset_tokens_per_second"], errors="coerce").to_numpy(np.float64)
    pred = _predict(args, grad_accum_steps, backward_factor, cal_dict)
    return compute_metrics(y_true, pred)
