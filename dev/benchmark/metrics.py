"""Unified accuracy metrics for the benchmark suite: MdAPE/MAPE/within-X and throughput↔latency conversion."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

METRIC_KEYS = (
    "n",
    "mdape",
    "mape",
    "r2",
    "rmse",
    "mae",
    "within_10",
    "within_20",
    "within_30",
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression accuracy metrics (METRIC_KEYS) over rows where y_true and y_pred > 0."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mask = (y_true > 0) & (y_pred > 0) & np.isfinite(y_true) & np.isfinite(y_pred)
    y_t = y_true[mask]
    y_p = y_pred[mask]

    if len(y_t) == 0:
        return {k: float("nan") for k in METRIC_KEYS}

    pct_errors = np.abs(y_p - y_t) / y_t * 100

    # R2 is undefined when SS_tot = 0 (constant y_true over >=2 samples); sklearn
    # returns a misleading 0.0/1.0 there, so report NaN. (<2 samples: already NaN.)
    if len(y_t) >= 2 and np.all(y_t == y_t[0]):
        r2 = float("nan")
    else:
        r2 = float(r2_score(y_t, y_p))

    return {
        "n": int(len(y_t)),
        "mdape": float(np.median(pct_errors)),
        "mape": float(np.mean(pct_errors)),
        "r2": r2,
        "rmse": float(np.sqrt(mean_squared_error(y_t, y_p))),
        "mae": float(mean_absolute_error(y_t, y_p)),
        "within_10": float(np.mean(pct_errors < 10) * 100),
        "within_20": float(np.mean(pct_errors < 20) * 100),
        "within_30": float(np.mean(pct_errors < 30) * 100),
    }


def throughput_to_latency(
    throughput: np.ndarray, batch_size: np.ndarray, tokens_per_sample: np.ndarray, total_gpus: np.ndarray
) -> np.ndarray:
    """Derive per-step latency (s): (batch_size * tokens_per_sample * total_gpus) / throughput."""
    total_tokens_per_step = (
        np.asarray(batch_size, dtype=np.float64)
        * np.asarray(tokens_per_sample, dtype=np.float64)
        * np.asarray(total_gpus, dtype=np.float64)
    )
    thr = np.asarray(throughput, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        latency = np.where(thr > 0, total_tokens_per_step / thr, np.nan)
    return latency


def ms_per_100_predictions(predict_time_s: float, n: int) -> float:
    """Wall time (ms) to complete 100 predictions, from ``n`` measured runs."""
    if n <= 0 or predict_time_s is None:
        return float("nan")
    return float(predict_time_s) / float(n) * 100.0 * 1000.0
