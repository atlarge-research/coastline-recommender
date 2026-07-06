#!/usr/bin/env python3
"""Exp1 timing: 12 models × N repeats -> timing_runs.csv + timing_summary.csv.

Each model is timed in its OWN subprocess (``--worker``) so the native ML backends
never co-load in one interpreter (co-loading several segfaults on macOS). The
measured ``predict_time_s`` is the in-loop prediction time, so subprocess startup
is not counted.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pandas as pd
from tqdm import tqdm

from benchmark.metrics import ms_per_100_predictions

OUT = Path(__file__).resolve().parents[2] / "reproducibility-capsule" / "data" / "exp1"
MODELS = "CacheLookup Kavier RandomForest XGBoost LightGBM CatBoost BayesianRidge SVR KNN GaussianProcess DeepLearning TabPFN".split()

_MARKER = "__TIMING__"


def _time_once(name: str, ml: dict) -> dict:
    """One timed evaluation of ``name`` over the test split (runs in a worker process)."""
    from benchmark.run_benchmark import (
        _ML_MODELS,
        evaluate_kavier,
        evaluate_ml_predictor,
        evaluate_tabpfn_batch,
    )

    if name == "Kavier":
        r = evaluate_kavier(ml)
    elif name == "TabPFN":
        r = evaluate_tabpfn_batch(ml)
    else:
        import importlib

        module, cls = _ML_MODELS[name]
        predictor = getattr(importlib.import_module(module), cls)()
        r = evaluate_ml_predictor(predictor, ml)
    t, n = float(r["predict_time_s"]), int(r["n"])
    ms = ms_per_100_predictions(t, n)
    return {"n": n, "predict_time_s": t, "ms_per_100": ms, "seconds_per_100": ms / 1000.0}


def _worker(name: str, repeats: int) -> None:
    """Worker: time one model ``repeats`` times; print each result as marked JSON."""
    import logging

    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    from benchmark.run_benchmark import prepare_ml_data

    ml = prepare_ml_data()
    for _ in range(repeats):
        print(_MARKER + json.dumps(_time_once(name, ml)), flush=True)


def _run_model_worker(name: str, repeats: int) -> list[dict]:
    """Spawn a child to time ``name`` ``repeats`` times in isolation; collect results."""
    proc = subprocess.run(
        [sys.executable, "-m", "benchmark.run_timing_study", "--worker", name, "--repeats", str(repeats)],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    return [json.loads(ln[len(_MARKER) :]) for ln in proc.stdout.splitlines() if ln.startswith(_MARKER)]


def run(n_repeats, runs_csv, summary_csv, force=False):
    if force and runs_csv.exists():
        runs_csv.unlink()
    df = pd.read_csv(runs_csv) if runs_csv.exists() and not force else pd.DataFrame()
    OUT.mkdir(parents=True, exist_ok=True)

    for name in tqdm(MODELS, desc="models", unit="model"):
        done = int((df["model"] == name).sum()) if len(df) and "model" in df else 0
        remaining = max(0, n_repeats - done)
        if remaining == 0:
            continue
        results = _run_model_worker(name, remaining)
        if len(results) < remaining:
            tqdm.write(f"  {name:16s} WARN: got {len(results)}/{remaining} timings")
        rows = []
        for k, res in enumerate(results, start=done + 1):
            rows.append({"model": name, "run_id": k, **res})
            ms = res["ms_per_100"]
            tqdm.write(f"  {name:16s} {ms:.4f} ms/100" if ms < 10 else f"  {name:16s} {ms:.2f} ms/100")
        if rows:
            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
            df.to_csv(runs_csv, index=False)

    g = df.groupby("model")["ms_per_100"]
    pd.DataFrame(
        {
            "model": g.mean().index,
            "n_runs": g.count().values,
            "mean_ms_per_100": g.mean().values,
            "std_ms_per_100": g.std(ddof=0).fillna(0).values,
            "median_ms_per_100": g.median().values,
            "min_ms_per_100": g.min().values,
            "max_ms_per_100": g.max().values,
            "mean_seconds_per_100": g.mean().values / 1000,
            "std_seconds_per_100": g.std(ddof=0).fillna(0).values / 1000,
        }
    ).to_csv(summary_csv, index=False)
    print(f"Saved {runs_csv} ({len(df)} rows), {summary_csv}", flush=True)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--output", type=Path, default=OUT / "timing_runs.csv")
    ap.add_argument("--summary", type=Path, default=OUT / "timing_summary.csv")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--worker", default=None, help="Internal: time one model in isolation.")
    a = ap.parse_args()
    if a.worker:
        _worker(a.worker, a.repeats)
        sys.exit(0)
    print(f"{len(MODELS)}×{a.repeats}={len(MODELS) * a.repeats} timings — make exp1-timing-100", flush=True)
    run(a.repeats, a.output, a.summary, a.force)
