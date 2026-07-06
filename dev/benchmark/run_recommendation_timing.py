#!/usr/bin/env python3
"""Exp1 recommendation timing: 4 policies × N repeats → recommendation_timing_runs.csv."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pandas as pd
from tqdm import tqdm

from benchmark.run_benchmark import prepare_ml_data
from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.policies.min_gpu import MinGPUStrategy
from coastline.sdk.policies.multi_objective import MultiObjectiveStrategy
from coastline.sdk.predictors.energy.kavier.kavier_power_predictor import KavierPowerPredictor
from coastline.sdk.predictors.performance.physics.kavier_predictor import KavierPredictor

OUT = Path(__file__).resolve().parents[2] / "trace-archive" / "out" / "exp1"

CONTEXT = SystemContext(
    available_gpu_models=["NVIDIA-A100-SXM4-80GB"],
    max_gpus=128,
    gpu_memory={"NVIDIA-A100-SXM4-80GB": 80},
    constraints=Constraints(max_gpus=128, gpus_per_node=8, max_nodes=16),
)

POLICIES = [
    ("min_gpu", None, None, None),
    ("balanced", "balanced", 0.5, 0.5),
    ("performance", "performance", 0.2, 0.8),
    ("energy", "energy", 0.8, 0.2),
]


def _build_workloads(ml_data: dict) -> list[WorkloadSpec]:
    full_test = ml_data["full_test"]
    workloads = []
    for idx in full_test.index:
        row = full_test.loc[idx]
        td = row.get("torch_dtype")
        td_s = None if pd.isna(td) else str(td).strip()
        roce = row.get("enable_roce")
        roce_b = None
        if not pd.isna(roce):
            try:
                roce_b = bool(float(roce) != 0.0)
            except (TypeError, ValueError):
                roce_b = None
        workloads.append(
            WorkloadSpec(
                llm_model=str(row["model_name"]),
                fine_tuning_method=str(row["method"]),
                gpu_model=str(row["gpu_model"]),
                tokens_per_sample=int(row["tokens_per_sample"]),
                batch_size=int(row["batch_size"]),
                gpus_per_node=int(row.get("number_gpus", 1)),
                number_of_nodes=int(row.get("number_nodes", 1)),
                torch_dtype=td_s,
                enable_roce=roce_b,
            )
        )
    return workloads


def _build_strategy(name, preset, alpha, beta, tp, pp):
    if name == "min_gpu":
        return MinGPUStrategy(
            throughput_predictor=tp,
            power_predictor=pp,
        )
    return MultiObjectiveStrategy(
        throughput_predictor=tp,
        power_predictor=pp,
        preset=preset,
        alpha=alpha,
        beta=beta,
    )


def _time_100_recommendations(strategy, workloads, context):
    n = len(workloads)
    successes = 0
    t0 = time.perf_counter()
    for w in workloads:
        try:
            strategy.recommend(w, context)
            successes += 1
        except Exception:
            pass
    elapsed = time.perf_counter() - t0
    ms_per_100 = (elapsed / n) * 100 * 1000 if n > 0 else 0.0
    return elapsed, n, successes, ms_per_100


def run(n_repeats: int, runs_csv: Path, summary_csv: Path, force: bool = False):
    if force and runs_csv.exists():
        runs_csv.unlink()
    df = pd.read_csv(runs_csv) if runs_csv.exists() and not force else pd.DataFrame()
    done = set()
    if len(df) and "run_id" in df.columns and "policy" in df.columns:
        done = set(zip(df["run_id"].astype(int), df["policy"]))

    ml_data = prepare_ml_data()
    workloads = _build_workloads(ml_data)
    print(f"  {len(workloads)} workloads loaded from test split")

    tp = KavierPredictor()
    pp = KavierPowerPredictor()

    OUT.mkdir(parents=True, exist_ok=True)

    pending_runs = sorted({r for r in range(1, n_repeats + 1) if any((r, p[0]) not in done for p in POLICIES)})

    for rid in tqdm(pending_runs, desc="runs", unit="run"):
        rows = []
        for pname, preset, alpha, beta in tqdm(POLICIES, desc=f"run {rid}", leave=False, unit="policy"):
            if (rid, pname) in done:
                continue
            strategy = _build_strategy(pname, preset, alpha, beta, tp, pp)
            elapsed, n, ok, ms100 = _time_100_recommendations(
                strategy,
                workloads,
                CONTEXT,
            )
            rows.append(
                {
                    "policy": pname,
                    "run_id": rid,
                    "n_workloads": n,
                    "n_success": ok,
                    "total_time_s": elapsed,
                    "ms_per_100": ms100,
                    "seconds_per_100": ms100 / 1000.0,
                }
            )
            tqdm.write(f"  {pname:16s} {ms100:>10.2f} ms/100  ({ok}/{n} ok)")

        if rows:
            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
            df.to_csv(runs_csv, index=False)

    g = df.groupby("policy")["ms_per_100"]
    summary = pd.DataFrame(
        {
            "policy": g.mean().index,
            "n_runs": g.count().values,
            "mean_ms_per_100": g.mean().values,
            "std_ms_per_100": g.std(ddof=0).fillna(0).values,
            "median_ms_per_100": g.median().values,
            "min_ms_per_100": g.min().values,
            "max_ms_per_100": g.max().values,
            "mean_seconds_per_100": g.mean().values / 1000,
            "std_seconds_per_100": g.std(ddof=0).fillna(0).values / 1000,
        }
    )
    summary.to_csv(summary_csv, index=False)
    print(f"\nSaved {runs_csv} ({len(df)} rows), {summary_csv}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--output", type=Path, default=OUT / "recommendation_timing_runs.csv")
    ap.add_argument("--summary", type=Path, default=OUT / "recommendation_timing_summary.csv")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    print(f"{len(POLICIES)}×{a.repeats}={len(POLICIES) * a.repeats} timings")
    run(a.repeats, a.output, a.summary, a.force)
