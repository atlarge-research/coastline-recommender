#!/usr/bin/env python3
"""Exp1 energy predictor comparison: Kavier-energy vs OpenDC.

Runs both energy predictors on the sfttrainer test workloads and records:
  1. Predicted power (watts) per workload from each predictor
  2. Per-prediction wall-clock time
  3. OpenDC speedup curve (1, 2, 4, 6, 8, 10, 12 workers)

Output CSVs → reproducibility-capsule/data/exp1/
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pandas as pd
from tqdm import tqdm

from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.energy.kavier.kavier_power_predictor import KavierPowerPredictor
from coastline.sdk.predictors.energy.opendc.predictor import OpenDCEnergyPredictor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reproducibility-capsule" / "data" / "exp1"
DATA_CSV = Path(os.environ.get("DATA_DIR", str(ROOT / "trace-archive"))) / "profiling-dataset" / "curated_trace.csv"

CONTEXT = SystemContext(
    available_gpu_models=["NVIDIA-A100-SXM4-80GB"],
    max_gpus=128,
    gpu_memory={"NVIDIA-A100-SXM4-80GB": 80},
    constraints=Constraints(max_gpus=128, gpus_per_node=8, max_nodes=16),
)


def _build_workloads() -> list[WorkloadSpec]:
    df = pd.read_csv(DATA_CSV)
    workloads = []
    for _, row in df.iterrows():
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


def run_predictions(workloads: list[WorkloadSpec], out_csv: Path):
    """Run both Kavier and OpenDC on every workload; save per-prediction CSV."""
    kavier = KavierPowerPredictor()
    opendc = OpenDCEnergyPredictor(max_workers=12)

    rows = []
    for i, w in enumerate(tqdm(workloads, desc="predictions")):
        total_gpus = (w.gpus_per_node or 1) * (w.number_of_nodes or 1)

        t0 = time.perf_counter()
        kp = kavier.predict(w, CONTEXT)
        kavier_ms = (time.perf_counter() - t0) * 1000
        kavier_power = kp.predicted_power if kp and kp.predicted_power else None
        kavier_total = kavier_power * total_gpus if kavier_power else None

        t0 = time.perf_counter()
        try:
            op = opendc.predict(w, CONTEXT)
            opendc_ms = (time.perf_counter() - t0) * 1000
            opendc_power = op.predicted_power if op else None
        except Exception as e:
            opendc_ms = (time.perf_counter() - t0) * 1000
            opendc_power = None
            tqdm.write(f"  OpenDC failed for workload {i}: {e}")

        rows.append(
            {
                "workload_id": i,
                "model": w.llm_model,
                "method": w.fine_tuning_method,
                "gpu_model": w.gpu_model,
                "batch_size": w.batch_size,
                "total_gpus": total_gpus,
                "kavier_power_per_gpu_w": kavier_power,
                "kavier_power_total_w": kavier_total,
                "opendc_power_total_w": opendc_power,
                "opendc_power_per_gpu_w": opendc_power / total_gpus if opendc_power else None,
                "kavier_time_ms": kavier_ms,
                "opendc_time_ms": opendc_ms,
            }
        )

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv} ({len(df)} rows)")

    ok = df.dropna(subset=["kavier_power_total_w", "opendc_power_total_w"])
    print(f"  Both predicted: {len(ok)}/{len(df)}")
    if len(ok):
        print(f"  Kavier total power:  median={ok['kavier_power_total_w'].median():.1f}W")
        print(f"  OpenDC total power:  median={ok['opendc_power_total_w'].median():.1f}W")
        print(f"  Kavier time:  median={ok['kavier_time_ms'].median():.3f} ms")
        print(f"  OpenDC time:  median={ok['opendc_time_ms'].median():.1f} ms")
    return df


def run_speedup(
    workloads: list[WorkloadSpec], out_csv: Path, worker_counts: list[int] | None = None, n_workloads: int = 24
):
    """OpenDC speedup curve: vary worker count on a fixed set of workloads."""
    if worker_counts is None:
        worker_counts = [1, 2, 4, 6, 8, 10, 12]

    subset = workloads[:n_workloads]
    print(f"\nSpeedup experiment: {len(subset)} workloads × {worker_counts} workers")

    rows = []
    for nw in tqdm(worker_counts, desc="speedup"):
        opendc = OpenDCEnergyPredictor(max_workers=nw)
        t0 = time.perf_counter()
        results = opendc.predict_batch(subset, CONTEXT)
        elapsed = time.perf_counter() - t0
        ok = sum(1 for r in results if r is not None)
        rows.append(
            {
                "workers": nw,
                "n_workloads": len(subset),
                "n_success": ok,
                "total_time_s": elapsed,
                "ms_per_prediction": elapsed / len(subset) * 1000,
            }
        )
        tqdm.write(
            f"  {nw:2d} workers: {elapsed:.1f}s total, {elapsed / len(subset) * 1000:.0f} ms/pred ({ok}/{len(subset)} ok)"
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    if len(df) > 1:
        t1 = df.loc[df["workers"] == 1, "total_time_s"].values
        if len(t1):
            df["speedup"] = t1[0] / df["total_time_s"]
            print(f"  Max speedup: {df['speedup'].max():.1f}× at {df.loc[df['speedup'].idxmax(), 'workers']} workers")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, default=OUT / "energy_comparison_predictions.csv")
    ap.add_argument("--speedup", type=Path, default=OUT / "energy_comparison_speedup.csv")
    ap.add_argument("--speedup-workloads", type=int, default=24, help="Number of workloads for speedup experiment")
    ap.add_argument("--skip-predictions", action="store_true")
    ap.add_argument("--skip-speedup", action="store_true")
    args = ap.parse_args()

    workloads = _build_workloads()
    print(f"{len(workloads)} workloads from {DATA_CSV.name}")

    if not args.skip_predictions:
        run_predictions(workloads, args.predictions)

    if not args.skip_speedup:
        run_speedup(workloads, args.speedup, n_workloads=args.speedup_workloads)
