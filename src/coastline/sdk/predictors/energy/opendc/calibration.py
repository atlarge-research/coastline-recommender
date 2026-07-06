"""Calibration engine: grid-searches calibrationFactor via parallel OpenDC sims, picks best MAPE.

NOTE: best_mape is IN-SAMPLE (minimized on the same actual_power curve; not held-out).
Strict: raises on failure. No fallbacks.
"""

from __future__ import annotations

import logging
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from coastline.sdk.io.odc_runner.topology import build_topology

from .mape import MapeComparator

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result from a single calibration simulation."""

    sim_number: int
    calibration_factor: float
    power_df: pd.DataFrame | None
    success: bool
    error_message: str | None = None


@dataclass
class CalibrationSweepResult:
    """Result of the full calibration sweep; best_mape_is_in_sample is always True."""

    best_calibration_factor: float
    best_mape: float
    all_results: list[dict[str, Any]]
    best_mape_is_in_sample: bool = True


def _run_single_calibration_sim(
    sim_number: int,
    calibration_factor: float,
    gpu_model: str,
    gpus_per_node: int,
    number_of_nodes: int,
    workload_dir: str,
    sim_dir: str,
    opendc_bin_path: str | None,
    timeout_seconds: int,
) -> CalibrationResult:
    """Run one OpenDC simulation with a specific calibrationFactor.

    Executed in a worker process via ProcessPoolExecutor.
    """
    from coastline.sdk.io.odc_runner.runner import OpenDCRunner

    try:
        bin_path = Path(opendc_bin_path) if opendc_bin_path else None
        runner = OpenDCRunner(bin_path)

        topology_dict = build_topology(
            gpu_model=gpu_model,
            gpus_per_node=gpus_per_node,
            number_of_nodes=number_of_nodes,
            calibration_factor=calibration_factor,
        )

        power_df = runner.run_simulation(
            workload_dir=Path(workload_dir),
            topology_dict=topology_dict,
            run_dir=Path(sim_dir),
            timeout_seconds=timeout_seconds,
        )

        return CalibrationResult(
            sim_number=sim_number,
            calibration_factor=calibration_factor,
            power_df=power_df,
            success=True,
        )

    except Exception as exc:
        logger.error("Calibration sim %d failed: %s", sim_number, exc)
        return CalibrationResult(
            sim_number=sim_number,
            calibration_factor=calibration_factor,
            power_df=None,
            success=False,
            error_message=str(exc),
        )


class CalibrationEngine:
    """Orchestrates parallel OpenDC calibration simulations."""

    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max_workers

    def run_calibration_sweep(
        self,
        gpu_model: str,
        gpus_per_node: int,
        number_of_nodes: int,
        workload_dir: Path,
        actual_power: pd.DataFrame,
        simulation_end_time: datetime,
        min_factor: float = 0.1,
        max_factor: float = 5.0,
        num_points: int = 10,
        mape_window_minutes: int = 60,
        opendc_bin_path: Path | None = None,
        timeout_seconds: int = 120,
    ) -> CalibrationSweepResult:
        """Run parallel OpenDC sims over a linspace of calibrationFactor values; return the best.

        Raises RuntimeError if all simulations fail or no valid MAPE is found.
        """
        factors = np.round(np.linspace(min_factor, max_factor, num_points), 3)

        logger.info(
            "Starting calibration sweep: %d points in [%.3f, %.3f]",
            num_points,
            min_factor,
            max_factor,
        )

        with tempfile.TemporaryDirectory(prefix="opendc_calib_") as tmp:
            tmp_path = Path(tmp)
            bin_str = str(opendc_bin_path) if opendc_bin_path else None

            results: list[CalibrationResult] = []
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for i, factor in enumerate(factors):
                    sim_dir = tmp_path / f"sim_{i}"
                    sim_dir.mkdir()
                    future = executor.submit(
                        _run_single_calibration_sim,
                        i,
                        float(factor),
                        gpu_model,
                        gpus_per_node,
                        number_of_nodes,
                        str(workload_dir),
                        str(sim_dir),
                        bin_str,
                        timeout_seconds,
                    )
                    futures[future] = (i, float(factor))

                for future in as_completed(futures):
                    sim_num, factor = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        status = "OK" if result.success else f"FAIL: {result.error_message}"
                        logger.info("Sim %d (factor=%.3f): %s", sim_num, factor, status)
                    except Exception as exc:
                        logger.error("Sim %d exception: %s", sim_num, exc)
                        results.append(
                            CalibrationResult(
                                sim_number=sim_num,
                                calibration_factor=factor,
                                power_df=None,
                                success=False,
                                error_message=str(exc),
                            )
                        )

        results.sort(key=lambda r: r.sim_number)

        comparator = MapeComparator(mape_window_minutes)
        all_mapes: list[dict[str, Any]] = []

        for r in results:
            entry: dict[str, Any] = {
                "sim_number": r.sim_number,
                "calibration_factor": r.calibration_factor,
                "success": r.success,
            }
            if r.success and r.power_df is not None:
                try:
                    mape_result = comparator.compare(r.power_df, actual_power, simulation_end_time)
                    entry["mape"] = mape_result["mape"]
                    entry.update(mape_result)
                except Exception as exc:
                    entry["mape"] = float("inf")
                    entry["mape_error"] = str(exc)
            else:
                entry["mape"] = float("inf")
                entry["error"] = r.error_message

            all_mapes.append(entry)

        valid = [m for m in all_mapes if m["mape"] < float("inf")]
        if not valid:
            raise RuntimeError(
                f"Calibration sweep failed: no valid MAPE results from {len(results)} simulations. "
                f"Errors: {[m.get('error') or m.get('mape_error') for m in all_mapes]}"
            )

        best = min(valid, key=lambda m: m["mape"])

        logger.info(
            # In-sample: this MAPE is the minimized fit on the SAME actual_power the
            # factor was selected against — not a held-out generalization estimate.
            "Calibration complete: best factor=%.3f, in-sample MAPE=%.2f%% (%d/%d succeeded)",
            best["calibration_factor"],
            best["mape"],
            len(valid),
            len(results),
        )

        return CalibrationSweepResult(
            best_calibration_factor=best["calibration_factor"],
            best_mape=best["mape"],
            all_results=all_mapes,
            best_mape_is_in_sample=True,
        )
