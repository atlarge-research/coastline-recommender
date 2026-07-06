"""OpenDC energy predictor: Kavier workload export -> OpenDC simulation -> per-GPU watts.

Strict: raises on failure. No fallbacks.
"""

from __future__ import annotations

import logging
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from coastline.sdk.io.odc_runner.runner import OpenDCRunner, OpenDCRunnerError
from coastline.sdk.io.odc_runner.topology import build_topology
from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.recommendation import Prediction
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.base import BasePredictor

logger = logging.getLogger(__name__)


def _runtime_or_none(train_runtime) -> Optional[float]:
    """Return positive runtime float, or None.

    OpenDC runs with total_tokens=None, so Kavier returns train_runtime=0.0 as a
    sentinel for "unknown" — surface that as None rather than a misleading 0.0s.
    """
    if train_runtime is None:
        return None
    try:
        value = float(train_runtime)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _fix_fragment_utilization(tasks_df, fragments_df):
    """Copy gpu_usage fraction into cpu_usage so OpenDC's MSE power model sees real utilisation.

    Kavier exports cpu_usage = 100% (capacity × GPUs); actual utilisation is in gpu_usage.
    """
    if fragments_df.empty or "gpu_usage" not in fragments_df.columns:
        return fragments_df
    gpu_cap = float(tasks_df["gpu_capacity"].iloc[0])
    cpu_cap = float(tasks_df["cpu_capacity"].iloc[0])
    total_gpus = int(tasks_df["gpu_count"].iloc[0])
    if gpu_cap <= 0 or total_gpus <= 0:
        return fragments_df
    utilisation = fragments_df["gpu_usage"] / (gpu_cap * total_gpus)
    fragments_df = fragments_df.copy()
    fragments_df["cpu_usage"] = utilisation * cpu_cap * total_gpus
    return fragments_df


def _run_single_opendc(
    workload_args: dict,
    calibration_factor: float,
    timeout_seconds: int,
    opendc_bin_path: str | None,
) -> dict:
    """Run one OpenDC simulation in a worker process. Returns a result dict."""
    from kavier.sdk.io.opendc.adapter import prepare_opendc_input
    from kavier.sdk.io.training_opendc import build_training_opendc_frames
    from kavier.sdk.training.core.engine import simulate_full_training, simulate_training_step

    from coastline.sdk.io.odc_runner.runner import OpenDCRunner
    from coastline.sdk.io.odc_runner.topology import build_topology

    bin_path = Path(opendc_bin_path) if opendc_bin_path else None

    w = workload_args
    gpus_per_node = w["gpus_per_node"]
    number_of_nodes = w["number_of_nodes"]

    with tempfile.TemporaryDirectory(prefix="opendc_energy_") as tmp:
        tmp_path = Path(tmp)
        workload_dir = tmp_path / "workload"
        workload_dir.mkdir()

        tasks_df, fragments_df, summary = build_training_opendc_frames(
            model_name=w["llm_model"],
            method=w["fine_tuning_method"],
            gpu_model=w["gpu_model"],
            tokens_per_sample=w["tokens_per_sample"],
            batch_size=w["batch_size"],
            number_gpus=gpus_per_node,
            number_nodes=number_of_nodes,
            total_tokens=None,
            task_id=w["task_id"],
            submission_time_ms=0,
            simulate_full_training_fn=simulate_full_training,
            simulate_training_step_fn=simulate_training_step,
        )

        fragments_df = _fix_fragment_utilization(tasks_df, fragments_df)
        prepare_opendc_input(tasks_df, fragments_df, str(workload_dir))

        topology_dict = build_topology(
            gpu_model=w["gpu_model"],
            gpus_per_node=gpus_per_node,
            number_of_nodes=number_of_nodes,
            calibration_factor=calibration_factor,
        )

        runner = OpenDCRunner(bin_path)
        run_dir = tmp_path / "run"
        power_df = runner.run_simulation(
            workload_dir=workload_dir,
            topology_dict=topology_dict,
            run_dir=run_dir,
            timeout_seconds=timeout_seconds,
        )

    if power_df.empty:
        return {"success": False, "error": "empty power trace", "task_id": w["task_id"]}

    mean_power = float(power_df["power_draw"].mean())
    return {
        "success": True,
        "task_id": w["task_id"],
        "mean_power": mean_power,
        "max_power": float(power_df["power_draw"].max()),
        "min_power": float(power_df["power_draw"].min()),
        "power_samples": len(power_df),
        "train_runtime": summary.get("train_runtime"),
    }


def _prefer_performance_cores() -> None:
    """ProcessPoolExecutor initializer: request USER_INITIATED QoS on macOS (advisory P-core bias).

    No-op off macOS. QoS is advisory only — no speed-up was measured, but it prevents
    E-core demotion (which only QOS_CLASS_BACKGROUND forces).
    """
    import sys

    if sys.platform != "darwin":
        return
    try:
        import ctypes

        _QOS_CLASS_USER_INITIATED = 0x19  # from <sys/qos.h>
        ctypes.CDLL("/usr/lib/libSystem.dylib").pthread_set_qos_class_self_np(
            ctypes.c_int(_QOS_CLASS_USER_INITIATED), ctypes.c_int(0)
        )
    except Exception:  # never fail the energy run over a scheduling hint
        pass


class OpenDCEnergyPredictor(BasePredictor):
    """Predicts power consumption via OpenDC datacenter simulation.

    Raises OpenDCRunnerError if anything fails. No fallbacks.
    """

    def __init__(
        self,
        calibration_factor: float = 1.0,
        opendc_bin_path: Path | None = None,
        timeout_seconds: int = 120,
        max_workers: int = 1,
    ) -> None:
        self._calibration_factor = calibration_factor
        self._timeout = timeout_seconds
        self._runner = OpenDCRunner(opendc_bin_path)
        self._bin_path = str(opendc_bin_path) if opendc_bin_path else None
        self._max_workers = max_workers
        logger.info(
            "OpenDCEnergyPredictor initialized (calibration=%.3f, workers=%d)",
            calibration_factor,
            max_workers,
        )

    def get_name(self) -> str:
        return "OpenDC Energy Simulator"

    def predict(
        self,
        workload: WorkloadSpec,
        context: SystemContext,
    ) -> Prediction:
        """Run one OpenDC simulation; return per-GPU watts. Raises OpenDCRunnerError on failure."""
        # kavier internals (there is no public training-OpenDC export yet) — imported
        # lazily so a kavier refactor breaks only this optional path at call time, never
        # the module import / test collection. See docs/kavier-integration.md.
        from kavier.sdk.io.opendc.adapter import prepare_opendc_input
        from kavier.sdk.io.training_opendc import build_training_opendc_frames
        from kavier.sdk.training.core.engine import simulate_full_training, simulate_training_step

        gpus_per_node = workload.gpus_per_node or 1
        number_of_nodes = workload.number_of_nodes or 1
        total_gpus = gpus_per_node * number_of_nodes

        with tempfile.TemporaryDirectory(prefix="opendc_energy_") as tmp:
            tmp_path = Path(tmp)
            workload_dir = tmp_path / "workload"
            workload_dir.mkdir()

            tasks_df, fragments_df, summary = build_training_opendc_frames(
                model_name=workload.llm_model,
                method=workload.fine_tuning_method,
                gpu_model=workload.gpu_model,
                tokens_per_sample=workload.tokens_per_sample,
                batch_size=workload.batch_size,
                number_gpus=gpus_per_node,
                number_nodes=number_of_nodes,
                total_tokens=None,
                task_id=0,
                submission_time_ms=0,
                simulate_full_training_fn=simulate_full_training,
                simulate_training_step_fn=simulate_training_step,
            )

            fragments_df = _fix_fragment_utilization(tasks_df, fragments_df)
            prepare_opendc_input(tasks_df, fragments_df, str(workload_dir))

            topology_dict = build_topology(
                gpu_model=workload.gpu_model,
                gpus_per_node=gpus_per_node,
                number_of_nodes=number_of_nodes,
                calibration_factor=self._calibration_factor,
            )

            run_dir = tmp_path / "run"
            power_df = self._runner.run_simulation(
                workload_dir=workload_dir,
                topology_dict=topology_dict,
                run_dir=run_dir,
                timeout_seconds=self._timeout,
            )

        if power_df.empty:
            raise OpenDCRunnerError("OpenDC returned empty power trace")

        mean_power = float(power_df["power_draw"].mean())

        if mean_power <= 0:
            raise OpenDCRunnerError(f"OpenDC returned non-positive mean power: {mean_power}W")

        # OpenDC's power_draw is the TOTAL datacenter draw (all GPUs). The
        # recommender's scoring (selection.normalize_candidates / _power_cost) and the
        # API's energy column treat predicted_power as PER-GPU watts (the convention
        # KavierPowerPredictor follows) and multiply back up by total_gpus. Divide to
        # per-GPU here so the two energy predictors are interchangeable and the total
        # power isn't double-counted (×total_gpus twice) under `energy: opendc`.
        per_gpu_power = mean_power / total_gpus if total_gpus > 0 else mean_power

        logger.info(
            "OpenDC prediction: %.1fW total (%.1fW/GPU, %d GPUs, %s)",
            mean_power,
            per_gpu_power,
            total_gpus,
            workload.gpu_model,
        )

        return Prediction(
            gpus_per_node=gpus_per_node,
            number_of_nodes=number_of_nodes,
            total_gpus=total_gpus,
            predicted_throughput=None,
            predicted_runtime_seconds=_runtime_or_none(summary.get("train_runtime")),
            predicted_power=per_gpu_power,
            metadata={
                "predictor": "opendc_energy",
                "power_model": "opendc_mse",
                "calibration_factor": self._calibration_factor,
                "power_samples": len(power_df),
                "mean_power_watts": mean_power,  # total datacenter draw
                "mean_power_watts_per_gpu": per_gpu_power,
                "max_power_watts": float(power_df["power_draw"].max()),
                "min_power_watts": float(power_df["power_draw"].min()),
            },
        )

    def predict_batch(
        self,
        workloads: list[WorkloadSpec],
        context: SystemContext,
    ) -> list[Prediction | None]:
        """Predict power for multiple workloads in parallel."""
        args_list = [
            {
                "task_id": i,
                "llm_model": w.llm_model,
                "fine_tuning_method": w.fine_tuning_method,
                "gpu_model": w.gpu_model,
                "tokens_per_sample": w.tokens_per_sample,
                "batch_size": w.batch_size,
                "gpus_per_node": w.gpus_per_node or 1,
                "number_of_nodes": w.number_of_nodes or 1,
            }
            for i, w in enumerate(workloads)
        ]

        results_by_id: dict[int, dict] = {}
        # Schedule OpenDC workers on the performance cores first (macOS QoS; see
        # _prefer_performance_cores). No-op off macOS.
        with ProcessPoolExecutor(max_workers=self._max_workers, initializer=_prefer_performance_cores) as executor:
            futures = {
                executor.submit(
                    _run_single_opendc,
                    args,
                    self._calibration_factor,
                    self._timeout,
                    self._bin_path,
                ): args["task_id"]
                for args in args_list
            }
            for future in as_completed(futures):
                tid = futures[future]
                try:
                    results_by_id[tid] = future.result()
                except Exception as e:
                    logger.error("OpenDC worker %d failed: %s", tid, e)
                    results_by_id[tid] = {"success": False, "task_id": tid}

        predictions: list[Prediction | None] = []
        for i, w in enumerate(workloads):
            r = results_by_id.get(i)
            if r is None or not r.get("success"):
                predictions.append(None)
                continue
            gpn = w.gpus_per_node or 1
            nn = w.number_of_nodes or 1
            total_gpus = gpn * nn
            # See predict(): OpenDC's mean_power is the TOTAL datacenter draw;
            # expose it per-GPU so it matches KavierPowerPredictor's convention.
            per_gpu_power = r["mean_power"] / total_gpus if total_gpus > 0 else r["mean_power"]
            predictions.append(
                Prediction(
                    gpus_per_node=gpn,
                    number_of_nodes=nn,
                    total_gpus=total_gpus,
                    predicted_throughput=None,
                    predicted_runtime_seconds=_runtime_or_none(r.get("train_runtime")),
                    predicted_power=per_gpu_power,
                    metadata={
                        "predictor": "opendc_energy",
                        "power_model": "opendc_mse",
                        "calibration_factor": self._calibration_factor,
                        "power_samples": r.get("power_samples", 0),
                        "mean_power_watts": r["mean_power"],  # total datacenter draw
                        "mean_power_watts_per_gpu": per_gpu_power,
                        "max_power_watts": r.get("max_power", 0),
                        "min_power_watts": r.get("min_power", 0),
                    },
                )
            )

        logger.info(
            "Batch prediction: %d/%d succeeded (%d workers)",
            sum(1 for p in predictions if p is not None),
            len(workloads),
            self._max_workers,
        )
        return predictions
