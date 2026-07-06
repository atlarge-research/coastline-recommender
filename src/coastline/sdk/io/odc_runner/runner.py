"""OpenDC Experiment Runner wrapper; raises on any failure."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from .java_home import detect_java_home
from .topology import write_topology_json

logger = logging.getLogger(__name__)

_DEFAULT_BIN = (
    Path(__file__).resolve().parents[6] / "opendc" / "bin" / "OpenDCExperimentRunner" / "bin" / "OpenDCExperimentRunner"
)


class OpenDCRunnerError(Exception):
    """Raised when an OpenDC simulation fails."""


class OpenDCRunner:
    """Wrapper around the OpenDC ExperimentRunner binary."""

    def __init__(self, opendc_bin_path: Path | None = None) -> None:
        if opendc_bin_path is None:
            env_path = os.environ.get("OPENDC_BIN_PATH")
            opendc_bin_path = Path(env_path) if env_path else _DEFAULT_BIN

        self.opendc_path = opendc_bin_path

        if not self.opendc_path.exists():
            raise FileNotFoundError(
                f"OpenDC binary not found at {self.opendc_path}. "
                "Set OPENDC_BIN_PATH env var or ensure the binary is available."
            )

        if not os.access(self.opendc_path, os.X_OK):
            try:
                os.chmod(self.opendc_path, 0o755)
            except OSError as exc:
                raise OpenDCRunnerError(f"OpenDC binary not executable and chmod failed: {exc}") from exc

        logger.info("OpenDCRunner initialized: %s", self.opendc_path)

    def run_simulation(
        self,
        workload_dir: Path,
        topology_dict: dict,
        run_dir: Path,
        timeout_seconds: int = 120,
    ) -> pd.DataFrame:
        """Run OpenDC simulation; returns [timestamp, power_draw] DataFrame."""
        input_dir = run_dir / "input"
        output_dir = run_dir / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        wl_input = input_dir / "workload"
        wl_input.mkdir(exist_ok=True)

        tasks_src = workload_dir / "tasks.parquet"
        frags_src = workload_dir / "fragments.parquet"
        if not tasks_src.exists():
            raise FileNotFoundError(f"tasks.parquet not found: {tasks_src}")
        if not frags_src.exists():
            raise FileNotFoundError(f"fragments.parquet not found: {frags_src}")

        shutil.copy2(tasks_src, wl_input / "tasks.parquet")
        shutil.copy2(frags_src, wl_input / "fragments.parquet")

        topology_path = input_dir / "topology.json"
        write_topology_json(topology_dict, topology_path)

        experiment = {
            "name": run_dir.name,
            "topologies": [{"pathToFile": str(topology_path)}],
            "workloads": [{"pathToFile": str(wl_input), "type": "ComputeWorkload"}],
            "outputFolder": str(output_dir),
            "exportModels": [
                {
                    "exportInterval": 150,
                    "filesToExport": ["powerSource", "host", "task", "service"],
                }
            ],
        }
        experiment_path = input_dir / "experiment.json"
        with open(experiment_path, "w") as f:
            json.dump(experiment, f, indent=2)

        self._execute(experiment_path, timeout_seconds)

        return self._read_power_output(output_dir)

    @staticmethod
    def _is_valid_java_home(java_home: str) -> bool:
        """True if ``<java_home>/bin/java`` exists and is executable."""
        if not java_home:
            return False
        java_bin = Path(java_home) / "bin" / "java"
        return java_bin.is_file() and os.access(java_bin, os.X_OK)

    def _execute(self, experiment_path: Path, timeout: int) -> None:
        """Execute the OpenDC binary. Raises on failure."""
        env = os.environ.copy()
        # Validate inherited JAVA_HOME — a bad value surfaces as a confusing JVM error
        # deep inside OpenDC; fall through to auto-detection instead.
        inherited = env.get("JAVA_HOME")
        if not self._is_valid_java_home(inherited):
            if inherited:
                logger.warning(
                    "Ignoring invalid JAVA_HOME=%r (no executable bin/java); falling back to JDK auto-detection.",
                    inherited,
                )
            env["JAVA_HOME"] = detect_java_home()

        command = [str(self.opendc_path), "--experiment-path", str(experiment_path)]
        logger.debug("Running OpenDC: %s", " ".join(command))

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise OpenDCRunnerError(f"OpenDC simulation timed out after {timeout}s") from exc

        if result.returncode != 0:
            raise OpenDCRunnerError(
                f"OpenDC exited with code {result.returncode}.\n"
                f"stdout: {result.stdout[:1000]}\n"
                f"stderr: {result.stderr[:1000]}"
            )

        logger.debug("OpenDC completed successfully")

    @staticmethod
    def _read_power_output(output_dir: Path) -> pd.DataFrame:
        """Read powerSource.parquet and return [timestamp, power_draw]."""
        power_files = list(output_dir.rglob("powerSource.parquet"))
        if not power_files:
            raise OpenDCRunnerError(f"No powerSource.parquet found in {output_dir}")

        df = pd.read_parquet(power_files[0])

        if "timestamp_absolute" not in df.columns:
            raise OpenDCRunnerError(
                f"powerSource.parquet missing 'timestamp_absolute' column. Columns: {list(df.columns)}"
            )
        if "power_draw" not in df.columns:
            raise OpenDCRunnerError(f"powerSource.parquet missing 'power_draw' column. Columns: {list(df.columns)}")

        df["timestamp"] = pd.to_datetime(df["timestamp_absolute"], unit="ms", utc=True)
        result = df[["timestamp", "power_draw"]].copy()

        logger.info(
            "Read %d power samples, mean=%.1fW",
            len(result),
            result["power_draw"].mean() if not result.empty else 0,
        )
        return result
