"""Tests for the OpenDC workload-export / runner integration.

Covers ``coastline.sdk.io.odc_runner`` (previously untested):

* ``topology.build_topology`` / ``write_topology_json`` — the GPU-cluster
  topology dict construction (power/memory/core scaling, calibration passthrough,
  unsupported-GPU error) and its JSON serialization.
* ``java_home.detect_java_home`` — JAVA_HOME fast path and the no-Java error.
* ``runner.OpenDCRunner`` — binary-presence checks, the experiment.json schema
  written for a run, parquet copying, ``_read_power_output`` parsing/validation,
  and ``_execute`` failure handling (non-zero exit, timeout).

These tests do NOT require the OpenDC Java binary: the ``OpenDCRunner`` is always
constructed against a fake executable in ``tmp_path``, and ``subprocess.run`` plus
``detect_java_home`` are mocked, so the suite is fast and deterministic. A real
end-to-end run is exercised only if the binary AND ``opendc`` package are present
(guarded with ``importorskip`` / ``skipif``); otherwise it is skipped.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from coastline.sdk.io.odc_runner import java_home as java_home_mod
from coastline.sdk.io.odc_runner import runner as runner_mod
from coastline.sdk.io.odc_runner.runner import OpenDCRunner, OpenDCRunnerError
from coastline.sdk.io.odc_runner.topology import (
    build_topology,
    write_topology_json,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_binary(tmp_path: Path, mode: int = 0o755) -> Path:
    """Create a fake, executable OpenDC binary file."""
    binary = tmp_path / "OpenDCExperimentRunner"
    binary.write_text("#!/bin/sh\nexit 0\n")
    os.chmod(binary, mode)
    return binary


def _make_runner(tmp_path: Path) -> OpenDCRunner:
    """OpenDCRunner pointed at a hermetic fake binary (no real OpenDC needed)."""
    return OpenDCRunner(opendc_bin_path=_make_fake_binary(tmp_path))


def _write_power_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)


def _write_workload(workload_dir: Path) -> None:
    """Write minimal tasks.parquet + fragments.parquet so copy succeeds."""
    workload_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": [0]}).to_parquet(workload_dir / "tasks.parquet")
    pd.DataFrame({"id": [0]}).to_parquet(workload_dir / "fragments.parquet")


# ===========================================================================
# topology.build_topology
# ===========================================================================


def test_build_topology_core_count_is_total_gpus():
    # coreCount must equal gpus_per_node * number_of_nodes (the whole fleet).
    host = build_topology("NVIDIA-A100-SXM4-80GB", gpus_per_node=8, number_of_nodes=3)["clusters"][0]["hosts"][0]
    assert host["cpu"]["coreCount"] == 24
    # coreSpeed is fixed at 1000.0 to match the workload export's cpu_capacity.
    assert host["cpu"]["coreSpeed"] == 1000.0


@pytest.mark.parametrize(
    "gpu,exp_power,exp_idle,exp_max,exp_mem",
    [
        # Oracle: datasheet watts × total_gpus (=4·2=8), computed by hand from the
        # published per-GPU specs — NOT read back from GPU_TOPOLOGY_SPECS, so a
        # corrupted catalog entry or a wrong scale factor fails this test.
        # A100-SXM4-80GB: 400·8, 75·8, 400·8; mem 80 GiB(=85_899_345_920 B)·8.
        ("NVIDIA-A100-SXM4-80GB", 3200.0, 600.0, 3200.0, 687_194_767_360),
        # A100-80GB-PCIe: 300·8, 60·8, 300·8; same 80 GiB·8.
        ("NVIDIA-A100-80GB-PCIe", 2400.0, 480.0, 2400.0, 687_194_767_360),
        # H100-PCIe: 350·8, 70·8, 700·8 (max ≠ nominal for the H100); same 80 GiB·8.
        ("NVIDIA-H100-PCIe", 2800.0, 560.0, 5600.0, 687_194_767_360),
    ],
)
def test_build_topology_power_and_memory_scaled_by_total_gpus(gpu, exp_power, exp_idle, exp_max, exp_mem):
    host = build_topology(gpu, gpus_per_node=4, number_of_nodes=2)["clusters"][0]["hosts"][0]

    pm = host["cpuPowerModel"]
    assert pm["modelType"] == "mse"
    assert pm["power"] == exp_power
    assert pm["idlePower"] == exp_idle
    assert pm["maxPower"] == exp_max
    assert host["memory"]["memorySize"] == exp_mem


def test_build_topology_custom_calibration_factor_passthrough():
    host = build_topology(
        "NVIDIA-A100-SXM4-80GB",
        gpus_per_node=1,
        number_of_nodes=1,
        calibration_factor=1.37,
    )["clusters"][0]["hosts"][0]
    assert host["cpuPowerModel"]["calibrationFactor"] == 1.37


def test_build_topology_unsupported_gpu_raises_valueerror():
    with pytest.raises(ValueError) as exc:
        build_topology("NVIDIA-RTX-4090", gpus_per_node=1, number_of_nodes=1)
    msg = str(exc.value)
    assert "NVIDIA-RTX-4090" in msg
    # The error advertises the supported set.
    assert "NVIDIA-A100-SXM4-80GB" in msg


# ===========================================================================
# topology.write_topology_json
# ===========================================================================


def test_write_topology_json_writes_file(tmp_path):
    topo = build_topology("NVIDIA-A100-SXM4-80GB", gpus_per_node=1, number_of_nodes=1)
    out = tmp_path / "topology.json"
    write_topology_json(topo, out)
    assert out.exists()
    assert json.loads(out.read_text()) == topo


def test_write_topology_json_creates_parent_dirs(tmp_path):
    topo = build_topology("NVIDIA-A100-SXM4-80GB", gpus_per_node=1, number_of_nodes=1)
    out = tmp_path / "nested" / "deeper" / "topology.json"
    write_topology_json(topo, out)
    assert out.exists()
    assert json.loads(out.read_text())["clusters"][0]["name"] == "cluster-0"


# ===========================================================================
# java_home.detect_java_home
# ===========================================================================


def test_detect_java_home_uses_existing_env(tmp_path, monkeypatch):
    # A JAVA_HOME with a valid executable bin/java is returned verbatim.
    java_bin = tmp_path / "bin" / "java"
    java_bin.parent.mkdir()
    java_bin.write_text("")
    java_bin.chmod(0o755)
    monkeypatch.setenv("JAVA_HOME", str(tmp_path))
    assert java_home_mod.detect_java_home() == str(tmp_path)


def test_detect_java_home_ignores_nonexistent_env_then_falls_back(monkeypatch):
    # JAVA_HOME set but missing on disk -> skip it, fall through to detection.
    monkeypatch.setenv("JAVA_HOME", "/no/such/java/home")
    fake = "/opt/java/home"

    # The bogus JAVA_HOME must NOT exist; the detected path must.
    def fake_exists(self):
        return str(self) != "/no/such/java/home"

    with (
        mock.patch.object(java_home_mod.subprocess, "run") as run,
        mock.patch.object(java_home_mod.Path, "exists", fake_exists),
    ):
        run.return_value = subprocess.CompletedProcess(["/usr/libexec/java_home"], 0, stdout=fake + "\n", stderr="")
        assert java_home_mod.detect_java_home() == fake


def test_detect_java_home_raises_when_nothing_found(monkeypatch):
    monkeypatch.delenv("JAVA_HOME", raising=False)
    # All subprocess probes fail and no common path exists.
    with (
        mock.patch.object(
            java_home_mod.subprocess,
            "run",
            side_effect=FileNotFoundError("no java"),
        ),
        mock.patch.object(java_home_mod.Path, "exists", return_value=False),
    ):
        with pytest.raises(RuntimeError, match="Could not detect JAVA_HOME"):
            java_home_mod.detect_java_home()


# ===========================================================================
# OpenDCRunner.__init__
# ===========================================================================


def test_init_with_explicit_existing_binary(tmp_path):
    binary = _make_fake_binary(tmp_path)
    r = OpenDCRunner(opendc_bin_path=binary)
    assert r.opendc_path == binary


def test_init_missing_binary_raises_filenotfound(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="OpenDC binary not found"):
        OpenDCRunner(opendc_bin_path=missing)


def test_init_chmods_non_executable_binary(tmp_path):
    # A present-but-non-executable binary gets chmod'd to 0755 rather than failing.
    binary = _make_fake_binary(tmp_path, mode=0o644)
    assert not os.access(binary, os.X_OK)
    r = OpenDCRunner(opendc_bin_path=binary)
    assert os.access(r.opendc_path, os.X_OK)


def test_init_reads_opendc_bin_path_env(tmp_path, monkeypatch):
    binary = _make_fake_binary(tmp_path)
    monkeypatch.setenv("OPENDC_BIN_PATH", str(binary))
    r = OpenDCRunner()  # no explicit arg -> must use env var
    assert r.opendc_path == binary


def test_init_falls_back_to_default_bin_when_no_env(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENDC_BIN_PATH", raising=False)
    fake_default = _make_fake_binary(tmp_path)
    # Redirect the module-level default to a hermetic fake.
    monkeypatch.setattr(runner_mod, "_DEFAULT_BIN", fake_default)
    r = OpenDCRunner()
    assert r.opendc_path == fake_default


# ===========================================================================
# OpenDCRunner._read_power_output  (static, parquet parsing/validation)
# ===========================================================================


def test_read_power_output_happy_path(tmp_path):
    out = tmp_path / "output"
    _write_power_parquet(
        out / "raw-output" / "0" / "powerSource.parquet",
        [
            {"timestamp_absolute": 0, "power_draw": 100.0},
            {"timestamp_absolute": 150_000, "power_draw": 200.0},
        ],
    )
    df = OpenDCRunner._read_power_output(out)
    assert list(df.columns) == ["timestamp", "power_draw"]
    assert len(df) == 2
    assert df["power_draw"].tolist() == [100.0, 200.0]
    # ms epoch -> tz-aware UTC datetime.
    assert str(df["timestamp"].dt.tz) == "UTC"
    assert df["timestamp"].iloc[1] == pd.Timestamp("1970-01-01 00:02:30", tz="UTC")


def test_read_power_output_missing_file_raises(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    with pytest.raises(OpenDCRunnerError, match="No powerSource.parquet"):
        OpenDCRunner._read_power_output(out)


def test_read_power_output_missing_timestamp_column_raises(tmp_path):
    out = tmp_path / "output"
    _write_power_parquet(out / "powerSource.parquet", [{"power_draw": 1.0}])
    with pytest.raises(OpenDCRunnerError, match="timestamp_absolute"):
        OpenDCRunner._read_power_output(out)


def test_read_power_output_missing_power_column_raises(tmp_path):
    out = tmp_path / "output"
    _write_power_parquet(out / "powerSource.parquet", [{"timestamp_absolute": 0}])
    with pytest.raises(OpenDCRunnerError, match="power_draw"):
        OpenDCRunner._read_power_output(out)


def test_read_power_output_finds_nested_parquet(tmp_path):
    # rglob must locate the file no matter how deep OpenDC nests its output.
    out = tmp_path / "output"
    _write_power_parquet(
        out / "a" / "b" / "c" / "powerSource.parquet",
        [{"timestamp_absolute": 0, "power_draw": 42.0}],
    )
    df = OpenDCRunner._read_power_output(out)
    assert df["power_draw"].iloc[0] == 42.0


# ===========================================================================
# OpenDCRunner.run_simulation  (orchestration; subprocess mocked)
# ===========================================================================


def test_run_simulation_missing_tasks_parquet_raises(tmp_path):
    r = _make_runner(tmp_path)
    workload = tmp_path / "wl"
    workload.mkdir()
    (workload / "fragments.parquet").write_bytes(b"x")  # frags present, tasks missing
    with pytest.raises(FileNotFoundError, match="tasks.parquet not found"):
        r.run_simulation(workload, {"clusters": []}, tmp_path / "run")


def test_run_simulation_missing_fragments_parquet_raises(tmp_path):
    r = _make_runner(tmp_path)
    workload = tmp_path / "wl"
    workload.mkdir()
    (workload / "tasks.parquet").write_bytes(b"x")  # tasks present, frags missing
    with pytest.raises(FileNotFoundError, match="fragments.parquet not found"):
        r.run_simulation(workload, {"clusters": []}, tmp_path / "run")


def test_run_simulation_writes_experiment_json_and_returns_power(tmp_path):
    r = _make_runner(tmp_path)
    workload = tmp_path / "wl"
    _write_workload(workload)
    run_dir = tmp_path / "run-42"
    topo = build_topology("NVIDIA-A100-SXM4-80GB", gpus_per_node=1, number_of_nodes=1)

    # Mock the binary execution: instead of running OpenDC, drop a powerSource
    # parquet where _read_power_output expects it, and report success.
    def fake_run(command, **kwargs):
        out_dir = run_dir / "output"
        _write_power_parquet(
            out_dir / "seed=0" / "powerSource.parquet",
            [{"timestamp_absolute": 0, "power_draw": 321.0}],
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    with (
        mock.patch.object(runner_mod.subprocess, "run", side_effect=fake_run) as run,
        mock.patch.object(runner_mod, "detect_java_home", return_value="/fake/java"),
    ):
        df = r.run_simulation(workload, topo, run_dir, timeout_seconds=5)

    # Returned the parsed power timeseries.
    assert list(df.columns) == ["timestamp", "power_draw"]
    assert df["power_draw"].iloc[0] == 321.0

    # Workload parquets were copied into input/workload/.
    wl_in = run_dir / "input" / "workload"
    assert (wl_in / "tasks.parquet").exists()
    assert (wl_in / "fragments.parquet").exists()

    # topology.json was written.
    assert (run_dir / "input" / "topology.json").exists()

    # experiment.json carries the documented schema.
    exp = json.loads((run_dir / "input" / "experiment.json").read_text())
    assert exp["name"] == "run-42"
    assert exp["topologies"][0]["pathToFile"] == str(run_dir / "input" / "topology.json")
    assert exp["workloads"][0]["type"] == "ComputeWorkload"
    assert exp["workloads"][0]["pathToFile"] == str(wl_in)
    assert exp["outputFolder"] == str(run_dir / "output")
    export = exp["exportModels"][0]
    assert export["exportInterval"] == 150
    assert "powerSource" in export["filesToExport"]

    # The binary was actually invoked with --experiment-path pointing at the json.
    (command, _kwargs) = run.call_args[0], run.call_args[1]
    invoked = command[0]
    assert str(r.opendc_path) == invoked[0]
    assert invoked[1] == "--experiment-path"
    assert invoked[2] == str(run_dir / "input" / "experiment.json")


def test_run_simulation_sets_java_home_when_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("JAVA_HOME", raising=False)
    r = _make_runner(tmp_path)
    workload = tmp_path / "wl"
    _write_workload(workload)
    run_dir = tmp_path / "run"
    topo = build_topology("NVIDIA-A100-SXM4-80GB", gpus_per_node=1, number_of_nodes=1)

    def fake_run(command, **kwargs):
        _write_power_parquet(
            run_dir / "output" / "powerSource.parquet",
            [{"timestamp_absolute": 0, "power_draw": 1.0}],
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with (
        mock.patch.object(runner_mod.subprocess, "run", side_effect=fake_run) as run,
        mock.patch.object(runner_mod, "detect_java_home", return_value="/detected/java") as det,
    ):
        r.run_simulation(workload, topo, run_dir)
    det.assert_called_once()
    assert run.call_args.kwargs["env"]["JAVA_HOME"] == "/detected/java"


def _make_fake_java_home(tmp_path: Path) -> Path:
    """A JAVA_HOME-shaped dir with an executable bin/java (passes validation)."""
    jh = tmp_path / "jdk"
    (jh / "bin").mkdir(parents=True)
    java = jh / "bin" / "java"
    java.write_text("#!/bin/sh\nexit 0\n")
    os.chmod(java, 0o755)
    return jh


def test_run_simulation_preserves_valid_java_home(tmp_path, monkeypatch):
    # A JAVA_HOME that points at a real JDK (bin/java present + executable) is kept
    # as-is and detection is NOT invoked.
    valid_jh = _make_fake_java_home(tmp_path)
    monkeypatch.setenv("JAVA_HOME", str(valid_jh))
    r = _make_runner(tmp_path)
    workload = tmp_path / "wl"
    _write_workload(workload)
    run_dir = tmp_path / "run"
    topo = build_topology("NVIDIA-A100-SXM4-80GB", gpus_per_node=1, number_of_nodes=1)

    def fake_run(command, **kwargs):
        _write_power_parquet(
            run_dir / "output" / "powerSource.parquet",
            [{"timestamp_absolute": 0, "power_draw": 1.0}],
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with (
        mock.patch.object(runner_mod.subprocess, "run", side_effect=fake_run) as run,
        mock.patch.object(runner_mod, "detect_java_home") as det,
    ):
        r.run_simulation(workload, topo, run_dir)
    # detect_java_home must NOT be called when JAVA_HOME is set AND valid.
    det.assert_not_called()
    assert run.call_args.kwargs["env"]["JAVA_HOME"] == str(valid_jh)


def test_run_simulation_invalid_java_home_falls_back_to_detection(tmp_path, monkeypatch):
    # JAVA_HOME set but bogus (no bin/java) -> do NOT trust it; fall through to the
    # same detection used when JAVA_HOME is absent.
    monkeypatch.setenv("JAVA_HOME", str(tmp_path / "not-a-jdk"))
    r = _make_runner(tmp_path)
    workload = tmp_path / "wl"
    _write_workload(workload)
    run_dir = tmp_path / "run"
    topo = build_topology("NVIDIA-A100-SXM4-80GB", gpus_per_node=1, number_of_nodes=1)

    def fake_run(command, **kwargs):
        _write_power_parquet(
            run_dir / "output" / "powerSource.parquet",
            [{"timestamp_absolute": 0, "power_draw": 1.0}],
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with (
        mock.patch.object(runner_mod.subprocess, "run", side_effect=fake_run) as run,
        mock.patch.object(runner_mod, "detect_java_home", return_value="/detected/java") as det,
    ):
        r.run_simulation(workload, topo, run_dir)
    # The bogus JAVA_HOME was rejected and detection ran instead.
    det.assert_called_once()
    assert run.call_args.kwargs["env"]["JAVA_HOME"] == "/detected/java"


def test_run_simulation_invalid_java_home_no_jdk_anywhere_raises(tmp_path, monkeypatch):
    # JAVA_HOME bogus AND detection finds nothing -> a clear RuntimeError (from
    # detect_java_home) rather than a confusing JVM failure deep inside OpenDC.
    monkeypatch.setenv("JAVA_HOME", str(tmp_path / "not-a-jdk"))
    r = _make_runner(tmp_path)
    with mock.patch.object(
        runner_mod,
        "detect_java_home",
        side_effect=RuntimeError("Could not detect JAVA_HOME"),
    ):
        with pytest.raises(RuntimeError, match="Could not detect JAVA_HOME"):
            r._execute(tmp_path / "experiment.json", timeout=5)


# ===========================================================================
# OpenDCRunner._execute  (failure handling)
# ===========================================================================


def test_execute_nonzero_exit_raises_with_output(tmp_path, monkeypatch):
    monkeypatch.setenv("JAVA_HOME", str(_make_fake_java_home(tmp_path)))
    r = _make_runner(tmp_path)
    with mock.patch.object(
        runner_mod.subprocess,
        "run",
        return_value=subprocess.CompletedProcess(["x"], 3, stdout="some-stdout", stderr="some-stderr"),
    ):
        with pytest.raises(OpenDCRunnerError) as exc:
            r._execute(tmp_path / "experiment.json", timeout=5)
    msg = str(exc.value)
    assert "code 3" in msg
    assert "some-stdout" in msg
    assert "some-stderr" in msg


def test_execute_timeout_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("JAVA_HOME", str(_make_fake_java_home(tmp_path)))
    r = _make_runner(tmp_path)
    with mock.patch.object(
        runner_mod.subprocess,
        "run",
        side_effect=subprocess.TimeoutExpired(cmd="opendc", timeout=5),
    ):
        with pytest.raises(OpenDCRunnerError, match="timed out after 5s"):
            r._execute(tmp_path / "experiment.json", timeout=5)


def test_execute_success_is_silent(tmp_path, monkeypatch):
    monkeypatch.setenv("JAVA_HOME", str(_make_fake_java_home(tmp_path)))
    r = _make_runner(tmp_path)
    with mock.patch.object(
        runner_mod.subprocess,
        "run",
        return_value=subprocess.CompletedProcess(["x"], 0, stdout="", stderr=""),
    ):
        # Returns None, raises nothing.
        assert r._execute(tmp_path / "experiment.json", timeout=5) is None


# ===========================================================================
# Optional real end-to-end (opt-in; needs the Java binary + a valid workload)
# ===========================================================================
#
# This is NOT run by default. The OpenDC Java binary expects tasks.parquet /
# fragments.parquet in OpenDC's own trace schema; fabricating that schema here
# would be brittle and non-deterministic (it would track OpenDC internals), so
# the test only runs when a caller points OPENDC_TEST_WORKLOAD at a directory of
# real, schema-correct parquet files. Without it, the test is skipped.

_REAL_BIN_EXISTS = runner_mod._DEFAULT_BIN.exists() or bool(os.environ.get("OPENDC_BIN_PATH"))
_REAL_WORKLOAD = os.environ.get("OPENDC_TEST_WORKLOAD")


@pytest.mark.skipif(
    not (_REAL_BIN_EXISTS and _REAL_WORKLOAD),
    reason=(
        "real OpenDC end-to-end requires the Java binary AND a schema-correct "
        "workload dir (set OPENDC_BIN_PATH + OPENDC_TEST_WORKLOAD to enable)"
    ),
)
def test_real_opendc_end_to_end(tmp_path):
    """End-to-end OpenDC run against a real workload (opt-in)."""
    workload = Path(_REAL_WORKLOAD)
    r = OpenDCRunner()
    topo = build_topology("NVIDIA-A100-SXM4-80GB", gpus_per_node=1, number_of_nodes=1)
    df = r.run_simulation(workload, topo, tmp_path / "run", timeout_seconds=120)
    assert list(df.columns) == ["timestamp", "power_draw"]
    assert not df.empty
