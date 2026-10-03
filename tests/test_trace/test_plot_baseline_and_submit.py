"""plot-trace: the --baseline layout and an explicit --submit-col of timestamps.

In the archived traces ``metadata.orig_number_gpus`` is the job's total GPU count (64 for 8 nodes
x 8 GPUs), and the baseline schedules that many GPUs per job. A submit column of ISO timestamps
is read as timestamps whatever its name; an explicit submit column with nothing readable in it
raises an error instead of placing every job at t=0.

Expected schedules, all jobs 1 h:
  baseline fixture on 64 GPUs (8 x 8): a 16-GPU job over 2 nodes at t=0 and an 8-GPU job at
  t=10 s run side by side, so the peak is 16 + 8 = 24 GPUs and the makespan 1 h (+10 s).
  ISO fixture on one 8-GPU node: two 8-GPU jobs submitted 10 h apart never queue; the makespan
  is 10 h + 1 h = 11 h.
"""

from __future__ import annotations

import ast

import pandas as pd
import pytest

from coastline.cli import main
from coastline.sdk.trace.plot import plot_trace_timeline

pytest.importorskip("matplotlib")

_DURATION = "metadata.output.extrapolated_duration"

_BASELINE_ROWS = [
    {
        "metadata.orig_number_gpus": 16,
        "metadata.orig_num_nodes": 2,
        "resources.num_gpus_per_node": 8,
        "resources.num_nodes": 2,
        _DURATION: 3600,
        "metadata.submission_time_issue_85_rescaled": 0,
    },
    {
        "metadata.orig_number_gpus": 8,
        "metadata.orig_num_nodes": 1,
        "resources.num_gpus_per_node": 8,
        "resources.num_nodes": 1,
        _DURATION: 3600,
        "metadata.submission_time_issue_85_rescaled": 10,
    },
]

_ISO_ROWS = [
    {
        "resources.num_gpus_per_node": 8,
        "resources.num_nodes": 1,
        _DURATION: 3600,
        "submitted_at": "2026-03-02T09:00:00Z",
        "metadata.submission_time": "2026-03-02T09:00:00Z",
    },
    {
        "resources.num_gpus_per_node": 8,
        "resources.num_nodes": 1,
        _DURATION: 3600,
        "submitted_at": "2026-03-02T19:00:00Z",
        "metadata.submission_time": "2026-03-02T19:00:00Z",
    },
]


def _csv(tmp_path, rows, name="trace.csv") -> str:
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _cli_stats(capsys, argv: list[str]) -> dict:
    main(["utils", "plot-trace", *argv])
    return ast.literal_eval(capsys.readouterr().out.strip().splitlines()[-1])


def test_the_baseline_schedules_the_original_total_gpus(tmp_path, capsys) -> None:
    trace = _csv(tmp_path, _BASELINE_ROWS)
    common = ["--input", trace, "--duration-col", _DURATION, "--cluster-gpus", "64", "--node-gpus", "8"]

    baseline = _cli_stats(capsys, [*common, "--output", str(tmp_path / "baseline.pdf"), "--baseline"])
    same_layout = _cli_stats(capsys, [*common, "--output", str(tmp_path / "resources.pdf")])

    assert baseline["peak_gpus"] == 24  # 16 + 8; reading 16 as GPUs per node would give 16 x 2 + 8
    # The original layout given as resources.* columns schedules identically.
    assert baseline == same_layout


def test_the_sdk_takes_a_total_gpu_column(tmp_path) -> None:
    stats = plot_trace_timeline(
        _csv(tmp_path, _BASELINE_ROWS),
        str(tmp_path / "t.pdf"),
        cluster_gpus=64,
        node_gpus=8,
        duration_col=_DURATION,
        total_gpus_col="metadata.orig_number_gpus",
        nodes_col="metadata.orig_num_nodes",
    )

    assert stats["peak_gpus"] == 24


def test_an_explicit_iso_submit_column_is_read_as_timestamps(tmp_path) -> None:
    trace = _csv(tmp_path, _ISO_ROWS)
    kwargs = {"cluster_gpus": 8, "node_gpus": 8, "duration_col": _DURATION}

    by_name = plot_trace_timeline(trace, str(tmp_path / "a.pdf"), submit_col="submitted_at", **kwargs)
    by_known_column = plot_trace_timeline(
        trace, str(tmp_path / "b.pdf"), submit_col="metadata.submission_time", **kwargs
    )

    assert by_name["makespan_h"] == pytest.approx(11.0)
    assert by_name["peak_queue"] == 0
    assert by_name == by_known_column


def test_an_explicit_submit_column_with_nothing_readable_is_an_error(tmp_path) -> None:
    rows = [{**row, "submitted_at": "not a time"} for row in _ISO_ROWS]

    with pytest.raises(ValueError, match="submitted_at"):
        plot_trace_timeline(
            _csv(tmp_path, rows),
            str(tmp_path / "c.pdf"),
            cluster_gpus=8,
            node_gpus=8,
            duration_col=_DURATION,
            submit_col="submitted_at",
        )


def test_unreadable_submit_values_raise_no_pandas_warning():
    """Strings that are neither numbers nor timestamps give NaN offsets without a pandas UserWarning."""
    import warnings

    from coastline.sdk.trace.plot import _column_seconds

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        secs = _column_seconds(pd.Series(["soon", "later"]))
    assert secs.isna().all()
