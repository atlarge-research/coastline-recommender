"""Plot the cluster timeline of a trace enriched by Coastline (``coastline utils plot-trace``).

The recommended configs are scheduled onto a fixed cluster, and the plot shows GPUs in use
(filled area) and jobs queued (line) over time, as in exp2 and exp4.

The scheduling (FIFO with backfill) and the timeline come from Kavier
(``kavier.sdk.cluster.schedule``); this module turns the trace into job rows and draws the result.

Input: a trace from ``coastline recommend-trace``, with the recommended layout columns and
``metadata.estimated_duration_<method>``. To plot a raw trace as a baseline, pass
``duration_col`` and ``submit_col``.
"""

from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd
from kavier.sdk.cluster import schedule as cluster_schedule

# Recommended layout written by coastline recommend-trace (originals are the fallback).
_GPN = "resources.num_gpus_per_node"
_NODES = "resources.num_nodes"
_ORIG_GPUS = "metadata.orig_number_gpus"  # the job's total GPUs, over _ORIG_NODES nodes
_ORIG_NODES = "metadata.orig_num_nodes"
# Submission-time columns, best first; jobs go to t=0 (row order) if none is usable.
_SUBMIT_COLS = (
    "metadata.submission_time_issue_85_rescaled",
    "metadata.submission_time_issue_85_original",
    "metadata.submission_time",
)
_GPU_FILL = "#0072B2"  # colourblind blue for the GPUs-in-use area


def _is_pdf(path: str) -> bool:
    return path.lower().endswith(".pdf")


def _savefig(fig, path: str) -> None:
    """Save a .pdf path as a vector PDF without a timestamp, so it is reproducible (callers omit
    the title there); any other path as a 130-dpi raster."""
    if _is_pdf(path):
        fig.savefig(path, metadata={"CreationDate": None})
    else:
        fig.savefig(path, dpi=130)


# Cluster timeline (scheduling and timeline come from kavier.sdk.cluster).


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Numeric column (NaN-coerced), or an all-NaN series when the column is absent."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([float("nan")] * len(df), index=df.index)


def _column_seconds(values: pd.Series) -> pd.Series:
    """Zero-based seconds from numeric seconds, else from timestamps; all NaN when neither parses."""
    secs = pd.to_numeric(values, errors="coerce")
    if secs.notna().any():
        return secs - secs.min()
    with warnings.catch_warnings():
        # pandas warns when it cannot infer one format; unreadable values become NaT either way.
        warnings.simplefilter("ignore", UserWarning)
        ts = pd.to_datetime(values, errors="coerce", utc=True)
    return (ts - ts.min()).dt.total_seconds()


def _submit_seconds(df: pd.DataFrame, submit_col: Optional[str] = None) -> list[float]:
    """Zero-based submission offsets [s] for FIFO ordering.

    A given ``submit_col`` must hold numeric seconds or ISO timestamps, else ValueError.
    Otherwise the first usable column of ``_SUBMIT_COLS`` with more than one distinct value is
    used (rescaled offset, then original offset, then ISO timestamp). With none, every job is
    submitted at t=0 and FIFO order is row order, as in the exp4 in-vitro replay."""
    cols = [submit_col] if submit_col else list(_SUBMIT_COLS)
    for col in cols:
        if col not in df.columns:
            continue
        secs = _column_seconds(df[col])
        if submit_col and secs.isna().all():
            raise ValueError(f"submit column '{col}' holds neither numeric seconds nor timestamps.")
        if secs.notna().any() and secs.dropna().nunique() > 1:
            return secs.fillna(0.0).tolist()
    return [0.0] * len(df)


def _trace_jobs(
    df: pd.DataFrame,
    method: str,
    *,
    duration_col: Optional[str] = None,
    submit_col: Optional[str] = None,
    gpus_per_node_col: Optional[str] = None,
    nodes_col: Optional[str] = None,
    total_gpus_col: Optional[str] = None,
) -> tuple[list[tuple], int]:
    """FIFO jobs from an enriched trace: the recommended layout's total GPUs scheduled
    for its predicted runtime at its submission offset.

    ``duration_col`` overrides the default ``metadata.estimated_duration_<method>`` column.
    ``submit_col`` overrides the default submission-time column resolution.
    ``gpus_per_node_col`` overrides the default ``resources.num_gpus_per_node`` column.
    ``nodes_col`` overrides the default ``resources.num_nodes`` column.
    ``total_gpus_col`` holds the job's total GPUs; when set, GPUs per node are not read.
    Returns ``(jobs, n_skipped)``; each job is ``(submit_s, gpus, duration_s, nodes)``."""
    dur_col = duration_col if duration_col else f"metadata.estimated_duration_{method}"
    dur = _num(df, dur_col)
    nodes = _num(df, nodes_col if nodes_col else _NODES)
    if total_gpus_col:
        total_gpus = _num(df, total_gpus_col)
    else:
        total_gpus = _num(df, gpus_per_node_col if gpus_per_node_col else _GPN) * nodes
    submit = _submit_seconds(df, submit_col)

    jobs, skipped = [], 0
    for i in range(len(df)):
        gpus, duration = total_gpus.iloc[i], dur.iloc[i]
        if not (pd.notna(gpus) and pd.notna(duration) and gpus > 0 and duration > 0):
            skipped += 1
            continue
        nd = nodes.iloc[i]
        nd = int(nd) if (pd.notna(nd) and nd >= 1) else 1
        jobs.append((float(submit[i]), int(gpus), float(duration), nd))
    return jobs, skipped


def plot_trace_timeline(
    enriched_csv: str,
    output_png: str,
    *,
    method: str = "kavier",
    cluster_gpus: int = 16,
    node_gpus: int = 8,
    duration_col: Optional[str] = None,
    submit_col: Optional[str] = None,
    gpus_per_node_col: Optional[str] = None,
    nodes_col: Optional[str] = None,
    label: Optional[str] = None,
    total_gpus_col: Optional[str] = None,
) -> dict:
    """FIFO-schedule the recommended configs of an enriched trace onto a ``cluster_gpus``-GPU
    cluster (``node_gpus`` per node, num_nodes = cluster // node) and draw the timeline: GPUs in
    use (filled area, left axis) and jobs queued (black line, right axis). Writes ``output_png``
    and returns stats (jobs, skipped, cluster_gpus, makespan_h, peak_gpus, peak_queue).

    ``duration_col``: job duration [s]; default ``metadata.estimated_duration_<method>``.
    ``submit_col``: submission time (numeric seconds or ISO timestamp); default: the first usable
        submission column. A given column that is absent or unreadable raises ValueError.
    ``gpus_per_node_col``: GPUs per node; default ``resources.num_gpus_per_node``.
    ``nodes_col``: number of nodes; default ``resources.num_nodes``.
    ``label``: legend and title label; default ``method``.
    ``total_gpus_col``: the job's total GPUs (for example ``metadata.orig_number_gpus``);
        replaces ``gpus_per_node_col``.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: write a file, never open a window
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    node_gpus = max(1, min(node_gpus, cluster_gpus))
    num_nodes = max(1, cluster_gpus // node_gpus)

    import sys

    df = pd.read_csv(enriched_csv, low_memory=False)
    effective_dur_col = duration_col if duration_col else f"metadata.estimated_duration_{method}"
    gpus_line = (
        f"  total GPUs    : {total_gpus_col}"
        if total_gpus_col
        else f"  gpus-per-node : {gpus_per_node_col if gpus_per_node_col else _GPN}"
    )
    effective_nodes_col = nodes_col if nodes_col else _NODES
    # Report the submit column; _submit_seconds makes the final choice.
    _submit_winner = next(
        (c for c in ([submit_col] if submit_col else list(_SUBMIT_COLS)) if c in df.columns),
        "row-order (t=0)",
    )
    print(
        f"plot-trace columns:\n"
        f"  duration      : {effective_dur_col}\n"
        f"{gpus_line}\n"
        f"  nodes         : {effective_nodes_col}\n"
        f"  submit-time   : {_submit_winner}",
        file=sys.stderr,
    )
    if effective_dur_col not in df.columns:
        raise SystemExit(
            f"{enriched_csv} has no '{effective_dur_col}' - either pass --duration-col or "
            f"run `coastline recommend-trace --method {method}` first."
        )
    # A given --submit-col that is missing raises, like the duration check above. Without
    # --submit-col the t=0 fallback applies.
    if submit_col and submit_col not in df.columns:
        raise ValueError(f"{enriched_csv} has no submit column '{submit_col}'; check the --submit-col value.")
    jobs, skipped = _trace_jobs(
        df,
        method,
        duration_col=duration_col,
        submit_col=submit_col,
        gpus_per_node_col=gpus_per_node_col,
        nodes_col=nodes_col,
        total_gpus_col=total_gpus_col,
    )
    if not jobs:
        raise SystemExit("no rows with both a positive estimated duration and a positive GPU count to schedule.")

    # Kavier schedules the jobs (node-aware backfill) and returns the GPUs-in-use and queue-depth
    # timeline.
    result = cluster_schedule(
        [
            {"submit_s": submit, "gpus": gpus, "duration_s": duration, "nodes": nodes}
            for submit, gpus, duration, nodes in jobs
        ],
        policy="distributed-backfill",
        num_nodes=num_nodes,
        node_gpus=node_gpus,
    )
    makespan_h = result.cluster.makespan_h
    gpu_t = q_t = result.timeline.times_h
    gpu_v = result.timeline.gpus_in_use
    q_v = result.timeline.queue_depth
    t_end = gpu_t[-1] if gpu_t else 0.0
    peak_gpus = result.cluster.peak_gpus
    peak_queue = result.cluster.peak_queue

    display_label = label if label else method
    fs_label, fs_tick, fs_legend, fs_title = 14, 12, 13, 13
    titled = not _is_pdf(output_png)
    gpu_dark = "#005a8d"  # darker blue: GPU area edge + left-axis label/ticks (matches the fill)
    queue_col = "black"  # the jobs-in-queue series + its (right) axis
    cap_col = "#8a8a8a"  # the dashed cluster-capacity line

    fig, ax = plt.subplots(figsize=(9, 3.9))

    # Light horizontal grid keyed to the GPU axis, kept behind the data.
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#d3d3d3", lw=0.6, alpha=0.7, zorder=0)

    # Left axis: GPUs in use, a filled blue area with a darker top edge.
    ax.fill_between(gpu_t, gpu_v, color=_GPU_FILL, alpha=0.45, lw=0, zorder=2)
    ax.plot(gpu_t, gpu_v, color=gpu_dark, lw=1.1, alpha=0.9, zorder=2.5)
    ax.axhline(cluster_gpus, ls=(0, (6, 4)), color=cap_col, lw=1.3, zorder=1)  # cluster cap
    ax.set_ylim(0, cluster_gpus * 1.08)
    ax.set_xlim(0, t_end if t_end > 0 else 1)
    ax.set_yticks(sorted({0, cluster_gpus // 2, cluster_gpus}))
    ax.set_xlabel("Time [h]", fontsize=fs_label)
    ax.set_ylabel("GPUs in use", fontsize=fs_label, color=gpu_dark)
    ax.tick_params(labelsize=fs_tick)
    ax.tick_params(axis="y", colors=gpu_dark)
    ax.spines["left"].set_color(gpu_dark)

    # Right axis: jobs in queue, a black line.
    queue_ax = ax.twinx()
    queue_ax.plot(
        q_t, q_v, color=queue_col, lw=1.7, alpha=0.9, solid_joinstyle="round", solid_capstyle="round", zorder=3
    )
    queue_ax.set_ylim(0, max(peak_queue * 1.15, 1))
    queue_ax.set_ylabel("Jobs in queue", fontsize=fs_label, color=queue_col)
    queue_ax.tick_params(labelsize=fs_tick)
    queue_ax.tick_params(axis="y", colors=queue_col)
    # Colour the visible spines to match each series; keep the shared left spine blue.
    queue_ax.spines["left"].set_color(gpu_dark)
    queue_ax.spines["right"].set_color(queue_col)
    # whole-number ticks on the queue axis
    queue_ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    handles = [
        Patch(facecolor=_GPU_FILL, alpha=0.45, edgecolor=gpu_dark, lw=1.1, label="GPUs in use"),
        Line2D([0], [0], color=queue_col, lw=1.7, label="Jobs in queue"),
        Line2D([0], [0], color=cap_col, lw=1.3, ls=(0, (6, 4)), label="Cluster capacity"),
    ]
    if titled:
        ax.set_title(
            f"Cluster timeline - {display_label}\n"
            f"{len(jobs)} jobs | {cluster_gpus} GPUs ({num_nodes}x{node_gpus}) | "
            f"makespan {makespan_h:.1f} h | peak queue {peak_queue:.0f}",
            fontsize=fs_title,
            pad=10,
        )
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        fontsize=fs_legend,
        handlelength=1.8,
        columnspacing=2.0,
        handletextpad=0.7,
    )
    # Leave room at the top for the legend (and the two-line title in PNG mode).
    fig.tight_layout(rect=(0, 0, 1, 0.84 if titled else 0.87))
    _savefig(fig, output_png)
    plt.close(fig)
    return {
        "jobs": len(jobs),
        "skipped": skipped,
        "cluster_gpus": cluster_gpus,
        "makespan_h": round(makespan_h, 2),
        "peak_gpus": int(peak_gpus),
        "peak_queue": int(peak_queue),
    }
