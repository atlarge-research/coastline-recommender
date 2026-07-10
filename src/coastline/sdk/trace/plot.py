"""Plot a coastline-enriched trace (``coastline plot-trace``): the operational
cluster timeline — FIFO-schedule the recommended configs onto a fixed cluster and
draw GPUs-in-use (filled area) + jobs-queued (line) over time, the exp2/exp4
"what does running these recommendations look like" view.

Input: a trace enriched by ``coastline enrich-trace`` (it carries the recommended
layout columns plus ``metadata.estimated_duration_<method>``).
"""

from __future__ import annotations

import heapq
import math

import pandas as pd

# Recommended layout written by coastline enrich-trace (originals are the fallback).
_GPN = "resources.num_gpus_per_node"
_NODES = "resources.num_nodes"
_ORIG_GPUS = "metadata.orig_number_gpus"
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
    """A .pdf path -> reproducible vector PDF (timestamp stripped), thesis-ready
    with the title omitted by the callers; otherwise a 130-dpi raster."""
    if _is_pdf(path):
        fig.savefig(path, metadata={"CreationDate": None})
    else:
        fig.savefig(path, dpi=130)


# Operational cluster-timeline view (FIFO scheduler ported from exp2/exp4).


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Numeric column (NaN-coerced), or an all-NaN series when the column is absent."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([float("nan")] * len(df), index=df.index)


def _placement(free: list[int], per_node: int, nodes: int) -> list[int] | None:
    """Tightest-fit node ids for a job needing `per_node` GPUs on `nodes` distinct
    nodes, or None if it does not fit the currently free GPUs."""
    fitting = sorted((f, n) for n, f in enumerate(free) if f >= per_node)
    if len(fitting) < nodes:
        return None
    return [n for _, n in fitting[:nodes]]


def schedule_backfill(jobs: list[tuple], node_gpus: int = 8, num_nodes: int = 2) -> list[dict]:
    """Best-effort FIFO with aggressive backfill on a ``num_nodes`` x ``node_gpus``
    cluster (ported from the exp4 in-vitro scheduler). Jobs are considered in
    submission order; one that does not fit is skipped so newer jobs that do fit can
    start. A job's GPUs are placed on whole nodes and capped to the cluster.

    jobs: list of ``(submit_s, gpus, duration_s[, nodes])``; nodes defaults to 1.
    Returns per-job records ``[{job, gpus, wait_h, start_h, end_h}]`` where ``gpus``
    is the count actually placed.
    """

    def unpack(job):
        submit, gpus, duration = job[0], job[1], job[2]
        nodes = max(1, min(int(job[3]) if len(job) > 3 else 1, num_nodes))
        per_node = min(math.ceil(gpus / nodes), node_gpus)
        return submit, per_node, nodes, duration

    arrivals = sorted(enumerate(jobs), key=lambda item: item[1][0])
    pending: list[tuple] = []  # FIFO queue of (index, submit, per_node, nodes, duration)
    running: list[tuple] = []  # min-heap of (end_s, per_node, node_ids)
    free = [node_gpus] * num_nodes
    next_arrival = 0
    done: dict[int, dict] = {}

    time = arrivals[0][1][0]
    while len(done) < len(jobs):
        while next_arrival < len(arrivals) and arrivals[next_arrival][1][0] <= time:
            index, job = arrivals[next_arrival]
            pending.append((index, *unpack(job)))
            next_arrival += 1
        while running and running[0][0] <= time:
            _, freed, node_ids = heapq.heappop(running)
            for n in node_ids:
                free[n] += freed
        admitted = []
        for queue_pos, (index, submit, per_node, nodes, duration) in enumerate(pending):
            node_ids = _placement(free, per_node, nodes)
            if node_ids is None:
                continue
            for n in node_ids:
                free[n] -= per_node
            heapq.heappush(running, (time + duration, per_node, tuple(node_ids)))
            done[index] = {
                "job": index,
                "gpus": per_node * len(node_ids),
                "wait_h": (time - submit) / 3600,
                "start_h": time / 3600,
                "end_h": (time + duration) / 3600,
            }
            admitted.append(queue_pos)
        for queue_pos in reversed(admitted):
            pending.pop(queue_pos)
        candidates = []
        if next_arrival < len(arrivals):
            candidates.append(arrivals[next_arrival][1][0])
        if running:
            candidates.append(running[0][0])
        if not candidates:
            break
        time = max(time, min(candidates))
    return [done[i] for i in sorted(done)]


def cumulative_steps(events: list[tuple], t_end: float) -> tuple[list[float], list[float]]:
    """Turn (time, change) events into a step line (times, values), netting
    same-instant events so a zero-wait job never makes the line dip below its true
    level. Ported from gen_exp2.cumulative_steps."""
    net: dict[float, float] = {}
    for time, change in events:
        net[time] = net.get(time, 0.0) + change
    times, values, running_sum = [0.0], [0.0], 0.0
    for time in sorted(net):
        change = net[time]
        if change == 0:
            continue
        times.append(time)
        values.append(running_sum)
        running_sum += change
        times.append(time)
        values.append(running_sum)
    times.append(t_end)
    values.append(running_sum)
    return times, values


def _submit_seconds(df: pd.DataFrame) -> list[float]:
    """Zero-based submission offsets [s] for FIFO ordering. Uses the first usable,
    non-degenerate submission column (rescaled offset > original offset > ISO
    timestamp); if none is usable or all values are equal, enqueues every job at t=0
    so FIFO order = row order (matches the exp4 in-vitro replay of this file)."""
    for col in _SUBMIT_COLS:
        if col not in df.columns:
            continue
        if col == "metadata.submission_time":
            ts = pd.to_datetime(df[col], errors="coerce", utc=True)
            secs = (ts - ts.min()).dt.total_seconds()
        else:
            secs = pd.to_numeric(df[col], errors="coerce")
            secs = secs - secs.min()
        if secs.notna().any() and secs.dropna().nunique() > 1:
            return secs.fillna(0.0).tolist()
    return [0.0] * len(df)


def _trace_jobs(df: pd.DataFrame, method: str) -> tuple[list[tuple], int]:
    """FIFO jobs from an enriched trace: the recommended layout's total GPUs scheduled
    for its predicted runtime ``estimated_duration_<method>`` at its submission offset.
    Returns ``(jobs, n_skipped)``; each job is ``(submit_s, gpus, duration_s, nodes)``."""
    dur = _num(df, f"metadata.estimated_duration_{method}")
    nodes = _num(df, _NODES)
    total_gpus = _num(df, _GPN) * nodes
    # fall back to the original layout where the recommended one is unusable
    total_gpus = total_gpus.where(total_gpus > 0, _num(df, _ORIG_GPUS))
    nodes = nodes.where(nodes >= 1, _num(df, _ORIG_NODES))
    submit = _submit_seconds(df)

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
    enriched_csv: str, output_png: str, *, method: str = "kavier", cluster_gpus: int = 16, node_gpus: int = 8
) -> dict:
    """FIFO-schedule the recommended configs from an enriched trace onto a
    ``cluster_gpus``-GPU cluster (``node_gpus`` per node, num_nodes = cluster // node)
    and draw the operational timeline: GPUs in use (filled area, primary axis) and
    jobs queued (black line, twin axis) over time. Writes output_png; returns a stats
    dict (jobs, skipped, cluster_gpus, makespan_h, peak_gpus, peak_queue)."""
    import matplotlib

    matplotlib.use("Agg")  # headless: write a file, never open a window
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    node_gpus = max(1, min(node_gpus, cluster_gpus))
    num_nodes = max(1, cluster_gpus // node_gpus)

    df = pd.read_csv(enriched_csv, low_memory=False)
    est_col = f"metadata.estimated_duration_{method}"
    if est_col not in df.columns:
        raise SystemExit(f"{enriched_csv} has no '{est_col}' — run `coastline enrich-trace --method {method}` first.")
    jobs, skipped = _trace_jobs(df, method)
    if not jobs:
        raise SystemExit("no rows with both a positive estimated duration and a positive GPU count to schedule.")

    records = schedule_backfill(jobs, node_gpus=node_gpus, num_nodes=num_nodes)
    t_end = max(r["end_h"] for r in records)
    makespan_h = t_end - min(r["start_h"] for r in records)

    gpu_events, queue_events = [], []
    for r in records:
        gpu_events.append((r["start_h"], r["gpus"]))
        gpu_events.append((r["end_h"], -r["gpus"]))
        submit_h = jobs[r["job"]][0] / 3600
        queue_events.append((submit_h, 1))  # +1 when submitted
        queue_events.append((r["start_h"], -1))  # -1 when it starts
    gpu_t, gpu_v = cumulative_steps(gpu_events, t_end)
    q_t, q_v = cumulative_steps(queue_events, t_end)
    peak_gpus = max(gpu_v) if gpu_v else 0.0
    peak_queue = max(q_v) if q_v else 0.0

    fs_label, fs_tick, fs_legend, fs_title = 14, 12, 13, 13
    titled = not _is_pdf(output_png)
    gpu_dark = "#005a8d"  # darker blue: GPU area edge + left-axis label/ticks (matches the fill)
    queue_col = "black"  # the jobs-in-queue series + its (right) axis
    cap_col = "#8a8a8a"  # the dashed cluster-capacity line

    fig, ax = plt.subplots(figsize=(9, 3.9))

    # Light horizontal grid keyed to the GPU axis, kept behind the data.
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#d3d3d3", lw=0.6, alpha=0.7, zorder=0)

    # Left axis — GPUs in use: filled blue area with a crisp darker-blue top edge.
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

    # Right axis — jobs in queue: a crisp black step line.
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
    # jobs are integers — keep the queue axis on whole numbers (no 2.5, 5.5, ...)
    queue_ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    handles = [
        Patch(facecolor=_GPU_FILL, alpha=0.45, edgecolor=gpu_dark, lw=1.1, label="GPUs in use"),
        Line2D([0], [0], color=queue_col, lw=1.7, label="Jobs in queue"),
        Line2D([0], [0], color=cap_col, lw=1.3, ls=(0, (6, 4)), label="Cluster capacity"),
    ]
    if titled:
        ax.set_title(
            f"Cluster timeline — {method}\n"
            f"{len(jobs)} jobs · {cluster_gpus} GPUs ({num_nodes}x{node_gpus}) · "
            f"makespan {makespan_h:.1f} h · peak queue {peak_queue:.0f}",
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
    # Reserve top room for the legend (and the two-line title in PNG mode) so nothing collides.
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
