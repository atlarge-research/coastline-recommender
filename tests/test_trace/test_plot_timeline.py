"""``plot_trace_timeline`` now delegates the cluster simulation to Kavier
(``kavier.sdk.cluster.schedule``) and only renders the result. This test pins that the
Coastline SDK path still produces the correct operational-timeline stats for a hand-derived
node-aware backfill schedule — i.e. Coastline correctly consumes Kavier's simulator.

Fixture on a 16-GPU (2x8) cluster, durations in whole hours:
  J0  8 GPU, 2 h, submit 0        J1 12 GPU (2 nodes), 1 h, submit 0
  J2  8 GPU, 1 h, submit 0        J3  4 GPU, 1 h, submit 1 h (arrives late)

Backfill (node-aware, no head-of-line reservation), traced by hand:
  t=0h  J0 -> node0 (8), J2 -> node1 (8); J1 needs two nodes with >=6 free -> blocked.
  t=1h  J2 frees node1; J3 arrives and backfills onto node1 (4); J1 still blocked.
  t=2h  J0 and J3 free both nodes; J1 finally runs on both nodes (12).
  => makespan 3 h; peak GPUs 16 (J0+J2 at t=0); peak queue 1 (J1 waiting from t=0).
"""

from __future__ import annotations

import pandas as pd
import pytest

from coastline.sdk.trace.plot import plot_trace_timeline

_ROWS = [
    # num_gpus_per_node, num_nodes, estimated_duration_kavier (s), submit (s)
    {
        "resources.num_gpus_per_node": 8,
        "resources.num_nodes": 1,
        "metadata.estimated_duration_kavier": 7200,
        "metadata.submission_time_issue_85_rescaled": 0,
    },
    {
        "resources.num_gpus_per_node": 6,
        "resources.num_nodes": 2,
        "metadata.estimated_duration_kavier": 3600,
        "metadata.submission_time_issue_85_rescaled": 0,
    },
    {
        "resources.num_gpus_per_node": 8,
        "resources.num_nodes": 1,
        "metadata.estimated_duration_kavier": 3600,
        "metadata.submission_time_issue_85_rescaled": 0,
    },
    {
        "resources.num_gpus_per_node": 4,
        "resources.num_nodes": 1,
        "metadata.estimated_duration_kavier": 3600,
        "metadata.submission_time_issue_85_rescaled": 3600,
    },
]


def test_plot_trace_timeline_stats_match_hand_derived_backfill(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    enriched = tmp_path / "enriched.csv"
    pd.DataFrame(_ROWS).to_csv(enriched, index=False)
    out_pdf = tmp_path / "timeline.pdf"

    stats = plot_trace_timeline(str(enriched), str(out_pdf), method="kavier", cluster_gpus=16, node_gpus=8)

    assert stats == {
        "jobs": 4,
        "skipped": 0,
        "cluster_gpus": 16,
        "makespan_h": 3.0,
        "peak_gpus": 16,
        "peak_queue": 1,
    }
    assert out_pdf.exists() and out_pdf.stat().st_size > 0  # the figure was actually written
