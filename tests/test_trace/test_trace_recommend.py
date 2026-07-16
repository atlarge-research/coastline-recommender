"""Tests for the ado trace-enrichment deliverable (coastline/sdk/trace/enrich.py).

Oracles used here:
  * _job_total_tokens / estimated_duration are hand-derived arithmetic (shown inline).
  * The recommended THROUGHPUT comes from Kavier (a black-box analytical engine), so
    real-engine tests never pin its magic number — they assert INVARIANTS (finite/positive,
    layout bounded by max_gpus, fallback preserves the original row) and a SCALING law
    (estimated_duration is linear in the job's actual work).
"""

import pandas as pd
import pytest

from coastline.sdk.trace import recommend as trace_recommend
from coastline.sdk.trace.recommend import (
    _METHOD_TO_PREDICTOR,
    _job_total_tokens,
    recommend_trace,
)

# One physical row: granite-8b / A100 / full. Kavier finds this feasible at 1 GPU.
_GOOD_ROW = {
    "metadata.model_name": "granite-3.1-8b-instruct",
    "metadata.method": "full",
    "resources.gpu_model": "NVIDIA-A100-SXM4-80GB",
    "metadata.tokens_per_sample": 4096,
    "metadata.batch_size": 8,
    "resources.num_gpus_per_node": 8,
    "resources.num_nodes": 1,
    "metadata.output.train_tokens_per_second": 15000.0,
    "metadata.train_runtime": 3600.0,
    "metadata.uid": "job-1",
}


def _write_csv(tmp_path, rows, name="trace.csv"):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_estimated_duration_scales_linearly_with_the_jobs_actual_work(tmp_path):
    """estimated_duration = job_total_tokens / recommended_throughput.

    Two rows with an IDENTICAL workload/layout differ only in train_runtime
    (3600 s vs 7200 s). The recommended throughput depends solely on the
    workload+GPU (not on the actual runtime), so it is the SAME for both rows;
    therefore estimated_duration must scale exactly with job_total_tokens =
    tps * runtime. Doubling the runtime doubles the work, so the second row's
    duration must be exactly 2x the first — a scaling oracle that needs no
    knowledge of Kavier's magic throughput number.
    """
    r1 = {**_GOOD_ROW, "metadata.uid": "one-hour", "metadata.train_runtime": 3600.0}
    r2 = {**_GOOD_ROW, "metadata.uid": "two-hour", "metadata.train_runtime": 7200.0}
    out = tmp_path / "enriched.csv"
    df = recommend_trace(str(_write_csv(tmp_path, [r1, r2])), str(out), method="kavier")

    col = "metadata.estimated_duration_kavier"
    assert col in df.columns
    one = df[df["metadata.uid"] == "one-hour"].iloc[0]
    two = df[df["metadata.uid"] == "two-hour"].iloc[0]

    # Both feasible -> finite, positive durations.
    assert one[col] > 0 and pd.notna(one[col])
    # 2x the actual work (7200 s vs 3600 s at the same tps) => exactly 2x duration.
    assert two[col] == pytest.approx(2.0 * one[col])

    # min_gpu picks the FEWEST feasible GPUs, so this single job stays small (<= 8) — well within
    # the cluster budget (infrastructure.yaml), which is the real ceiling now, not the job's footprint.
    total = int(df["resources.num_gpus_per_node"].iloc[0]) * int(df["resources.num_nodes"].iloc[0])
    assert 1 <= total <= 8

    # Unrelated column survives; round-trips to disk with the same row count.
    assert df["metadata.uid"].iloc[0] == "one-hour"
    assert out.exists() and len(pd.read_csv(out)) == 2


def test_recommendations_never_exceed_the_cluster_budget(tmp_path):
    """The cluster GPU budget is a hard ceiling: with cluster_gpus=2, no job — even under the
    scale-out `performance` goal — is recommended more than 2 GPUs. The cluster size comes from the
    argument (infrastructure.yaml in production), never from the trace."""
    rows = [{**_GOOD_ROW, "metadata.uid": f"job-{i}", "resources.num_gpus_per_node": 8} for i in range(3)]
    out = tmp_path / "rec.csv"
    df = recommend_trace(
        str(_write_csv(tmp_path, rows)),
        str(out),
        method="kavier",
        goal="performance",
        feasibility="rules",
        cluster_gpus=2,
    )
    total = pd.to_numeric(df["resources.num_gpus_per_node"]) * pd.to_numeric(df["resources.num_nodes"])
    assert (total <= 2).all()  # never exceeds the 2-GPU cluster
    assert (total >= 1).all()


# --------------------------------------------------------------------------- #
# Failure isolation: an unrecommendable / incomplete row must never crash the
# trace — the job appears UNCHANGED: original layout + observed duration
# (extrapolated_duration, else train_runtime), so the timeline still receives it.
# --------------------------------------------------------------------------- #


def test_incomplete_layout_row_short_circuits_before_the_recommender(tmp_path, monkeypatch):
    """A row missing an integer layout field (blank gpus_per_node) must never
    reach coastline.recommend: the guard `not (tokens and batch and gpn and
    nodes)` keeps the job unchanged — original layout, and the OBSERVED duration
    (the fixture's train_runtime = 3600 s, since it has no extrapolated_duration).
    We prove the short-circuit by making recommend a tripwire that fails the test
    if called (this distinguishes 'guard skipped it' from 'exception was swallowed')."""

    def _tripwire(*_a, **_k):
        raise AssertionError("recommend must not be called for an incomplete-layout row")

    monkeypatch.setattr(trace_recommend.coastline, "recommend", _tripwire)

    row = {**_GOOD_ROW, "resources.num_gpus_per_node": None, "resources.num_nodes": 1, "metadata.uid": "incomplete"}
    out = tmp_path / "out.csv"
    df = recommend_trace(str(_write_csv(tmp_path, [row])), str(out), method="kavier")

    # Fallback duration = observed train_runtime (3600.0), not null: the job stays in the replay.
    assert df["metadata.estimated_duration_kavier"].iloc[0] == pytest.approx(3600.0)
    # Original layout is preserved untouched (num_nodes stays 1).
    assert int(df["resources.num_nodes"].iloc[0]) == 1
    assert df["metadata.uid"].iloc[0] == "incomplete"


def test_mixed_trace_recommends_good_row_and_preserves_the_unrecommendable_one(tmp_path):
    """One good row + one unrecommendable row (unknown model/GPU Kavier has no
    library for) in the same trace. The good row gets a positive predicted
    duration; the bad row appears UNCHANGED — its EXACT original layout
    (4 gpn x 2 nodes, batch 8) and its OBSERVED duration (extrapolated_duration
    = 1234.5, which must win over train_runtime = 3600). Contract oracle:
    per-row failure isolation — one bad row changes nothing about the good one
    and leaves its own inputs untouched."""
    good = {**_GOOD_ROW, "metadata.uid": "good"}
    bad = {
        "metadata.model_name": "totally-unknown-model-xyz",
        "metadata.method": "full",
        "resources.gpu_model": "NoSuchGPU-9999",
        "metadata.tokens_per_sample": 4096,
        "metadata.batch_size": 8,
        "resources.num_gpus_per_node": 4,
        "resources.num_nodes": 2,
        "metadata.output.train_tokens_per_second": 15000.0,
        "metadata.train_runtime": 3600.0,
        "metadata.output.extrapolated_duration": 1234.5,
        "metadata.uid": "bad",
    }
    out = tmp_path / "mixed_out.csv"
    df = recommend_trace(str(_write_csv(tmp_path, [good, bad])), str(out), method="kavier")

    assert len(df) == 2 and out.exists()
    g = df[df["metadata.uid"] == "good"].iloc[0]
    b = df[df["metadata.uid"] == "bad"].iloc[0]

    col = "metadata.estimated_duration_kavier"
    assert g[col] > 0 and pd.notna(g[col])
    # Unchanged job: observed extrapolated_duration wins over train_runtime.
    assert b[col] == pytest.approx(1234.5)
    assert "unchanged" in str(b["metadata.recommendation_note"])
    # Bad row's original layout AND batch survive verbatim.
    assert int(b["resources.num_gpus_per_node"]) == 4
    assert int(b["resources.num_nodes"]) == 2
    assert int(b["metadata.batch_size"]) == 8


# --------------------------------------------------------------------------- #
# _job_total_tokens — config-independent "work" = throughput x runtime
# --------------------------------------------------------------------------- #


def test_job_total_tokens_is_throughput_times_runtime():
    row = pd.Series(
        {
            "metadata.output.train_tokens_per_second": 15000.0,
            "metadata.train_runtime": 3600.0,
        }
    )
    # By hand: 15000 tok/s sustained for 3600 s (one hour) = 54,000,000 tokens.
    # tot_tokens_col=None triggers the legacy tps×runtime path.
    assert _job_total_tokens(row, None) == pytest.approx(54_000_000.0)


@pytest.mark.parametrize(
    "tps,rt",
    [
        (None, 3600.0),  # missing throughput
        (15000.0, None),  # missing runtime
        (0.0, 3600.0),  # zero throughput -> no ground truth
        (15000.0, 0.0),  # zero runtime -> no ground truth
        (-5.0, 3600.0),  # negative throughput rejected by the >0 guard
        ("n/a", "n/a"),  # unparseable strings coerce to NaN
    ],
)
def test_job_total_tokens_returns_none_for_missing_or_nonpositive(tps, rt):
    fields = {}
    if tps is not None:
        fields["metadata.output.train_tokens_per_second"] = tps
    if rt is not None:
        fields["metadata.train_runtime"] = rt
    assert _job_total_tokens(pd.Series(fields), None) is None


# --------------------------------------------------------------------------- #
# _METHOD_TO_PREDICTOR — method-name -> coastline predictor key (incl. xgb alias)
# --------------------------------------------------------------------------- #


def test_method_to_predictor_map_is_exactly_the_declared_aliases():
    # The exact contract: only these four keys, and xgb is an alias of xgboost.
    assert _METHOD_TO_PREDICTOR == {
        "kavier": "kavier",
        "tabpfn": "tabpfn",
        "xgb": "xgboost",
        "xgboost": "xgboost",
    }


def test_enrich_resolves_predictor_and_computes_duration_from_recommended_throughput(tmp_path, monkeypatch):
    """recommend_trace maps method 'xgb' -> predictor 'xgboost', passes it plus the
    feasibility choice to coastline.recommend, replaces the layout with the
    recommendation, and derives estimated_duration = job_total_tokens /
    recommended_throughput."""
    captured: dict[str, object] = {}

    def _fake_recommend(workloads, *, predictor, goal, max_gpus, top_k, feasibility, **_):
        captured["predictor"] = predictor
        captured["feasibility"] = feasibility
        # A feasible top-1 recommendation: 1 node x 8 gpus, throughput 15000 tok/s.
        return pd.DataFrame(
            [
                {
                    "feasible": True,
                    "number_of_nodes": 1,
                    "gpus_per_node": 8,
                    "batch_size": 8,
                    "throughput_tok_s": 15000.0,
                }
            ]
        )

    monkeypatch.setattr(trace_recommend.coastline, "recommend", _fake_recommend)

    df = recommend_trace(str(_write_csv(tmp_path, [_GOOD_ROW])), str(tmp_path / "o1.csv"), method="xgb")
    assert captured["predictor"] == "xgboost"
    assert captured["feasibility"] == "autoconf"  # default is the real OOM check
    # Layout replaced by the recommendation.
    assert int(df["resources.num_nodes"].iloc[0]) == 1
    assert int(df["resources.num_gpus_per_node"].iloc[0]) == 8
    # Duration = job_total_tokens / throughput. By hand:
    #   job_total_tokens = 15000 tok/s * 3600 s = 54,000,000 tokens
    #   duration = 54,000,000 / 15000 tok/s = 3600 s  (equals the real runtime,
    #   as expected when predicted throughput matches the observed throughput).
    assert df["metadata.estimated_duration_xgb"].iloc[0] == pytest.approx(3600.0)

    # Caller can opt out to the rules-only feasibility path.
    recommend_trace(str(_write_csv(tmp_path, [_GOOD_ROW])), str(tmp_path / "o2.csv"), method="xgb", feasibility="rules")
    assert captured["feasibility"] == "rules"

    # An unmapped method falls through lowercased, unchanged.
    recommend_trace(str(_write_csv(tmp_path, [_GOOD_ROW])), str(tmp_path / "o3.csv"), method="SomeModel")
    assert captured["predictor"] == "somemodel"


def test_setup_time_col_adds_overhead_to_estimated_duration(tmp_path, monkeypatch):
    """When setup_time_col is provided the duration formula is:
        estimated_duration = setup_time + extrapolated_num_tokens / throughput

    Oracle (all numbers hand-computed):
        extrapolated_num_tokens = 54,000,000  (from tot_tokens_col)
        setup_time              = 120.0 s     (from setup_time_col)
        recommended_throughput  = 15000 tok/s (fake recommender)
        training_time           = 54,000,000 / 15000 = 3600 s
        estimated_duration      = 120 + 3600 = 3720 s

    Without setup_time_col the duration must be just 3600 s (regression guard).
    """

    def _fake_recommend(workloads, *, predictor, goal, max_gpus, top_k, feasibility, **_):
        return pd.DataFrame(
            [{"feasible": True, "number_of_nodes": 1, "gpus_per_node": 8, "batch_size": 8, "throughput_tok_s": 15000.0}]
        )

    monkeypatch.setattr(trace_recommend.coastline, "recommend", _fake_recommend)

    row = {
        **_GOOD_ROW,
        "my_tot_tokens": 54_000_000,
        "my_setup_time": 120.0,
    }

    # With setup_time_col: 120 + 54_000_000 / 15000 = 3720 s
    df_with = recommend_trace(
        str(_write_csv(tmp_path, [row])),
        str(tmp_path / "with_setup.csv"),
        method="kavier",
        tot_tokens_col="my_tot_tokens",
        setup_time_col="my_setup_time",
    )
    assert df_with["metadata.estimated_duration_kavier"].iloc[0] == pytest.approx(3720.0)

    # Without setup_time_col: 54_000_000 / 15000 = 3600 s
    df_without = recommend_trace(
        str(_write_csv(tmp_path, [row])),
        str(tmp_path / "without_setup.csv"),
        method="kavier",
        tot_tokens_col="my_tot_tokens",
    )
    assert df_without["metadata.estimated_duration_kavier"].iloc[0] == pytest.approx(3600.0)


# --------------------------------------------------------------------------- #
# main() / the coastline recommend-trace CLI (via monkeypatched argv)
# --------------------------------------------------------------------------- #


def test_main_cli_enriches_trace_and_reports_the_derived_row_count(tmp_path, capsys):
    src = _write_csv(tmp_path, [_GOOD_ROW])
    out = tmp_path / "cli_out.csv"
    argv = [
        "recommend-trace",
        "--input",
        str(src),
        "--output",
        str(out),
        "--method",
        "kavier",
        "--goal",
        "min_gpu",
        # rules feasibility keeps this test install-agnostic (no AutoConf needed).
        "--feasibility",
        "rules",
    ]
    from coastline.cli import main

    main(argv)

    assert out.exists()
    enriched = pd.read_csv(out)
    # throughput is always written; duration is written via legacy tps×runtime fallback
    assert "metadata.estimated_throughput_kavier" in enriched.columns
    assert "metadata.estimated_duration_kavier" in enriched.columns
    assert len(enriched) == 1
    # The single granite/A100 row is feasible: 1 throughput and 1 duration produced.
    printed = capsys.readouterr().out
    assert str(out) in printed
    assert "1 rows" in printed
    assert "1 with estimated_throughput" in printed
    assert "1 with estimated_duration" in printed
