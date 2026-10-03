"""recommend-trace: a whole-file mistake fails the run; a per-row failure note gives the engine's reason.

A missing identity column or a mistyped --feasibility or --method affects the whole call, so it
raises (exit 2 from the CLI) instead of failing mid-trace or writing an 'infeasible' note on every
row. A row the engine cannot recommend keeps its original config, and its note carries the
engine's reason, also when it suggests rerunning with kavier; the infeasible note is kept for a
grid with no feasible config. Real runs pin Kavier and the `rules` checker.
"""

from __future__ import annotations

import pandas as pd
import pytest

from coastline.cli import main
from coastline.sdk.trace import recommend as trace_recommend
from coastline.sdk.trace.recommend import recommend_trace

_ROW = {
    "metadata.model_name": "mistral-7b-v0.1",
    "metadata.method": "lora",
    "resources.gpu_model": "NVIDIA-A100-SXM4-80GB",
    "metadata.tokens_per_sample": 1024,
    "metadata.batch_size": 8,
    "resources.num_gpus_per_node": 8,
    "resources.num_nodes": 1,
    "metadata.output.extrapolated_duration": 3600.0,
}

_NOTE = "metadata.recommendation_note"


def _write(tmp_path, rows, name="trace.csv"):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


@pytest.mark.parametrize("column", ["metadata.model_name", "metadata.method", "resources.gpu_model"])
def test_a_missing_identity_column_is_named(tmp_path, column) -> None:
    trace = _write(tmp_path, [{k: v for k, v in _ROW.items() if k != column}])

    with pytest.raises(ValueError, match=column):
        recommend_trace(trace, str(tmp_path / "out.csv"), feasibility="rules")


def test_the_cli_reports_a_missing_identity_column_as_a_usage_error(tmp_path, capsys) -> None:
    trace = _write(tmp_path, [{k: v for k, v in _ROW.items() if k != "metadata.model_name"}])

    with pytest.raises(SystemExit) as excinfo:
        main(["recommend-trace", "--input", trace, "--output", str(tmp_path / "o.csv"), "--feasibility", "rules"])

    assert excinfo.value.code == 2
    assert "metadata.model_name" in capsys.readouterr().err


@pytest.mark.parametrize("mode", ["bogus", "Autoconf"])
def test_an_unknown_feasibility_mode_fails_the_call(tmp_path, mode) -> None:
    trace = _write(tmp_path, [_ROW])

    with pytest.raises(ValueError, match="feasibility"):
        recommend_trace(trace, str(tmp_path / "out.csv"), feasibility=mode)
    assert not (tmp_path / "out.csv").exists()


def test_the_cli_rejects_an_unknown_feasibility_mode(tmp_path, capsys) -> None:
    trace = _write(tmp_path, [_ROW])

    with pytest.raises(SystemExit) as excinfo:
        main(["recommend-trace", "--input", trace, "--output", str(tmp_path / "o.csv"), "--feasibility", "bogus"])

    assert excinfo.value.code == 2
    assert "--feasibility" in capsys.readouterr().err


def test_an_unknown_method_fails_the_call(tmp_path) -> None:
    trace = _write(tmp_path, [_ROW])

    with pytest.raises(ValueError, match="unknown predictor"):
        recommend_trace(trace, str(tmp_path / "out.csv"), method="kavierr", feasibility="rules")


def test_a_row_the_engine_rejects_carries_the_engine_reason(tmp_path) -> None:
    good = dict(_ROW)
    unknown = {**_ROW, "metadata.model_name": "not-a-model"}
    df = recommend_trace(_write(tmp_path, [good, unknown]), str(tmp_path / "out.csv"), feasibility="rules")

    assert pd.isna(df[_NOTE].iloc[0])  # the good row is recommended
    note = str(df[_NOTE].iloc[1])
    assert "not-a-model" in note  # the predictor's own reason
    assert "structural feasibility guards" not in note
    assert "unchanged" in note  # and the row is still kept for the replay


def test_a_grid_with_no_feasible_config_keeps_the_infeasible_note(tmp_path, monkeypatch) -> None:
    """The engine's 'no feasible candidates' error is the case the infeasible note describes."""

    def no_feasible_config(batch, **kwargs):
        return pd.DataFrame(
            [
                {
                    **batch[0],
                    "feasible": False,
                    "error": "Workflow (min_gpu): no feasible candidates found in grid of 6 configurations.",
                }
            ]
        )

    monkeypatch.setattr(trace_recommend.coastline, "recommend", no_feasible_config)
    df = recommend_trace(_write(tmp_path, [_ROW]), str(tmp_path / "out.csv"), feasibility="autoconf", cluster_gpus=8)

    assert str(df[_NOTE].iloc[0]).startswith("infeasible within 8 GPUs: every config would run out of GPU memory")


def test_a_row_kavier_can_handle_keeps_the_method_reason_and_the_hint(tmp_path, monkeypatch) -> None:
    """Another method fails on a row that Kavier can predict: the note gives that method's own error
    as well as the hint to rerun with kavier."""
    reason = "the tabpfn model is not in this install (looked for /models/tabpfn.pkl)"
    real_recommend = trace_recommend.coastline.recommend

    def recommend(batch, **kwargs):
        if kwargs["predictor"] == "kavier":
            return real_recommend(batch, **kwargs)
        return pd.DataFrame([{**batch[0], "feasible": False, "error": reason}])

    monkeypatch.setattr(trace_recommend.coastline, "recommend", recommend)
    df = recommend_trace(
        _write(tmp_path, [_ROW]), str(tmp_path / "out.csv"), method="tabpfn", feasibility="rules", cluster_gpus=8
    )

    note = str(df[_NOTE].iloc[0])
    assert note.startswith(f"'tabpfn' could not predict this workload: {reason}")
    assert "rerun with --method kavier" in note
