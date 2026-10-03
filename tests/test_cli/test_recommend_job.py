"""Mode selection in `coastline recommend-job`.

The command has three input modes (`--interactive`, `--config`, `--input/--output`); with no
flags it runs the declared job of the default config. These tests check the routing only. The
default config selects AutoConf and the cache predictor, so the engine call is stubbed and real
runs are left to the pipeline tests.
"""

from __future__ import annotations

import json

import pytest
import yaml

from coastline.cli import main
from coastline.sdk.io.run_config import default_experiment_path
from coastline.sdk.models.recommendation import Recommendation
from coastline.sdk.recommend import engine


def _payload(capsys) -> dict:
    """The recommendation JSON printed to stdout, past the run's log lines (same stream)."""
    out = capsys.readouterr().out
    return json.loads(out[out.index("{") :])


@pytest.fixture
def captured_request(monkeypatch):
    """Stub the engine call that every mode goes through and keep the request it receives."""
    captured: dict = {}

    def fake_run_request(request):
        captured["request"] = request
        return [
            Recommendation(
                gpus_per_node=4,
                number_of_nodes=1,
                total_gpus=4,
                strategy="multi_objective_balanced",
                predicted_throughput=1234.5,
                metadata={"predicted_power_watts": 200.0, "tokens_per_watt": 6.17},
            )
        ], {}

    monkeypatch.setattr(engine, "run_request", fake_run_request)
    return captured


def test_bare_recommend_job_runs_the_default_configuration(capsys, captured_request) -> None:
    """With no flags the command runs the declared job of the default config and does not exit 2."""
    main(["recommend-job"])

    declared = (yaml.safe_load(default_experiment_path().read_text(encoding="utf-8")) or {})["workload"]
    workload = captured_request["request"].workload
    assert workload.llm_model == declared["llm_model"]
    assert workload.fine_tuning_method == declared["fine_tuning_method"]
    assert workload.tokens_per_sample == declared["tokens_per_sample"]
    assert workload.batch_size == declared["batch_size"]

    # The recommendation reaches stdout, as with an explicit --config.
    assert _payload(capsys)["configuration"]["total_gpus"] == 4


def test_bare_recommend_job_matches_an_explicit_default_config(capsys, captured_request) -> None:
    main(["recommend-job"])
    bare = _payload(capsys)

    main(["recommend-job", "--config", str(default_experiment_path())])
    explicit = _payload(capsys)

    del bare["timestamp"], explicit["timestamp"]
    assert bare == explicit


def test_recommend_job_still_rejects_an_output_without_an_input(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["recommend-job", "--output", "recs.csv"])

    assert excinfo.value.code == 2
    assert "--input" in capsys.readouterr().err


def test_recommend_job_still_rejects_a_batch_without_a_config(capsys) -> None:
    """The config carries the recommendation policy; a batch must not silently take the default."""
    with pytest.raises(SystemExit) as excinfo:
        main(["recommend-job", "--input", "workloads.csv", "--output", "recs.csv"])

    assert excinfo.value.code == 2
    assert "--config" in capsys.readouterr().err


def test_recommend_job_still_rejects_a_json_override_without_a_config(capsys) -> None:
    """--input alone overrides a config's workload, so it needs a config."""
    with pytest.raises(SystemExit) as excinfo:
        main(["recommend-job", "--input", "job.json"])

    assert excinfo.value.code == 2
    assert "--config" in capsys.readouterr().err
