"""Defaults that work from an installed wheel, which has the package but no config/ directory.

The default run database and the default experiment config ship inside the package, so after a
pip install `--lookup default` finds its database and a bare `coastline recommend-job` runs.
"""

from __future__ import annotations

from pathlib import Path

import yaml

import coastline
from coastline.cli import main
from coastline.cli.run import _DEFAULT_WORKLOAD
from coastline.sdk import policies
from coastline.sdk.io import run_config
from coastline.sdk.io.sample_data import default_run_database_path
from coastline.sdk.models.recommendation import Recommendation
from coastline.sdk.policies import PolicyFactory
from coastline.sdk.predictors.performance.retrieval.cache_predictor import RetrievalPredictor
from coastline.sdk.recommend import engine

_PACKAGE_DIR = Path(coastline.__file__).resolve().parent


def _without_repository_config(tmp_path, monkeypatch) -> None:
    """What an installed wheel sees: no EXPERIMENT_CONFIG and no config/ next to the package."""
    monkeypatch.delenv("EXPERIMENT_CONFIG", raising=False)
    monkeypatch.setattr(run_config, "_CANONICAL_CONFIG", tmp_path / "config" / "experiment.yaml")


# The default run database (lookup: default)
def test_the_default_run_database_ships_inside_the_package():
    path = default_run_database_path()
    assert path.is_file()
    assert _PACKAGE_DIR in path.resolve().parents


def test_the_default_run_database_loads_as_a_lookup_database():
    predictor = RetrievalPredictor(dataset_path=default_run_database_path())
    assert len(predictor.config_index) > 0


def test_lookup_default_is_the_packaged_run_database():
    assert PolicyFactory._lookup_path({"lookup": "default"}) == default_run_database_path()


def test_lookup_default_serves_cache_hits_without_the_repository_config(tmp_path, monkeypatch):
    # What an installed wheel sees: no config/ beside the package. The job is a run in the
    # packaged database, so the cache predictor alone recommends it.
    monkeypatch.setattr(policies, "_REPO_ROOT", tmp_path)
    job = {
        "llm_model": "granite-3.1-2b",
        "fine_tuning_method": "full",
        "gpu_model": "NVIDIA-A100-SXM4-80GB",
        "tokens_per_sample": 4096,
        "batch_size": 1,
    }

    out = coastline.recommend([job], predictor="cache", lookup="default", feasibility="rules", max_gpus=8)

    assert out["feasible"].tolist() == [True], out["error"].tolist()
    assert out["throughput_tok_s"].iloc[0] > 0


# The default experiment config
def test_the_repository_config_is_the_default_when_present(monkeypatch):
    monkeypatch.delenv("EXPERIMENT_CONFIG", raising=False)
    assert run_config._CANONICAL_CONFIG.is_file()
    assert run_config.default_experiment_path() == run_config._CANONICAL_CONFIG


def test_without_the_repository_config_the_bundled_default_is_used(tmp_path, monkeypatch):
    _without_repository_config(tmp_path, monkeypatch)
    path = run_config.default_experiment_path()
    assert path.is_file()
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == run_config.builtin_default_config()


def test_a_missing_experiment_config_override_is_returned_as_given(tmp_path, monkeypatch):
    # An explicit path the user set is not replaced; the caller reports it missing.
    missing = tmp_path / "missing.yaml"
    monkeypatch.setenv("EXPERIMENT_CONFIG", str(missing))
    assert run_config.default_experiment_path() == missing


def test_bare_recommend_job_runs_without_the_repository_config(tmp_path, monkeypatch, capsys):
    _without_repository_config(tmp_path, monkeypatch)
    captured = {}

    def fake_run_request(request):
        captured["request"] = request
        return [
            Recommendation(
                gpus_per_node=2,
                number_of_nodes=1,
                total_gpus=2,
                strategy="multi_objective_balanced",
                predicted_throughput=100.0,
                metadata={"predicted_power_watts": 200.0, "tokens_per_watt": 0.5},
            )
        ], {}

    monkeypatch.setattr(engine, "run_request", fake_run_request)
    main(["recommend-job"])

    request = captured["request"]
    default = run_config.builtin_default_config()
    assert request.config["predictors"] == default["predictors"]
    assert request.config["strategy"] == default["strategy"]
    # The bundled default declares no job, so run.py's own default workload runs.
    assert request.workload.llm_model == _DEFAULT_WORKLOAD["llm_model"]
    assert '"total_gpus": 2' in capsys.readouterr().out


def test_a_missing_default_infrastructure_file_is_not_a_warning(tmp_path, monkeypatch, caplog):
    """An installed wheel has no config/ directory, so the built-in caps apply without a warning."""
    from coastline.sdk.io import infrastructure

    monkeypatch.delenv("INFRASTRUCTURE_CONFIG", raising=False)
    monkeypatch.setattr(infrastructure, "_config_path", lambda: tmp_path / "missing.yaml")
    infrastructure.load_infrastructure.cache_clear()
    try:
        with caplog.at_level("INFO", logger=infrastructure.logger.name):
            infra = infrastructure.load_infrastructure()
    finally:
        infrastructure.load_infrastructure.cache_clear()
    assert infra.total_gpus >= 1
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_a_missing_infrastructure_override_is_a_warning(tmp_path, monkeypatch, caplog):
    """INFRASTRUCTURE_CONFIG pointing at a missing file is a user error, so it is reported."""
    from coastline.sdk.io import infrastructure

    monkeypatch.setenv("INFRASTRUCTURE_CONFIG", str(tmp_path / "missing.yaml"))
    infrastructure.load_infrastructure.cache_clear()
    try:
        with caplog.at_level("INFO", logger=infrastructure.logger.name):
            infrastructure.load_infrastructure()
    finally:
        infrastructure.load_infrastructure.cache_clear()
    assert [r for r in caplog.records if r.levelname == "WARNING" and "not found" in r.getMessage()]
