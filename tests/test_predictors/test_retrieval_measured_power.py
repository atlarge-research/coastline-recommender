"""A cache hit on a measured run yields a recommendation even when Kavier has no power model.

The pipeline drops a candidate without power, and Kavier declines models outside its catalog,
such as the bundled sample's demo-llm-3b. A hit carries the run's gpu_power_watts_avg in
``metadata['measured_power_watts']``. The pipeline and ``coastline simulate`` use it when the power
predictor gives no power, and a workload Kavier covers keeps Kavier's power. The lookup calls no
Kavier itself, so a hit costs one hash lookup.
"""

from pathlib import Path

import pandas as pd
import pytest

import coastline
from coastline.sdk.io.sample_data import sample_raw_trace_path
from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.pipeline.workflow import simulate_one
from coastline.sdk.predictors.energy import KavierPowerPredictor
from coastline.sdk.predictors.performance.physics import kavier_predictor
from coastline.sdk.predictors.performance.retrieval.cache_predictor import RetrievalPredictor
from coastline.sdk.recommend import simulate as sdk_simulate

_GPU = "NVIDIA-A100-SXM4-80GB"
_CONTEXT = SystemContext.for_gpus([_GPU], max_gpus=8)


def _workload(model: str, gpus: int, batch_size: int = 8) -> WorkloadSpec:
    return WorkloadSpec(
        llm_model=model,
        fine_tuning_method="lora",
        gpu_model=_GPU,
        tokens_per_sample=2048,
        batch_size=batch_size,
        gpus_per_node=gpus,
        number_of_nodes=1,
    )


def _measured_run(model: str, power: float | None = None) -> dict:
    """One lookup row on 4 GPUs: 9000 tok/s and 100 s, with a measured power when given."""
    row = dict(
        model_name=model,
        method="lora",
        gpu_model=_GPU,
        number_nodes=1,
        number_gpus=4,
        tokens_per_sample=2048,
        batch_size=8,
        dataset_tokens_per_second=9000.0,
        train_runtime=100.0,
    )
    if power is not None:
        row["gpu_power_watts_avg"] = power
    return row


def _lookup_csv(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_hit_on_a_model_kavier_declines_uses_the_measured_power():
    cache = RetrievalPredictor(dataset_path=sample_raw_trace_path())
    workload = _workload("demo-llm-3b", gpus=4)  # sample row: 14000 tok/s, 115 s, 200 W
    assert KavierPowerPredictor().predict(workload, _CONTEXT).predicted_power is None

    hit = cache.predict(workload, _CONTEXT)
    assert hit.metadata["measured_power_watts"] == pytest.approx(200.0)
    assert simulate_one(cache, KavierPowerPredictor(), workload, _CONTEXT) == pytest.approx((14000.0, 200.0, 115.0))


@pytest.mark.parametrize("predictor", ["cache", "intelligent"])
@pytest.mark.parametrize("goal", ["balanced", "min_gpu"])
def test_the_bundled_sample_produces_recommendations(monkeypatch, predictor, goal):
    monkeypatch.setenv("DATA_DIR", "/nonexistent/coastline-no-trace")  # force the bundled sample
    recs = coastline.Coastline(predictor=predictor, feasibility="rules").recommend(
        {
            "llm_model": "demo-llm-3b",
            "fine_tuning_method": "lora",
            "gpu_model": _GPU,
            "tokens_per_sample": 2048,
            "batch_size": 8,
        },
        goal=goal,
        max_gpus=8,
        batch_sizes=[8, 16],
    )
    # The sample measures 5 of the 8 grid configurations; each one is a candidate.
    measured = {(1, 8): 300.0, (2, 8): 280.0, (4, 8): 200.0, (8, 8): 155.0, (1, 16): 320.0}
    assert recs
    for rec in recs:
        key = (rec.total_gpus, rec.metadata["batch_size"])
        assert rec.metadata["predicted_power_watts"] == pytest.approx(measured[key])


def test_hit_on_a_model_kavier_covers_keeps_kavier_power(tmp_path):
    workload = _workload("mistral-7b-v0.1", gpus=4)
    cache = RetrievalPredictor(
        dataset_path=_lookup_csv(tmp_path / "lookup.csv", [_measured_run(workload.llm_model, 123.0)])
    )

    hit = cache.predict(workload, _CONTEXT)
    assert hit.metadata["measured_power_watts"] == pytest.approx(123.0)
    kavier_power = KavierPowerPredictor().predict(workload, _CONTEXT).predicted_power
    assert kavier_power != pytest.approx(123.0)
    assert simulate_one(cache, KavierPowerPredictor(), workload, _CONTEXT) == pytest.approx(
        (9000.0, kavier_power, 100.0)
    )


def test_lookup_without_a_power_column_still_drops_the_candidate(tmp_path):
    cache = RetrievalPredictor(dataset_path=_lookup_csv(tmp_path / "lookup.csv", [_measured_run("demo-llm-3b")]))
    workload = _workload("demo-llm-3b", gpus=4)

    assert "measured_power_watts" not in cache.predict(workload, _CONTEXT).metadata
    assert simulate_one(cache, KavierPowerPredictor(), workload, _CONTEXT) is None


def test_the_lookup_calls_no_kavier(monkeypatch, tmp_path):
    """Hits with and without a model Kavier covers, and a miss, are answered from the index."""
    calls = []
    performance = kavier_predictor._kavier_training.performance
    monkeypatch.setattr(
        kavier_predictor._kavier_training, "performance", lambda row: calls.append(row) or performance(row)
    )
    rows = [_measured_run("demo-llm-3b", 200.0), _measured_run("mistral-7b-v0.1", 123.0)]
    cache = RetrievalPredictor(dataset_path=_lookup_csv(tmp_path / "lookup.csv", rows))

    for model in ("demo-llm-3b", "mistral-7b-v0.1"):
        assert cache.predict(_workload(model, gpus=4), _CONTEXT) is not None
    assert cache.predict(_workload("mistral-7b-v0.1", gpus=2), _CONTEXT) is None
    assert calls == []

    # The counter sees a Kavier call made through the power predictor.
    KavierPowerPredictor().predict(_workload("mistral-7b-v0.1", gpus=4), _CONTEXT)
    assert len(calls) == 1


def test_simulate_reports_the_measured_power_when_kavier_has_none(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "/nonexistent/coastline-no-trace")  # force the bundled sample

    result = sdk_simulate.simulate_one(
        _workload("demo-llm-3b", gpus=4), _CONTEXT, predictor="cache", feasibility="none"
    )

    assert result["error"] is None
    assert result["predicted_throughput"] == pytest.approx(14000.0)
    assert result["predicted_power_watts"] == pytest.approx(200.0)
    assert result["cluster_power_watts"] == pytest.approx(800.0)


def test_simulate_keeps_kavier_power_for_a_model_kavier_covers(monkeypatch, tmp_path):
    workload = _workload("mistral-7b-v0.1", gpus=4)
    (tmp_path / "profiling-dataset").mkdir()
    _lookup_csv(tmp_path / "profiling-dataset" / "raw_trace.csv", [_measured_run(workload.llm_model, 123.0)])
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    result = sdk_simulate.simulate_one(workload, _CONTEXT, predictor="cache", feasibility="none")

    assert result["predicted_throughput"] == pytest.approx(9000.0)
    assert result["predicted_power_watts"] == pytest.approx(
        KavierPowerPredictor().predict(workload, _CONTEXT).predicted_power
    )
