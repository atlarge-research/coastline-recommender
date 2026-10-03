"""AutoConf's OOM verdict for a feasibility_model does not depend on how the model is spelled.

AutoConf's name mapper uses ^-anchored lowercase patterns, so every spelling has to reach it as
'mistral-7b-v0.1'. The classifier below is a stand-in that, like AutoConf, only recognises the
lowercase short name.
"""

import pytest

import coastline.sdk.predictors.feasibility.autoconf as af
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.feasibility.autoconf import AutoconfFeasibilityChecker


class _JobConfig:
    def __init__(self, fields):
        self.fields = fields

    @classmethod
    def model_validate(cls, data):
        return cls(dict(data))


def _modules(seen):
    def load_model(model_version):
        return object()

    def get_model_prediction_and_metadata(job_config, predictor):
        seen.append(job_config.fields["model_name"])
        return (1 if job_config.fields["model_name"] == "mistral-7b-v0.1" else 0), {}

    return load_model, _JobConfig, get_model_prediction_and_metadata


@pytest.mark.parametrize("spelling", ["mistral-7b-v0.1", "mistralai/Mistral-7B-v0.1", "Mistral-7B-v0.1"])
def test_every_spelling_reaches_autoconf_as_the_same_model(monkeypatch, spelling):
    seen: list = []
    monkeypatch.setattr(af, "_autoconf_modules", lambda: _modules(seen))
    workload = WorkloadSpec(
        llm_model="anon-proxy",
        fine_tuning_method="lora",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=2048,
        batch_size=16,
        gpus_per_node=8,
        number_of_nodes=1,
        feasibility_model=spelling,
    )

    feasible, _ = AutoconfFeasibilityChecker().is_feasible(workload)

    assert seen == ["mistral-7b-v0.1"]
    assert feasible is True
