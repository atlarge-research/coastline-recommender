"""One GPU vocabulary: every name the context accepts can be predicted, and Kavier's GPUs are accepted.

Coastline's GPU names and aliases match Kavier's GPU catalog, so a name such as 'A100-SXM4-80GB'
that passes the context check also works in Kavier, and Kavier names such as 'A100-80GB' or
'H100-SXM' pass the context check.
"""

import math

import pytest
from kavier.sdk.library import GPU_SPEC_LIBRARY

import coastline
from coastline.sdk.exceptions import UnsupportedGPUError
from coastline.sdk.library.hardware import (
    GPU_ALIASES,
    get_gpu_idle_power,
    get_gpu_memory,
    get_gpu_tdp,
    list_supported_gpus,
)
from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.performance.data_driven.ml_common import gpu_spec_features
from coastline.sdk.predictors.performance.physics.kavier_predictor import KavierPredictor

_ACCEPTED = sorted({*list_supported_gpus(), *GPU_ALIASES})


def _workload(gpu: str) -> WorkloadSpec:
    return WorkloadSpec(
        llm_model="mistral-7b-v0.1",
        fine_tuning_method="lora",
        gpu_model=gpu,
        tokens_per_sample=2048,
        batch_size=8,
        gpus_per_node=4,
        number_of_nodes=1,
    )


@pytest.mark.parametrize("gpu", _ACCEPTED)
def test_every_name_the_context_accepts_is_predicted(gpu):
    context = SystemContext.for_gpus([gpu], max_gpus=8)
    prediction = KavierPredictor().predict(_workload(gpu), context)
    assert math.isfinite(prediction.predicted_throughput) and prediction.predicted_throughput > 0
    assert prediction.predicted_power > 0


@pytest.mark.parametrize("gpu", sorted(GPU_SPEC_LIBRARY))
def test_every_kavier_gpu_is_accepted_with_its_memory(gpu):
    context = SystemContext.for_gpus([gpu], max_gpus=8)
    assert context.gpu_memory == {gpu: int(GPU_SPEC_LIBRARY[gpu].memory_gb)}


@pytest.mark.parametrize("alias, name", sorted(GPU_ALIASES.items()))
def test_an_alias_is_the_same_gpu_everywhere(alias, name):
    assert name in GPU_SPEC_LIBRARY
    assert _workload(alias).gpu_model == name
    assert SystemContext.for_gpus([alias], max_gpus=8).available_gpu_models == [name]
    assert get_gpu_memory(alias) == get_gpu_memory(name)
    assert gpu_spec_features(alias) == gpu_spec_features(name)


def test_a_gpu_kavier_cannot_predict_is_refused_up_front():
    # The SXM4 40 GB part has no Kavier entry; its 400 W TDP rules out the 250 W 'A100-40GB'.
    with pytest.raises(UnsupportedGPUError, match="A100-SXM4-40GB") as err:
        SystemContext.for_gpus(["A100-SXM4-40GB"], max_gpus=8)
    assert "NVIDIA-A100-SXM4-80GB" in str(err.value) and "H100-SXM" in str(err.value)


@pytest.mark.parametrize(
    "gpu, memory, tdp, idle",
    [
        ("NVIDIA-A100-SXM4-80GB", 80, 400.0, 75.0),
        ("NVIDIA-A100-80GB-PCIe", 80, 300.0, 60.0),
        ("L40S", 48, 350.0, 40.0),
        ("NVIDIA-H100-PCIe", 80, 350.0, 50.0),
    ],
)
def test_the_datasheet_values_of_the_existing_names_are_unchanged(gpu, memory, tdp, idle):
    assert (get_gpu_memory(gpu), get_gpu_tdp(gpu), get_gpu_idle_power(gpu)) == (memory, tdp, idle)


@pytest.mark.parametrize("gpu", ["A100-SXM4-80GB", "A100-80GB", "H100-SXM"])
def test_the_facade_recommends_on_these_gpus(gpu):
    recs = coastline.Coastline(predictor="kavier", feasibility="rules").recommend(
        {
            "llm_model": "mistral-7b-v0.1",
            "fine_tuning_method": "lora",
            "gpu_model": gpu,
            "tokens_per_sample": 2048,
            "batch_size": 8,
        },
        max_gpus=8,
        batch_sizes=[4, 8],
    )
    assert recs and all(rec.predicted_throughput > 0 for rec in recs)


def test_an_alias_recommends_exactly_like_its_kavier_name():
    def recommend(gpu):
        recs = coastline.Coastline(predictor="kavier", feasibility="rules").recommend(
            {
                "llm_model": "mistral-7b-v0.1",
                "fine_tuning_method": "lora",
                "gpu_model": gpu,
                "tokens_per_sample": 2048,
                "batch_size": 8,
            },
            max_gpus=8,
            batch_sizes=[4, 8],
        )
        return [(r.total_gpus, r.metadata["batch_size"], r.predicted_throughput) for r in recs]

    assert recommend("A100-SXM4-80GB") == recommend("NVIDIA-A100-SXM4-80GB")
