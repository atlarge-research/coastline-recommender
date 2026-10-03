"""Model names in Kavier's catalog, and the HuggingFace ids of its models, reach the catalog entry.

WorkloadSpec lowercases llm_model, while the catalog has mixed-case keys ('Llama-3-8B',
'BLOOM-176B', ...), and an HF id such as 'meta-llama/Llama-3.2-3B' lowercases to 'llama-3.2-3b'
where the catalog key is 'llama3.2-3b'. Each case is compared with Kavier called on the catalog
key. The HF ids map to keys as in ADO's sfttrainer catalog (config/models.yaml).
"""

import math

import numpy as np
import pytest
from kavier import training as kavier_training
from kavier.sdk.library import LLM_SPEC_LIBRARY, UnknownSpecError, get_gpu, get_llm

import coastline
from coastline.sdk.library.llm_names import LLM_ALIASES
from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.performance.data_driven.ml_common import llm_spec_features
from coastline.sdk.predictors.performance.physics.kavier_predictor import KavierPredictor

_GPU = "NVIDIA-A100-SXM4-80GB"
_CONTEXT = SystemContext.for_gpus([_GPU], max_gpus=8)


def _workload(model: str, method: str = "lora", gpu: str = _GPU) -> WorkloadSpec:
    return WorkloadSpec(
        llm_model=model,
        fine_tuning_method=method,
        gpu_model=gpu,
        tokens_per_sample=1024,
        batch_size=4,
        gpus_per_node=2,
        number_of_nodes=1,
    )


def _kavier_on_key(key: str, method: str = "lora") -> float:
    row = {"model": key, "gpu": _GPU, "method": method, "seq_len": 1024, "batch_size": 4, "num_gpus": 2, "num_nodes": 1}
    return float(kavier_training.performance(row).iloc[0]["train_tokens_per_second"])


_MIXED_CASE_KEYS = sorted(key for key in LLM_SPEC_LIBRARY if key != key.lower())

# HF id, and the catalog key ADO's sfttrainer uses for that checkpoint.
_HF_IDS = [
    ("meta-llama/Llama-3.2-3B", "llama3.2-3b"),
    ("meta-llama/Llama-3.2-1B", "llama3.2-1b"),
    ("meta-llama/Llama-3.1-8B", "llama3.1-8b"),
    ("meta-llama/Llama-3.1-70B", "llama3.1-70b"),
    ("ibm-granite/granite-3.3-8b-base", "granite-3.3-8b"),
    ("ibm-granite/granite-3.0-8b-base", "granite-3-8b"),
    ("ibm-granite/granite-3.1-2b-base", "granite-3.1-2b"),
    ("mistralai/Mistral-7B-v0.1", "mistral-7b-v0.1"),
]


def test_the_catalog_has_mixed_case_keys():
    assert "Llama-3-8B" in _MIXED_CASE_KEYS and "BLOOM-176B" in _MIXED_CASE_KEYS


@pytest.mark.parametrize("key", _MIXED_CASE_KEYS)
def test_a_mixed_case_catalog_key_is_predicted(key):
    prediction = KavierPredictor().predict(_workload(key), _CONTEXT)
    assert prediction.predicted_throughput == pytest.approx(_kavier_on_key(key))


@pytest.mark.parametrize("hf_id, key", _HF_IDS)
def test_an_hf_id_is_predicted_as_its_catalog_entry(hf_id, key):
    prediction = KavierPredictor().predict(_workload(hf_id), _CONTEXT)
    assert prediction.predicted_throughput == pytest.approx(_kavier_on_key(key))


@pytest.mark.parametrize("method", ["LoRA", "LORA", "Full"])
def test_method_case_does_not_matter(method):
    prediction = KavierPredictor().predict(_workload("mistral-7b-v0.1", method), _CONTEXT)
    assert prediction.predicted_throughput == pytest.approx(_kavier_on_key("mistral-7b-v0.1", method.lower()))


def _kavier_message(lookup, name: str) -> str:
    with pytest.raises(UnknownSpecError) as err:
        lookup(name)
    return str(err.value)


def test_an_unknown_model_reports_kavier_s_own_catalog_listing():
    prediction = KavierPredictor().predict(_workload("not-a-real-model-7b"), _CONTEXT)
    assert prediction.metadata["error"] == "unsupported_config"
    # The detail is Kavier's own message, which lists its catalog.
    assert prediction.metadata["error_detail"] == _kavier_message(get_llm, "not-a-real-model-7b")


def test_an_unknown_gpu_reports_kavier_s_own_catalog_listing():
    prediction = KavierPredictor().predict(_workload("mistral-7b-v0.1", gpu="FAKE-GPU-1"), _CONTEXT)
    assert prediction.metadata["error_detail"] == _kavier_message(get_gpu, "FAKE-GPU-1")


def test_every_alias_points_at_a_catalog_entry():
    assert set(LLM_ALIASES.values()) <= set(LLM_SPEC_LIBRARY)


@pytest.mark.parametrize("name, key", [("Llama-3-8B", "Llama-3-8B"), ("meta-llama/Llama-3.2-3B", "llama3.2-3b")])
def test_ml_spec_features_resolve_the_same_names(name, key):
    resolved = llm_spec_features(_workload(name).llm_model)
    assert all(math.isfinite(value) for value in resolved.values())
    assert resolved == llm_spec_features(key)


def test_ml_spec_features_stay_nan_for_an_unknown_model():
    assert all(np.isnan(value) for value in llm_spec_features("not-a-real-model-7b").values())


@pytest.mark.parametrize("model, method", [("Llama-3-8B", "lora"), ("mistral-7b-v0.1", "LoRA")])
def test_the_facade_recommends_for_these_spellings(model, method):
    recs = coastline.Coastline(predictor="kavier", feasibility="rules").recommend(
        {
            "llm_model": model,
            "fine_tuning_method": method,
            "gpu_model": _GPU,
            "tokens_per_sample": 2048,
            "batch_size": 8,
        },
        max_gpus=8,
        batch_sizes=[4, 8],
    )
    assert recs and all(rec.predicted_throughput > 0 for rec in recs)
