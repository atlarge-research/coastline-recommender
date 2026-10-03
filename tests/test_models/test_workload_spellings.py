"""WorkloadSpec gives every consumer (Kavier, the cache, AutoConf, the ML models) one spelling.

It canonicalizes llm_model, feasibility_model, fine_tuning_method and gpu_model at ingestion.
Otherwise 'mistralai/Mistral-7B-v0.1' would get a different AutoConf verdict than
'mistral-7b-v0.1', 'LoRA' would make Kavier raise, and 'A100-SXM4-80GB' would fail every
prediction.
"""

import pytest

from coastline.sdk.models import WorkloadSpec


def _workload(**overrides):
    base = dict(
        llm_model="mistral-7b-v0.1",
        fine_tuning_method="lora",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=2048,
        batch_size=8,
    )
    base.update(overrides)
    return WorkloadSpec(**base)


class TestFeasibilityModel:
    @pytest.mark.parametrize("raw", ["mistralai/Mistral-7B-v0.1", "Mistral-7B-v0.1", "mistral-7b-v0.1"])
    def test_is_canonicalized_like_llm_model(self, raw):
        assert _workload(feasibility_model=raw).feasibility_model == "mistral-7b-v0.1"
        assert _workload(feasibility_model=raw).feasibility_model == _workload(llm_model=raw).llm_model

    def test_unset_stays_unset(self):
        assert _workload().feasibility_model is None

    @pytest.mark.parametrize("name", ["llama3.1-8b", "granite-3.3-8b-base", "granite-2b-base", "llama-3.1-8b"])
    def test_lowercase_names_from_the_in_vitro_traces_are_unchanged(self, name):
        # Exp4 sets feasibility_model from these trace columns, so they must stay as they are.
        assert _workload(feasibility_model=name).feasibility_model == name


class TestFineTuningMethod:
    @pytest.mark.parametrize(
        "raw, expected",
        [("LoRA", "lora"), ("FULL", "full"), ("GPTQ-LoRA", "gptq-lora"), (" lora ", "lora"), ("lora", "lora")],
    )
    def test_is_lowercased(self, raw, expected):
        assert _workload(fine_tuning_method=raw).fine_tuning_method == expected


class TestGpuModel:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("A100-SXM4-80GB", "NVIDIA-A100-SXM4-80GB"),
            ("A100-PCIE-80GB", "NVIDIA-A100-80GB-PCIe"),
            ("A100-PCIE-40GB", "A100-40GB"),
        ],
    )
    def test_coastline_spellings_map_to_kavier_names(self, raw, expected):
        assert _workload(gpu_model=raw).gpu_model == expected

    @pytest.mark.parametrize("name", ["NVIDIA-A100-SXM4-80GB", "NVIDIA-A100-80GB-PCIe", "L40S", "H100-SXM", "FAKE-GPU"])
    def test_other_names_are_unchanged(self, name):
        assert _workload(gpu_model=name).gpu_model == name
