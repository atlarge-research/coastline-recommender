"""Tests for Coastline's own logic in coastline.sdk.models; pydantic's behaviour is not retested.

- WorkloadSpec.total_gpus, a computed field (gpus_per_node x number_of_nodes).
- The Prediction and Recommendation check that total_gpus == gpus_per_node x number_of_nodes.
- A JSON round trip, since the API sends these models over the wire.
"""

import json

import pytest
from pydantic import ValidationError

from coastline.sdk.models import Prediction, Recommendation, WorkloadSpec


def _workload(**overrides):
    base = dict(
        llm_model="mistral-7b-v0.1",
        fine_tuning_method="full",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=512,
        batch_size=4,
    )
    base.update(overrides)
    return WorkloadSpec(**base)


class TestWorkloadTotalGpus:
    """total_gpus = gpus_per_node * number_of_nodes, each defaulting to 1."""

    @pytest.mark.parametrize(
        "gpn, nodes, expected",
        [
            (None, None, 1),
            (4, None, 4),
            (None, 4, 4),
            (8, 2, 16),
            (2, 3, 6),
        ],
    )
    def test_is_product_with_one_defaults(self, gpn, nodes, expected):
        kw = {}
        if gpn is not None:
            kw["gpus_per_node"] = gpn
        if nodes is not None:
            kw["number_of_nodes"] = nodes
        assert _workload(**kw).total_gpus == expected

    def test_inbound_total_gpus_is_ignored(self):
        # total_gpus is computed, so an input value does not override the product.
        assert _workload(gpus_per_node=2, number_of_nodes=3, total_gpus=999).total_gpus == 6


@pytest.mark.parametrize(
    "Model, extra",
    [
        (Prediction, {}),
        (Recommendation, {"strategy": "min_gpu"}),
    ],
    ids=["prediction", "recommendation"],
)
class TestTotalGpusInvariant:
    """Prediction/Recommendation enforce total_gpus == gpus_per_node * number_of_nodes."""

    def test_rejects_inconsistent(self, Model, extra):
        with pytest.raises(ValidationError):
            Model(gpus_per_node=4, number_of_nodes=2, total_gpus=7, **extra)

    def test_accepts_consistent(self, Model, extra):
        assert Model(gpus_per_node=4, number_of_nodes=2, total_gpus=8, **extra).total_gpus == 8


class TestWorkloadCanonicalization:
    """WorkloadSpec canonicalizes llm_model at ingestion: it drops everything up to and including
    the last '/', then lowercases."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            # last segment "Mistral-7B-v0.1", lowercased
            ("mistralai/Mistral-7B-v0.1", "mistral-7b-v0.1"),
            # already short and lowercase: unchanged
            ("mistral-7b-v0.1", "mistral-7b-v0.1"),
            # uppercase short form: lowercased only
            ("LLAMA-2-7B", "llama-2-7b"),
            # org path: only the final segment is kept
            ("meta-llama/Llama-2-70b-hf", "llama-2-70b-hf"),
        ],
    )
    def test_llm_model_is_canonicalized_at_ingestion(self, raw, expected):
        assert _workload(llm_model=raw).llm_model == expected


def test_prediction_round_trips_through_json():
    """A Prediction survives a JSON round trip unchanged, including nested metadata and the float
    throughput. The API sends Predictions over the wire."""
    p = Prediction(
        gpus_per_node=8, number_of_nodes=2, total_gpus=16, predicted_throughput=999.0, metadata={"predictor": "kavier"}
    )
    restored = Prediction(**json.loads(p.model_dump_json()))
    assert restored == p
    # The nested and numeric fields, checked one by one.
    assert restored.metadata == {"predictor": "kavier"}
    assert restored.predicted_throughput == 999.0


# An empty model key is rejected at construction time


class TestWorkloadEmptyModelKey:
    """WorkloadSpec rejects llm_model values that canonicalize to an empty string."""

    @pytest.mark.parametrize(
        "bad_model",
        [
            "someorg/",  # trailing slash: the canonical form is ''
            "/",  # slash only
            "//",  # double slash
        ],
    )
    def test_trailing_slash_raises_value_error(self, bad_model):
        with pytest.raises(ValidationError) as exc_info:
            _workload(llm_model=bad_model)
        # The error message says why.
        assert "empty after canonicalization" in str(exc_info.value)
