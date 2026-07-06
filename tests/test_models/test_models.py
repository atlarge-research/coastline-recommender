"""Tests for the *custom* logic in coastline.sdk.models.

Pydantic mechanics (required-field enforcement, type coercion, default factories,
plain serialization) are the library's job and are deliberately NOT re-tested
here. What IS Coastline logic, and is tested:
  - WorkloadSpec.total_gpus — a computed field (gpus_per_node * number_of_nodes).
  - Prediction/Recommendation — the total_gpus == gpus_per_node*number_of_nodes
    invariant validator (these models otherwise let it drift freely).
Plus one round-trip smoke, since the API serializes these models over the wire.
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
        # Computed/read-only: an inbound total_gpus must not override the product.
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
    """WorkloadSpec._canonicalize_llm_model = split('/')[-1].lower() at ingestion.

    Oracle is the documented rule applied by hand: drop everything up to and
    including the last '/', then lowercase. A different form than the impl would be
    "take the last path segment and lowercase it".
    """

    @pytest.mark.parametrize(
        "raw, expected",
        [
            # "mistralai/Mistral-7B-v0.1" -> last segment "Mistral-7B-v0.1" -> lower
            ("mistralai/Mistral-7B-v0.1", "mistral-7b-v0.1"),
            # already short + lowercase -> unchanged (idempotent)
            ("mistral-7b-v0.1", "mistral-7b-v0.1"),
            # uppercase short form -> only lowercased, no org prefix to drop
            ("LLAMA-2-7B", "llama-2-7b"),
            # nested org path -> still only the final segment survives
            ("meta-llama/Llama-2-70b-hf", "llama-2-70b-hf"),
        ],
    )
    def test_llm_model_is_canonicalized_at_ingestion(self, raw, expected):
        assert _workload(llm_model=raw).llm_model == expected


def test_prediction_round_trips_through_json():
    """JSON round-trip is identity: deserialize(serialize(p)) == p (invariant).

    The API ships Predictions over the wire, so every field — including nested
    metadata and the float throughput — must survive intact. A bug that dropped or
    coerced any field (e.g. serialized metadata to a string, or lost total_gpus)
    breaks the equality, and total_gpus=16 would additionally fail the 8*2 validator
    on the way back in.
    """
    p = Prediction(
        gpus_per_node=8, number_of_nodes=2, total_gpus=16, predicted_throughput=999.0, metadata={"predictor": "kavier"}
    )
    restored = Prediction(**json.loads(p.model_dump_json()))
    assert restored == p
    # Pin the nested + numeric fields explicitly so a partial-drop bug is legible.
    assert restored.metadata == {"predictor": "kavier"}
    assert restored.predicted_throughput == 999.0


# --------------------------------------------------------------------------- #
# L2 — empty/degenerate model key is rejected at construction time
# --------------------------------------------------------------------------- #


class TestWorkloadEmptyModelKey:
    """WorkloadSpec rejects llm_model values that canonicalize to an empty string."""

    @pytest.mark.parametrize(
        "bad_model",
        [
            "someorg/",  # trailing slash -> canonical '' after split('/')[-1].lower()
            "/",  # slash only
            "//",  # double slash
        ],
    )
    def test_trailing_slash_raises_value_error(self, bad_model):
        with pytest.raises(ValidationError) as exc_info:
            _workload(llm_model=bad_model)
        # The error message must be informative, not a bare pydantic boilerplate.
        assert "empty after canonicalization" in str(exc_info.value)
