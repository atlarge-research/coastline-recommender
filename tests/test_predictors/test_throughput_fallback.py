"""The 'intelligent' throughput predictor is a cache->physics cascade: it returns
an exact cache match (a measured past run) when one exists, else the Kavier
analytical predictor. These tests pin the cascade contract with independent
oracles (a value we planted in a controlled cache; the physics-vs-cache
provenance recorded in metadata) and the factory's name->predictor-class map.
"""

import math

import pandas as pd
import pytest

from coastline.sdk.library.hardware import get_gpu_memory
from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.policies import PolicyFactory
from coastline.sdk.predictors.factory import create_physics_driven
from coastline.sdk.predictors.performance.composite import CacheThenPhysicsPredictor
from coastline.sdk.predictors.performance.retrieval.cache_predictor import RetrievalPredictor

_GPU = "NVIDIA-A100-SXM4-80GB"  # a Kavier-supported GPU
_MODEL = "mistral-7b-v0.1"  # a Kavier-supported model, already in canonical form


def _workload(batch_size: int = 8):
    return WorkloadSpec(
        llm_model=_MODEL,
        fine_tuning_method="lora",
        gpu_model=_GPU,
        tokens_per_sample=1024,
        batch_size=batch_size,
        gpus_per_node=4,
        number_of_nodes=1,
    )


def _context():
    return SystemContext(
        available_gpu_models=[_GPU],
        max_gpus=32,
        gpu_memory={_GPU: get_gpu_memory(_GPU)},
        constraints=Constraints(max_gpus=32, gpus_per_node=8, max_nodes=4),
    )


def _cache_row(batch_size: int, throughput: float, runtime: float) -> dict:
    """One RetrievalPredictor-indexable run. number_gpus is the PER-NODE count
    (it is hashed as gpus_per_node), so 1 node x 4 gpus matches _workload()."""
    return {
        "model_name": _MODEL,
        "method": "lora",
        "gpu_model": _GPU,
        "number_nodes": 1,
        "number_gpus": 4,
        "tokens_per_sample": 1024,
        "batch_size": batch_size,
        "dataset_tokens_per_second": throughput,
        "train_runtime": runtime,
    }


def _cache_over(tmp_path, rows: list[dict]) -> RetrievalPredictor:
    csv = tmp_path / "raw_trace.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return RetrievalPredictor(dataset_path=csv)


def test_intelligent_returns_the_recorded_cache_value_on_an_exact_hit(tmp_path):
    # Oracle: the cache is the deployment's memory of a measured run. We plant a
    # run for exactly _workload()'s config with a distinctive throughput 4242.0
    # tokens/s that the physics engine would never coincidentally emit. On an
    # exact hit the cascade must surface THAT recorded number verbatim, proving
    # the cache short-circuits before physics is consulted.
    cache = _cache_over(tmp_path, [_cache_row(batch_size=8, throughput=4242.0, runtime=600.0)])
    physics = create_physics_driven()
    intelligent = CacheThenPhysicsPredictor(cache=cache, physics=physics)

    out = intelligent.predict(_workload(batch_size=8), _context())

    assert out.predicted_throughput == pytest.approx(4242.0)
    # Cross-check that this is genuinely cache-over-physics, not a coincidence:
    # physics alone on the same supported config yields a different number.
    physics_out = physics.predict(_workload(batch_size=8), _context())
    assert physics_out.predicted_throughput != pytest.approx(4242.0)


def test_intelligent_falls_through_to_physics_on_a_cache_miss(tmp_path):
    # Oracle: the cache holds a run for batch_size=999 only, so _workload()'s
    # batch_size=8 config MISSES. A miss must not return None nor the wrong
    # cached row; it must yield the Kavier physics estimate. We assert the
    # provenance (metadata predictor == "kavier") plus a finite positive
    # throughput -- the contract of the fallback branch.
    cache = _cache_over(tmp_path, [_cache_row(batch_size=999, throughput=4242.0, runtime=600.0)])
    intelligent = CacheThenPhysicsPredictor(cache=cache, physics=create_physics_driven())

    out = intelligent.predict(_workload(batch_size=8), _context())

    assert out is not None
    assert out.metadata.get("predictor") == "kavier"  # came from physics, not the cache
    assert out.predicted_throughput > 0 and math.isfinite(out.predicted_throughput)
    # And it is emphatically NOT the mismatched cache row.
    assert out.predicted_throughput != pytest.approx(4242.0)


@pytest.mark.parametrize(
    "name, expected_cls",
    [
        # physics aliases all resolve to the one Kavier predictor
        ("kavier", "KavierPredictor"),
        ("physics", "KavierPredictor"),
        ("physics_driven", "KavierPredictor"),
        ("cache", "RetrievalPredictor"),
        # "intelligent" is the cache->physics cascade
        ("intelligent", "CacheThenPhysicsPredictor"),
        # named ML models must reach the ML branch, not collapse to the composite or
        # to CatBoost. The six portfolio models share SklearnPortfolioPredictor (they
        # stay distinguishable by get_name; see test_config_predictor_selection); the
        # distinct-runtime models keep their own class.
        ("xgboost", "SklearnPortfolioPredictor"),
        ("tabpfn", "TabPFNPredictor"),
        ("deep_learning", "DeepLearningPredictor"),
        # an unknown name falls back to the intelligent default (policies L117-118)
        ("totally-not-a-real-predictor", "CacheThenPhysicsPredictor"),
    ],
)
def test_factory_resolves_each_name_to_its_own_predictor_class(name, expected_cls):
    pred = PolicyFactory.throughput_predictor({"performance": name})
    assert type(pred).__name__ == expected_cls, f"{name} -> {type(pred).__name__}"
