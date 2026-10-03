"""Tests for the 'intelligent' throughput predictor and the lookup cache behind it.

'intelligent' returns an exact cache match (a measured past run) when there is one, else the
Kavier estimate. The file also checks the factory's map from name to predictor class.
"""

import math

import pandas as pd
import pytest

from coastline.sdk.library.hardware import get_gpu_memory
from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.policies import PolicyFactory
from coastline.sdk.predictors.factory import create_physics_driven
from coastline.sdk.predictors.performance.composite import CacheThenSimulatePredictor
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
    """One run the RetrievalPredictor can index. ``number_gpus`` is the per-node count (hashed as
    gpus_per_node), so 1 node x 4 GPUs matches ``_workload()``."""
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
    # The cache holds a run for _workload()'s config with throughput 4242.0 tokens/s, a value
    # Kavier does not produce for it. An exact hit returns that recorded number.
    cache = _cache_over(tmp_path, [_cache_row(batch_size=8, throughput=4242.0, runtime=600.0)])
    physics = create_physics_driven()
    intelligent = CacheThenSimulatePredictor(cache=cache, fallback=physics)

    out = intelligent.predict(_workload(batch_size=8), _context())

    assert out.predicted_throughput == pytest.approx(4242.0)
    # Kavier alone gives a different number for the same config.
    physics_out = physics.predict(_workload(batch_size=8), _context())
    assert physics_out.predicted_throughput != pytest.approx(4242.0)


def test_intelligent_falls_through_to_physics_on_a_cache_miss(tmp_path):
    # The cache holds a run for batch_size=999 only, so batch_size=8 misses and the result is the
    # Kavier estimate: metadata predictor "kavier" and a finite positive throughput.
    cache = _cache_over(tmp_path, [_cache_row(batch_size=999, throughput=4242.0, runtime=600.0)])
    intelligent = CacheThenSimulatePredictor(cache=cache, fallback=create_physics_driven())

    out = intelligent.predict(_workload(batch_size=8), _context())

    assert out is not None
    assert out.metadata.get("predictor") == "kavier"  # from Kavier
    assert out.predicted_throughput > 0 and math.isfinite(out.predicted_throughput)
    # The cached row for the other config is not used.
    assert out.predicted_throughput != pytest.approx(4242.0)


def _row_with_custom_cols() -> dict:
    """An indexable run with throughput and duration under custom headers (tps, dur)."""
    return {
        "model_name": _MODEL,
        "method": "lora",
        "gpu_model": _GPU,
        "number_nodes": 1,
        "number_gpus": 4,
        "tokens_per_sample": 1024,
        "batch_size": 8,
        "tps": 3131.0,  # throughput, in place of dataset_tokens_per_second
        "dur": 720.0,  # duration, in place of train_runtime
    }


def test_lookup_reads_configurable_throughput_and_runtime_columns(tmp_path):
    # A lookup CSV may use any headers for throughput and duration; throughput_col and runtime_col
    # name them. The default columns are absent here, so reading them would raise.
    csv = tmp_path / "custom_cols.csv"
    pd.DataFrame([_row_with_custom_cols()]).to_csv(csv, index=False)
    cache = RetrievalPredictor(dataset_path=csv, throughput_col="tps", runtime_col="dur")

    out = cache.predict(_workload(batch_size=8), _context())

    assert out is not None, "an exact config match must hit even under custom column names"
    assert out.predicted_throughput == pytest.approx(3131.0)
    assert out.predicted_runtime_seconds == pytest.approx(720.0)


def test_lookup_column_keys_thread_through_the_policy_factory(tmp_path):
    # The predictors.lookup_throughput_col and lookup_runtime_col config keys reach the
    # RetrievalPredictor, and a hit returns the value stored under them.
    csv = tmp_path / "custom_cols.csv"
    pd.DataFrame([_row_with_custom_cols()]).to_csv(csv, index=False)
    cache = PolicyFactory.throughput_predictor(
        {
            "performance": "cache",
            "lookup": str(csv),
            "lookup_throughput_col": "tps",
            "lookup_runtime_col": "dur",
        }
    )
    assert cache._throughput_col == "tps" and cache._runtime_col == "dur"
    out = cache.predict(_workload(batch_size=8), _context())
    assert out is not None and out.predicted_throughput == pytest.approx(3131.0)


def test_lookup_missing_named_column_raises_clear_error(tmp_path):
    # A lookup_throughput_col or lookup_runtime_col missing from the CSV raises a ValueError that
    # names the column. This CSV has the default columns but not "tps" or "dur".
    csv = tmp_path / "wrong_cols.csv"
    pd.DataFrame([_cache_row(batch_size=8, throughput=100.0, runtime=60.0)]).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="tps"):
        RetrievalPredictor(dataset_path=csv, throughput_col="tps", runtime_col="dur")


@pytest.mark.parametrize(
    "name, expected_cls",
    [
        # physics aliases all resolve to the one Kavier predictor
        ("kavier", "KavierPredictor"),
        ("physics", "KavierPredictor"),
        ("physics_driven", "KavierPredictor"),
        ("cache", "RetrievalPredictor"),
        # "intelligent" is the cache-then-Kavier cascade
        ("intelligent", "CacheThenSimulatePredictor"),
        # Named ML models get their own predictor. The six portfolio models share
        # SklearnPortfolioPredictor and differ by get_name; tabpfn and deep_learning keep their
        # own class.
        ("xgboost", "SklearnPortfolioPredictor"),
        ("tabpfn", "TabPFNPredictor"),
        ("deep_learning", "DeepLearningPredictor"),
    ],
)
def test_factory_resolves_each_name_to_its_own_predictor_class(name, expected_cls):
    pred = PolicyFactory.throughput_predictor({"performance": name})
    assert type(pred).__name__ == expected_cls, f"{name} -> {type(pred).__name__}"


def test_factory_rejects_an_unknown_name():
    # An unknown name raises.
    with pytest.raises(ValueError, match="unknown predictor"):
        PolicyFactory.throughput_predictor({"performance": "totally-not-a-real-predictor"})
