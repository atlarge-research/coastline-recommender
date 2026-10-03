"""Unit tests for RetrievalPredictor, the exact-match cache, on a small synthetic CSV.

Behaviour under test (``cache_predictor.py``):
  * An exact hit returns the first recorded run for a config, so re-predicting a stored config
    has about 0% error.
  * A miss returns ``None``, and the caller falls back to simulation.
  * Configs are keyed by a SHA256 over the 7 config fields, with ints and floats normalised.
  * Multi-run statistics (median, std, min, max, count, CV) are kept in the metadata and index.
"""

import pandas as pd
import pytest

from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.performance.retrieval.cache_predictor import RetrievalPredictor

# Synthetic dataset
# Config A (mistral-7b-v0.1/lora, 2 nodes x 8 GPUs, 512 tokens, batch 16): three valid runs,
#   throughputs 100/200/300 (median 200, first 100) and runtimes 1000/2000/3000 (median 2000,
#   first 1000). Its first row comes first in the CSV, and groupby keeps row order in a group.
#   Two more config A rows must be dropped by the loader: is_valid == 0 with throughput 50000,
#   and zero throughput.
# Config B (granite-3.3-8b/full, 1 node x 1 GPU, 1024 tokens, batch 8): one run, throughput 555,
#   runtime 4242, so count 1 and std 0.
# Config D (phi-4/lora, 1 node x 2 GPUs, 256 tokens, batch 32): runs recorded as
#   [200, 100, 300, 400], so min 100, median 250, max 400. The first run, 200, equals none of
#   them; config A cannot show this because its first run is also its min.

_COLUMNS = [
    "model_name",
    "method",
    "gpu_model",
    "number_nodes",
    "number_gpus",
    "tokens_per_sample",
    "batch_size",
    "dataset_tokens_per_second",
    "train_runtime",
    "is_valid",
]

_GPU = "NVIDIA-A100-SXM4-80GB"

# Rows are not sorted; the order within a config matters.
_ROWS = [
    # --- config A, the first valid run (throughput 100) ---
    ["mistral-7b-v0.1", "lora", _GPU, 2, 8, 512, 16, 100.0, 1000.0, 1.0],
    # an invalid row for config A; the loader drops it
    ["mistral-7b-v0.1", "lora", _GPU, 2, 8, 512, 16, 50000.0, 7.0, 0.0],
    # config D, the first valid run (throughput 200, a middle value)
    ["phi-4", "lora", _GPU, 1, 2, 256, 32, 200.0, 1500.0, 1.0],
    # config B single run (interleaved to test stable grouping)
    ["granite-3.3-8b", "full", _GPU, 1, 1, 1024, 8, 555.0, 4242.0, 1.0],
    # --- config A, remaining valid runs ---
    ["mistral-7b-v0.1", "lora", _GPU, 2, 8, 512, 16, 200.0, 2000.0, 1.0],
    # a zero-throughput row for config A, dropped by the > 0 filter
    ["mistral-7b-v0.1", "lora", _GPU, 2, 8, 512, 16, 0.0, 9000.0, 1.0],
    ["mistral-7b-v0.1", "lora", _GPU, 2, 8, 512, 16, 300.0, 3000.0, 1.0],
    # config D, remaining valid runs: full order [200 (first), 100, 300, 400]
    ["phi-4", "lora", _GPU, 1, 2, 256, 32, 100.0, 3000.0, 1.0],
    ["phi-4", "lora", _GPU, 1, 2, 256, 32, 300.0, 1000.0, 1.0],
    ["phi-4", "lora", _GPU, 1, 2, 256, 32, 400.0, 750.0, 1.0],
]


@pytest.fixture
def synthetic_csv(tmp_path):
    """Write the synthetic curated-runs CSV and return its path."""
    df = pd.DataFrame(_ROWS, columns=_COLUMNS)
    path = tmp_path / "synthetic_valid_runs.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def predictor(synthetic_csv):
    """RetrievalPredictor backed by the synthetic CSV (no real data)."""
    return RetrievalPredictor(dataset_path=synthetic_csv)


@pytest.fixture
def context():
    """Minimal system context (unused by retrieval, but required by predict())."""
    return SystemContext(
        available_gpu_models=[_GPU],
        max_gpus=32,
        gpu_memory={_GPU: 80},
        constraints=Constraints(max_gpus=32, gpus_per_node=8, max_nodes=4),
    )


def _config_a_workload():
    return WorkloadSpec(
        llm_model="mistral-7b-v0.1",
        fine_tuning_method="lora",
        gpu_model=_GPU,
        tokens_per_sample=512,
        batch_size=16,
        gpus_per_node=8,
        number_of_nodes=2,
    )


def _config_b_workload():
    return WorkloadSpec(
        llm_model="granite-3.3-8b",
        fine_tuning_method="full",
        gpu_model=_GPU,
        tokens_per_sample=1024,
        batch_size=8,
        gpus_per_node=1,
        number_of_nodes=1,
    )


def _config_d_workload():
    return WorkloadSpec(
        llm_model="phi-4",
        fine_tuning_method="lora",
        gpu_model=_GPU,
        tokens_per_sample=256,
        batch_size=32,
        gpus_per_node=2,
        number_of_nodes=1,
    )


# Loading / filtering
def test_loads_only_valid_positive_rows(predictor):
    """Rows with is_valid == 0 or zero throughput are dropped; the rest are kept."""
    # 10 rows - 1 (is_valid == 0) - 1 (zero throughput) = 8: config A keeps 3, B 1, D 4.
    assert len(predictor.dataset) == 8
    assert (predictor.dataset["dataset_tokens_per_second"] > 0).all()
    assert (predictor.dataset["is_valid"] == 1.0).all()


# Exact-match hit: the first recorded run
def test_hit_returns_first_run_not_median(predictor, context):
    """A hit on config A ([100, 200, 300] after filtering) returns the first run: throughput 100
    and runtime 1000."""
    pred = predictor.predict(_config_a_workload(), context)
    assert pred is not None, "expected a cache HIT for config A"

    # The first run, below the median of 200.
    assert pred.predicted_throughput == 100.0
    assert pred.predicted_runtime_seconds == 1000.0
    assert pred.predicted_throughput != 200.0  # would be the median

    # Hit flag and total GPUs (2 nodes x 8 = 16).
    assert pred.metadata["cache_hit"] is True
    assert pred.metadata["predictor"] == "retrieval"
    assert pred.total_gpus == 16


def test_hit_returns_recorded_first_not_min_median_or_max(predictor, context):
    """Config D, recorded as [200, 100, 300, 400], returns its first run, 200, which is neither the
    min (100), the median (250) nor the max (400)."""
    pred = predictor.predict(_config_d_workload(), context)
    assert pred is not None, "expected a cache HIT for config D"

    assert pred.predicted_throughput == 200.0  # first recorded run
    assert pred.predicted_throughput != 100.0  # not min
    assert pred.predicted_throughput != 250.0  # not median
    assert pred.predicted_throughput != 400.0  # not max
    assert pred.predicted_runtime_seconds == 1500.0  # runtime of the first row

    # total GPUs = 1 node x 2 GPU = 2 (distinct from config A's 16 and B's 1).
    assert pred.total_gpus == 2
    assert pred.metadata["run_count"] == 4


def test_single_run_hit_returns_that_run(predictor, context):
    """Single-run config B returns its only measurement; count==1, std==0."""
    pred = predictor.predict(_config_b_workload(), context)
    assert pred is not None
    assert pred.predicted_throughput == 555.0
    assert pred.predicted_runtime_seconds == 4242.0
    assert pred.metadata["run_count"] == 1
    assert pred.metadata["throughput_std"] == 0.0
    # For a single run, first == min == max.
    assert pred.metadata["throughput_min"] == 555.0
    assert pred.metadata["throughput_max"] == 555.0
    assert pred.total_gpus == 1


# Cache miss: None
def test_miss_unknown_model_returns_none(predictor, context):
    """A config missing from the dataset returns None, which signals the fallback."""
    wl = WorkloadSpec(
        llm_model="totally-unknown-model",
        fine_tuning_method="full",
        gpu_model=_GPU,
        tokens_per_sample=512,
        batch_size=16,
        gpus_per_node=8,
        number_of_nodes=2,
    )
    assert predictor.predict(wl, context) is None


def test_miss_when_one_field_differs_returns_none(predictor, context):
    """Changing one config field (batch_size) gives a miss (None)."""
    wl = _config_a_workload().model_copy(update={"batch_size": 17})
    assert predictor.predict(wl, context) is None


# Multi-run aggregation stats
def test_aggregation_stats_config_a(predictor, context):
    """Config A [100,200,300]: median 200, min 100, max 300, count 3, std ~81.65."""
    pred = predictor.predict(_config_a_workload(), context)
    md = pred.metadata
    assert md["run_count"] == 3
    assert md["throughput_min"] == 100.0
    assert md["throughput_max"] == 300.0
    # Population std of [100, 200, 300]: sqrt((100^2 + 0 + 100^2) / 3) = sqrt(6666.667) = 81.649658.
    assert md["throughput_std"] == pytest.approx(81.649658, rel=1e-6)
    # CV = std / median = 81.649658 / 200 = 0.40824829.
    assert md["coefficient_of_variation"] == pytest.approx(0.40824829, rel=1e-6)

    # Runtime aggregates [1000,2000,3000] (median 2000) live in the index.
    h = next(iter(k for k, v in predictor.config_index.items() if v["llm_model"] == "mistral-7b-v0.1"))
    idx = predictor.config_index[h]
    assert idx["runtime_median"] == 2000.0
    assert idx["runtime_min"] == 1000.0
    assert idx["runtime_max"] == 3000.0
    # The index keeps the median too; predict() returns the first run.
    assert idx["throughput_median"] == 200.0
    assert idx["throughput_first"] == 100.0


# Cache canonicalization: an uppercase model_name in the dataset still hits


def test_uppercase_model_name_in_dataset_still_hits(tmp_path, context):
    """A CSV whose model_name is 'Mistral-7B-v0.1' still hits for the canonical 'mistral-7b-v0.1':
    ``_build_index`` canonicalizes the name before hashing it."""
    columns = [
        "model_name",
        "method",
        "gpu_model",
        "number_nodes",
        "number_gpus",
        "tokens_per_sample",
        "batch_size",
        "dataset_tokens_per_second",
        "train_runtime",
        "is_valid",
    ]
    rows = [
        # Mixed-case model_name; its canonical form is 'mistral-7b-v0.1'.
        ["Mistral-7B-v0.1", "lora", _GPU, 2, 8, 512, 16, 777.0, 5000.0, 1.0],
    ]
    df = pd.DataFrame(rows, columns=columns)
    path = tmp_path / "uppercase_model.csv"
    df.to_csv(path, index=False)

    predictor = RetrievalPredictor(dataset_path=path)

    # The WorkloadSpec canonicalizes to 'mistral-7b-v0.1' via its field_validator.
    wl = WorkloadSpec(
        llm_model="mistral-7b-v0.1",
        fine_tuning_method="lora",
        gpu_model=_GPU,
        tokens_per_sample=512,
        batch_size=16,
        gpus_per_node=8,
        number_of_nodes=2,
    )
    pred = predictor.predict(wl, context)
    assert pred is not None, "cache MISS for uppercase CSV model_name - canonicalization in _build_index is broken"
    assert pred.predicted_throughput == 777.0
