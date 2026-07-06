"""Focused unit tests for RetrievalPredictor (cache/exact-match predictor).

These tests build a *small synthetic* curated-runs CSV via ``tmp_path`` and feed
it to ``RetrievalPredictor(dataset_path=...)`` so they are fully self-contained
and do NOT depend on the real trace-archive data.

Behaviours under test (see cache_predictor.py):
  * Exact-match hit returns the *recorded first* run for a config (not the
    median), so a true hit against the deduplicated target is ~0% error.
  * Cache miss returns ``None`` (orchestrator falls back to simulation).
  * Config-hash keying: int/float-normalised SHA256 over the 7 config fields.
  * Multi-run aggregation stats (median/std/min/max/count/CV) and metadata.

A companion broad test lives in ``test_retrieval_predictor.py`` (uses the real
dataset); this file deliberately does not touch it.
"""

import pandas as pd
import pytest

from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.performance.retrieval.cache_predictor import RetrievalPredictor

# --------------------------------------------------------------------------- #
# Synthetic dataset
# --------------------------------------------------------------------------- #
# Config A ("mistral-7b-v0.1"/lora, 2 nodes x 8 GPU, 512 tok, bs 16):
#   THREE valid runs with throughputs 100/200/300 -> median 200, FIRST 100.
#   Runtimes 1000/2000/3000 -> median 2000, FIRST 1000.
#   The first row appears *before* the others in the CSV so "recorded first" is
#   unambiguous; groupby is stable and preserves within-group row order.
#
# We also inject rows that must be DROPPED by the loader's validity filters so
# they cannot leak into the "first" value:
#   - an is_valid==0 row for config A with a giant throughput (50000),
#   - a zero-throughput row for config A.
# If filtering were broken, the recorded-first / aggregates would shift.
#
# Config B ("granite-3.3-8b"/full, 1 node x 1 GPU, 1024 tok, bs 8):
#   a SINGLE run (throughput 555, runtime 4242) -> count==1, std==0.
#
# Config D ("phi-4"/lora, 1 node x 2 GPU, 256 tok, bs 32):
#   FOUR runs recorded in order [200, 100, 300, 400]. Sorted -> min 100,
#   median (100+... ) = (200+300)/2 = 250, max 400. The FIRST recorded run (200)
#   is a MIDDLE value: it equals NONE of {min, median, max}. This is the strong
#   oracle for "returns the recorded FIRST run" — a `return min(...)`, `return
#   max(...)` or `return median(...)` bug all differ from 200 and go red.
#   (Config A cannot catch a return-min bug: its first run 100 == its min.)

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

# rows are intentionally NOT pre-sorted; ordering within a config matters.
_ROWS = [
    # --- config A, the recorded-first valid run (throughput 100) ---
    ["mistral-7b-v0.1", "lora", _GPU, 2, 8, 512, 16, 100.0, 1000.0, 1.0],
    # an invalid row for config A that would corrupt aggregates if not filtered
    ["mistral-7b-v0.1", "lora", _GPU, 2, 8, 512, 16, 50000.0, 7.0, 0.0],
    # config D, recorded-first valid run (throughput 200 == MIDDLE value)
    ["phi-4", "lora", _GPU, 1, 2, 256, 32, 200.0, 1500.0, 1.0],
    # config B single run (interleaved to test stable grouping)
    ["granite-3.3-8b", "full", _GPU, 1, 1, 1024, 8, 555.0, 4242.0, 1.0],
    # --- config A, remaining valid runs ---
    ["mistral-7b-v0.1", "lora", _GPU, 2, 8, 512, 16, 200.0, 2000.0, 1.0],
    # a zero-throughput row for config A (must be dropped by the >0 filter)
    ["mistral-7b-v0.1", "lora", _GPU, 2, 8, 512, 16, 0.0, 9000.0, 1.0],
    ["mistral-7b-v0.1", "lora", _GPU, 2, 8, 512, 16, 300.0, 3000.0, 1.0],
    # config D remaining valid runs -> full order [200(first),100,300,400]
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


# --------------------------------------------------------------------------- #
# Loading / filtering
# --------------------------------------------------------------------------- #
def test_loads_only_valid_positive_rows(predictor):
    """is_valid==0 and zero-throughput rows are filtered; the rest survive."""
    # 10 input rows - 1 (is_valid==0, the 50000 row) - 1 (zero throughput) = 8.
    # By hand: config A keeps 3 (100/200/300), B keeps 1, D keeps 4 -> 8.
    assert len(predictor.dataset) == 8
    assert (predictor.dataset["dataset_tokens_per_second"] > 0).all()
    assert (predictor.dataset["is_valid"] == 1.0).all()


# --------------------------------------------------------------------------- #
# Exact-match HIT -> recorded FIRST run, not the median
# --------------------------------------------------------------------------- #
def test_hit_returns_first_run_not_median(predictor, context):
    """A true hit on a multi-run config returns the recorded *first* run.

    Config A runs (after filtering): [100, 200, 300] -> median 200, first 100.
    The predictor must return 100 (and runtime 1000), so re-predicting a
    deduplicated stored config yields ~0% error rather than median-vs-run noise.
    """
    pred = predictor.predict(_config_a_workload(), context)
    assert pred is not None, "expected a cache HIT for config A"

    # The load-bearing assertion: FIRST, not MEDIAN.
    assert pred.predicted_throughput == 100.0
    assert pred.predicted_runtime_seconds == 1000.0
    assert pred.predicted_throughput != 200.0  # would be the median

    # Hit flag + total GPUs (2 nodes x 8 = 16).
    assert pred.metadata["cache_hit"] is True
    assert pred.metadata["predictor"] == "retrieval"
    assert pred.total_gpus == 16


def test_hit_returns_recorded_first_not_min_median_or_max(predictor, context):
    """Config D recorded [200, 100, 300, 400]: the return is the FIRST row (200).

    200 is a *middle* value, so this simultaneously rejects the three plausible
    aggregation bugs: returning min (100), median (250), or max (400). Config A
    cannot do this because its first run equals its min. Hand-derived from the
    input order: throughputs [200,100,300,400] -> sorted [100,200,300,400] ->
    min 100, median (200+300)/2 = 250, max 400; the recorded first is 200.
    """
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


# --------------------------------------------------------------------------- #
# Cache MISS -> None
# --------------------------------------------------------------------------- #
def test_miss_unknown_model_returns_none(predictor, context):
    """A config not present in the dataset returns None (signals fallback)."""
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
    """Changing a single config field (batch_size) misses -> None.

    Proves keying uses *all* seven fields, not a subset.
    """
    wl = _config_a_workload().model_copy(update={"batch_size": 17})
    assert predictor.predict(wl, context) is None


# --------------------------------------------------------------------------- #
# Multi-run aggregation stats
# --------------------------------------------------------------------------- #
def test_aggregation_stats_config_a(predictor, context):
    """Config A [100,200,300]: median 200, min 100, max 300, count 3, std ~81.65."""
    pred = predictor.predict(_config_a_workload(), context)
    md = pred.metadata
    assert md["run_count"] == 3
    assert md["throughput_min"] == 100.0
    assert md["throughput_max"] == 300.0
    # Population std of [100,200,300]: deviations -100/0/+100 -> sqrt((10000+0+10000)/3)
    #   = sqrt(20000/3) = sqrt(6666.667) = 81.649658  (hand-computed literal, NOT np.std).
    assert md["throughput_std"] == pytest.approx(81.649658, rel=1e-6)
    # CV = std / median = 81.649658 / 200 = 0.40824829.
    assert md["coefficient_of_variation"] == pytest.approx(0.40824829, rel=1e-6)

    # Runtime aggregates [1000,2000,3000] (median 2000) live in the index.
    h = next(iter(k for k, v in predictor.config_index.items() if v["llm_model"] == "mistral-7b-v0.1"))
    idx = predictor.config_index[h]
    assert idx["runtime_median"] == 2000.0
    assert idx["runtime_min"] == 1000.0
    assert idx["runtime_max"] == 3000.0
    # median is stored separately and is NOT what predict() returns.
    assert idx["throughput_median"] == 200.0
    assert idx["throughput_first"] == 100.0


# --------------------------------------------------------------------------- #
# L1 — cache canonicalization: uppercase model_name in dataset still hits
# --------------------------------------------------------------------------- #


def test_uppercase_model_name_in_dataset_still_hits(tmp_path, context):
    """A custom DATA_DIR whose CSV uses mixed-case model_name must still yield a
    cache HIT when the lookup WorkloadSpec carries the canonical (lowercased) key.

    Before the fix _build_index hashed the raw CSV value ('Mistral-7B-v0.1'),
    while predict() hashed the already-canonicalized workload.llm_model
    ('mistral-7b-v0.1'), producing a hash mismatch and a silent fallback.
    """
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
        # Upper-cased model_name in the CSV — matches canonical 'mistral-7b-v0.1'.
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
    assert pred is not None, "cache MISS for uppercase CSV model_name — canonicalization in _build_index is broken"
    assert pred.predicted_throughput == 777.0
