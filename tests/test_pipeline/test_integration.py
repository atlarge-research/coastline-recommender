"""Integration tests for the RetrievalPredictor exact-match cache.

Oracles here are *recorded values*: the predictor is a SHA256 exact-match lookup
that returns the FIRST recorded run of a matching configuration verbatim. The
expected numbers are read straight out of the bundled trace CSV
(``src/coastline/sdk/io/data/sample_raw_trace.csv``), never copied from predictor
output, so a bug that fabricates, aggregates, or mis-indexes values goes red.
"""

import logging

import pandas as pd
import pytest

from coastline.sdk.io.sample_data import sample_raw_trace_path
from coastline.sdk.logging import setup_logging
from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.performance.retrieval.cache_predictor import RetrievalPredictor

setup_logging()
logger = logging.getLogger(__name__)


# Ground truth read directly out of the bundled sample trace. Each tuple is a
# distinct configuration; the predictor maps WorkloadSpec.gpus_per_node onto the
# trace's ``number_gpus`` column and returns dataset_tokens_per_second /
# train_runtime of the (single) matching row. All sample rows have number_nodes=1,
# so total_gpus = number_nodes * number_gpus = gpus_per_node.
#   (gpus_per_node, batch_size, dataset_tokens_per_second, train_runtime, total_gpus)
SAMPLE_ROWS = [
    (1, 8, 12000.0, 120.0, 1),  # synthetic-sample-0001
    (2, 8, 12500.0, 118.0, 2),  # synthetic-sample-0002
    (4, 8, 14000.0, 115.0, 4),  # synthetic-sample-0003
    (8, 8, 15500.0, 110.0, 8),  # synthetic-sample-0004
    (1, 16, 8500.0, 150.0, 1),  # synthetic-sample-0005
]


@pytest.fixture
def system_context():
    return SystemContext(
        available_gpu_models=["NVIDIA-A100-SXM4-80GB"],
        max_gpus=32,
        gpu_memory={"NVIDIA-A100-SXM4-80GB": 80},
        constraints=Constraints(max_gpus=32, gpus_per_node=8, max_nodes=4),
    )


@pytest.fixture
def sample_predictor():
    """RetrievalPredictor pinned to the bundled 5-row sample.

    A bare RetrievalPredictor() resolves a sibling ../trace-archive when one
    exists (dev layout), whose real trace has no demo-llm-3b row -> spurious miss.
    Pinning dataset_path keeps every assertion hermetic and reproducible.
    """
    return RetrievalPredictor(dataset_path=sample_raw_trace_path())


def _demo_workload(gpus_per_node: int, batch_size: int) -> WorkloadSpec:
    return WorkloadSpec(
        llm_model="demo-llm-3b",
        fine_tuning_method="lora",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=2048,
        batch_size=batch_size,
        gpus_per_node=gpus_per_node,
        number_of_nodes=1,
    )


@pytest.mark.parametrize(
    "gpus_per_node,batch_size,exp_throughput,exp_runtime,exp_total_gpus",
    SAMPLE_ROWS,
)
def test_cache_hit_returns_the_exact_recorded_row(
    sample_predictor, system_context, gpus_per_node, batch_size, exp_throughput, exp_runtime, exp_total_gpus
):
    """Each catalog config resolves to ITS OWN recorded throughput/runtime.

    Oracle: the values are the CSV's dataset_tokens_per_second / train_runtime for
    that row (independent of the predictor). Because the five configs map to five
    different recorded values, a bug that returns a constant, the wrong row, or an
    aggregate (median) instead of the first run cannot pass this parametrization.
    """
    prediction = sample_predictor.predict(_demo_workload(gpus_per_node, batch_size), system_context)

    assert prediction is not None, "exact-match config must cache-hit"
    assert prediction.predicted_throughput == pytest.approx(exp_throughput)
    assert prediction.predicted_runtime_seconds == pytest.approx(exp_runtime)
    # total_gpus = number_nodes(1) * number_gpus, enforced consistent by the model.
    assert prediction.total_gpus == exp_total_gpus
    # The prediction echoes the requested layout verbatim.
    assert prediction.gpus_per_node == gpus_per_node
    assert prediction.number_of_nodes == 1
    assert prediction.metadata.get("cache_hit") is True


def test_single_run_config_reports_zero_spread_metadata(sample_predictor, system_context):
    """For a config with exactly one recorded run, the spread stats collapse.

    Row synthetic-sample-0003 (gpus_per_node=4, batch=8) is the ONLY run of its
    config, so by hand: run_count=1, min=max=first=14000, std=0, cv=std/median=0.
    Falsifies a bug that mis-counts runs or computes spread over the wrong group.
    """
    prediction = sample_predictor.predict(_demo_workload(4, 8), system_context)

    md = prediction.metadata
    assert md["run_count"] == 1
    assert md["throughput_min"] == pytest.approx(14000.0)
    assert md["throughput_max"] == pytest.approx(14000.0)
    assert md["throughput_std"] == pytest.approx(0.0)
    assert md["coefficient_of_variation"] == pytest.approx(0.0)


def test_cache_returns_first_run_not_median_on_duplicate_config(tmp_path, system_context):
    """When a config has multiple runs, the hit returns the FIRST, not the median.

    Two rows of one identical config with throughput 100 then 300 (median 200) and
    runtime 50 then 70. The documented contract returns the first recorded run, so
    the oracle is throughput=100 / runtime=50 -- deliberately != the median 200,
    which a mean/median-returning regression would emit instead.
    """
    trace = tmp_path / "dup.csv"
    pd.DataFrame(
        {
            "model_name": ["tinyllm", "tinyllm"],
            "method": ["lora", "lora"],
            "gpu_model": ["NVIDIA-A100-SXM4-80GB", "NVIDIA-A100-SXM4-80GB"],
            "number_nodes": [1.0, 1.0],
            "number_gpus": [2.0, 2.0],
            "tokens_per_sample": [2048.0, 2048.0],
            "batch_size": [8.0, 8.0],
            "dataset_tokens_per_second": [100.0, 300.0],
            "train_runtime": [50.0, 70.0],
            "is_valid": [1.0, 1.0],
        }
    ).to_csv(trace, index=False)

    workload = WorkloadSpec(
        llm_model="tinyllm",
        fine_tuning_method="lora",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=2048,
        batch_size=8,
        gpus_per_node=2,
        number_of_nodes=1,
    )
    prediction = RetrievalPredictor(dataset_path=trace).predict(workload, system_context)

    assert prediction is not None
    # first run, not the median (200) or max (300).
    assert prediction.predicted_throughput == pytest.approx(100.0)
    assert prediction.predicted_runtime_seconds == pytest.approx(50.0)
    # aggregation still spans both runs: min=100, max=300, count=2.
    assert prediction.metadata["run_count"] == 2
    assert prediction.metadata["throughput_min"] == pytest.approx(100.0)
    assert prediction.metadata["throughput_max"] == pytest.approx(300.0)


def test_exact_match_discriminates_on_batch_size(sample_predictor, system_context):
    """A near-miss that differs in only one hashed field must MISS, not hit.

    Config matches synthetic-sample-0003 exactly except batch_size=7 (absent from
    the trace). A hit here would mean the hash ignores batch_size; the contract is
    an exact match, so the oracle is None (no fabricated nearest-neighbour result).
    """
    assert sample_predictor.predict(_demo_workload(4, 7), system_context) is None


def test_unknown_model_returns_none(sample_predictor, system_context):
    """An entirely unknown config returns None so the orchestrator falls back.

    Oracle: contract of a miss is None (never a fabricated prediction).
    """
    workload = WorkloadSpec(
        llm_model="nonexistent-model",
        fine_tuning_method="full",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=9999,
        batch_size=999,
        gpus_per_node=1,
        number_of_nodes=1,
    )
    assert sample_predictor.predict(workload, system_context) is None
