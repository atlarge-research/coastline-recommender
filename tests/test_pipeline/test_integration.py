"""Integration tests for the RetrievalPredictor exact-match cache.

The predictor looks a configuration up by its SHA256 hash and returns the first recorded run.
The expected numbers come from the bundled trace ``src/coastline/sdk/io/data/sample_raw_trace.csv``.
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


# Rows of the sample trace, one per configuration. The predictor matches
# WorkloadSpec.gpus_per_node against the trace's number_gpus column. Every row has
# number_nodes=1, so total_gpus = gpus_per_node.
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
    """RetrievalPredictor on the bundled 5-row sample.

    Without dataset_path it would use a sibling ../trace-archive when one exists, and that
    trace has no demo-llm-3b row.
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
    """Each configuration returns the throughput and runtime recorded in its own trace row."""
    prediction = sample_predictor.predict(_demo_workload(gpus_per_node, batch_size), system_context)

    assert prediction is not None, "exact-match config must cache-hit"
    assert prediction.predicted_throughput == pytest.approx(exp_throughput)
    assert prediction.predicted_runtime_seconds == pytest.approx(exp_runtime)
    # total_gpus = number_nodes (1) * number_gpus
    assert prediction.total_gpus == exp_total_gpus
    # The prediction keeps the requested layout.
    assert prediction.gpus_per_node == gpus_per_node
    assert prediction.number_of_nodes == 1
    assert prediction.metadata.get("cache_hit") is True


def test_single_run_config_reports_zero_spread_metadata(sample_predictor, system_context):
    """A configuration with one recorded run reports run_count=1 and zero spread.

    synthetic-sample-0003 (gpus_per_node=4, batch 8) is the only run of its configuration:
    min = max = 14000, std = 0 and cv = std / median = 0.
    """
    prediction = sample_predictor.predict(_demo_workload(4, 8), system_context)

    md = prediction.metadata
    assert md["run_count"] == 1
    assert md["throughput_min"] == pytest.approx(14000.0)
    assert md["throughput_max"] == pytest.approx(14000.0)
    assert md["throughput_std"] == pytest.approx(0.0)
    assert md["coefficient_of_variation"] == pytest.approx(0.0)


def test_cache_returns_first_run_not_median_on_duplicate_config(tmp_path, system_context):
    """With several runs of one configuration, a hit returns the first run.

    The two runs record throughput 100 then 300 and runtime 50 then 70, so the expected
    values are 100 and 50; the median throughput would be 200.
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
    # the first run; the median is 200 and the maximum 300
    assert prediction.predicted_throughput == pytest.approx(100.0)
    assert prediction.predicted_runtime_seconds == pytest.approx(50.0)
    # the spread covers both runs: min 100, max 300, count 2
    assert prediction.metadata["run_count"] == 2
    assert prediction.metadata["throughput_min"] == pytest.approx(100.0)
    assert prediction.metadata["throughput_max"] == pytest.approx(300.0)


def test_exact_match_discriminates_on_batch_size(sample_predictor, system_context):
    """A configuration that differs from a trace row only in batch_size is a miss.

    It matches synthetic-sample-0003 except for batch_size=7, which the trace does not have.
    """
    assert sample_predictor.predict(_demo_workload(4, 7), system_context) is None


def test_unknown_model_returns_none(sample_predictor, system_context):
    """An unknown configuration returns None, so the caller can fall back to another predictor."""
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
