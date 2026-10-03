"""
Tests for the DeepLearningPredictor wrapper.

Kept apart from test_ml_predictors.py because importing torch can segfault on systems with a
broken MPS backend; the tests skip when torch is missing. Exact network outputs are not checked.
"""

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not available or unstable on this system")


from coastline.sdk.predictors.performance.data_driven.deep_learning_predictor import DeepLearningPredictor  # noqa: E402


@pytest.fixture
def dl_predictor():
    return DeepLearningPredictor()


def test_model_path_is_weights_file_inside_model_dir():
    """model_path is '<model_dir>/performance_deep_learning.pth' for any model_dir."""
    custom_dir = Path("/tmp/does-not-need-to-exist")
    predictor = DeepLearningPredictor(model_dir=custom_dir)
    assert predictor.model_path == custom_dir / "performance_deep_learning.pth"
    assert predictor.model_path.name == "performance_deep_learning.pth"


def test_default_artifacts_are_bundled_and_loadable(dl_predictor):
    """The default predictor points at the bundled weights file, which exists on disk."""
    assert dl_predictor.model_path.name == "performance_deep_learning.pth"
    assert dl_predictor.model_path.exists(), f"DL weights missing at {dl_predictor.model_path}"


def test_in_library_workload_yields_finite_positive_throughput(dl_predictor, a100_context, known_workload):
    """A model and GPU in Kavier's library give a finite, positive throughput."""
    prediction = dl_predictor.predict(known_workload, a100_context)

    assert prediction is not None, "supported mistral-7b/A100 workload must be predictable"
    thr = prediction.predicted_throughput
    assert thr is not None and math.isfinite(thr) and thr > 0.0
    assert prediction.metadata.get("predictor") == "deep_learning"


def test_prediction_geometry_matches_workload_not_net_output(dl_predictor, a100_context, known_workload):
    """The Prediction's GPU layout comes from the WorkloadSpec: 8 GPUs per node x 2 nodes = 16."""
    prediction = dl_predictor.predict(known_workload, a100_context)

    assert prediction is not None
    assert prediction.gpus_per_node == 8
    assert prediction.number_of_nodes == 2
    assert prediction.total_gpus == 16  # 8 per node * 2 nodes


def test_metadata_dual_output_flag_agrees_with_runtime_field(dl_predictor, a100_context, known_workload):
    """The 'dual_output' metadata flag is True when the prediction has a runtime, else False."""
    prediction = dl_predictor.predict(known_workload, a100_context)

    assert prediction is not None
    has_runtime = prediction.predicted_runtime_seconds is not None
    assert prediction.metadata.get("dual_output") is has_runtime
    # The deep-learning predictor has no cache.
    assert prediction.metadata.get("cache_hit") is False


def test_out_of_library_workload_returns_none(dl_predictor, a100_context, unknown_workload):
    """A model missing from Kavier's spec library has NaN spec features, so predict returns None."""
    assert dl_predictor.predict(unknown_workload, a100_context) is None


def test_get_name_is_stable_registry_key(dl_predictor):
    """get_name() returns 'DeepLearningPredictor', the registry key."""
    assert dl_predictor.get_name() == "DeepLearningPredictor"
