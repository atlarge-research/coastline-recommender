"""
Tests for the DeepLearningPredictor wrapper.

Separated from test_ml_predictors.py because importing torch can segfault
on systems with broken MPS backends. These tests are gated with importorskip.

Oracles used here (the DL net itself is a black box, so no magic-number pins):
  * path-construction contract (model_path = model_dir / "performance_deep_learning.pth")
  * ML in-library contract: finite, positive throughput
  * ML out-of-library contract: unknown model/GPU specs => predict returns None
  * Prediction geometry derived from the WorkloadSpec, independent of the net output
    (total_gpus = gpus_per_node * number_of_nodes)
  * metadata invariant: the dual_output flag must agree with the runtime field
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
    """model_path must be '<model_dir>/performance_deep_learning.pth' regardless of dir.

    Oracle: hand-constructed path join with a known model_dir. Independent of where
    the default artifacts happen to live, so it falsifies a renamed weights file or a
    broken path join (e.g. joining the .pkl artifacts name instead).
    """
    custom_dir = Path("/tmp/does-not-need-to-exist")
    predictor = DeepLearningPredictor(model_dir=custom_dir)
    assert predictor.model_path == custom_dir / "performance_deep_learning.pth"
    assert predictor.model_path.name == "performance_deep_learning.pth"


def test_default_artifacts_are_bundled_and_loadable(dl_predictor):
    """The default predictor points at real, on-disk weights (wheel bundles them).

    Guard test: the basename is the oracle (contract); existence confirms the bundled
    artifact ships so the predict tests below can actually load a model.
    """
    assert dl_predictor.model_path.name == "performance_deep_learning.pth"
    assert dl_predictor.model_path.exists(), f"DL weights missing at {dl_predictor.model_path}"


def test_in_library_workload_yields_finite_positive_throughput(dl_predictor, a100_context, known_workload):
    """In-library (model+GPU in Kavier library) => a real, usable throughput prediction.

    Oracle = the ML contract: throughput must be finite and strictly positive (a net
    that emitted NaN/inf or a non-positive rate would be unusable downstream). No
    magic value is pinned because the net is a black box.
    """
    prediction = dl_predictor.predict(known_workload, a100_context)

    assert prediction is not None, "supported mistral-7b/A100 workload must be predictable"
    thr = prediction.predicted_throughput
    assert thr is not None and math.isfinite(thr) and thr > 0.0
    assert prediction.metadata.get("predictor") == "deep_learning"


def test_prediction_geometry_matches_workload_not_net_output(dl_predictor, a100_context, known_workload):
    """Prediction node/GPU counts come from the WorkloadSpec, not from the net.

    Oracle hand-derived from known_workload: gpus_per_node=8, number_of_nodes=2
    => total_gpus = 8 * 2 = 16. Independent of whatever throughput the net emits.
    """
    prediction = dl_predictor.predict(known_workload, a100_context)

    assert prediction is not None
    assert prediction.gpus_per_node == 8
    assert prediction.number_of_nodes == 2
    assert prediction.total_gpus == 16  # 8 per node * 2 nodes


def test_metadata_dual_output_flag_agrees_with_runtime_field(dl_predictor, a100_context, known_workload):
    """The 'dual_output' metadata flag must match whether a runtime was produced.

    Oracle = internal-consistency invariant: dual_output is True iff the prediction
    actually carries a runtime. Falsifies a flag that drifts from the payload.
    """
    prediction = dl_predictor.predict(known_workload, a100_context)

    assert prediction is not None
    has_runtime = prediction.predicted_runtime_seconds is not None
    assert prediction.metadata.get("dual_output") is has_runtime
    # DL never serves from cache — it always runs the net.
    assert prediction.metadata.get("cache_hit") is False


def test_out_of_library_workload_returns_none(dl_predictor, a100_context, unknown_workload):
    """Unknown LLM/GPU (absent from Kavier spec library) => predict returns None.

    Oracle = the out-of-library contract: 'nonexistent-model' has no spec row, so its
    SPEC_NUMERICAL features are NaN and the predictor must bail rather than feed NaNs
    to the net. Falsifies a predictor that hallucinates a number for unknown hardware.
    """
    assert dl_predictor.predict(unknown_workload, a100_context) is None


def test_get_name_is_stable_registry_key(dl_predictor):
    """get_name() is the resolver/registry key; it must stay exactly 'DeepLearningPredictor'."""
    assert dl_predictor.get_name() == "DeepLearningPredictor"
