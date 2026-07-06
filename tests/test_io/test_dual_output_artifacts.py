"""
Regression tests for dual-output artifact format compatibility.

Tests verify that GaussianProcessPredictor and BayesianRidgePredictor can:
1. Load and predict from new dual-output artifact format (dict with separate throughput/runtime models)
2. Return throughput predictions
3. Return runtime predictions when runtime submodel exists
4. Handle backward-compatible metadata without breaking
5. Not crash on dual-output artifact structure

These tests use lightweight mocks/stubs to avoid retraining real models.

Oracle strategy: each mock submodel stores log1p(value) in the GP/BR target
space; the predictor's job is to invert that with expm1. Since expm1 is the
exact inverse of log1p, the recovered throughput/runtime must equal the value
we injected -- an independent round-trip oracle that goes red if the predictor
drops the transform. The dual_output flag is checked against its derivation
(runtime head present <-> flag True) via a dual/single counterexample pair, and
uncertainty (std) is checked to pass through in log-space (never expm1'd).
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.performance.data_driven.bayesian_ridge_predictor import BayesianRidgePredictor
from coastline.sdk.predictors.performance.data_driven.gaussian_process_predictor import GaussianProcessPredictor

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def test_workload():
    """Standard test workload.

    Uses an in-library model (llama3.2-3b) so the deep LLM/GPU spec features
    resolve to real values rather than NaN. The non-tree sklearn predictors
    (GP, Bayesian Ridge) now bail out with None on NaN specs (unknown
    model/GPU), so these dual-output *plumbing* tests must use a known,
    modelable workload to exercise the mocked predict path.
    """
    return WorkloadSpec(
        llm_model="llama3.2-3b",
        fine_tuning_method="lora",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=16,
        gpus_per_node=8,
        number_of_nodes=1,
    )


@pytest.fixture
def test_context():
    """Standard test context."""
    return SystemContext(
        available_gpu_models=["NVIDIA-A100-SXM4-80GB"],
        max_gpus=8,
        gpu_memory={"NVIDIA-A100-SXM4-80GB": 80},
        constraints=Constraints(
            max_gpus=8,
            gpus_per_node=8,
            max_nodes=1,
        ),
    )


def create_mock_gp_model(predict_value=1000.0, std_value=50.0):
    """Create a mock Gaussian Process model for testing."""
    mock_gp = MagicMock()
    mock_gp.predict = MagicMock(
        side_effect=lambda X, return_std=False: (
            (np.array([np.log1p(predict_value)]), np.array([std_value]))
            if return_std
            else np.array([np.log1p(predict_value)])
        )
    )
    return mock_gp


def create_mock_gp_pipeline(predict_value=1000.0, std_value=50.0):
    """Create a mock GP pipeline with scaler and GP model."""
    mock_pipeline = MagicMock()
    mock_pipeline.named_steps = {
        "scaler": MagicMock(transform=lambda X: X),
        "gp": create_mock_gp_model(predict_value, std_value),
    }
    return mock_pipeline


def create_mock_bayesian_model(predict_value=1000.0, std_value=50.0):
    """Create a mock Bayesian Ridge model for testing."""
    mock_model = MagicMock()
    mock_model.predict = MagicMock(
        side_effect=lambda X, return_std=False: (
            (np.array([np.log1p(predict_value)]), np.array([std_value]))
            if return_std
            else np.array([np.log1p(predict_value)])
        )
    )
    return mock_model


def create_mock_encoders():
    """Create mock label encoders for categorical features."""
    mock_encoder = MagicMock()
    mock_encoder.classes_ = ["lora", "full", "NVIDIA-A100-SXM4-80GB", "llama", "unknown"]
    mock_encoder.transform = MagicMock(return_value=[0])

    return {
        "method": mock_encoder,
        "gpu_model": mock_encoder,
        "model_type": mock_encoder,
    }


def setup_gp_predictor_with_dual_output(predictor, throughput_value=1500.0, runtime_value=3600.0, std_value=50.0):
    """Setup GP predictor with dual-output artifact structure."""
    predictor._model = {
        "throughput": create_mock_gp_pipeline(throughput_value, std_value),
        "runtime": create_mock_gp_pipeline(runtime_value, std_value),
    }
    predictor._encoders = create_mock_encoders()
    predictor._cat_features = ["method", "gpu_model", "model_type"]
    predictor._num_features = ["number_nodes", "number_gpus", "tokens_per_sample", "batch_size"]
    predictor._test_metrics = {
        "throughput": {"original_space": {"mdape": 15.2, "r2": 0.92}},
        "runtime": {"original_space": {"mdape": 12.8, "r2": 0.94}},
    }
    predictor._kernel = "RBF"
    predictor._uncertainty_correlation = 0.85
    predictor._is_dual_output = True
    predictor._loaded = True


def setup_gp_predictor_with_single_output(predictor, throughput_value=1300.0, std_value=50.0):
    """Setup GP predictor with single-output (legacy) artifact structure."""
    predictor._model = {"throughput": create_mock_gp_pipeline(throughput_value, std_value)}
    predictor._encoders = create_mock_encoders()
    predictor._cat_features = ["method", "gpu_model", "model_type"]
    predictor._num_features = ["number_nodes", "number_gpus", "tokens_per_sample", "batch_size"]
    predictor._test_metrics = {"original_space": {"r2": 0.84}}
    predictor._kernel = "RBF"
    predictor._uncertainty_correlation = 0.75
    predictor._is_dual_output = False
    predictor._loaded = True


def setup_bayesian_predictor_with_dual_output(predictor, throughput_value=1400.0, runtime_value=4200.0, std_value=50.0):
    """Setup Bayesian Ridge predictor with dual-output artifact structure."""
    predictor._model = {
        "throughput": create_mock_bayesian_model(throughput_value, std_value),
        "runtime": create_mock_bayesian_model(runtime_value, std_value),
    }
    predictor._cat_features = ["method", "gpu_model", "model_type"]
    predictor._num_features = ["number_nodes", "number_gpus", "tokens_per_sample", "batch_size"]
    predictor._cat_indices = [0, 1, 2]
    predictor._num_indices = [3, 4, 5, 6]
    predictor._test_metrics = {
        "throughput": {"original_space": {"mdape": 16.8, "r2": 0.88}},
        "runtime": {"original_space": {"mdape": 13.5, "r2": 0.90}},
    }
    predictor._best_params = {"throughput": {"poly__degree": 2}, "runtime": {"poly__degree": 2}}
    predictor._alpha = 1.5
    predictor._lambda = 0.001
    predictor._uncertainty_correlation = 0.79
    predictor._is_dual_output = True
    predictor._loaded = True


def setup_bayesian_predictor_with_single_output(predictor, throughput_value=1250.0, std_value=50.0):
    """Setup Bayesian Ridge predictor with single-output (legacy) artifact structure."""
    predictor._model = {"throughput": create_mock_bayesian_model(throughput_value, std_value)}
    predictor._cat_features = ["method", "gpu_model", "model_type"]
    predictor._num_features = ["number_nodes", "number_gpus", "tokens_per_sample", "batch_size"]
    predictor._cat_indices = [0, 1, 2]
    predictor._num_indices = [3, 4, 5, 6]
    predictor._test_metrics = {"original_space": {"r2": 0.82}}
    predictor._best_params = {"poly__degree": 2}
    predictor._alpha = 1.4
    predictor._lambda = 0.0016
    predictor._uncertainty_correlation = 0.73
    predictor._is_dual_output = False
    predictor._loaded = True


# ============================================================================
# GaussianProcessPredictor Dual-Output Tests
# ============================================================================


def test_gp_dual_output_inverts_log1p_for_both_heads(test_workload, test_context):
    """The dual-output GP mock emits log1p(value) for each head; the predictor
    must apply expm1 to recover the original throughput AND runtime exactly.
    This pins the target-transform inversion, not just sign/None.

    Oracle: the mock stores log1p(2000)=ln(2001)=7.6014 for the throughput head
    and log1p(5400)=ln(5401)=8.5943 for the runtime head. expm1 is the exact
    inverse of log1p, so expm1(7.6014)=2000 and expm1(8.5943)=5400. A predictor
    that forgot expm1 would surface ~7.60 / ~8.59 instead.
    """
    predictor = GaussianProcessPredictor()
    setup_gp_predictor_with_dual_output(predictor, throughput_value=2000.0, runtime_value=5400.0)

    prediction = predictor.predict(test_workload, test_context)

    assert prediction is not None  # guard: load path succeeded
    assert prediction.predicted_throughput == pytest.approx(2000.0, rel=1e-3), (
        "log1p-encoded throughput must invert exactly via expm1"
    )
    assert prediction.predicted_runtime_seconds is not None
    assert prediction.predicted_runtime_seconds == pytest.approx(5400.0, rel=1e-3), (
        "log1p-encoded runtime must invert exactly via expm1"
    )


def test_gp_dual_output_flag_true_when_runtime_head_present(test_workload, test_context):
    """The dual_output flag is DERIVED, not stored: predict() sets it to
    (runtime_seconds is not None). With a runtime head present the runtime
    prediction is non-None, so the flag must be True. The single-output test
    is the paired counterexample (no runtime head -> flag False), so together
    they pin the flag to the presence of the runtime submodel.

    Also pins the metadata contract surfaced from the loaded artifact:
    the kernel string ("RBF", as loaded) and the identity labels. cache_hit
    is always False for an ML predictor (only the cache predictor sets it True).
    Because return_std defaults False, no uncertainty must leak into metadata.
    """
    predictor = GaussianProcessPredictor()
    setup_gp_predictor_with_dual_output(predictor, throughput_value=1600.0, runtime_value=4800.0)

    prediction = predictor.predict(test_workload, test_context)

    assert prediction is not None
    assert prediction.metadata["predictor"] == "gaussian_process"
    assert prediction.metadata["algorithm"] == "gaussian_process"
    assert prediction.metadata["dual_output"] is True, "dual_output must be True when a runtime head exists"
    assert prediction.metadata["kernel"] == "RBF", "loaded kernel must be surfaced verbatim"
    assert prediction.metadata["cache_hit"] is False
    assert "std" not in prediction.metadata, "return_std defaulted False -> no uncertainty key"


def test_gp_dual_output_honors_runtime_seconds_key(test_workload, test_context):
    """The runtime head may be stored under either 'runtime_seconds' or 'runtime'
    (predict() prefers 'runtime_seconds'). A model dict keyed 'runtime_seconds'
    must still yield a runtime prediction; a bug that only checked 'runtime'
    would drop it to None. Oracle: expm1(log1p(4321))=4321, dual_output True.
    """
    predictor = GaussianProcessPredictor()
    predictor._model = {
        "throughput": create_mock_gp_pipeline(1234.0),
        "runtime_seconds": create_mock_gp_pipeline(4321.0),
    }
    predictor._encoders = create_mock_encoders()
    predictor._cat_features = ["method", "gpu_model", "model_type"]
    predictor._num_features = ["number_nodes", "number_gpus", "tokens_per_sample", "batch_size"]
    predictor._kernel = "RBF"
    predictor._uncertainty_correlation = 0.85
    predictor._loaded = True

    prediction = predictor.predict(test_workload, test_context)

    assert prediction is not None
    assert prediction.predicted_throughput == pytest.approx(1234.0, rel=1e-3)
    assert prediction.predicted_runtime_seconds == pytest.approx(4321.0, rel=1e-3)
    assert prediction.metadata["dual_output"] is True


def test_gp_dual_output_surfaces_logspace_std_unchanged(test_workload, test_context):
    """With return_std=True the GP returns a log-space std alongside the mean.
    That std is a variance estimate in the model's target space and must NOT be
    run through expm1 (the mean is; the std is not). The mock emits std=85.0, so
    metadata['std'] must be exactly 85.0 -- an expm1'd std would be ~1e37.
    uncertainty_correlation is the loaded artifact value (0.85), surfaced as-is.
    """
    predictor = GaussianProcessPredictor()
    setup_gp_predictor_with_dual_output(predictor, throughput_value=1700.0, runtime_value=5100.0, std_value=85.0)

    prediction = predictor.predict(test_workload, test_context, return_std=True)

    assert prediction is not None
    assert prediction.metadata["std"] == pytest.approx(85.0), "log-space std must pass through untransformed"
    assert prediction.metadata["uncertainty_correlation"] == 0.85


# ============================================================================
# BayesianRidgePredictor Dual-Output Tests
# ============================================================================


def test_bayesian_dual_output_inverts_log1p_for_both_heads(test_workload, test_context):
    """Same log1p-inversion contract for Bayesian Ridge: expm1 recovers both
    the throughput and runtime mock values exactly.

    Oracle: mock stores log1p(1900)=ln(1901)=7.5502 and log1p(5250)=ln(5251)=8.5661.
    expm1 inverts log1p exactly, so the heads must decode to 1900 and 5250.
    """
    predictor = BayesianRidgePredictor()
    setup_bayesian_predictor_with_dual_output(predictor, throughput_value=1900.0, runtime_value=5250.0)

    prediction = predictor.predict(test_workload, test_context)

    assert prediction is not None
    assert prediction.predicted_throughput == pytest.approx(1900.0, rel=1e-3)
    assert prediction.predicted_runtime_seconds is not None
    assert prediction.predicted_runtime_seconds == pytest.approx(5250.0, rel=1e-3)


def test_bayesian_dual_output_reports_throughput_head_poly_degree(test_workload, test_context):
    """When best_params is a per-head dict, the metadata polynomial_degree must
    come from the THROUGHPUT head, not the runtime head. Here throughput uses
    degree 3 and runtime uses degree 2; the reported degree must be 3. A bug
    that read the runtime sub-dict (or the top-level 'poly__degree', absent
    here -> 'N/A') would surface 2 or 'N/A' instead.

    alpha (1.5) and lambda (0.001) are the loaded hyperparameters surfaced
    verbatim; dual_output is True because the runtime head is present.
    """
    predictor = BayesianRidgePredictor()
    setup_bayesian_predictor_with_dual_output(predictor, throughput_value=1650.0, runtime_value=4950.0)
    predictor._best_params = {"throughput": {"poly__degree": 3}, "runtime": {"poly__degree": 2}}

    prediction = predictor.predict(test_workload, test_context)

    assert prediction is not None
    assert prediction.metadata["predictor"] == "bayesian_ridge"
    assert prediction.metadata["algorithm"] == "bayesian_linear_regression"
    assert prediction.metadata["dual_output"] is True, "dual_output must be True when a runtime head exists"
    assert prediction.metadata["polynomial_degree"] == 3, (
        "degree must come from the throughput head (3), not runtime (2)"
    )
    assert prediction.metadata["alpha"] == 1.5, "loaded alpha surfaced verbatim"
    assert prediction.metadata["lambda"] == 0.001, "loaded lambda surfaced verbatim"
    assert prediction.metadata["cache_hit"] is False


def test_bayesian_dual_output_surfaces_logspace_std_unchanged(test_workload, test_context):
    """return_std=True surfaces the throughput head's log-space std verbatim
    (no expm1 on the std). The mock emits std=92.5, so metadata['std'] must be
    exactly 92.5. uncertainty_correlation is the loaded value (0.84), as-is.
    """
    predictor = BayesianRidgePredictor()
    setup_bayesian_predictor_with_dual_output(predictor, throughput_value=1850.0, runtime_value=5550.0, std_value=92.5)
    predictor._uncertainty_correlation = 0.84

    prediction = predictor.predict(test_workload, test_context, return_std=True)

    assert prediction is not None
    assert prediction.metadata["std"] == pytest.approx(92.5), "log-space std must pass through untransformed"
    assert prediction.metadata["uncertainty_correlation"] == 0.84


# ============================================================================
# Backward Compatibility Tests
# ============================================================================


def test_gp_backward_compatible_single_output(test_workload, test_context):
    """Legacy single-output artifact: only a throughput head, no runtime head.
    The throughput must still decode (expm1(log1p(1300))=1300), runtime must be
    None (no head to predict it), and dual_output must therefore be False. This
    is the paired counterexample to the dual-output flag test.
    """
    predictor = GaussianProcessPredictor()
    setup_gp_predictor_with_single_output(predictor, throughput_value=1300.0)

    prediction = predictor.predict(test_workload, test_context)

    assert prediction is not None
    assert prediction.predicted_throughput == pytest.approx(1300.0, rel=1e-3)
    assert prediction.predicted_runtime_seconds is None, "no runtime head -> runtime must be None"
    assert prediction.metadata["dual_output"] is False, "dual_output must be False without a runtime head"


def test_bayesian_backward_compatible_single_output(test_workload, test_context):
    """Legacy single-output artifact for Bayesian Ridge: throughput decodes
    (expm1(log1p(1250))=1250), runtime is None, dual_output is False.
    """
    predictor = BayesianRidgePredictor()
    setup_bayesian_predictor_with_single_output(predictor, throughput_value=1250.0)

    prediction = predictor.predict(test_workload, test_context)

    assert prediction is not None
    assert prediction.predicted_throughput == pytest.approx(1250.0, rel=1e-3)
    assert prediction.predicted_runtime_seconds is None, "no runtime head -> runtime must be None"
    assert prediction.metadata["dual_output"] is False, "dual_output must be False without a runtime head"
