"""Tests that GaussianProcessPredictor and BayesianRidgePredictor read both artifact formats.

A dual-output artifact holds separate throughput and runtime models; a single-output (legacy)
artifact holds only the throughput model. Mocks stand in for the trained models. Each mock
returns log1p(value) and the predictor inverts it with expm1, so the prediction equals the
injected value. The dual_output flag is True when a runtime head is present and False
otherwise, and the returned std stays in log space.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from coastline.sdk.models.context import Constraints, SystemContext
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.performance.data_driven.bayesian_ridge_predictor import BayesianRidgePredictor
from coastline.sdk.predictors.performance.data_driven.gaussian_process_predictor import GaussianProcessPredictor


@pytest.fixture
def test_workload():
    """A workload whose LLM and GPU spec features are known (llama3.2-3b on an A100).

    The GP and Bayesian ridge predictors return None when a spec feature is NaN, so an unknown
    model would never reach the mocked predict path.
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
    mock_pipeline = MagicMock()
    mock_pipeline.named_steps = {
        "scaler": MagicMock(transform=lambda X: X),
        "gp": create_mock_gp_model(predict_value, std_value),
    }
    return mock_pipeline


def create_mock_bayesian_model(predict_value=1000.0, std_value=50.0):
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
    mock_encoder = MagicMock()
    mock_encoder.classes_ = ["lora", "full", "NVIDIA-A100-SXM4-80GB", "llama", "unknown"]
    mock_encoder.transform = MagicMock(return_value=[0])

    return {
        "method": mock_encoder,
        "gpu_model": mock_encoder,
        "model_type": mock_encoder,
    }


def setup_gp_predictor_with_dual_output(predictor, throughput_value=1500.0, runtime_value=3600.0, std_value=50.0):
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


def test_gp_dual_output_inverts_log1p_for_both_heads(test_workload, test_context):
    """The GP predictor applies expm1 to both heads of a dual-output artifact.

    The mock returns log1p(2000) = 7.6014 and log1p(5400) = 8.5943, so the predictions are 2000
    and 5400; without expm1 they would be about 7.60 and 8.59.
    """
    predictor = GaussianProcessPredictor()
    setup_gp_predictor_with_dual_output(predictor, throughput_value=2000.0, runtime_value=5400.0)

    prediction = predictor.predict(test_workload, test_context)

    assert prediction is not None  # the load path succeeded
    assert prediction.predicted_throughput == pytest.approx(2000.0, rel=1e-3), (
        "log1p-encoded throughput must invert exactly via expm1"
    )
    assert prediction.predicted_runtime_seconds is not None
    assert prediction.predicted_runtime_seconds == pytest.approx(5400.0, rel=1e-3), (
        "log1p-encoded runtime must invert exactly via expm1"
    )


def test_gp_dual_output_flag_true_when_runtime_head_present(test_workload, test_context):
    """With a runtime head, dual_output is True and the metadata carries the loaded kernel.

    predict() sets dual_output to (runtime prediction is not None);
    test_gp_backward_compatible_single_output covers the False case. cache_hit is False for an
    ML predictor, and without return_std the metadata has no std.
    """
    predictor = GaussianProcessPredictor()
    setup_gp_predictor_with_dual_output(predictor, throughput_value=1600.0, runtime_value=4800.0)

    prediction = predictor.predict(test_workload, test_context)

    assert prediction is not None
    assert prediction.metadata["predictor"] == "gaussian_process"
    assert prediction.metadata["algorithm"] == "gaussian_process"
    assert prediction.metadata["dual_output"] is True, "dual_output must be True when a runtime head exists"
    assert prediction.metadata["kernel"] == "RBF", "loaded kernel passed through unchanged"
    assert prediction.metadata["cache_hit"] is False
    assert "std" not in prediction.metadata, "return_std defaulted False -> no uncertainty key"


def test_gp_dual_output_honors_runtime_seconds_key(test_workload, test_context):
    """predict() reads the runtime head from 'runtime_seconds' as well as 'runtime'.

    The model here has only 'runtime_seconds'; the runtime decodes to 4321 and dual_output is True.
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
    """With return_std=True the GP's std stays in log space: expm1 applies to the mean alone.

    The mock returns std=85.0, so metadata['std'] is 85.0 (expm1 would give about 1e37).
    uncertainty_correlation is the loaded value, 0.85.
    """
    predictor = GaussianProcessPredictor()
    setup_gp_predictor_with_dual_output(predictor, throughput_value=1700.0, runtime_value=5100.0, std_value=85.0)

    prediction = predictor.predict(test_workload, test_context, return_std=True)

    assert prediction is not None
    assert prediction.metadata["std"] == pytest.approx(85.0), "log-space std must pass through untransformed"
    assert prediction.metadata["uncertainty_correlation"] == 0.85


def test_bayesian_dual_output_inverts_log1p_for_both_heads(test_workload, test_context):
    """The Bayesian ridge predictor applies expm1 to both heads.

    The mock returns log1p(1900) = 7.5502 and log1p(5250) = 8.5661, so the heads decode to 1900
    and 5250.
    """
    predictor = BayesianRidgePredictor()
    setup_bayesian_predictor_with_dual_output(predictor, throughput_value=1900.0, runtime_value=5250.0)

    prediction = predictor.predict(test_workload, test_context)

    assert prediction is not None
    assert prediction.predicted_throughput == pytest.approx(1900.0, rel=1e-3)
    assert prediction.predicted_runtime_seconds is not None
    assert prediction.predicted_runtime_seconds == pytest.approx(5250.0, rel=1e-3)


def test_bayesian_dual_output_reports_throughput_head_poly_degree(test_workload, test_context):
    """With per-head best_params, polynomial_degree comes from the throughput head.

    Throughput uses degree 3 and runtime degree 2, so the reported degree is 3 (the runtime head
    would give 2, and the absent top-level 'poly__degree' would give 'N/A'). alpha (1.5) and
    lambda (0.001) are the loaded values; dual_output is True because the runtime head is present.
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
        "degree comes from the throughput head (3); the runtime head has 2"
    )
    assert prediction.metadata["alpha"] == 1.5, "loaded alpha passed through unchanged"
    assert prediction.metadata["lambda"] == 0.001, "loaded lambda passed through unchanged"
    assert prediction.metadata["cache_hit"] is False


def test_bayesian_dual_output_surfaces_logspace_std_unchanged(test_workload, test_context):
    """With return_std=True the throughput head's std stays in log space (no expm1).

    The mock returns std=92.5, so metadata['std'] is 92.5. uncertainty_correlation is the loaded
    value, 0.84.
    """
    predictor = BayesianRidgePredictor()
    setup_bayesian_predictor_with_dual_output(predictor, throughput_value=1850.0, runtime_value=5550.0, std_value=92.5)
    predictor._uncertainty_correlation = 0.84

    prediction = predictor.predict(test_workload, test_context, return_std=True)

    assert prediction is not None
    assert prediction.metadata["std"] == pytest.approx(92.5), "log-space std must pass through untransformed"
    assert prediction.metadata["uncertainty_correlation"] == 0.84


def test_gp_backward_compatible_single_output(test_workload, test_context):
    """A single-output (legacy) GP artifact has a throughput head and no runtime head.

    The throughput decodes to 1300, the runtime is None and dual_output is False.
    """
    predictor = GaussianProcessPredictor()
    setup_gp_predictor_with_single_output(predictor, throughput_value=1300.0)

    prediction = predictor.predict(test_workload, test_context)

    assert prediction is not None
    assert prediction.predicted_throughput == pytest.approx(1300.0, rel=1e-3)
    assert prediction.predicted_runtime_seconds is None, "no runtime head -> runtime must be None"
    assert prediction.metadata["dual_output"] is False, "dual_output must be False without a runtime head"


def test_bayesian_backward_compatible_single_output(test_workload, test_context):
    """A single-output (legacy) Bayesian ridge artifact: the throughput decodes to 1250, the
    runtime is None and dual_output is False.
    """
    predictor = BayesianRidgePredictor()
    setup_bayesian_predictor_with_single_output(predictor, throughput_value=1250.0)

    prediction = predictor.predict(test_workload, test_context)

    assert prediction is not None
    assert prediction.predicted_throughput == pytest.approx(1250.0, rel=1e-3)
    assert prediction.predicted_runtime_seconds is None, "no runtime head -> runtime must be None"
    assert prediction.metadata["dual_output"] is False, "dual_output must be False without a runtime head"
