"""A data-driven model whose file is not in this install fails with a message the user can act on.

The PyPI wheel leaves out several model files (tabpfn, random_forest, gaussian_process, svr,
knn); selecting one raises ``ModelNotShippedError``, which points to Zenodo and PORTFOLIO_DIR.
A model file that is present but unusable (a Git LFS pointer in a checkout without LFS, or a
file without the model in it) gives a prediction with no numbers whose reason names the file,
so the pipeline error names it too. No real model file is unpickled, so no native ML runtime
loads.
"""

import pickle

import pytest

import coastline.sdk.predictors.performance.data_driven.ml_common as ml_common
from coastline.sdk.exceptions import NoPredictionError
from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.performance.data_driven.ml_common import ModelNotShippedError

_GPU = "NVIDIA-A100-SXM4-80GB"
_WORKLOAD = WorkloadSpec(
    llm_model="mistral-7b-v0.1",
    fine_tuning_method="lora",
    gpu_model=_GPU,
    tokens_per_sample=2048,
    batch_size=8,
    gpus_per_node=4,
    number_of_nodes=1,
)
_CONTEXT = SystemContext.for_gpus([_GPU], max_gpus=8)

# What a checkout without Git LFS holds in place of a model file.
_LFS_POINTER = "version https://git-lfs.github.com/spec/v1\noid sha256:" + "0" * 64 + "\nsize 1234\n"


def _assert_actionable(message: str, model: str) -> None:
    assert f"the {model} model is not in this install" in message
    assert "Zenodo" in message and "PORTFOLIO_DIR" in message
    assert "trainer.main" not in message


@pytest.fixture
def empty_install(monkeypatch, tmp_path):
    """An install whose portfolio holds no model files, like the wheel for knn or tabpfn."""
    monkeypatch.setattr(ml_common, "PORTFOLIO_DIR", tmp_path)
    monkeypatch.setattr(ml_common, "_BUNDLED_PORTFOLIO_DIR", tmp_path)
    return tmp_path


def test_missing_portfolio_model_raises_instead_of_returning_none(empty_install):
    from coastline.sdk.policies import _build_named_ml_predictor

    predictor = _build_named_ml_predictor("knn")
    assert predictor._model_path == empty_install / "knn.pkl"  # not custom/knn.pkl
    with pytest.raises(ModelNotShippedError) as err:
        predictor.predict(_WORKLOAD, _CONTEXT)
    _assert_actionable(str(err.value), "knn")


def test_recommend_reports_the_missing_model_not_an_empty_grid(empty_install):
    import coastline

    with pytest.raises(ModelNotShippedError) as err:
        coastline.Coastline(predictor="knn", feasibility="rules").recommend(
            {
                "llm_model": "mistral-7b-v0.1",
                "fine_tuning_method": "lora",
                "gpu_model": _GPU,
                "tokens_per_sample": 2048,
                "batch_size": 8,
            },
            max_gpus=8,
            batch_sizes=[4, 8],
        )
    _assert_actionable(str(err.value), "knn")


@pytest.mark.parametrize(
    "module, cls, model",
    [
        ("tabpfn_predictor", "TabPFNPredictor", "tabpfn"),
        ("gaussian_process_predictor", "GaussianProcessPredictor", "gaussian_process"),
        ("bayesian_ridge_predictor", "BayesianRidgePredictor", "bayesian_ridge"),
    ],
)
def test_dedicated_predictors_raise_the_same_error(tmp_path, module, cls, model):
    import importlib

    predictor_cls = getattr(importlib.import_module(f"coastline.sdk.predictors.performance.data_driven.{module}"), cls)
    with pytest.raises(ModelNotShippedError) as err:
        predictor_cls(model_path=tmp_path / f"{model}.pkl").predict(_WORKLOAD, _CONTEXT)
    _assert_actionable(str(err.value), model)


def test_deep_learning_predictor_raises_the_same_error(tmp_path):
    pytest.importorskip("torch", reason="PyTorch not available")
    from coastline.sdk.predictors.performance.data_driven.deep_learning_predictor import DeepLearningPredictor

    with pytest.raises(ModelNotShippedError) as err:
        DeepLearningPredictor(model_dir=tmp_path / "deep_learning").predict(_WORKLOAD, _CONTEXT)
    _assert_actionable(str(err.value), "deep_learning")


def _assert_names_the_file(prediction, path) -> None:
    assert prediction is not None
    assert prediction.predicted_throughput is None
    assert str(path) in prediction.metadata["error_detail"]


def test_a_missing_model_file_is_named_in_the_error(empty_install):
    from coastline.sdk.policies import _build_named_ml_predictor

    with pytest.raises(ModelNotShippedError) as err:
        _build_named_ml_predictor("knn").predict(_WORKLOAD, _CONTEXT)
    assert str(empty_install / "knn.pkl") in str(err.value)


def test_an_unreadable_model_file_is_named_in_the_prediction(empty_install):
    from coastline.sdk.policies import _build_named_ml_predictor

    (empty_install / "knn.pkl").write_text(_LFS_POINTER)

    _assert_names_the_file(_build_named_ml_predictor("knn").predict(_WORKLOAD, _CONTEXT), empty_install / "knn.pkl")


def test_a_model_file_without_the_model_is_named_in_the_prediction(empty_install):
    from coastline.sdk.policies import _build_named_ml_predictor

    with open(empty_install / "knn.pkl", "wb") as f:
        pickle.dump({"model": None, "cat_features": None, "num_features": None, "encoders": None}, f)

    _assert_names_the_file(_build_named_ml_predictor("knn").predict(_WORKLOAD, _CONTEXT), empty_install / "knn.pkl")


@pytest.mark.parametrize(
    "module, cls, model",
    [
        ("tabpfn_predictor", "TabPFNPredictor", "tabpfn"),
        ("gaussian_process_predictor", "GaussianProcessPredictor", "gaussian_process"),
        ("bayesian_ridge_predictor", "BayesianRidgePredictor", "bayesian_ridge"),
    ],
)
def test_dedicated_predictors_name_an_unreadable_model_file(tmp_path, module, cls, model):
    import importlib

    path = tmp_path / f"{model}.pkl"
    path.write_text(_LFS_POINTER)
    predictor_cls = getattr(importlib.import_module(f"coastline.sdk.predictors.performance.data_driven.{module}"), cls)

    _assert_names_the_file(predictor_cls(model_path=path).predict(_WORKLOAD, _CONTEXT), path)


def test_deep_learning_predictor_names_an_unreadable_model_folder(tmp_path):
    pytest.importorskip("torch", reason="PyTorch not available")
    from coastline.sdk.predictors.performance.data_driven.deep_learning_predictor import DeepLearningPredictor

    model_dir = tmp_path / "deep_learning"
    model_dir.mkdir()
    for name in ("performance_deep_learning.pth", "performance_deep_learning_artifacts.pkl"):
        (model_dir / name).write_text(_LFS_POINTER)

    _assert_names_the_file(DeepLearningPredictor(model_dir=model_dir).predict(_WORKLOAD, _CONTEXT), model_dir)


def test_recommend_names_an_unreadable_model_file(empty_install):
    import coastline

    (empty_install / "knn.pkl").write_text(_LFS_POINTER)

    with pytest.raises(NoPredictionError) as err:
        coastline.Coastline(predictor="knn", feasibility="rules").recommend(
            {
                "llm_model": "mistral-7b-v0.1",
                "fine_tuning_method": "lora",
                "gpu_model": _GPU,
                "tokens_per_sample": 2048,
                "batch_size": 8,
            },
            max_gpus=8,
            batch_sizes=[4, 8],
        )
    assert str(empty_install / "knn.pkl") in str(err.value)


def test_the_error_survives_a_worker_process_boundary():
    import pickle

    error = ModelNotShippedError("the knn model is not in this install.")
    restored = pickle.loads(pickle.dumps(error))
    assert type(restored) is ModelNotShippedError and str(restored) == str(error)
