"""A TabPFN pickle written by another scikit-learn version loads with one log line.

The bundled tabpfn.pkl was written by scikit-learn 1.8.0 and loads under 1.7.2, where
scikit-learn warns once per estimator class in it. The loader logs one INFO line with the file and
both versions instead, and any other warning raised while unpickling still reaches the caller. The
pickles here hold small fitted scalers with version 9.9.9 recorded in them, so no TabPFN model or
torch loads.
"""

import logging
import pickle
import warnings

import pytest

sklearn_base = pytest.importorskip("sklearn.base")
from sklearn import __version__ as SKLEARN_VERSION  # noqa: E402
from sklearn.exceptions import InconsistentVersionWarning  # noqa: E402
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # noqa: E402

from coastline.sdk.models.context import SystemContext  # noqa: E402
from coastline.sdk.models.workload import WorkloadSpec  # noqa: E402
from coastline.sdk.predictors.performance.data_driven.tabpfn_predictor import TabPFNPredictor  # noqa: E402

_GPU = "NVIDIA-A100-SXM4-80GB"
_CONTEXT = SystemContext.for_gpus([_GPU], max_gpus=8)
# An unknown model stops predict() after loading, before the model would be called.
_UNKNOWN_MODEL = WorkloadSpec(
    llm_model="not-a-real-model",
    fine_tuning_method="lora",
    gpu_model=_GPU,
    tokens_per_sample=1024,
    batch_size=8,
    gpus_per_node=8,
    number_of_nodes=1,
)


class _WarnsWhenLoaded:
    """Warns when unpickled, as a library object might about its own state."""

    def __init__(self):
        self.state = "cached"  # pickle calls __setstate__ only for a non-empty state

    def __setstate__(self, state):
        warnings.warn("cached state is stale", UserWarning)
        self.__dict__.update(state)


def _write_pickle(path, monkeypatch, *objects) -> None:
    """Pickle ``{"model": [...]}`` with scikit-learn reporting version 9.9.9."""
    with monkeypatch.context() as patched:
        patched.setattr(sklearn_base, "__version__", "9.9.9")
        path.write_bytes(pickle.dumps({"model": list(objects)}))


def test_a_pickle_from_another_scikit_learn_logs_one_line_and_no_version_warning(tmp_path, monkeypatch, caplog):
    path = tmp_path / "tabpfn.pkl"
    _write_pickle(path, monkeypatch, StandardScaler().fit([[0.0], [1.0]]), MinMaxScaler().fit([[0.0], [1.0]]))

    with warnings.catch_warnings(record=True) as caught, caplog.at_level(logging.INFO):
        warnings.simplefilter("always")
        assert TabPFNPredictor(model_path=path).predict(_UNKNOWN_MODEL, _CONTEXT) is None

    assert [w for w in caught if issubclass(w.category, InconsistentVersionWarning)] == []
    lines = [record for record in caplog.records if "9.9.9" in record.getMessage()]
    assert len(lines) == 1
    assert lines[0].levelno == logging.INFO
    assert str(path) in lines[0].getMessage()
    assert SKLEARN_VERSION in lines[0].getMessage()


def test_another_warning_raised_while_loading_still_reaches_the_caller(tmp_path, monkeypatch):
    path = tmp_path / "tabpfn.pkl"
    _write_pickle(path, monkeypatch, StandardScaler().fit([[0.0], [1.0]]), _WarnsWhenLoaded())

    with pytest.warns(UserWarning, match="cached state is stale"):
        TabPFNPredictor(model_path=path).predict(_UNKNOWN_MODEL, _CONTEXT)
