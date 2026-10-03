"""Tests for the feasibility checkers in ``sdk/predictors/feasibility/autoconf.py``.

Each checker's ``is_feasible(workload)`` returns ``(bool, dict)``:

* ``AutoconfFeasibilityChecker`` wraps ADO's AutoGluon validity classifier. It loads the model
  lazily and returns ``(False, {"error": "autoconf_unavailable"})`` when ``autogluon`` or the ADO
  artifacts are missing.
* ``RulesFeasibilityChecker`` checks ``total_gpus >= 1`` and per-device ``batch_size >= 1``. It
  has no divisibility, memory or OOM check.
* ``NoOpFeasibilityChecker`` accepts everything.

The AutoGluon model and ADO artifacts may be absent in CI, so ``_autoconf_modules`` is
monkeypatched with fakes of ``load_model``, ``JobConfig`` and ``get_model_prediction_and_metadata``.
"""

from __future__ import annotations

import logging

import pytest
from pydantic import BaseModel

import coastline.sdk.predictors.feasibility.autoconf as af
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.feasibility.autoconf import (
    AutoconfFeasibilityChecker,
    NoOpFeasibilityChecker,
    RulesFeasibilityChecker,
)

_GPU = "NVIDIA-A100-SXM4-80GB"


def _workload(batch_size: int = 8, gpus_per_node: int = 8, number_of_nodes: int = 1):
    return WorkloadSpec(
        llm_model="mistral-7b-v0.1",
        fine_tuning_method="full",
        gpu_model=_GPU,
        tokens_per_sample=512,
        batch_size=batch_size,
        gpus_per_node=gpus_per_node,
        number_of_nodes=number_of_nodes,
    )


# Fakes for the ADO autoconf modules.
class _FakeJobConfig:
    """Stand-in for ``autoconf.utils.pydantic_models.JobConfig`` that records the fields it gets."""

    def __init__(self, **fields):
        self.fields = fields

    @classmethod
    def model_validate(cls, data):
        return cls(**data)


class _StrictInt(BaseModel):
    """Pydantic model whose validation gives a real ``ValidationError``."""

    n: int


class _RaisingJobConfig:
    """JobConfig stand-in whose ``model_validate`` raises a pydantic ``ValidationError``, as the
    real JobConfig does for an invalid job layout."""

    @classmethod
    def model_validate(cls, data):
        _StrictInt.model_validate({"n": "not-an-int"})  # raises ValidationError
        raise AssertionError("unreachable")  # pragma: no cover


class _FakePredictor:
    """Sentinel object returned by the fake ``load_model``."""

    def __init__(self, model_version: str):
        self.model_version = model_version


def _make_mods(valid_flag, metadata, *, load_calls=None, predict_calls=None):
    """Build a (load_model, JobConfig, get_model_prediction_and_metadata) triple.

    The prediction function returns ``valid_flag`` and ``metadata``; ``load_calls`` and
    ``predict_calls`` record call arguments.
    """

    def load_model(model_version):
        if load_calls is not None:
            load_calls.append(model_version)
        return _FakePredictor(model_version)

    def get_model_prediction_and_metadata(job_config, predictor):
        if predict_calls is not None:
            predict_calls.append((job_config, predictor))
        return valid_flag, metadata

    return load_model, _FakeJobConfig, get_model_prediction_and_metadata


# AutoconfFeasibilityChecker: fallback when autoconf is unavailable.
def test_autoconf_unavailable_is_infeasible_with_error(monkeypatch):
    """When the autoconf modules cannot be imported, ``is_feasible`` returns
    (False, {"error": "autoconf_unavailable"}) and does not raise."""
    monkeypatch.setattr(af, "_autoconf_modules", lambda: None)
    feasible, meta = AutoconfFeasibilityChecker().is_feasible(_workload())
    assert feasible is False
    assert meta == {"error": "autoconf_unavailable"}


# AutoconfFeasibilityChecker: model-available paths, with fakes.
def test_autoconf_feasible_when_classifier_returns_valid(monkeypatch):
    """A classifier flag of 1 means feasible, and the classifier metadata is returned unchanged."""
    meta = {"score": 0.91, "min_gpus": 4}
    monkeypatch.setattr(af, "_autoconf_modules", lambda: _make_mods(1, meta))
    feasible, out = AutoconfFeasibilityChecker().is_feasible(_workload())
    assert feasible is True
    assert out == meta


def test_autoconf_infeasible_when_classifier_returns_invalid(monkeypatch):
    """A classifier flag of 0 means infeasible, and the metadata is still returned."""
    meta = {"reason": "out_of_memory"}
    monkeypatch.setattr(af, "_autoconf_modules", lambda: _make_mods(0, meta))
    feasible, out = AutoconfFeasibilityChecker().is_feasible(_workload())
    assert feasible is False
    assert out == meta


def test_autoconf_none_metadata_normalized_to_empty_dict(monkeypatch):
    """A feasible verdict with None metadata comes back as (True, {})."""
    monkeypatch.setattr(af, "_autoconf_modules", lambda: _make_mods(1, None))
    feasible, out = AutoconfFeasibilityChecker().is_feasible(_workload())
    assert feasible is True
    assert out == {}


def test_autoconf_maps_workload_fields_into_job_config(monkeypatch):
    """Workload fields map onto JobConfig: ``number_gpus`` is gpus_per_node x number_of_nodes, and
    ``batch_size`` is the total batch (per-device batch x GPUs). Other fields pass through unchanged.
    """
    predict_calls: list = []
    monkeypatch.setattr(af, "_autoconf_modules", lambda: _make_mods(1, {}, predict_calls=predict_calls))
    wl = _workload(batch_size=16, gpus_per_node=4, number_of_nodes=2)  # per-device batch 16 on 4 x 2 = 8 GPUs
    AutoconfFeasibilityChecker().is_feasible(wl)

    assert len(predict_calls) == 1
    job_config, _ = predict_calls[0]
    assert job_config.fields == {
        "model_name": "mistral-7b-v0.1",
        "method": "full",
        "gpu_model": _GPU,
        "tokens_per_sample": 512,
        "batch_size": 128,  # per-device 16 x 8 GPUs; AutoConf's batch_size is the total
        "number_gpus": 8,  # 4 GPUs/node x 2 nodes
    }


def test_autoconf_feasibility_model_overrides_llm_model_in_job_config(monkeypatch):
    """The JobConfig model name is ``feasibility_model or llm_model``, so a set feasibility_model
    overrides a proxy llm_model. Both are canonicalized at ingestion ("anon-model" and
    "mistral-7b-v0.1" here).
    """
    predict_calls: list = []
    monkeypatch.setattr(af, "_autoconf_modules", lambda: _make_mods(1, {}, predict_calls=predict_calls))
    wl = WorkloadSpec(
        llm_model="proxy/Anon-Model",  # canonicalized to "anon-model"
        fine_tuning_method="lora",
        gpu_model=_GPU,
        tokens_per_sample=256,
        batch_size=8,
        gpus_per_node=8,
        number_of_nodes=1,
        feasibility_model="mistralai/Mistral-7B-v0.1",
    )
    AutoconfFeasibilityChecker().is_feasible(wl)

    job_config, _ = predict_calls[0]
    # model_name is the canonical feasibility_model.
    assert job_config.fields["model_name"] == "mistral-7b-v0.1"
    assert job_config.fields["method"] == "lora"


def test_autoconf_loads_model_once_across_multiple_predictions(monkeypatch):
    """The AutoGluon model loads once per checker and is reused for later candidates."""
    load_calls: list = []
    monkeypatch.setattr(af, "_autoconf_modules", lambda: _make_mods(1, {}, load_calls=load_calls))
    checker = AutoconfFeasibilityChecker()
    checker.is_feasible(_workload(batch_size=8))
    checker.is_feasible(_workload(batch_size=16))
    assert len(load_calls) == 1


def test_autoconf_invalid_job_config_rejected_before_classifier(monkeypatch):
    """An invalid JobConfig gives (False, error) with an ``invalid_job_config:`` tag, and the
    classifier is not called."""
    predict_calls: list = []

    def load_model(model_version):
        return _FakePredictor(model_version)

    def predict(job_config, predictor):
        predict_calls.append((job_config, predictor))
        return 1, {}

    monkeypatch.setattr(af, "_autoconf_modules", lambda: (load_model, _RaisingJobConfig, predict))
    feasible, out = AutoconfFeasibilityChecker().is_feasible(_workload())
    assert feasible is False
    assert out["error"].startswith("invalid_job_config:")
    assert predict_calls == []  # the classifier was not called


def test_autoconf_exception_during_prediction_is_caught(monkeypatch):
    """An exception from the classifier comes back as (False, {"error": <message>}), with no tag."""

    def boom(job_config, predictor):
        raise RuntimeError("autogluon exploded")

    load_model, JobConfig, _ = _make_mods(1, {})
    monkeypatch.setattr(af, "_autoconf_modules", lambda: (load_model, JobConfig, boom))
    feasible, out = AutoconfFeasibilityChecker().is_feasible(_workload())
    assert feasible is False
    assert out == {"error": "autogluon exploded"}


# AutoconfFeasibilityChecker: a GPU outside AutoConf's training data.
def _extrapolation_warnings(caplog, gpu: str) -> list[str]:
    """The warnings about ``gpu`` itself (the message also lists the GPUs AutoConf knows)."""
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and f"GPU {gpu!r}" in r.getMessage()]


@pytest.mark.parametrize("valid_flag", [1, 0])
def test_a_gpu_autoconf_never_saw_warns_once_and_keeps_the_verdict(monkeypatch, caplog, valid_flag):
    """H100-SXM is in Kavier's catalog but not in AutoConf's training data. The classifier's verdict
    is kept, and one warning per GPU name calls it an extrapolation."""
    monkeypatch.setattr(af, "_autoconf_modules", lambda: _make_mods(valid_flag, {}))
    monkeypatch.setattr(af, "_WARNED_GPUS", set())
    monkeypatch.setenv("COASTLINE_NO_AUTOCONF_BATCH", "1")  # check_chunk asks the fake per candidate
    checker = AutoconfFeasibilityChecker()
    h100_sxm = _workload().model_copy(update={"gpu_model": "H100-SXM"})

    with caplog.at_level(logging.WARNING, logger=af.__name__):
        verdicts = [checker.is_feasible(h100_sxm), checker.is_feasible(h100_sxm.model_copy(update={"batch_size": 4}))]
        verdicts += checker.check_chunk([h100_sxm])

    assert verdicts == [(valid_flag == 1, {})] * 3
    (message,) = _extrapolation_warnings(caplog, "H100-SXM")
    assert "extrapolation" in message


def test_each_unknown_gpu_gets_its_own_warning(monkeypatch, caplog):
    monkeypatch.setattr(af, "_autoconf_modules", lambda: _make_mods(1, {}))
    monkeypatch.setattr(af, "_WARNED_GPUS", set())
    checker = AutoconfFeasibilityChecker()

    with caplog.at_level(logging.WARNING, logger=af.__name__):
        for gpu in ("H100-SXM", "L4", "H100-SXM"):
            checker.is_feasible(_workload().model_copy(update={"gpu_model": gpu}))

    assert len(_extrapolation_warnings(caplog, "H100-SXM")) == 1
    assert len(_extrapolation_warnings(caplog, "L4")) == 1


@pytest.mark.parametrize("gpu", ["NVIDIA-A100-SXM4-80GB", "NVIDIA-A100-80GB-PCIe", "NVIDIA-H100-PCIe", "L40S"])
def test_a_gpu_in_autoconf_training_data_does_not_warn(monkeypatch, caplog, gpu):
    monkeypatch.setattr(af, "_autoconf_modules", lambda: _make_mods(1, {}))
    monkeypatch.setattr(af, "_WARNED_GPUS", set())

    with caplog.at_level(logging.WARNING, logger=af.__name__):
        AutoconfFeasibilityChecker().is_feasible(_workload().model_copy(update={"gpu_model": gpu}))

    assert not [r for r in caplog.records if "extrapolation" in r.getMessage()]


# RulesFeasibilityChecker: per-device sanity guards, no divisibility rule.
def test_rules_feasible_for_valid_per_device_workload():
    # A valid per-device workload is feasible, with empty metadata.
    feasible, meta = RulesFeasibilityChecker().is_feasible(_workload(batch_size=8, gpus_per_node=8, number_of_nodes=1))
    assert feasible is True
    assert meta == {}


def test_rules_feasible_when_batch_not_divisible_by_gpus():
    # batch_size is per device, so a batch of 7 on 8 GPUs is feasible.
    feasible, meta = RulesFeasibilityChecker().is_feasible(_workload(batch_size=7, gpus_per_node=8, number_of_nodes=1))
    assert feasible is True
    assert meta == {}


def test_rules_per_device_batch_feasible_regardless_of_gpu_count():
    """A per-device batch of 8 is feasible on 4 GPUs (1 node) and on 16 GPUs (4 nodes); it need not
    divide the GPU count."""
    one_node = RulesFeasibilityChecker().is_feasible(_workload(batch_size=8, gpus_per_node=4, number_of_nodes=1))
    four_nodes = RulesFeasibilityChecker().is_feasible(_workload(batch_size=8, gpus_per_node=4, number_of_nodes=4))
    assert one_node[0] is True
    assert four_nodes[0] is True


# NoOpFeasibilityChecker: accepts everything.
def test_noop_accepts_any_config():
    """The no-op checker accepts every workload (used in tests or when feasibility is off)."""
    feasible, meta = NoOpFeasibilityChecker().is_feasible(_workload(batch_size=7, gpus_per_node=8, number_of_nodes=1))
    assert feasible is True
    assert meta == {}
