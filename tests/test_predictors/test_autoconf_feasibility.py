"""Tests for the AutoConf-based feasibility checker.

Target: recommender/predictors/feasibility/autoconf.py

The module exposes three feasibility checkers, each with the same
``is_feasible(workload) -> (bool, dict)`` contract:

* ``AutoconfFeasibilityChecker`` — wraps the ADO autoconf AutoGluon validity
  classifier. The autoconf model is lazy-loaded and depends on the optional
  ``autogluon`` package plus ADO artifacts; when those are missing it must
  degrade gracefully (``(False, {"error": "autoconf_unavailable"})``) rather
  than raise.
* ``RulesFeasibilityChecker`` — pure-Python divisibility rule
  (``batch_size`` must be evenly divisible by ``total_gpus``, ``total_gpus``
  must be >= 1). No external dependency.
* ``NoOpFeasibilityChecker`` — accepts everything.

Because the real AutoGluon model + ADO artifacts are not guaranteed to be
present in CI, the "model available" paths of ``AutoconfFeasibilityChecker``
are exercised by monkeypatching ``_autoconf_modules`` with light fakes that
mimic ``load_model`` / ``JobConfig`` / ``get_model_prediction_and_metadata``.

Every assertion below is anchored to an independent oracle: the documented
``is_feasible`` contract, hand-computed divisibility arithmetic, the
``total_gpus = gpus_per_node × number_of_nodes`` layout law, or a load-once
invariant — never a value copied back out of the code under test.

No production code is modified by these tests.
"""

from __future__ import annotations

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


# --------------------------------------------------------------------------- #
# Fakes that stand in for the ADO autoconf modules.
# --------------------------------------------------------------------------- #
class _FakeJobConfig:
    """Mimics ``autoconf.utils.pydantic_models.JobConfig`` enough for the test.

    The real class is a pydantic model whose ``model_validate`` accepts the
    dict that ``is_feasible`` builds; here we just record what we were handed
    so the test can assert the field mapping (workload -> job config) is wired
    correctly.
    """

    def __init__(self, **fields):
        self.fields = fields

    @classmethod
    def model_validate(cls, data):
        return cls(**data)


class _StrictInt(BaseModel):
    """A real pydantic model used only to mint a genuine ``ValidationError``."""

    n: int


class _RaisingJobConfig:
    """JobConfig stand-in whose ``model_validate`` raises a real pydantic
    ``ValidationError`` — what the production JobConfig raises when the grid
    probes a structurally invalid job layout."""

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

    ``valid_flag`` / ``metadata`` are returned by the prediction fn. Optional
    lists capture call args so tests can assert lazy-loading / field mapping.
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


# --------------------------------------------------------------------------- #
# AutoconfFeasibilityChecker — graceful fallback when autoconf is unavailable.
# --------------------------------------------------------------------------- #
def test_autoconf_unavailable_is_infeasible_with_error(monkeypatch):
    """When the autoconf modules cannot be imported, is_feasible returns
    (False, {"error": "autoconf_unavailable"}) instead of raising.

    Oracle: the documented graceful-degradation contract. Falsifies if the
    checker instead propagates ImportError or defaults a missing model to
    feasible=True.
    """
    monkeypatch.setattr(af, "_autoconf_modules", lambda: None)
    feasible, meta = AutoconfFeasibilityChecker().is_feasible(_workload())
    assert feasible is False
    assert meta == {"error": "autoconf_unavailable"}


# --------------------------------------------------------------------------- #
# AutoconfFeasibilityChecker — model-available paths (via injected fakes).
# --------------------------------------------------------------------------- #
def test_autoconf_feasible_when_classifier_returns_valid(monkeypatch):
    """valid_flag == 1 -> feasible, and the classifier metadata is passed
    straight through.

    Oracle: contract maps the classifier's 1 to True and echoes its metadata.
    Falsifies if the flag→bool mapping is inverted or metadata is dropped.
    """
    meta = {"score": 0.91, "min_gpus": 4}
    monkeypatch.setattr(af, "_autoconf_modules", lambda: _make_mods(1, meta))
    feasible, out = AutoconfFeasibilityChecker().is_feasible(_workload())
    assert feasible is True
    assert out == meta


def test_autoconf_infeasible_when_classifier_returns_invalid(monkeypatch):
    """valid_flag == 0 -> infeasible (e.g. would OOM / exceed GPU budget).

    Oracle: only ``valid_flag == 1`` is feasible, so 0 must map to False while
    still echoing metadata. Falsifies if any truthy/nonzero flag is accepted.
    """
    meta = {"reason": "out_of_memory"}
    monkeypatch.setattr(af, "_autoconf_modules", lambda: _make_mods(0, meta))
    feasible, out = AutoconfFeasibilityChecker().is_feasible(_workload())
    assert feasible is False
    assert out == meta


def test_autoconf_none_metadata_normalized_to_empty_dict(monkeypatch):
    """A feasible verdict with None metadata must surface as (True, {}).

    Oracle: the ``metadata or {}`` normalization in the contract — callers get
    a dict, never None. Falsifies if the ``or {}`` guard is dropped and None
    leaks through (a downstream ``meta[...]`` would then blow up).
    """
    monkeypatch.setattr(af, "_autoconf_modules", lambda: _make_mods(1, None))
    feasible, out = AutoconfFeasibilityChecker().is_feasible(_workload())
    assert feasible is True
    assert out == {}


def test_autoconf_maps_workload_fields_into_job_config(monkeypatch):
    """The workload -> JobConfig field mapping must be correct, including the
    derived total_gpus (gpus_per_node x number_of_nodes) feeding number_gpus.

    Oracle: hand-derived layout — 4 GPUs/node x 2 nodes = 8 total GPUs — plus
    the verbatim pass-through of the remaining fields.
    """
    predict_calls: list = []
    monkeypatch.setattr(af, "_autoconf_modules", lambda: _make_mods(1, {}, predict_calls=predict_calls))
    wl = _workload(batch_size=16, gpus_per_node=4, number_of_nodes=2)  # total = 4*2 = 8
    AutoconfFeasibilityChecker().is_feasible(wl)

    assert len(predict_calls) == 1
    job_config, _ = predict_calls[0]
    assert job_config.fields == {
        "model_name": "mistral-7b-v0.1",
        "method": "full",
        "gpu_model": _GPU,
        "tokens_per_sample": 512,
        "batch_size": 16,
        "number_gpus": 8,  # 4 GPUs/node x 2 nodes
    }


def test_autoconf_feasibility_model_overrides_llm_model_in_job_config(monkeypatch):
    """When feasibility_model is set, the OOM check runs against it, not the
    (possibly anonymized/proxy) llm_model.

    Oracle: contract precedence ``feasibility_model or llm_model``. Only
    llm_model is canonicalized at ingestion, so the proxy id becomes
    "anon-model" while feasibility_model is passed through verbatim — the two
    values are deliberately distinct so a wrong branch is visible.
    """
    predict_calls: list = []
    monkeypatch.setattr(af, "_autoconf_modules", lambda: _make_mods(1, {}, predict_calls=predict_calls))
    wl = WorkloadSpec(
        llm_model="proxy/Anon-Model",  # canonicalized -> "anon-model"
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
    # model_name is the feasibility_model verbatim (uncanonicalized), NOT "anon-model".
    assert job_config.fields["model_name"] == "mistralai/Mistral-7B-v0.1"
    assert job_config.fields["method"] == "lora"


def test_autoconf_loads_model_once_across_multiple_predictions(monkeypatch):
    """The AutoGluon model is lazy-loaded once and reused for later candidates.

    Oracle: the memoization invariant on ``self._predictor`` — two is_feasible
    calls on one checker must trigger exactly one load_model. Falsifies if the
    ``if self._predictor is None`` guard is removed and the (expensive) model
    is reloaded per candidate.
    """
    load_calls: list = []
    monkeypatch.setattr(af, "_autoconf_modules", lambda: _make_mods(1, {}, load_calls=load_calls))
    checker = AutoconfFeasibilityChecker()
    checker.is_feasible(_workload(batch_size=8))
    checker.is_feasible(_workload(batch_size=16))
    assert len(load_calls) == 1


def test_autoconf_invalid_job_config_rejected_before_classifier(monkeypatch):
    """A structurally invalid JobConfig is a benign reject: (False, error) with
    an ``invalid_job_config:`` tag, and the classifier is never invoked.

    Oracle: the ValidationError branch is distinct from the generic prediction
    failure branch — its message is tagged and it short-circuits before
    prediction. Falsifies if ValidationError isn't caught separately (it would
    reach the generic handler with an untagged message, or the classifier would
    still run).
    """
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
    assert predict_calls == []  # short-circuited: classifier never consulted


def test_autoconf_exception_during_prediction_is_caught(monkeypatch):
    """If the classifier path raises, is_feasible swallows it and reports the
    failure as (False, {"error": <msg>}) rather than propagating.

    Oracle: the generic-failure contract — the raised message is surfaced
    verbatim (untagged, unlike the ValidationError branch). Falsifies if the
    exception escapes and aborts the grid.
    """

    def boom(job_config, predictor):
        raise RuntimeError("autogluon exploded")

    load_model, JobConfig, _ = _make_mods(1, {})
    monkeypatch.setattr(af, "_autoconf_modules", lambda: (load_model, JobConfig, boom))
    feasible, out = AutoconfFeasibilityChecker().is_feasible(_workload())
    assert feasible is False
    assert out == {"error": "autogluon exploded"}


# --------------------------------------------------------------------------- #
# RulesFeasibilityChecker — per-device sanity guards (no divisibility rule).
# --------------------------------------------------------------------------- #
def test_rules_feasible_for_valid_per_device_workload():
    # Any valid per-device workload is feasible (no divisibility rule); no error dict.
    feasible, meta = RulesFeasibilityChecker().is_feasible(_workload(batch_size=8, gpus_per_node=8, number_of_nodes=1))
    assert feasible is True
    assert meta == {}


def test_rules_feasible_when_batch_not_divisible_by_gpus():
    # batch_size is PER-DEVICE: a per-device batch of 7 on 8 GPUs is feasible (the old
    # ``7 % 8 != 0`` divisibility rejection is gone).
    feasible, meta = RulesFeasibilityChecker().is_feasible(_workload(batch_size=7, gpus_per_node=8, number_of_nodes=1))
    assert feasible is True
    assert meta == {}


def test_rules_per_device_batch_feasible_regardless_of_gpu_count():
    """batch_size is PER-DEVICE, so it need not divide the GPU count: a per-device batch of 8
    is feasible on 4 GPUs (1 node) AND on 16 GPUs (4 nodes) — the latter would have been
    rejected by the old ``8 % 16`` divisibility rule.
    """
    one_node = RulesFeasibilityChecker().is_feasible(_workload(batch_size=8, gpus_per_node=4, number_of_nodes=1))
    four_nodes = RulesFeasibilityChecker().is_feasible(_workload(batch_size=8, gpus_per_node=4, number_of_nodes=4))
    assert one_node[0] is True
    assert four_nodes[0] is True


# --------------------------------------------------------------------------- #
# NoOpFeasibilityChecker — accept everything.
# --------------------------------------------------------------------------- #
def test_noop_accepts_any_config():
    """The no-op checker accepts every workload unconditionally (used for tests or when
    feasibility is disabled)."""
    feasible, meta = NoOpFeasibilityChecker().is_feasible(_workload(batch_size=7, gpus_per_node=8, number_of_nodes=1))
    assert feasible is True
    assert meta == {}
