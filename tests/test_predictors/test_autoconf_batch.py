"""Tests for the batched AutoConf feasibility path.

The batched methods are ``AutoconfFeasibilityChecker.check_chunk``,
``_RulesThenAutoconfChecker.check_chunk``, ``GuardedFeasibilityChecker.check_chunk`` and
``evaluate_chunk``. Each result is compared with the per-row path, which runs a copy of ado's
``get_model_prediction_and_metadata`` (``autoconf/utils/recommender.py``). The AutoGluon model and
the ADO modules are faked, so the tests do not depend on ADO being installed.

A real ``WorkloadSpec`` cannot produce an invalid JobConfig or a batch that fails ado's
divisibility rule, so those two branches are reached with JobConfig stand-ins.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Optional

import pandas as pd
import pytest
from pydantic import BaseModel, Field

import coastline.sdk.predictors.feasibility.autoconf as af
from coastline.sdk.constants import (
    BATCHABLE_AUTOCONF_MODEL_VERSIONS,
    DEFAULT_AUTOCONF_MODEL_VERSION,
    EMPIRICAL_OOM_TOKEN_BUDGET,
)
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.pipeline.feasibility import _RulesThenAutoconfChecker, evaluate_chunk
from coastline.sdk.predictors.feasibility.autoconf import AutoconfFeasibilityChecker
from coastline.sdk.predictors.feasibility.token_budget import (
    GuardedFeasibilityChecker,
    TokenBudgetFeasibilityChecker,
)

_GPU = "NVIDIA-A100-SXM4-80GB"

#: ado's rule-based classifier error, copied from autoconf/utils/rule_based_classifier.py.
_RULE_ERROR = "Rule-based classifier error: batch_size must be evenly divisible by number_gpus."

#: The fake classifier calls a row valid while total batch x tokens_per_sample is at most this.
#: The rule is arbitrary; the tests only need it to be row-wise and deterministic.
_CLASSIFIER_CEILING = 1_000_000

#: Sentinel model name that ``_SelectiveJobConfig`` refuses to validate.
_NOT_A_JOB = "not-a-job"

#: The spy backends' metadata. ``guard`` is also a token-budget guard key, so a test can see which
#: side wins the clash.
_SPY_METADATA = {"backend": "spy", "guard": "backend-wins"}


def _workload(
    batch_size: int = 8,
    tokens_per_sample: int = 512,
    gpus_per_node: int = 8,
    number_of_nodes: int = 1,
    llm_model: str = "mistral-7b-v0.1",
) -> WorkloadSpec:
    return WorkloadSpec(
        llm_model=llm_model,
        fine_tuning_method="full",
        gpu_model=_GPU,
        tokens_per_sample=tokens_per_sample,
        batch_size=batch_size,
        gpus_per_node=gpus_per_node,
        number_of_nodes=number_of_nodes,
    )


def _chunk() -> list[WorkloadSpec]:
    """Five candidates on 8 GPUs, built fresh on every call.

    The fake classifier calls them valid, invalid, valid, valid, invalid, so a reordered result
    shows. The token guard vetoes the second, fourth and fifth (262,144, 65,536 and 262,144
    tokens/device against 60,224); the classifier would pass the fourth. The per-device batches
    mix multiples and non-multiples of the GPU count.
    """
    return [
        _workload(batch_size=1, tokens_per_sample=512),  # total batch 8: 8 x 512 = 4,096
        _workload(batch_size=64, tokens_per_sample=4096),  # total batch 512: 2,097,152
        _workload(batch_size=4, tokens_per_sample=1024),  # total batch 32: 32,768
        _workload(batch_size=16, tokens_per_sample=4096),  # total batch 128: 524,288
        _workload(batch_size=32, tokens_per_sample=8192),  # total batch 256: 2,097,152
    ]


# Fakes for the ADO autoconf modules.
class _FakeJobConfig(BaseModel):
    """Stand-in for ``autoconf.utils.pydantic_models.JobConfig`` with the same fields and bounds.

    ``model_validate`` raises a real ``ValidationError``, and ``model_dump()`` gives the row shape
    the batched path builds, including the ``is_valid`` default.
    """

    model_name: str
    method: str
    gpu_model: str
    tokens_per_sample: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    is_valid: Optional[int] = None
    number_gpus: Optional[int] = Field(default=None, ge=1)


class _PositiveInt(BaseModel):
    """Pydantic model whose validation gives a real ``ValidationError``."""

    n: int = Field(gt=0)


class _SelectiveJobConfig(_FakeJobConfig):
    """JobConfig stand-in that rejects the candidate whose model is ``_NOT_A_JOB``.

    The chunk tests need one bad candidate among good ones.
    """

    @classmethod
    def model_validate(cls, data: Any, **kwargs: Any) -> "_SelectiveJobConfig":
        if data["model_name"] == _NOT_A_JOB:
            _PositiveInt.model_validate({"n": -1})  # raises ValidationError
            raise AssertionError("unreachable")  # pragma: no cover
        return super().model_validate(data, **kwargs)


class _UnconvertedJobConfig(_FakeJobConfig):
    """JobConfig stand-in that undoes the per-device to total batch conversion.

    ado's rule rejects ``batch_size % number_gpus != 0``. The converted batch,
    ``per_device x total_gpus``, always divides by ``total_gpus``, so the rule is reached by
    putting the per-device batch back on the row.
    """

    @classmethod
    def model_validate(cls, data: Any, **kwargs: Any) -> "_UnconvertedJobConfig":
        undone = {**data, "batch_size": data["batch_size"] // data["number_gpus"]}
        return super().model_validate(undone, **kwargs)


class _FakePredictor:
    """Stand-in for the AutoGluon ``TabularPredictor``.

    ``predict`` is row-wise and deterministic, so one-row and N-row frames agree, as measured for
    the real 3.1.0 model. Every frame passed to it is recorded.
    """

    def __init__(self, model_version: str | None = None, fail_on_batch: bool = False) -> None:
        self.model_version = model_version
        self.fail_on_batch = fail_on_batch
        self.frames: list[pd.DataFrame] = []

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        self.frames.append(frame.copy())
        if self.fail_on_batch and len(frame) > 1:
            raise RuntimeError("autogluon refused a multi-row frame")
        return pd.Series([self._verdict(row) for _, row in frame.iterrows()], index=frame.index)

    @staticmethod
    def _verdict(row: pd.Series) -> int:
        return 1 if row["batch_size"] * row["tokens_per_sample"] <= _CLASSIFIER_CEILING else 0

    @property
    def rows_seen(self) -> list[int]:
        """Rows per predict call: [N] when batched, [1, 1, ...] when not."""
        return [len(frame) for frame in self.frames]

    def batches_of(self, column: str) -> list[list[Any]]:
        return [list(frame[column]) for frame in self.frames]


def _is_row_valid(config: Any, err_prefix: str = "Rule-based classifier error: ") -> tuple[bool, list[str]]:
    """Copy of ado's ``is_row_valid`` and ``to_series``.

    The one-row check keeps the rule stage per row: a whole frame raises ``ValueError``.
    """
    if len(config) != 1:
        raise ValueError(f"DataFrame must have exactly 1 row, got {len(config)}")
    row = config.iloc[0]
    errors: list[str] = []
    if row["batch_size"] % row["number_gpus"] != 0:
        errors.append(err_prefix + "batch_size must be evenly divisible by number_gpus.")
    return len(errors) == 0, errors


def _ado_prediction_and_metadata(config: Any, predictor: _FakePredictor) -> tuple[int, dict[str, Any]]:
    """Copy of ado's ``get_model_prediction_and_metadata``, the per-row reference for batching."""
    frame = pd.DataFrame([config.model_dump()], index=[0])
    metadata: dict[str, Any] = {}
    pred: Any = None
    machine_learning_classifier_error: Optional[str] = None
    row_is_valid, rule_based_classifier_error = _is_row_valid(frame)
    if int(row_is_valid) == 1:
        try:
            pred = predictor.predict(frame).values[0]
        except Exception as exc:
            machine_learning_classifier_error = str(exc)
    metadata["Rule-Based Classifier error"] = " ".join(rule_based_classifier_error)
    metadata["Predictive Model Classifier error"] = machine_learning_classifier_error
    pred = int(pred) if pred else 0
    return pred, metadata


def _mods(predictor: _FakePredictor, job_config_cls: type = _FakeJobConfig, load_calls: list | None = None):
    """A (load_model, JobConfig, get_model_prediction_and_metadata) triple for ``_autoconf_modules``."""

    def load_model(model_version: str) -> _FakePredictor:
        if load_calls is not None:
            load_calls.append(model_version)
        return predictor

    return load_model, job_config_cls, _ado_prediction_and_metadata


@pytest.fixture(autouse=True)
def _hermetic_autoconf(monkeypatch):
    """Fake the rule module ``check_chunk`` imports at call time, and clear the batching opt-out.

    This keeps the tests independent of an installed ADO and of the environment.
    """
    module = types.ModuleType("autoconf.utils.rule_based_classifier")
    module.is_row_valid = _is_row_valid  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "autoconf.utils.rule_based_classifier", module)
    monkeypatch.delenv("COASTLINE_NO_AUTOCONF_BATCH", raising=False)


def _install(monkeypatch, predictor: _FakePredictor, job_config_cls: type = _FakeJobConfig) -> None:
    monkeypatch.setattr(af, "_autoconf_modules", lambda: _mods(predictor, job_config_cls))


def _batchable() -> AutoconfFeasibilityChecker:
    return AutoconfFeasibilityChecker(model_version=DEFAULT_AUTOCONF_MODEL_VERSION)


def _chain() -> _RulesThenAutoconfChecker:
    return _RulesThenAutoconfChecker(model_version=DEFAULT_AUTOCONF_MODEL_VERSION)


# Batched output equals per-row output.
def test_batched_verdicts_and_metadata_equal_the_per_row_path(monkeypatch):
    """``check_chunk`` returns the same (verdict, metadata) pairs, in order, as ``is_feasible`` per row."""
    batched_predictor, per_row_predictor = _FakePredictor(), _FakePredictor()
    workloads = _chunk()

    _install(monkeypatch, per_row_predictor)
    expected = [_batchable().is_feasible(workload) for workload in workloads]

    _install(monkeypatch, batched_predictor)
    actual = _batchable().check_chunk(workloads)

    assert actual == expected
    assert [verdict for verdict, _ in actual] == [True, False, True, True, False]


def test_the_batched_metadata_uses_ado_s_two_classifier_error_keys(monkeypatch):
    """Each candidate carries ado's two metadata keys as spelled in ``autoconf/utils/recommender.py``.

    Downstream code and recorded traces index on these keys. A clean candidate has an empty rule
    string and a None model error.
    """
    _install(monkeypatch, _FakePredictor())

    results = _batchable().check_chunk(_chunk())

    for _, metadata in results:
        assert set(metadata) == {"Rule-Based Classifier error", "Predictive Model Classifier error"}
    assert results[0][1] == {"Rule-Based Classifier error": "", "Predictive Model Classifier error": None}


def test_the_whole_chunk_is_decided_by_one_classifier_call(monkeypatch):
    """N candidates reach the classifier in one predict call with N rows.

    A per-row loop would give the same verdicts, so the test checks the call log.
    """
    predictor = _FakePredictor()
    _install(monkeypatch, predictor)
    workloads = _chunk()

    _batchable().check_chunk(workloads)

    assert predictor.rows_seen == [len(workloads)]


# Rejected candidates stay out of the classifier.
def test_an_invalid_job_config_is_rejected_exactly_as_the_per_row_path_does(monkeypatch):
    """A candidate whose JobConfig fails validation gets the per-row ``invalid_job_config:`` error,
    keeps its position and stays out of the classifier frame."""
    workloads = _chunk()
    workloads.insert(2, _workload(llm_model=_NOT_A_JOB))
    batched_predictor, per_row_predictor = _FakePredictor(), _FakePredictor()

    _install(monkeypatch, per_row_predictor, _SelectiveJobConfig)
    expected = [_batchable().is_feasible(workload) for workload in workloads]

    _install(monkeypatch, batched_predictor, _SelectiveJobConfig)
    actual = _batchable().check_chunk(workloads)

    assert actual == expected
    assert actual[2][0] is False
    assert actual[2][1]["error"].startswith("invalid_job_config:")
    # The classifier saw the other five candidates and only those.
    assert batched_predictor.rows_seen == [len(workloads) - 1]
    assert _NOT_A_JOB not in batched_predictor.batches_of("model_name")[0]


def test_a_rule_invalid_candidate_skips_the_classifier_and_carries_the_rule_error(monkeypatch):
    """A row that fails ado's divisibility rule is infeasible, carries the rule error and a None
    model error, and stays out of the classifier frame. The results match ``is_feasible``.
    """
    workloads = _chunk()  # per-device batches 1, 64, 4, 16, 32 on 8 GPUs
    batched_predictor, per_row_predictor = _FakePredictor(), _FakePredictor()

    _install(monkeypatch, per_row_predictor, _UnconvertedJobConfig)
    expected = [_batchable().is_feasible(workload) for workload in workloads]

    _install(monkeypatch, batched_predictor, _UnconvertedJobConfig)
    actual = _batchable().check_chunk(workloads)

    assert actual == expected
    # 1 % 8 and 4 % 8 are non-zero; 64, 16 and 32 are multiples of 8.
    for position in (0, 2):
        assert actual[position] == (
            False,
            {"Rule-Based Classifier error": _RULE_ERROR, "Predictive Model Classifier error": None},
        )
    assert batched_predictor.rows_seen == [3]
    assert batched_predictor.batches_of("batch_size")[0] == [64, 16, 32]


# Failure handling.
def test_a_failing_batched_predict_falls_back_to_the_per_row_path(monkeypatch):
    """When the batched predict raises, the chunk is re-run one candidate at a time.

    A batch failure does not say which row caused it, so each candidate gets its own verdict.
    """
    predictor = _FakePredictor(fail_on_batch=True)
    workloads = _chunk()

    _install(monkeypatch, predictor)
    expected = [_batchable().is_feasible(workload) for workload in workloads]
    per_row_frames = len(predictor.frames)

    actual = _batchable().check_chunk(workloads)

    assert actual == expected
    assert len({verdict for verdict, _ in actual}) == 2  # both True and False appear
    # One rejected batch of 5, then a one-row call per candidate.
    assert predictor.rows_seen[per_row_frames:] == [len(workloads), 1, 1, 1, 1, 1]


def test_a_model_that_will_not_load_marks_the_whole_chunk_infeasible(monkeypatch):
    """A model load failure gives every candidate (False, {"error": <message>}) and does not raise."""

    def exploding_load_model(model_version: str):
        raise RuntimeError("model artifacts missing")

    monkeypatch.setattr(
        af, "_autoconf_modules", lambda: (exploding_load_model, _FakeJobConfig, _ado_prediction_and_metadata)
    )
    workloads = _chunk()

    results = _batchable().check_chunk(workloads)

    assert results == [(False, {"error": "model artifacts missing"})] * len(workloads)


def test_an_unavailable_autoconf_marks_every_candidate_in_the_chunk(monkeypatch):
    """Without the ADO modules, every candidate gets (False, {"error": "autoconf_unavailable"}), as
    with ``is_feasible``."""
    monkeypatch.setattr(af, "_autoconf_modules", lambda: None)
    workloads = _chunk()

    results = _batchable().check_chunk(workloads)

    assert results == [(False, {"error": "autoconf_unavailable"})] * len(workloads)


# The batching gate: batching was measured per model version and can be switched off.
def test_a_non_batchable_model_version_falls_back_to_one_call_per_candidate(monkeypatch):
    """A model version outside ``BATCHABLE_AUTOCONF_MODEL_VERSIONS`` runs one predict per
    candidate and gives the per-row verdicts."""
    other_version = "2.0.0"
    assert other_version not in BATCHABLE_AUTOCONF_MODEL_VERSIONS

    batched_predictor, per_row_predictor = _FakePredictor(), _FakePredictor()
    workloads = _chunk()

    _install(monkeypatch, per_row_predictor)
    expected = [AutoconfFeasibilityChecker(model_version=other_version).is_feasible(w) for w in workloads]

    _install(monkeypatch, batched_predictor)
    checker = AutoconfFeasibilityChecker(model_version=other_version)
    assert checker._can_batch() is False

    assert checker.check_chunk(workloads) == expected
    assert batched_predictor.rows_seen == [1] * len(workloads)


def test_the_opt_out_environment_variable_disables_batching(monkeypatch):
    """With ``COASTLINE_NO_AUTOCONF_BATCH=1`` set after import, the batchable version runs one
    predict per candidate and gives the same verdicts."""
    batched_predictor, per_row_predictor = _FakePredictor(), _FakePredictor()
    workloads = _chunk()

    _install(monkeypatch, per_row_predictor)
    expected = [_batchable().is_feasible(workload) for workload in workloads]

    monkeypatch.setenv("COASTLINE_NO_AUTOCONF_BATCH", "1")
    _install(monkeypatch, batched_predictor)
    checker = _batchable()
    assert checker._can_batch() is False

    assert checker.check_chunk(workloads) == expected
    assert batched_predictor.rows_seen == [1] * len(workloads)


# Per-device to total batch conversion in the batched path.
def test_the_batched_frame_carries_the_total_batch_not_the_per_device_batch(monkeypatch):
    """The batched frame carries the total batch: ``WorkloadSpec.batch_size`` is per device and
    AutoConf's ``JobConfig.batch_size`` is the total. 4 GPUs/node x 2 nodes = 8 GPUs, so a
    per-device batch of 16 becomes 128.
    """
    predictor = _FakePredictor()
    _install(monkeypatch, predictor)
    workload = _workload(batch_size=16, tokens_per_sample=512, gpus_per_node=4, number_of_nodes=2)

    _batchable().check_chunk([workload])

    row = predictor.frames[0].iloc[0]
    assert dict(row) == {
        "model_name": "mistral-7b-v0.1",
        "method": "full",
        "gpu_model": _GPU,
        "tokens_per_sample": 512,
        "batch_size": 128,  # per-device 16 x 8 GPUs
        "is_valid": None,
        "number_gpus": 8,  # 4 GPUs/node x 2 nodes
    }
    assert row["batch_size"] == workload.batch_size * workload.total_gpus != workload.batch_size


def test_every_batched_row_stays_divisible_so_ado_s_rule_cannot_fire(monkeypatch):
    """Every row sent to the classifier has ``batch_size`` divisible by ``number_gpus``, so ado's
    rule cannot fire."""
    predictor = _FakePredictor()
    _install(monkeypatch, predictor)

    _batchable().check_chunk(_chunk())

    frame = predictor.frames[0]
    assert len(frame) == 5
    assert all(row["batch_size"] % row["number_gpus"] == 0 for _, row in frame.iterrows())


# The chain wrapper: rules first, then one classifier call for the survivors.
def test_the_chain_batches_the_survivors_and_matches_its_own_per_row_path(monkeypatch):
    """``_RulesThenAutoconfChecker.check_chunk`` matches its own ``is_feasible`` and makes one
    AutoConf predict call. The pipeline holds this wrapper, so it has to pass batching through.
    """
    batched_predictor, per_row_predictor = _FakePredictor(), _FakePredictor()
    workloads = _chunk()

    _install(monkeypatch, per_row_predictor)
    expected = [_chain().is_feasible(workload) for workload in workloads]

    _install(monkeypatch, batched_predictor)
    actual = _chain().check_chunk(workloads)

    assert actual == expected
    assert batched_predictor.rows_seen == [len(workloads)]


def test_the_chain_keeps_a_rules_rejected_candidate_away_from_the_classifier(monkeypatch):
    """A candidate the rules reject keeps its position and the rules' error, and stays out of the
    classifier frame.

    ``WorkloadSpec`` refuses a per-device batch of 0, so the test sets it after construction.
    """
    predictor = _FakePredictor()
    _install(monkeypatch, predictor)
    workloads = _chunk()
    workloads[1].batch_size = 0

    results = _chain().check_chunk(workloads)

    assert len(results) == len(workloads)
    assert results[1] == (False, {"error": "batch_size must be >= 1 (per-device)"})
    assert predictor.rows_seen == [len(workloads) - 1]
    assert 0 not in predictor.batches_of("batch_size")[0]


def test_evaluate_chunk_prefers_the_batched_path_and_still_matches_per_row(monkeypatch):
    """``evaluate_chunk`` uses ``check_chunk`` when the checker has it, with the same result as
    calling ``is_feasible`` per candidate."""
    batched_predictor, per_row_predictor = _FakePredictor(), _FakePredictor()
    workloads = _chunk()

    _install(monkeypatch, per_row_predictor)
    expected = [_batchable().is_feasible(workload) for workload in workloads]

    _install(monkeypatch, batched_predictor)
    actual = evaluate_chunk(_batchable(), workloads)

    assert actual == expected
    assert batched_predictor.rows_seen == [len(workloads)]


# The guard wrapper: veto first, then the backend, batched or not.
class _SinglesOnlyBackend:
    """Backend without ``check_chunk``.

    Its metadata reuses the guard's ``guard`` key, so the merge order is visible.
    """

    EXPENSIVE = True

    def __init__(self) -> None:
        self.singles: list[WorkloadSpec] = []
        self.chunks: list[list[WorkloadSpec]] = []

    def is_feasible(self, workload: WorkloadSpec) -> tuple[bool, dict[str, Any]]:
        self.singles.append(workload)
        return True, dict(_SPY_METADATA)


class _SpyBackend(_SinglesOnlyBackend):
    """The same backend with ``check_chunk``."""

    def check_chunk(self, workloads) -> list[tuple[bool, dict[str, Any]]]:
        self.chunks.append(list(workloads))
        return [(True, dict(_SPY_METADATA)) for _ in workloads]


def _guarded(backend: Any, threshold: int = EMPIRICAL_OOM_TOKEN_BUDGET) -> GuardedFeasibilityChecker:
    return GuardedFeasibilityChecker(TokenBudgetFeasibilityChecker(threshold), backend)


def test_the_guard_merges_metadata_exactly_as_its_is_feasible_does(monkeypatch):
    """``GuardedFeasibilityChecker.check_chunk`` matches its own ``is_feasible`` per candidate, with
    the guard's and AutoConf's metadata keys merged."""
    batched_predictor, per_row_predictor = _FakePredictor(), _FakePredictor()
    workloads = _chunk()

    _install(monkeypatch, per_row_predictor)
    expected = [_guarded(_batchable()).is_feasible(workload) for workload in workloads]

    _install(monkeypatch, batched_predictor)
    actual = _guarded(_batchable()).check_chunk(workloads)

    assert actual == expected
    # A candidate the guard passed carries both sets of keys.
    assert set(actual[0][1]) == {
        "guard",
        "tokens_per_device",
        "token_budget_threshold",
        "Rule-Based Classifier error",
        "Predictive Model Classifier error",
    }
    assert actual[0][1]["tokens_per_device"] == 1 * 512


def test_the_backend_wins_a_metadata_key_clash_in_both_paths(monkeypatch):
    """Both paths merge ``{**guard_metadata, **backend_metadata}``, so the backend's value wins the
    shared ``guard`` key."""
    workloads = [_workload(batch_size=1, tokens_per_sample=512)]

    per_row = _guarded(_SpyBackend()).is_feasible(workloads[0])
    batched = _guarded(_SpyBackend()).check_chunk(workloads)

    assert batched == [per_row]
    assert batched[0][1]["guard"] == "backend-wins"
    assert batched[0][1]["tokens_per_device"] == 512  # the guard's other keys are kept


def test_a_guard_vetoed_candidate_never_reaches_the_backend(monkeypatch):
    """A candidate the token guard vetoes stays out of the backend batch and carries only guard
    metadata. The fourth candidate (16 x 4096 = 65,536 tokens/device, over 60,224) is one the spy
    backend would pass.
    """
    backend = _SpyBackend()
    workloads = _chunk()

    results = _guarded(backend).check_chunk(workloads)

    assert results[3][0] is False
    assert "backend" not in results[3][1]
    assert results[3][1]["tokens_per_device"] == 16 * 4096 > EMPIRICAL_OOM_TOKEN_BUDGET
    assert backend.chunks == [[workloads[0], workloads[2]]]
    assert backend.singles == []
    # Every survivor was decided by the backend, in its original slot.
    assert [verdict for verdict, _ in results] == [True, False, True, False, False]


def test_the_guard_falls_back_to_is_feasible_for_a_backend_without_check_chunk(monkeypatch):
    """For a backend without ``check_chunk`` (``rules``, ``none``), the guard calls ``is_feasible``
    once per candidate it did not veto."""
    backend = _SinglesOnlyBackend()
    workloads = _chunk()

    results = _guarded(backend).check_chunk(workloads)

    assert backend.chunks == []
    assert backend.singles == [workloads[0], workloads[2]]
    assert [verdict for verdict, _ in results] == [True, False, True, False, False]
