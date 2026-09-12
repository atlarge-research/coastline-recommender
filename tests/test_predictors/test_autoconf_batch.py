"""Tests for the batched AutoConf feasibility path.

Targets:

* ``sdk/predictors/feasibility/autoconf.py::AutoconfFeasibilityChecker.check_chunk`` — decides a
  whole run of candidates with ONE AutoGluon predict instead of one predict per candidate.
* ``sdk/pipeline/feasibility.py::_RulesThenAutoconfChecker.check_chunk`` and ``evaluate_chunk``.
* ``sdk/predictors/feasibility/token_budget.py::GuardedFeasibilityChecker.check_chunk``.

Batching is only sound if it is *indistinguishable* from the per-row path, so the oracle for
nearly every test below is the per-row path itself: ado's own
``get_model_prediction_and_metadata``, reproduced verbatim from ``autoconf/utils/recommender.py``
in ``_ado_prediction_and_metadata`` — its two metadata keys, its ``" ".join(errors)`` rule string,
its capture of a prediction failure, and its ``pred = int(pred) if pred else 0`` rule. Nothing is
asserted against a value copied back out of ``check_chunk``.

As in ``test_autoconf_feasibility.py``, neither the real AutoGluon model nor the ADO artifacts are
assumed to be present: ``_autoconf_modules`` is monkeypatched with light fakes, and the
``autoconf.utils.rule_based_classifier`` module that the batched path imports at call time is
pre-seeded into ``sys.modules`` with a verbatim copy of ado's rule (``to_series``'s
one-row-only guard included). The tests are hermetic and give the same answer whether or not ADO
is installed.

Two branches cannot be provoked through a real ``WorkloadSpec``, because the boundary that builds
the JobConfig makes them unreachable: a JobConfig that fails validation (WorkloadSpec already
constrains batch and tokens to > 0 and total_gpus to >= 1) and a row that fails ado's
divisibility rule (the per-device -> total conversion sends ``per_device x total_gpus``, divisible
by ``total_gpus`` for every integer pair). Both are still live branches of ``check_chunk``, and
both are reached here with a JobConfig stand-in, exactly as ``test_autoconf_feasibility.py``
reaches the first of them with ``_RaisingJobConfig``.

No production code is modified by these tests.
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

#: ado's rule-based classifier error, string for string (autoconf/utils/rule_based_classifier.py).
_RULE_ERROR = "Rule-based classifier error: batch_size must be evenly divisible by number_gpus."

#: The fake classifier calls a row valid while total_batch x tokens_per_sample stays under this.
#: An arbitrary but row-wise deterministic rule — that is the only property the oracle needs.
_CLASSIFIER_CEILING = 1_000_000

#: Sentinel model name that ``_SelectiveJobConfig`` refuses to validate.
_NOT_A_JOB = "not-a-job"

#: The spy backends' metadata. ``guard`` is deliberately a key the token-budget guard also emits,
#: so which side wins a clash is observable rather than an untested implementation detail.
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
    """Five candidates on 8 GPUs, freshly built on every call.

    Chosen so the chunk is not uniform in any dimension the code could get away with ignoring:
    the fake classifier says valid / invalid / valid / valid / invalid (alternating, so a
    reordered gather is visible); the empirical token guard vetoes the second, fourth and fifth
    (262,144 / 65,536 / 262,144 tokens/device against a 60,224 budget), the fourth being one the
    classifier would have passed; and the per-device batches are a mix of multiples and
    non-multiples of the GPU count.
    """
    return [
        _workload(batch_size=1, tokens_per_sample=512),  # total batch 8    -> 4,096
        _workload(batch_size=64, tokens_per_sample=4096),  # total batch 512  -> 2,097,152
        _workload(batch_size=4, tokens_per_sample=1024),  # total batch 32   -> 32,768
        _workload(batch_size=16, tokens_per_sample=4096),  # total batch 128  -> 524,288
        _workload(batch_size=32, tokens_per_sample=8192),  # total batch 256  -> 2,097,152
    ]


# --------------------------------------------------------------------------- #
# Fakes that stand in for the ADO autoconf modules.
# --------------------------------------------------------------------------- #
class _FakeJobConfig(BaseModel):
    """Stands in for ``autoconf.utils.pydantic_models.JobConfig``.

    A real pydantic model with the production field set and bounds, so ``model_validate`` raises a
    genuine ``ValidationError`` and ``model_dump()`` produces the same row shape the batched path
    feeds to the frame (``is_valid`` default included).
    """

    model_name: str
    method: str
    gpu_model: str
    tokens_per_sample: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    is_valid: Optional[int] = None
    number_gpus: Optional[int] = Field(default=None, ge=1)


class _PositiveInt(BaseModel):
    """A real pydantic model used only to mint a genuine ``ValidationError``."""

    n: int = Field(gt=0)


class _SelectiveJobConfig(_FakeJobConfig):
    """A JobConfig stand-in that rejects exactly one designated candidate.

    ``test_autoconf_feasibility.py`` reaches the ValidationError branch with a stand-in that
    always raises; a chunk needs one bad candidate *among good ones*, since surviving that is the
    bookkeeping under test. Rejecting the candidate whose model is ``_NOT_A_JOB`` does that.
    """

    @classmethod
    def model_validate(cls, data: Any, **kwargs: Any) -> "_SelectiveJobConfig":
        if data["model_name"] == _NOT_A_JOB:
            _PositiveInt.model_validate({"n": -1})  # raises ValidationError
            raise AssertionError("unreachable")  # pragma: no cover
        return super().model_validate(data, **kwargs)


class _UnconvertedJobConfig(_FakeJobConfig):
    """A JobConfig stand-in that undoes the per-device -> total batch conversion.

    ado's rule is ``batch_size % number_gpus != 0`` and the boundary makes it unfireable: it sends
    ``per_device x total_gpus``, divisible by ``total_gpus`` for every integer pair. The rule stage
    is still live — it is precisely what guards a caller that hands AutoConf a per-device batch —
    so this stand-in puts the unconverted batch back on the row to reach it.
    """

    @classmethod
    def model_validate(cls, data: Any, **kwargs: Any) -> "_UnconvertedJobConfig":
        undone = {**data, "batch_size": data["batch_size"] // data["number_gpus"]}
        return super().model_validate(undone, **kwargs)


class _FakePredictor:
    """Stands in for the AutoGluon ``TabularPredictor``.

    ``predict`` is row-wise and deterministic, so a one-row frame and an N-row frame agree by
    construction — the property the real 3.1.0 model was measured to have and the one the batched
    path trades on. Every frame handed to it is recorded, so a test can see exactly what (and how
    much) reached the classifier.
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
        """How many rows each predict call carried — [N] when batched, [1, 1, ...] when not."""
        return [len(frame) for frame in self.frames]

    def batches_of(self, column: str) -> list[list[Any]]:
        return [list(frame[column]) for frame in self.frames]


def _is_row_valid(config: Any, err_prefix: str = "Rule-based classifier error: ") -> tuple[bool, list[str]]:
    """ado's ``is_row_valid`` + ``to_series``, reproduced verbatim.

    The one-row-only guard matters: it is what forces the rule stage to stay per row while the
    classifier call is batched, and a regression that handed it the whole frame must fail loudly
    here rather than silently evaluate row 0 for everyone.
    """
    if len(config) != 1:
        raise ValueError(f"DataFrame must have exactly 1 row, got {len(config)}")
    row = config.iloc[0]
    errors: list[str] = []
    if row["batch_size"] % row["number_gpus"] != 0:
        errors.append(err_prefix + "batch_size must be evenly divisible by number_gpus.")
    return len(errors) == 0, errors


def _ado_prediction_and_metadata(config: Any, predictor: _FakePredictor) -> tuple[int, dict[str, Any]]:
    """ado's ``get_model_prediction_and_metadata``, reproduced verbatim.

    This is the independent oracle: the batched path claims to reproduce this function's verdict
    and metadata for every candidate, so the per-row path is run through the real thing's logic
    rather than through anything the code under test supplies.
    """
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
    """Pin the two things ``check_chunk`` reaches for outside its own module.

    It imports ``autoconf.utils.rule_based_classifier`` at call time (so the import is paid only
    on the batched path); pre-seeding ``sys.modules`` short-circuits that import, so these tests
    neither need ADO installed nor silently pick up a real one that happens to be. Clearing the
    opt-out keeps the batching gate independent of the ambient environment.
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


# --------------------------------------------------------------------------- #
# The guarantee: batched output is indistinguishable from per-row output.
# --------------------------------------------------------------------------- #
def test_batched_verdicts_and_metadata_equal_the_per_row_path(monkeypatch):
    """One predict for the chunk must produce the same (verdict, metadata) pairs, in the same
    order, as ado's own one-predict-per-candidate path.

    Oracle: ``_ado_prediction_and_metadata`` driven per row through ``is_feasible`` — the code
    the batched path replaces. Falsifies if a verdict flips, metadata is dropped or renamed, or
    results come back reordered.
    """
    batched_predictor, per_row_predictor = _FakePredictor(), _FakePredictor()
    workloads = _chunk()

    _install(monkeypatch, per_row_predictor)
    expected = [_batchable().is_feasible(workload) for workload in workloads]

    _install(monkeypatch, batched_predictor)
    actual = _batchable().check_chunk(workloads)

    assert actual == expected
    assert [verdict for verdict, _ in actual] == [True, False, True, True, False]


def test_the_batched_metadata_uses_ado_s_two_classifier_error_keys(monkeypatch):
    """Every candidate carries exactly ado's two metadata keys, spelled as ado spells them.

    Oracle: the literal keys in ``autoconf/utils/recommender.py``. Downstream code and the
    recorded traces index on these strings, so a re-spelling ("rule_based_error") would be a
    silent schema break. A clean candidate has an empty rule string and a None model error.
    """
    _install(monkeypatch, _FakePredictor())

    results = _batchable().check_chunk(_chunk())

    for _, metadata in results:
        assert set(metadata) == {"Rule-Based Classifier error", "Predictive Model Classifier error"}
    assert results[0][1] == {"Rule-Based Classifier error": "", "Predictive Model Classifier error": None}


def test_the_whole_chunk_is_decided_by_one_classifier_call(monkeypatch):
    """The point of the change: N candidates, one predict, N rows in it.

    Oracle: the recorded frames. Falsifies if the batched path quietly degrades to a loop (the
    verdicts would still be right, so only the call log can catch it).
    """
    predictor = _FakePredictor()
    _install(monkeypatch, predictor)
    workloads = _chunk()

    _batchable().check_chunk(workloads)

    assert predictor.rows_seen == [len(workloads)]


# --------------------------------------------------------------------------- #
# Rejects that must never reach the classifier.
# --------------------------------------------------------------------------- #
def test_an_invalid_job_config_is_rejected_exactly_as_the_per_row_path_does(monkeypatch):
    """A candidate whose JobConfig fails validation gets the per-row path's error metadata,
    keeps its position, never reaches the classifier, and does not disturb its neighbours.

    Oracle: ``is_feasible`` on the same candidate, compared whole — same ``invalid_job_config:``
    tag, same pydantic message. Falsifies if the batched path drops the candidate (shortening the
    result list and shifting every later verdict), lets it through to the frame, or reports it
    through the untagged generic-failure branch.
    """
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
    """A row ado's divisibility rule rejects is decided by the rule alone: infeasible, the rule
    error in ``Rule-Based Classifier error``, ``None`` for the model error, and no seat in the
    frame.

    Oracle: ado's own rule string, plus ``is_feasible`` over the same candidates — ado's single-row
    function reaches the identical verdict through ``pred = int(pred) if pred else 0``. Falsifies
    if a rule-rejected row is batched anyway (it would be scored by a classifier ado never
    consults) or if its metadata loses the rule text.
    """
    workloads = _chunk()  # per-device batches 1, 64, 4, 16, 32 on 8 GPUs
    batched_predictor, per_row_predictor = _FakePredictor(), _FakePredictor()

    _install(monkeypatch, per_row_predictor, _UnconvertedJobConfig)
    expected = [_batchable().is_feasible(workload) for workload in workloads]

    _install(monkeypatch, batched_predictor, _UnconvertedJobConfig)
    actual = _batchable().check_chunk(workloads)

    assert actual == expected
    # 1 % 8 and 4 % 8 are non-zero; 64, 16 and 32 divide 8 evenly.
    for position in (0, 2):
        assert actual[position] == (
            False,
            {"Rule-Based Classifier error": _RULE_ERROR, "Predictive Model Classifier error": None},
        )
    assert batched_predictor.rows_seen == [3]
    assert batched_predictor.batches_of("batch_size")[0] == [64, 16, 32]


# --------------------------------------------------------------------------- #
# Failure handling.
# --------------------------------------------------------------------------- #
def test_a_failing_batched_predict_falls_back_to_the_per_row_path(monkeypatch):
    """When the batched predict raises, the chunk is re-run one candidate at a time.

    A batched failure says nothing about which row caused it, so a blanket error for the whole
    chunk would condemn candidates the model can decide perfectly well. Oracle: the per-row path
    over the same candidates — every one must come back with its own verdict, not a shared one.
    Falsifies if the exception escapes, or if the fallback scores the chunk as a block.
    """
    predictor = _FakePredictor(fail_on_batch=True)
    workloads = _chunk()

    _install(monkeypatch, predictor)
    expected = [_batchable().is_feasible(workload) for workload in workloads]
    per_row_frames = len(predictor.frames)

    actual = _batchable().check_chunk(workloads)

    assert actual == expected
    assert len({verdict for verdict, _ in actual}) == 2  # genuinely per-candidate, not one verdict
    # One rejected batch of 5, then one one-row call per candidate.
    assert predictor.rows_seen[per_row_frames:] == [len(workloads), 1, 1, 1, 1, 1]


def test_a_model_that_will_not_load_marks_the_whole_chunk_infeasible(monkeypatch):
    """A load failure is not a verdict about any one candidate, so every candidate in the chunk
    gets the same surfaced error and the grid continues.

    Oracle: ``is_feasible``'s generic-failure contract — (False, {"error": <message>}) — applied
    to each candidate. Falsifies if the load error escapes and aborts the stage.
    """

    def exploding_load_model(model_version: str):
        raise RuntimeError("model artifacts missing")

    monkeypatch.setattr(
        af, "_autoconf_modules", lambda: (exploding_load_model, _FakeJobConfig, _ado_prediction_and_metadata)
    )
    workloads = _chunk()

    results = _batchable().check_chunk(workloads)

    assert results == [(False, {"error": "model artifacts missing"})] * len(workloads)


def test_an_unavailable_autoconf_marks_every_candidate_in_the_chunk(monkeypatch):
    """With the ADO modules unimportable, the chunk degrades exactly as ``is_feasible`` does,
    one verdict per candidate.

    Oracle: the documented graceful-degradation contract, (False, {"error":
    "autoconf_unavailable"}). Falsifies if the chunk returns a single verdict, an empty list, or
    raises ImportError.
    """
    monkeypatch.setattr(af, "_autoconf_modules", lambda: None)
    workloads = _chunk()

    results = _batchable().check_chunk(workloads)

    assert results == [(False, {"error": "autoconf_unavailable"})] * len(workloads)


# --------------------------------------------------------------------------- #
# The batching gate: batching is version-specific evidence, so it must be revocable.
# --------------------------------------------------------------------------- #
def test_a_non_batchable_model_version_falls_back_to_one_call_per_candidate(monkeypatch):
    """Batched == per-row was measured on 3.1.0 only, so any other model version must not batch —
    and must still return correct verdicts.

    Oracle: the per-row path over the same candidates, and the call log (one row per call).
    Falsifies if the gate is dropped (an unmeasured model would be batched) or if turning
    batching off changes an answer.
    """
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
    """``COASTLINE_NO_AUTOCONF_BATCH=1`` is the escape hatch for the batchable model itself; with
    it set, the batchable version still runs one call per candidate and still agrees.

    Oracle: the per-row path, plus the call log. Falsifies if the opt-out is read once at import
    (it would not take effect here), spelled differently, or ignored on the batchable version.
    """
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


# --------------------------------------------------------------------------- #
# The per-device -> total batch boundary, inside the batched path.
# --------------------------------------------------------------------------- #
def test_the_batched_frame_carries_the_total_batch_not_the_per_device_batch(monkeypatch):
    """``WorkloadSpec.batch_size`` is PER-DEVICE and AutoConf's ``JobConfig.batch_size`` is
    TOTAL. The conversion lives at this boundary and the batched path must not skip it.

    Oracle: hand-derived layout — 4 GPUs/node x 2 nodes = 8 total GPUs, effective batch =
    per-device 16 x 8 = 128 — asserted on the row that actually reached the classifier. Getting
    this wrong silently changes every feasibility verdict, so it is pinned on both sides of the
    batching switch.
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
        "batch_size": 128,  # per-device 16 -> effective 16 x 8 GPUs
        "is_valid": None,
        "number_gpus": 8,  # 4 GPUs/node x 2 nodes
    }
    assert row["batch_size"] == workload.batch_size * workload.total_gpus != workload.batch_size


def test_every_batched_row_stays_divisible_so_ado_s_rule_cannot_fire(monkeypatch):
    """The conversion is also what keeps ado's rule-based classifier quiet: ``per_device x
    total_gpus`` is divisible by ``total_gpus`` for every candidate in the chunk.

    Oracle: the divisibility arithmetic on the rows the classifier was handed. This is the reason
    a rule rejection has to be provoked with a stand-in above, and it is the property that would
    break first if the per-device conversion were ever dropped.
    """
    predictor = _FakePredictor()
    _install(monkeypatch, predictor)

    _batchable().check_chunk(_chunk())

    frame = predictor.frames[0]
    assert len(frame) == 5
    assert all(row["batch_size"] % row["number_gpus"] == 0 for _, row in frame.iterrows())


# --------------------------------------------------------------------------- #
# The chain wrapper: rules first, then ONE classifier call for the survivors.
# --------------------------------------------------------------------------- #
def test_the_chain_batches_the_survivors_and_matches_its_own_per_row_path(monkeypatch):
    """``_RulesThenAutoconfChecker.check_chunk`` must agree with its own ``is_feasible``, and
    collapse the AutoConf leg to a single call.

    Oracle: the chain's per-row path over the same candidates, plus the call log. Falsifies if
    the chain loses the batching (the wrapper is what the pipeline actually holds, so hiding
    ``check_chunk`` here would silently undo the whole change).
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
    """A candidate the cheap rules reject keeps its position, keeps the rules' own metadata, and
    never costs a classifier seat.

    A per-device batch of 0 is the rules' own guard (``batch_size must be >= 1``); WorkloadSpec
    forbids it at construction, so it is set afterwards — as ``test_token_budget_guard.py`` does
    for the node count. Oracle: ``RulesFeasibilityChecker``'s documented error, and the frame the
    classifier was handed. Falsifies if rules-rejected candidates are still batched or if the
    rejection is written into the wrong slot.
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
    """The pipeline's dispatcher must pick up ``check_chunk`` when a checker offers it, and the
    result must be what the old per-candidate loop produced.

    Oracle: ``[checker.is_feasible(w) for w in workloads]`` — literally the branch
    ``evaluate_chunk`` takes for a checker without ``check_chunk``. Falsifies if the dispatcher
    ignores the batched method (no speed-up) or mis-handles its return value.
    """
    batched_predictor, per_row_predictor = _FakePredictor(), _FakePredictor()
    workloads = _chunk()

    _install(monkeypatch, per_row_predictor)
    expected = [_batchable().is_feasible(workload) for workload in workloads]

    _install(monkeypatch, batched_predictor)
    actual = evaluate_chunk(_batchable(), workloads)

    assert actual == expected
    assert batched_predictor.rows_seen == [len(workloads)]


# --------------------------------------------------------------------------- #
# The guard wrapper: veto first, then the backend — batched or not.
# --------------------------------------------------------------------------- #
class _SinglesOnlyBackend:
    """A backend with no ``check_chunk`` — the shape every checker had before this change.

    Its metadata deliberately reuses the guard's ``guard`` key, so the merge order is observable.
    """

    EXPENSIVE = True

    def __init__(self) -> None:
        self.singles: list[WorkloadSpec] = []
        self.chunks: list[list[WorkloadSpec]] = []

    def is_feasible(self, workload: WorkloadSpec) -> tuple[bool, dict[str, Any]]:
        self.singles.append(workload)
        return True, dict(_SPY_METADATA)


class _SpyBackend(_SinglesOnlyBackend):
    """The same backend, offering a batched entry point as well."""

    def check_chunk(self, workloads) -> list[tuple[bool, dict[str, Any]]]:
        self.chunks.append(list(workloads))
        return [(True, dict(_SPY_METADATA)) for _ in workloads]


def _guarded(backend: Any, threshold: int = EMPIRICAL_OOM_TOKEN_BUDGET) -> GuardedFeasibilityChecker:
    return GuardedFeasibilityChecker(TokenBudgetFeasibilityChecker(threshold), backend)


def test_the_guard_merges_metadata_exactly_as_its_is_feasible_does(monkeypatch):
    """``GuardedFeasibilityChecker.check_chunk`` must produce, candidate for candidate, what its
    own ``is_feasible`` produces: guard metadata merged under the backend's, backend wins on a
    key clash.

    Oracle: the guard's per-row path over the same candidates, backed by the real AutoConf
    checker so the merged dict carries both the guard's keys and ado's. Falsifies if the chunked
    merge drops the guard's keys, drops the backend's, or merges in the other order.
    """
    batched_predictor, per_row_predictor = _FakePredictor(), _FakePredictor()
    workloads = _chunk()

    _install(monkeypatch, per_row_predictor)
    expected = [_guarded(_batchable()).is_feasible(workload) for workload in workloads]

    _install(monkeypatch, batched_predictor)
    actual = _guarded(_batchable()).check_chunk(workloads)

    assert actual == expected
    # A candidate the guard passed carries both halves.
    assert set(actual[0][1]) == {
        "guard",
        "tokens_per_device",
        "token_budget_threshold",
        "Rule-Based Classifier error",
        "Predictive Model Classifier error",
    }
    assert actual[0][1]["tokens_per_device"] == 1 * 512


def test_the_backend_wins_a_metadata_key_clash_in_both_paths(monkeypatch):
    """``is_feasible`` merges ``{**guard_metadata, **backend_metadata}``, so the backend's value
    for a shared key survives. ``check_chunk`` must merge the same way round.

    Oracle: the spy's ``guard`` entry, which collides with the token-budget guard's own ``guard``
    key. Without a collision the merge is order-insensitive and this ordering would be untested;
    with one, a flipped merge in the chunked path is immediately visible.
    """
    workloads = [_workload(batch_size=1, tokens_per_sample=512)]

    per_row = _guarded(_SpyBackend()).is_feasible(workloads[0])
    batched = _guarded(_SpyBackend()).check_chunk(workloads)

    assert batched == [per_row]
    assert batched[0][1]["guard"] == "backend-wins"
    assert batched[0][1]["tokens_per_device"] == 512  # the guard's non-clashing keys survive


def test_a_guard_vetoed_candidate_never_reaches_the_backend(monkeypatch):
    """The guard is cheap arithmetic in front of an AutoGluon model: a candidate it vetoes must
    not be in the batch at all, and its metadata must be the guard's alone.

    Oracle: three candidates are over the 60,224 budget — the fourth at 16 x 4096 = 65,536
    tokens/device is one the spy backend would have passed — so a leak is visible in the verdict
    as well as in the spy's log. Falsifies if the guard runs after the backend or merely
    overrides its answer.
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
    """Not every checker offers a batched path (``rules``/``none`` do not), so the guard must
    still ask them one candidate at a time — and only about the survivors.

    Oracle: the spy's own log. Falsifies if the guard assumes ``check_chunk`` exists (an
    AttributeError on the default configuration) or asks about a candidate it already vetoed.
    """
    backend = _SinglesOnlyBackend()
    workloads = _chunk()

    results = _guarded(backend).check_chunk(workloads)

    assert backend.chunks == []
    assert backend.singles == [workloads[0], workloads[2]]
    assert [verdict for verdict, _ in results] == [True, False, True, False, False]
