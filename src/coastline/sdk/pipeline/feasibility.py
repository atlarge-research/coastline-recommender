"""Feasibility checker factory."""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol, Sequence

from coastline.sdk.constants import (
    DEFAULT_AUTOCONF_MODEL_VERSION,
    EMPIRICAL_OOM_TOKEN_BUDGET,
    FeasibilityMode,
)
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.feasibility.autoconf import (
    AutoconfFeasibilityChecker,
    NoOpFeasibilityChecker,
    RulesFeasibilityChecker,
)
from coastline.sdk.predictors.feasibility.token_budget import (
    GuardedFeasibilityChecker,
    TokenBudgetFeasibilityChecker,
)

logger = logging.getLogger(__name__)


class FeasibilityChecker(Protocol):
    def is_feasible(self, workload: WorkloadSpec) -> tuple[bool, dict[str, Any]]: ...


def evaluate_chunk(checker: FeasibilityChecker, workloads: Sequence[WorkloadSpec]) -> list[tuple[bool, dict[str, Any]]]:
    """Verdicts for a run of candidates, in input order.

    A checker may provide ``check_chunk`` to judge the whole run at once (the AutoConf backend
    batches its classifier this way); otherwise each candidate is checked in turn. This is a
    function because checkers match ``FeasibilityChecker`` structurally and do not inherit
    Protocol methods.
    """
    batched = getattr(checker, "check_chunk", None)
    if batched is not None:
        verdicts = batched(workloads)
        if len(verdicts) != len(workloads):  # pragma: no cover - defensive
            raise RuntimeError(
                f"{type(checker).__name__}.check_chunk returned {len(verdicts)} verdicts "
                f"for {len(workloads)} candidates"
            )
        return list(verdicts)
    return [checker.is_feasible(workload) for workload in workloads]


def is_expensive(checker: FeasibilityChecker) -> bool:
    """Whether one call costs enough to send to a worker process.

    Only ``EXPENSIVE is True`` counts; any other value, truthy or not, means no.
    """
    return getattr(checker, "EXPENSIVE", False) is True


class _RulesThenAutoconfChecker:
    """Structural sanity guards first, then the AutoConf OOM classifier.

    The guards are the two checks of :class:`RulesFeasibilityChecker` (a positive GPU count and a
    per-device batch of at least 1). They have no memory model; they keep invalid jobs, which the
    classifier was not trained on, away from it. The empirical per-device token budget is a
    separate, opt-in layer (see :func:`_wrap_with_empirical_guard`).
    """

    def __init__(self, model_version: str):
        self._rules = RulesFeasibilityChecker()
        self._autoconf = AutoconfFeasibilityChecker(model_version=model_version)

    @property
    def EXPENSIVE(self) -> bool:  # noqa: N802 (same name as the class attribute of other checkers)
        """Expensive when the AutoConf checker is; the guards are two integer comparisons."""
        return bool(getattr(self._autoconf, "EXPENSIVE", False))

    def is_feasible(self, workload: WorkloadSpec) -> tuple[bool, dict[str, Any]]:
        ok, meta = self._rules.is_feasible(workload)
        if not ok:
            return ok, meta
        return self._autoconf.is_feasible(workload)

    def batches(self) -> bool:
        """Whether the AutoConf checker judges a chunk in one call."""
        return bool(self._autoconf.batches())

    def check_chunk(self, workloads: Sequence[WorkloadSpec]) -> list[tuple[bool, dict[str, Any]]]:
        """Guards per candidate, then one classifier call for all candidates that pass them."""
        results: list[Any] = [None] * len(workloads)
        survivors: list[WorkloadSpec] = []
        positions: list[int] = []
        for position, workload in enumerate(workloads):
            ok, meta = self._rules.is_feasible(workload)
            if ok:
                survivors.append(workload)
                positions.append(position)
            else:
                results[position] = (ok, meta)
        verdicts = self._autoconf.check_chunk(survivors)
        if len(verdicts) != len(survivors):  # pragma: no cover - the backend asserts this itself
            raise RuntimeError(f"AutoConf returned {len(verdicts)} verdicts for {len(survivors)} rule-valid candidates")
        for position, verdict in zip(positions, verdicts):
            results[position] = verdict
        return results


def _wrap_with_empirical_guard(checker: FeasibilityChecker, predictor_config: dict) -> FeasibilityChecker:
    """Add the empirical per-device token ceiling on top of the selected backend.

    Off unless ``predictors.empirical_oom_guard: true``: the ceiling was fitted on one cluster's
    campaigns and can only turn feasible candidates infeasible, so turning it on by default would
    change recommendations. See EMPIRICAL_OOM_TOKEN_BUDGET.
    """
    if not predictor_config.get("empirical_oom_guard", False):
        return checker
    threshold = int(predictor_config.get("empirical_oom_token_budget", EMPIRICAL_OOM_TOKEN_BUDGET))
    logger.info("Empirical OOM guard enabled at %d tokens/device", threshold)
    return GuardedFeasibilityChecker(TokenBudgetFeasibilityChecker(threshold), checker)


def create_feasibility_checker(predictor_config: dict) -> FeasibilityChecker:
    """Build the feasibility checker from the config (predictors.feasibility: autoconf|rules|none).

    ``predictors.empirical_oom_guard: true`` adds the measured per-device token ceiling on top of
    the selected backend; it is off by default.
    """
    mode = predictor_config.get("feasibility", FeasibilityMode.AUTOCONF.value)
    version = predictor_config.get("autoconf_model_version", DEFAULT_AUTOCONF_MODEL_VERSION)

    if mode == FeasibilityMode.AUTOCONF:
        if AutoconfFeasibilityChecker.available():
            return _wrap_with_empirical_guard(_RulesThenAutoconfChecker(model_version=version), predictor_config)
        if os.environ.get("COASTLINE_ALLOW_RULES_FALLBACK") == "1":
            logger.warning(
                "AutoConf requested but unavailable; falling back to rules (COASTLINE_ALLOW_RULES_FALLBACK=1)"
            )
            return _wrap_with_empirical_guard(RulesFeasibilityChecker(), predictor_config)
        raise RuntimeError(
            "feasibility=autoconf requested but the AutoConf model cannot be loaded "
            "(needs Python >= 3.10 and the ado autoconf package: "
            "pip install 'coastline-recommender[autoconf]'). "
            "Set COASTLINE_ALLOW_RULES_FALLBACK=1 to knowingly degrade to the rules backend: "
            "structural sanity guards only (positive GPU count, per-device batch >= 1), "
            "no memory model and no OOM check."
        )

    if mode == FeasibilityMode.RULES:
        return _wrap_with_empirical_guard(RulesFeasibilityChecker(), predictor_config)

    if mode == FeasibilityMode.NONE:
        return _wrap_with_empirical_guard(NoOpFeasibilityChecker(), predictor_config)

    # An unknown mode (such as "Autoconf", "auto-conf" or "strict") raises, so a typo cannot
    # skip the OOM check.
    raise ValueError(f"unknown feasibility mode {mode!r}: expected one of {[m.value for m in FeasibilityMode]}")
