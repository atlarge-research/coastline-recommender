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

    A free function, NOT a Protocol default: ``FeasibilityChecker`` is satisfied structurally and
    no implementer inherits from it, so a method body added to the Protocol would reach none of
    them. A checker may offer ``check_chunk`` to evaluate a whole run at once (the AutoConf
    backend batches its classifier call that way); otherwise each candidate is checked in turn,
    which is exactly today's behaviour.
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
    """Whether one call costs enough to be worth shipping to a worker process.

    Strictly ``True``: a checker that happens to carry a truthy attribute of some other shape
    has not declared anything, and defaulting it to "fork me" would be the wrong way round.
    """
    return getattr(checker, "EXPENSIVE", False) is True


class _RulesThenAutoconfChecker:
    """Divisibility rules first, then AutoConf OOM classifier. Rules guard configs the classifier never trained on."""

    def __init__(self, model_version: str):
        self._rules = RulesFeasibilityChecker()
        self._autoconf = AutoconfFeasibilityChecker(model_version=model_version)

    @property
    def EXPENSIVE(self) -> bool:  # noqa: N802 — mirrors the class-level flag on plain checkers
        """Worth forking exactly when the AutoConf leg is: the rules leg is a modulo."""
        return bool(getattr(self._autoconf, "EXPENSIVE", False))

    def is_feasible(self, workload: WorkloadSpec) -> tuple[bool, dict[str, Any]]:
        ok, meta = self._rules.is_feasible(workload)
        if not ok:
            return ok, meta
        return self._autoconf.is_feasible(workload)

    def batches(self) -> bool:
        """One classifier call per chunk exactly when the AutoConf leg can batch."""
        return bool(self._autoconf.batches())

    def check_chunk(self, workloads: Sequence[WorkloadSpec]) -> list[tuple[bool, dict[str, Any]]]:
        """Rules per candidate, then ONE classifier call for everything the rules let through."""
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
    """Layer the empirical per-device token ceiling on top of the selected backend.

    Opt-in (``predictors.empirical_oom_guard: true``), because it is a blunt instrument derived
    from one cluster's campaigns: it only ever turns feasible into infeasible, and enabling it by
    default would silently change every recommendation. See EMPIRICAL_OOM_TOKEN_BUDGET.
    """
    if not predictor_config.get("empirical_oom_guard", False):
        return checker
    threshold = int(predictor_config.get("empirical_oom_token_budget", EMPIRICAL_OOM_TOKEN_BUDGET))
    logger.info("Empirical OOM guard enabled at %d tokens/device", threshold)
    return GuardedFeasibilityChecker(TokenBudgetFeasibilityChecker(threshold), checker)


def create_feasibility_checker(predictor_config: dict) -> FeasibilityChecker:
    """Build feasibility checker from config (predictors.feasibility: autoconf|rules|none).

    ``predictors.empirical_oom_guard: true`` additionally layers the measured per-device token
    ceiling over whichever backend is selected; it is off by default.
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
            "Set COASTLINE_ALLOW_RULES_FALLBACK=1 to knowingly degrade to divisibility-only rules."
        )

    if mode == FeasibilityMode.RULES:
        return _wrap_with_empirical_guard(RulesFeasibilityChecker(), predictor_config)

    if mode == FeasibilityMode.NONE:
        return _wrap_with_empirical_guard(NoOpFeasibilityChecker(), predictor_config)

    # A typo (e.g. "Autoconf", "auto-conf", "strict") must fail loudly rather than fall
    # through to the rules checker and silently bypass the OOM veto.
    raise ValueError(f"unknown feasibility mode {mode!r}: expected one of {[m.value for m in FeasibilityMode]}")
