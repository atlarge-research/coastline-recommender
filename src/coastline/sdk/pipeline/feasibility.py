"""Feasibility checker factory."""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol

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


class _RulesThenAutoconfChecker:
    """Divisibility rules first, then AutoConf OOM classifier. Rules guard configs the classifier never trained on."""

    def __init__(self, model_version: str):
        self._rules = RulesFeasibilityChecker()
        self._autoconf = AutoconfFeasibilityChecker(model_version=model_version)

    def is_feasible(self, workload: WorkloadSpec) -> tuple[bool, dict[str, Any]]:
        ok, meta = self._rules.is_feasible(workload)
        if not ok:
            return ok, meta
        return self._autoconf.is_feasible(workload)


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
