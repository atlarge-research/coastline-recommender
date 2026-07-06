"""Feasibility checker factory."""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol

from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.feasibility.autoconf import (
    AutoconfFeasibilityChecker,
    NoOpFeasibilityChecker,
    RulesFeasibilityChecker,
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


_VALID_FEASIBILITY_MODES = ("autoconf", "rules", "none")


def create_feasibility_checker(predictor_config: dict) -> FeasibilityChecker:
    """Build feasibility checker from config (predictors.feasibility: autoconf|rules|none)."""
    mode = predictor_config.get("feasibility", "autoconf")
    version = predictor_config.get("autoconf_model_version", "3.1.0")

    if mode == "autoconf":
        if AutoconfFeasibilityChecker.available():
            return _RulesThenAutoconfChecker(model_version=version)
        if os.environ.get("COASTLINE_ALLOW_RULES_FALLBACK") == "1":
            logger.warning(
                "AutoConf requested but unavailable; falling back to rules (COASTLINE_ALLOW_RULES_FALLBACK=1)"
            )
            return RulesFeasibilityChecker()
        raise RuntimeError(
            "feasibility=autoconf requested but the AutoConf model cannot be loaded "
            "(needs Python >= 3.10 and the ado autoconf package: "
            "pip install 'coastline-recommender[autoconf]'). "
            "Set COASTLINE_ALLOW_RULES_FALLBACK=1 to knowingly degrade to divisibility-only rules."
        )

    if mode == "rules":
        return RulesFeasibilityChecker()

    if mode == "none":
        return NoOpFeasibilityChecker()

    # A typo (e.g. "Autoconf", "auto-conf", "strict") used to fall through to the
    # divisibility-only rules checker, silently bypassing the OOM veto. Fail loudly.
    raise ValueError(
        f"unknown feasibility mode {mode!r}: expected one of {', '.join(repr(m) for m in _VALID_FEASIBILITY_MODES)}"
    )
