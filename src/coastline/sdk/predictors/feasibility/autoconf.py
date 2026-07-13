"""AutoConf feasibility checker — wraps ADO autoconf validity classifier (lazy-loaded)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

from coastline.sdk.constants import DEFAULT_AUTOCONF_MODEL_VERSION
from coastline.sdk.models.workload import WorkloadSpec

logger = logging.getLogger(__name__)

# The installed ``ado-autoconf`` package is the normal import path; these paths only help a
# source checkout of ADO. ``ADO_ROOT`` overrides; else guess the sibling ``../ado`` of the
# coastline repo in the dev superproject (feasibility/ -> predictors -> sdk -> coastline -> src
# -> repo -> superproject == parents[6]).
_ADO_ROOT = Path(os.environ.get("ADO_ROOT") or Path(__file__).resolve().parents[6] / "ado")
_ADO_AUTOCONF_PARENT = _ADO_ROOT / "plugins" / "custom_experiments" / "autoconf"

_AUTOCONF_AVAILABLE: Optional[bool] = None


def _autoconf_modules():
    """Import ADO autoconf on demand (memoizes success only, so a transient import failure
    retries); None if unavailable."""
    global _AUTOCONF_AVAILABLE
    for path in (str(_ADO_ROOT), str(_ADO_AUTOCONF_PARENT)):
        if path not in sys.path:
            sys.path.insert(0, path)

    try:
        from autoconf.min_gpu_recommender import load_model
        from autoconf.utils.pydantic_models import JobConfig
        from autoconf.utils.recommender import get_model_prediction_and_metadata

        _AUTOCONF_AVAILABLE = True
        return load_model, JobConfig, get_model_prediction_and_metadata
    except Exception as exc:
        # Failure not memoized -> next call retries (may be transient).
        logger.warning("AutoConf import failed (will retry on next call): %s", exc)
        return None


class AutoconfFeasibilityChecker:
    """Rule + AutoGluon validity check for a single candidate layout."""

    def __init__(self, model_version: str = DEFAULT_AUTOCONF_MODEL_VERSION):
        self.model_version = model_version
        self._predictor: Any = None

    def _ensure_predictor(self) -> Any:
        mods = _autoconf_modules()
        if mods is None:
            raise ImportError("AutoConf is not available. Set predictors.feasibility to 'rules'.")
        load_model, _, _ = mods
        if self._predictor is None:
            logger.info("Loading AutoConf model %s for feasibility", self.model_version)
            self._predictor = load_model(model_version=self.model_version)
        return self._predictor

    def is_feasible(self, workload: WorkloadSpec) -> tuple[bool, dict[str, Any]]:
        mods = _autoconf_modules()
        if mods is None:
            return False, {"error": "autoconf_unavailable"}

        _, JobConfig, get_model_prediction_and_metadata = mods
        try:
            job_config = JobConfig.model_validate(
                {
                    # feasibility_model lets the OOM check use the real model when the perf
                    # predictor uses a proxy; else llm_model.
                    "model_name": workload.feasibility_model or workload.llm_model,
                    "method": workload.fine_tuning_method,
                    "gpu_model": workload.gpu_model,
                    "tokens_per_sample": workload.tokens_per_sample,
                    "batch_size": workload.batch_size,
                    "number_gpus": workload.total_gpus,
                }
            )
        except ValidationError as exc:
            # Invalid JobConfig = a real "not a valid job" reject (debug: the grid legitimately probes such configs).
            logger.debug("AutoConf rejected candidate (invalid JobConfig): %s", exc)
            return False, {"error": f"invalid_job_config: {exc}"}

        try:
            predictor = self._ensure_predictor()
            valid_flag, metadata = get_model_prediction_and_metadata(job_config, predictor)
            return valid_flag == 1, metadata or {}
        except Exception as exc:
            # Load/predict failure != a benign reject — warn (so a broken model is noticed),
            # then treat as infeasible so the grid continues.
            logger.warning(
                "AutoConf prediction failed for %s/%s on %s (treating candidate as infeasible): %s",
                workload.llm_model,
                workload.fine_tuning_method,
                workload.gpu_model,
                exc,
            )
            return False, {"error": str(exc)}

    @staticmethod
    def available() -> bool:
        return _autoconf_modules() is not None


class RulesFeasibilityChecker:
    """Lightweight feasibility without AutoConf (divisibility rule only)."""

    def is_feasible(self, workload: WorkloadSpec) -> tuple[bool, dict[str, Any]]:
        total = workload.total_gpus
        if total < 1:
            return False, {"error": "invalid total_gpus"}
        if workload.batch_size % total != 0:
            return (
                False,
                {"error": "batch_size must be evenly divisible by number_gpus"},
            )
        return True, {}


class NoOpFeasibilityChecker:
    """Accept all candidates (for tests or when feasibility is disabled)."""

    def is_feasible(self, workload: WorkloadSpec) -> tuple[bool, dict[str, Any]]:
        return True, {}
