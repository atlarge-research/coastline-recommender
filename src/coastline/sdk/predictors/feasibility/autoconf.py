"""AutoConf feasibility checker — wraps ADO autoconf validity classifier (lazy-loaded)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from pydantic import ValidationError

from coastline.sdk.constants import BATCHABLE_AUTOCONF_MODEL_VERSION, DEFAULT_AUTOCONF_MODEL_VERSION
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

    #: An AutoGluon predict per candidate (~3.3 ms) dominates the cost of shipping the candidate
    #: to a worker process, so this backend is worth forking across candidates.
    EXPENSIVE = True

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

    def _job_config(self, JobConfig: Any, workload: WorkloadSpec) -> Any:
        """The AutoConf JobConfig for one candidate. Raises ValidationError for a non-job."""
        return JobConfig.model_validate(
            {
                # feasibility_model lets the OOM check use the real model when the perf
                # predictor uses a proxy; else llm_model.
                "model_name": workload.feasibility_model or workload.llm_model,
                "method": workload.fine_tuning_method,
                "gpu_model": workload.gpu_model,
                "tokens_per_sample": workload.tokens_per_sample,
                # AutoConf's JobConfig.batch_size is the TOTAL/effective batch — it divides by
                # number_gpus internally (and its rule-based classifier requires divisibility).
                # WorkloadSpec.batch_size is PER-DEVICE, so convert at this boundary:
                # effective = per_device × total_gpus (always divisible by total_gpus).
                "batch_size": workload.batch_size * workload.total_gpus,
                "number_gpus": workload.total_gpus,
            }
        )

    def _can_batch(self) -> bool:
        """Whether one classifier call may decide a whole chunk of candidates.

        Measured on this model: over 3,339 real grid candidates plus 81,928 synthetic ones, a
        batched predict and N one-row predicts agree on every verdict; probabilities drift by at
        most 6e-7 while the candidate nearest the 0.5 decision threshold sits 1.8e-3 away, a
        margin of ~2,900x. That evidence is version-specific, so a different model (or an
        explicit opt-out) falls back to one call per candidate.
        """
        if os.environ.get("COASTLINE_NO_AUTOCONF_BATCH") == "1":
            return False
        return self.model_version == BATCHABLE_AUTOCONF_MODEL_VERSION

    def check_chunk(self, workloads: "Sequence[WorkloadSpec]") -> list[tuple[bool, dict[str, Any]]]:
        """Verdicts for a run of candidates, in input order, with ONE classifier call.

        The AutoGluon predict is ~99.9% of a recommendation and is a vectorised model, so a chunk
        of candidates costs about what one candidate costs. The rule-based classifier cannot be
        batched — ado's ``to_series`` rejects a multi-row frame — so it still runs per row, which
        is cheap (one modulo). Everything else reproduces ado's
        ``get_model_prediction_and_metadata`` exactly, including its metadata keys and its
        ``pred = int(pred) if pred else 0`` rule.
        """
        import pandas as pd

        mods = _autoconf_modules()
        if mods is None:
            return [(False, {"error": "autoconf_unavailable"}) for _ in workloads]
        _, JobConfig, get_model_prediction_and_metadata = mods

        results: list[Optional[tuple[bool, dict[str, Any]]]] = [None] * len(workloads)
        configs: list[Any] = []
        positions: list[int] = []
        for position, workload in enumerate(workloads):
            try:
                configs.append(self._job_config(JobConfig, workload))
                positions.append(position)
            except ValidationError as exc:
                # Invalid JobConfig = a real "not a valid job" reject (debug: the grid
                # legitimately probes such configs).
                logger.debug("AutoConf rejected candidate (invalid JobConfig): %s", exc)
                results[position] = (False, {"error": f"invalid_job_config: {exc}"})
        if not positions:
            return [result for result in results if result is not None]  # type: ignore[misc]

        try:
            predictor = self._ensure_predictor()
        except Exception as exc:  # model unavailable: every candidate in the chunk is undecided
            logger.warning("AutoConf model unavailable (treating candidates as infeasible): %s", exc)
            for position in positions:
                results[position] = (False, {"error": str(exc)})
            return [result for result in results if result is not None]  # type: ignore[misc]

        if not self._can_batch():
            for position, config in zip(positions, configs):
                results[position] = self._decide_one(config, predictor, get_model_prediction_and_metadata)
            return [result for result in results if result is not None]  # type: ignore[misc]

        # Rule stage, per row (ado's is_row_valid takes exactly one row).
        from autoconf.utils.rule_based_classifier import is_row_valid

        rule_errors: dict[int, str] = {}
        batched_positions: list[int] = []
        batched_rows: list[dict[str, Any]] = []
        for position, config in zip(positions, configs):
            row = config.model_dump()
            row_valid, errors = is_row_valid(pd.DataFrame([row], index=[0]))
            rule_errors[position] = " ".join(errors)
            if int(row_valid) == 1:
                batched_positions.append(position)
                batched_rows.append(row)
            else:
                results[position] = (
                    False,
                    {"Rule-Based Classifier error": rule_errors[position], "Predictive Model Classifier error": None},
                )

        if batched_rows:
            frame = pd.DataFrame(batched_rows)
            try:
                predictions = list(predictor.predict(frame).values)
                errors_by_position: dict[int, Optional[str]] = {p: None for p in batched_positions}
            except Exception as exc:
                # A batched failure says nothing about which row caused it, so fall back to the
                # per-row path for this chunk: that is what attributes the error to one candidate.
                logger.warning("AutoConf batched prediction failed, falling back to per-row: %s", exc)
                for position, config in zip(positions, configs):
                    if position in rule_errors and results[position] is None:
                        results[position] = self._decide_one(config, predictor, get_model_prediction_and_metadata)
                return [result for result in results if result is not None]  # type: ignore[misc]
            for position, prediction in zip(batched_positions, predictions):
                flag = int(prediction) if prediction else 0
                results[position] = (
                    flag == 1,
                    {
                        "Rule-Based Classifier error": rule_errors[position],
                        "Predictive Model Classifier error": errors_by_position[position],
                    },
                )
        return [result for result in results if result is not None]  # type: ignore[misc]

    def _decide_one(self, config: Any, predictor: Any, get_model_prediction_and_metadata: Any):
        """One candidate through ado's own single-row path."""
        try:
            valid_flag, metadata = get_model_prediction_and_metadata(config, predictor)
            return valid_flag == 1, metadata or {}
        except Exception as exc:
            logger.warning("AutoConf prediction failed (treating candidate as infeasible): %s", exc)
            return False, {"error": str(exc)}

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
                    # AutoConf's JobConfig.batch_size is the TOTAL/effective batch — it divides by
                    # number_gpus internally (and its rule-based classifier requires divisibility).
                    # WorkloadSpec.batch_size is PER-DEVICE, so convert at this boundary:
                    # effective = per_device × total_gpus (always divisible by total_gpus).
                    "batch_size": workload.batch_size * workload.total_gpus,
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
    """Lightweight feasibility without AutoConf (basic per-device sanity guards; no OOM check)."""

    #: Two integer comparisons — dispatching them to a worker would cost more than the work.
    EXPENSIVE = False

    def is_feasible(self, workload: WorkloadSpec) -> tuple[bool, dict[str, Any]]:
        # batch_size is PER-DEVICE (Kavier's convention — it multiplies by total GPUs
        # internally): a per-device batch need not divide the GPU count, so the old
        # ``batch_size % total_gpus`` rule was wrong. Keep only basic sanity guards; the
        # real OOM constraint is the AutoConf checker's job.
        if workload.total_gpus < 1:
            return False, {"error": "invalid total_gpus"}
        if workload.batch_size < 1:
            return False, {"error": "batch_size must be >= 1 (per-device)"}
        return True, {}


class NoOpFeasibilityChecker:
    """Accept all candidates (for tests or when feasibility is disabled)."""

    EXPENSIVE = False

    def is_feasible(self, workload: WorkloadSpec) -> tuple[bool, dict[str, Any]]:
        return True, {}
