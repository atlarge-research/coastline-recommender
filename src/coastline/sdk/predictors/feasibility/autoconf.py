"""AutoConf feasibility checker: wraps ADO's autoconf validity classifier (lazy-loaded)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from pydantic import ValidationError

from coastline.sdk.constants import BATCHABLE_AUTOCONF_MODEL_VERSIONS, DEFAULT_AUTOCONF_MODEL_VERSION
from coastline.sdk.models.workload import WorkloadSpec

logger = logging.getLogger(__name__)

# Normally autoconf comes from the installed ``ado-autoconf`` package; these paths are for a
# source checkout of ADO. ``ADO_ROOT`` overrides; otherwise use ``ado/`` in the superproject that
# holds this repo (parents[6] of this file).
_ADO_ROOT = Path(os.environ.get("ADO_ROOT") or Path(__file__).resolve().parents[6] / "ado")
_ADO_AUTOCONF_PARENT = _ADO_ROOT / "plugins" / "custom_experiments" / "autoconf"

_AUTOCONF_AVAILABLE: Optional[bool] = None

# The GPUs in AutoConf's training data: the gpu_model categories of its 3.0.0 and 3.1.0 models,
# also listed as autoconf.min_gpu_recommender.GPUModel. The classifier has seen no run on any
# other GPU, so its OOM verdict there is an extrapolation.
AUTOCONF_TRAINED_GPUS = frozenset({"L40S", "NVIDIA-A100-80GB-PCIe", "NVIDIA-A100-SXM4-80GB", "NVIDIA-H100-PCIe"})

# GPU names already warned about, so each one is logged once per process.
_WARNED_GPUS: set[str] = set()


def _warn_if_untrained_gpu(gpu_model: str) -> None:
    """Log once per GPU name when AutoConf has no training data for it. The verdict is unchanged."""
    if gpu_model in AUTOCONF_TRAINED_GPUS or gpu_model in _WARNED_GPUS:
        return
    _WARNED_GPUS.add(gpu_model)
    logger.warning(
        "AutoConf has no training data for GPU %r (it knows %s), so its OOM verdict for this GPU is an extrapolation.",
        gpu_model,
        ", ".join(sorted(AUTOCONF_TRAINED_GPUS)),
    )


def _autoconf_modules():
    """Import ADO autoconf on demand; None if unavailable. A failed import is retried on the
    next call."""
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
        # Not cached, so the next call retries (the failure may be transient).
        logger.warning("AutoConf import failed (will retry on next call): %s", exc)
        return None


def _settled(
    results: "list[Optional[tuple[bool, dict[str, Any]]]]", workloads: "Sequence[WorkloadSpec]"
) -> list[tuple[bool, dict[str, Any]]]:
    """Check that every candidate got a verdict and return the verdicts in candidate order.

    The chunk paths fill a list of placeholders, so a gap means a dropped candidate. Removing
    the gap would misalign every later verdict, so this raises while the position is known.
    """
    missing = [index for index, result in enumerate(results) if result is None]
    if missing or len(results) != len(workloads):
        raise RuntimeError(
            f"AutoConf decided {len(results) - len(missing)} of {len(workloads)} candidates "
            f"(no verdict at positions {missing[:5]})"
        )
    return [result for result in results if result is not None]


class AutoconfFeasibilityChecker:
    """Rule + AutoGluon validity check for a single candidate layout."""

    #: One AutoGluon predict per candidate costs more than shipping the candidate to a worker
    #: process, so this backend is worth forking across candidates.
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
        """The AutoConf JobConfig for one candidate. Raises ValidationError for an invalid job."""
        return JobConfig.model_validate(
            {
                # feasibility_model lets the OOM check use the real model when the perf
                # predictor uses a proxy; else llm_model.
                "model_name": workload.feasibility_model or workload.llm_model,
                "method": workload.fine_tuning_method,
                "gpu_model": workload.gpu_model,
                "tokens_per_sample": workload.tokens_per_sample,
                # AutoConf's JobConfig.batch_size is the total (effective) batch: AutoConf divides
                # it by number_gpus and its rule-based classifier requires divisibility.
                # WorkloadSpec.batch_size is per device, so convert here:
                # effective = per_device x total_gpus, which total_gpus always divides.
                "batch_size": workload.batch_size * workload.total_gpus,
                "number_gpus": workload.total_gpus,
            }
        )

    def batches(self) -> bool:
        """Whether one classifier call decides a whole chunk (see :meth:`_can_batch`)."""
        return self._can_batch()

    def _can_batch(self) -> bool:
        """Whether one classifier call may decide a whole chunk of candidates.

        True for the model versions checked against the per-row path
        (:data:`BATCHABLE_AUTOCONF_MODEL_VERSIONS`). Other versions, or
        ``COASTLINE_NO_AUTOCONF_BATCH=1``, use one call per candidate.
        """
        if os.environ.get("COASTLINE_NO_AUTOCONF_BATCH") == "1":
            return False
        return self.model_version in BATCHABLE_AUTOCONF_MODEL_VERSIONS

    def check_chunk(self, workloads: "Sequence[WorkloadSpec]") -> list[tuple[bool, dict[str, Any]]]:
        """Verdicts for a run of candidates, in input order, from one classifier call.

        The AutoGluon predict takes most of a recommendation's time and is vectorised, so a chunk
        costs about as much as one candidate. ado's rule-based classifier takes one row at a time
        (its ``to_series`` rejects a multi-row frame); its rule is one modulo, checked below for
        the whole frame. The rest matches ado's ``get_model_prediction_and_metadata``, including
        its metadata keys and its ``pred = int(pred) if pred else 0`` rule.
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
                _warn_if_untrained_gpu(workload.gpu_model)
            except ValidationError as exc:
                # An invalid JobConfig is a real reject; logged at debug level because the grid
                # probes such configs.
                logger.debug("AutoConf rejected candidate (invalid JobConfig): %s", exc)
                results[position] = (False, {"error": f"invalid_job_config: {exc}"})
        if not positions:
            return _settled(results, workloads)

        try:
            predictor = self._ensure_predictor()
        except Exception as exc:  # model unavailable: every candidate in the chunk is undecided
            logger.warning("AutoConf model unavailable (treating candidates as infeasible): %s", exc)
            for position in positions:
                results[position] = (False, {"error": str(exc)})
            return _settled(results, workloads)

        if not self._can_batch():
            for position, config in zip(positions, configs):
                results[position] = self._decide_one(config, predictor, get_model_prediction_and_metadata)
            return _settled(results, workloads)

        # Rule stage. ado's is_row_valid takes one row, and building a one-row DataFrame per
        # candidate would dominate a batched gate. The rule is one modulo, so it is checked over the
        # whole frame here, and only rows this fast check cannot clear go to ado, which decides their
        # verdict and error text. In production no row goes to ado: the effective batch is
        # per_device x total_gpus, so total_gpus divides it.
        from autoconf.utils.rule_based_classifier import is_row_valid

        rows = [config.model_dump() for config in configs]
        gpus = pd.to_numeric(pd.Series([row.get("number_gpus") for row in rows]), errors="coerce")
        batch = pd.to_numeric(pd.Series([row.get("batch_size") for row in rows]), errors="coerce")
        # Rows with a non-numeric value, a GPU count below 1 or a non-zero remainder go to ado.
        clearly_valid = ((gpus > 0) & (batch % gpus == 0)).fillna(False).to_numpy()

        rule_errors: dict[int, str] = {}
        batched_positions: list[int] = []
        batched_rows: list[dict[str, Any]] = []
        for offset, (position, row) in enumerate(zip(positions, rows)):
            if clearly_valid[offset]:
                rule_errors[position] = ""
                batched_positions.append(position)
                batched_rows.append(row)
                continue
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
            except Exception as exc:
                # A batched failure does not say which row caused it, so rerun this chunk per row
                # to tie the error to one candidate.
                logger.warning("AutoConf batched prediction failed, falling back to per-row: %s", exc)
                for position, config in zip(positions, configs):
                    if results[position] is None:
                        results[position] = self._decide_one(config, predictor, get_model_prediction_and_metadata)
                return _settled(results, workloads)
            for position, prediction in zip(batched_positions, predictions):
                flag = int(prediction) if prediction else 0
                results[position] = (
                    flag == 1,
                    {
                        "Rule-Based Classifier error": rule_errors[position],
                        "Predictive Model Classifier error": None,
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
                    # AutoConf's JobConfig.batch_size is the total (effective) batch: AutoConf divides
                    # it by number_gpus and its rule-based classifier requires divisibility.
                    # WorkloadSpec.batch_size is per device, so convert here:
                    # effective = per_device x total_gpus, which total_gpus always divides.
                    "batch_size": workload.batch_size * workload.total_gpus,
                    "number_gpus": workload.total_gpus,
                }
            )
        except ValidationError as exc:
            # An invalid JobConfig is a real reject; logged at debug level because the grid probes such configs.
            logger.debug("AutoConf rejected candidate (invalid JobConfig): %s", exc)
            return False, {"error": f"invalid_job_config: {exc}"}
        _warn_if_untrained_gpu(workload.gpu_model)

        try:
            predictor = self._ensure_predictor()
            valid_flag, metadata = get_model_prediction_and_metadata(job_config, predictor)
            return valid_flag == 1, metadata or {}
        except Exception as exc:
            # A load or predict failure is a warning, so a broken model gets noticed; the candidate
            # is then treated as infeasible so the grid continues.
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

    #: Two integer comparisons; a worker dispatch would cost more than the work.
    EXPENSIVE = False

    def is_feasible(self, workload: WorkloadSpec) -> tuple[bool, dict[str, Any]]:
        # batch_size is per device (Kavier multiplies it by the GPU count), so it need not divide
        # total_gpus. These are sanity guards only; the AutoConf checker does the OOM check.
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
