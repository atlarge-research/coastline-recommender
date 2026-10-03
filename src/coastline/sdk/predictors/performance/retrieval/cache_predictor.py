"""Retrieval-based predictor: cache-first exact-match lookup over the run database."""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from coastline.sdk.io.sample_data import sample_raw_trace_path
from coastline.sdk.library.hardware import canonical_gpu_name
from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.recommendation import Prediction
from coastline.sdk.models.workload import WorkloadSpec, canonical_method_name, canonical_model_name
from coastline.sdk.predictors.base import BasePredictor

logger = logging.getLogger(__name__)


# Default source columns for a hit; a lookup CSV with other headers passes its own names.
_DEFAULT_THROUGHPUT_COL = "dataset_tokens_per_second"
_DEFAULT_RUNTIME_COL = "train_runtime"
# Measured per-GPU power of the run (optional column; the raw trace and the bundled sample have it).
_DEFAULT_POWER_COL = "gpu_power_watts_avg"

#: Metadata key of a hit's measured per-GPU power in watts, present when the run recorded one.
MEASURED_POWER_KEY = "measured_power_watts"

_CONFIG_COLS = [
    "model_name",
    "method",
    "gpu_model",
    "number_nodes",
    "number_gpus",
    "tokens_per_sample",
    "batch_size",
]
_NUMERIC_CONFIG_COLS = ["number_nodes", "number_gpus", "tokens_per_sample", "batch_size"]


def _measured_power(values: np.ndarray) -> Optional[float]:
    """The first run's power, else the median of the runs that recorded one, else None."""
    valid = values[np.isfinite(values) & (values > 0)]
    if not len(valid):
        return None
    first = values[0]
    return float(first) if np.isfinite(first) and first > 0 else float(np.median(valid))


class RetrievalPredictor(BasePredictor):
    """Exact-match cache over the curated run database, indexed by SHA256 hash.

    On a hit, returns the first recorded run's throughput and runtime (~0% error); on a miss,
    returns None so the orchestrator falls back to a simulation predictor. ``throughput_col``
    and ``runtime_col`` name the columns a hit reads (the defaults match the run database), for
    lookup CSVs that use other headers.

    Power comes from the power predictor (Kavier). A hit also carries the run's measured power
    from ``power_col`` in ``metadata[MEASURED_POWER_KEY]``, which the pipeline uses when the power
    predictor has none (a model outside Kavier's catalog), so the measured run can still be
    recommended.
    """

    #: The reason a recommendation reports when every candidate missed.
    MISS_REASON = "no measured run in the lookup database matches"

    #: A hash lookup in a prebuilt index, ~4.6 us; not worth a worker dispatch.
    EXPENSIVE = False

    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        *,
        throughput_col: Optional[str] = None,
        runtime_col: Optional[str] = None,
        power_col: Optional[str] = None,
    ):
        """Load the raw trace (rather than the ML subset) and build the hash index."""
        self._throughput_col = throughput_col or _DEFAULT_THROUGHPUT_COL
        self._runtime_col = runtime_col or _DEFAULT_RUNTIME_COL
        self._power_col = power_col or _DEFAULT_POWER_COL
        if dataset_path is None:
            env = os.environ.get("DATA_DIR")
            # parents[7] is the superproject that holds trace-archive/.
            data_dir = Path(env) if env else Path(__file__).resolve().parents[7] / "trace-archive"
            full_trace = data_dir / "profiling-dataset" / "raw_trace.csv"
            # Without the full trace, use the bundled sample, so a `pip install` still serves a few
            # exact-match cache hits.
            dataset_path = full_trace if full_trace.exists() else sample_raw_trace_path()

        self.dataset_path = dataset_path
        self.dataset: Optional[pd.DataFrame] = None
        self.config_index: Dict[str, Dict[str, Any]] = {}
        self.aggregation_stats: Dict[str, Dict[str, float]] = {}

        self._load_dataset()
        self._build_index()

        logger.info(f"RetrievalPredictor initialized with {len(self.dataset) if self.dataset is not None else 0} runs")
        logger.info(f"Indexed {len(self.config_index)} unique configurations")

    def _load_dataset(self):
        """Load and validate the curated dataset."""
        try:
            dataset = pd.read_csv(self.dataset_path)
            if not isinstance(dataset, pd.DataFrame):
                raise TypeError("Expected pandas DataFrame from retrieval dataset")

            logger.info(f"Loaded dataset from {self.dataset_path}")
            logger.info(f"Dataset shape: {dataset.shape}")

            required_cols = [*_CONFIG_COLS, self._throughput_col, self._runtime_col]
            missing_cols = [col for col in required_cols if col not in dataset.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Drop invalid runs, in case the file still has any
            if "is_valid" in dataset.columns:
                initial_count = len(dataset)
                dataset = dataset.loc[dataset["is_valid"] == 1.0].copy()
                filtered_count = initial_count - len(dataset)
                if filtered_count > 0:
                    logger.warning(f"Filtered out {filtered_count} invalid runs")

            # A cell such as 'OOM' or '' in a numeric column becomes NaN, so its row is dropped
            # below instead of failing the comparisons (the same coercion `coastline utils tune` uses).
            numeric_cols = [*_NUMERIC_CONFIG_COLS, self._throughput_col, self._runtime_col]
            if self._power_col in dataset.columns:
                numeric_cols.append(self._power_col)
            for col in numeric_cols:
                dataset[col] = pd.to_numeric(dataset[col], errors="coerce")

            initial_count = len(dataset)
            dataset = dataset.loc[dataset[_NUMERIC_CONFIG_COLS].notna().all(axis=1)].copy()
            config_filtered = initial_count - len(dataset)
            if config_filtered > 0:
                logger.warning(
                    f"Filtered out {config_filtered} rows with a missing or non-numeric "
                    f"{', '.join(_NUMERIC_CONFIG_COLS)}"
                )

            # Filter out rows with missing/invalid throughput or runtime values
            initial_count = len(dataset)
            dataset = dataset.loc[
                dataset[self._throughput_col].notna()
                & (dataset[self._throughput_col] > 0)
                & dataset[self._runtime_col].notna()
                & (dataset[self._runtime_col] > 0)
            ].copy()
            nan_filtered = initial_count - len(dataset)
            if nan_filtered > 0:
                logger.warning(f"Filtered out {nan_filtered} rows with missing/zero throughput or runtime")

            self.dataset = dataset

        except FileNotFoundError:
            logger.error(f"Dataset not found at {self.dataset_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def _build_index(self):
        """Build the SHA256 hash index (O(1) lookup), with stats and first-run values per config."""
        if self.dataset is None or len(self.dataset) == 0:
            logger.warning("Cannot build index: dataset is empty")
            return

        has_power = self._power_col in self.dataset.columns

        for config_key, group in self.dataset.groupby(_CONFIG_COLS):
            config_tuple = config_key if isinstance(config_key, tuple) else (config_key,)
            if len(config_tuple) != 7:
                logger.warning(f"Skipping malformed grouped key: {config_tuple}")
                continue

            # Keyed on the spellings WorkloadSpec stores, so 'LoRA' or a GPU alias in the CSV still hits.
            config_hash = self._hash_configuration(
                canonical_model_name(str(config_tuple[0])),
                canonical_method_name(str(config_tuple[1])),
                canonical_gpu_name(str(config_tuple[2])),
                float(config_tuple[3]),
                float(config_tuple[4]),
                float(config_tuple[5]),
                float(config_tuple[6]),
            )

            throughputs = pd.to_numeric(group[self._throughput_col], errors="coerce").to_numpy(dtype=float)
            runtimes = pd.to_numeric(group[self._runtime_col], errors="coerce").to_numpy(dtype=float)
            measured_power = _measured_power(group[self._power_col].to_numpy(dtype=float)) if has_power else None

            throughput_median = float(np.median(throughputs))
            throughput_std = float(np.std(throughputs))
            throughput_min = float(np.min(throughputs))
            throughput_max = float(np.max(throughputs))
            runtime_median = float(np.median(runtimes))
            runtime_std = float(np.std(runtimes))
            runtime_min = float(np.min(runtimes))
            runtime_max = float(np.max(runtimes))
            run_count = int(len(throughputs))
            # Return the first recorded measurement (matches the deduplicated test
            # target, so a true hit is ~0% error) rather than the median over runs.
            throughput_first = float(throughputs[0])
            runtime_first = float(runtimes[0])
            cv = float(throughput_std / throughput_median) if throughput_median > 0 else 0.0

            self.config_index[config_hash] = {
                "llm_model": str(config_tuple[0]),
                "fine_tuning_method": str(config_tuple[1]),
                "gpu_model": str(config_tuple[2]),
                "number_of_nodes": float(config_tuple[3]),
                "gpus_per_node": float(config_tuple[4]),
                "tokens_per_sample": float(config_tuple[5]),
                "batch_size": float(config_tuple[6]),
                "throughput_median": throughput_median,
                "throughput_std": throughput_std,
                "throughput_min": throughput_min,
                "throughput_max": throughput_max,
                "runtime_median": runtime_median,
                "runtime_std": runtime_std,
                "runtime_min": runtime_min,
                "runtime_max": runtime_max,
                "run_count": run_count,
                "throughput_first": throughput_first,
                "runtime_first": runtime_first,
                "measured_power_watts": measured_power,
            }

            self.aggregation_stats[config_hash] = {
                "median": throughput_median,
                "std": throughput_std,
                "count": run_count,
                "cv": cv,
            }

        logger.info(f"Built index with {len(self.config_index)} unique configurations")

        multi_run_configs = {k: v for k, v in self.aggregation_stats.items() if v["count"] > 1}
        if multi_run_configs:
            logger.info(f"Found {len(multi_run_configs)} configurations with multiple runs")

    def _hash_configuration(
        self,
        llm_model: str,
        method: str,
        gpu_model: str,
        number_of_nodes: float,
        gpus_per_node: float,
        tokens_per_sample: float,
        batch_size: float,
    ) -> str:
        """Deterministic SHA256 hash of a configuration (ints normalize float noise)."""
        config_str = (
            f"{llm_model}|{method}|{gpu_model}|"
            f"{int(number_of_nodes)}|{int(gpus_per_node)}|"
            f"{int(tokens_per_sample)}|{int(batch_size)}"
        )
        return hashlib.sha256(config_str.encode()).hexdigest()

    def predict(self, workload: WorkloadSpec, context: SystemContext) -> Optional[Prediction]:
        """Return first-run throughput on exact match, or None on miss."""
        config_hash = self._hash_configuration(
            llm_model=workload.llm_model,
            method=workload.fine_tuning_method,
            gpu_model=workload.gpu_model,
            number_of_nodes=workload.number_of_nodes or 1,
            gpus_per_node=workload.gpus_per_node or 1,
            tokens_per_sample=workload.tokens_per_sample,
            batch_size=workload.batch_size,
        )

        if config_hash not in self.config_index:
            logger.info(
                f"Cache MISS: {workload.llm_model} ({workload.fine_tuning_method}) - "
                f"Configuration not found in database"
            )
            return None

        config_data = self.config_index[config_hash]
        stats = self.aggregation_stats[config_hash]

        logger.info(
            f"Cache HIT: {workload.llm_model} ({workload.fine_tuning_method}) - "
            f"Throughput: {config_data['throughput_median']:.0f} tokens/sec, "
            f"Runtime: {config_data['runtime_median']:.1f}s "
            f"(+/-{config_data['throughput_std']:.0f} tokens/sec, n={stats['count']})"
        )

        metadata = {
            "predictor": "retrieval",
            "cache_hit": True,
            "throughput_std": config_data["throughput_std"],
            "runtime_std": config_data["runtime_std"],
            "run_count": stats["count"],
            "throughput_min": config_data["throughput_min"],
            "throughput_max": config_data["throughput_max"],
            "runtime_min": config_data["runtime_min"],
            "runtime_max": config_data["runtime_max"],
            "coefficient_of_variation": stats["cv"],
        }
        if config_data["measured_power_watts"] is not None:
            metadata[MEASURED_POWER_KEY] = config_data["measured_power_watts"]

        return Prediction(
            gpus_per_node=workload.gpus_per_node or 1,
            number_of_nodes=workload.number_of_nodes or 1,
            total_gpus=int(config_data["number_of_nodes"] * config_data["gpus_per_node"]),
            predicted_throughput=config_data["throughput_first"],
            predicted_runtime_seconds=config_data["runtime_first"],
            metadata=metadata,
        )

    def get_name(self) -> str:
        return "RetrievalPredictor"

    def get_statistics(self) -> Dict[str, Any]:
        """Return coverage and aggregation-quality statistics for the database."""
        if self.dataset is None or len(self.dataset) == 0:
            return {"total_runs": 0, "unique_configurations": 0, "coverage": {}}

        coverage = {
            "models": self.dataset["model_name"].nunique(),
            "methods": self.dataset["method"].nunique(),
            "gpu_models": self.dataset["gpu_model"].nunique(),
            "node_counts": self.dataset["number_nodes"].nunique(),
            "gpu_counts": self.dataset["number_gpus"].nunique(),
            "token_lengths": self.dataset["tokens_per_sample"].nunique(),
            "batch_sizes": self.dataset["batch_size"].nunique(),
        }

        multi_run_count = sum(1 for stats in self.aggregation_stats.values() if stats["count"] > 1)
        avg_runs_per_config = np.mean([stats["count"] for stats in self.aggregation_stats.values()])
        avg_cv = np.mean([stats["cv"] for stats in self.aggregation_stats.values()])

        return {
            "total_runs": len(self.dataset),
            "unique_configurations": len(self.config_index),
            "configurations_with_multiple_runs": multi_run_count,
            "avg_runs_per_configuration": avg_runs_per_config,
            "avg_coefficient_of_variation": avg_cv,
            "coverage": coverage,
            "dataset_path": str(self.dataset_path),
        }

    def find_similar_configurations(
        self, workload: WorkloadSpec, max_results: int = 5
    ) -> list[Tuple[Dict[str, Any], float]]:
        """Score indexed configs by weighted categorical match + GPU-count proximity; return top-k."""
        if self.dataset is None or len(self.dataset) == 0:
            return []

        similar = []
        for config_data in self.config_index.values():
            similarity = 0.0
            if config_data["llm_model"] == workload.llm_model:
                similarity += 0.4
            if config_data["fine_tuning_method"] == workload.fine_tuning_method:
                similarity += 0.3
            if config_data["gpu_model"] == workload.gpu_model:
                similarity += 0.2

            target_gpus = (workload.number_of_nodes or 1) * (workload.gpus_per_node or 1)
            config_total_gpus = config_data["number_of_nodes"] * config_data["gpus_per_node"]
            gpu_diff = abs(config_total_gpus - target_gpus) / max(target_gpus, 1)
            similarity += 0.1 * (1.0 / (1.0 + gpu_diff))

            if similarity > 0:
                similar.append((config_data, similarity))

        similar.sort(key=lambda x: x[1], reverse=True)
        return similar[:max_results]
