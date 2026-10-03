"""Physics-based predictor using Kavier simulator."""

import logging
from typing import Any, Dict, Optional

# Kavier is a separate package (version range pinned in pyproject.toml). This module calls only
# its public training API (kavier.training.performance), so changes to Kavier's internals do not
# break it. Without Kavier, KAVIER_AVAILABLE is False and predict() returns an error Prediction.
try:
    from kavier import training as _kavier_training
    from kavier.sdk.library import GPU_SPEC_LIBRARY as _KAVIER_GPUS
    from kavier.sdk.library import LLM_SPEC_LIBRARY as _KAVIER_LLMS
    from kavier.sdk.library import UnknownSpecError

    KAVIER_AVAILABLE = True
except ImportError as e:
    KAVIER_AVAILABLE = False
    _import_error = str(e)

from coastline.sdk.library.hardware import canonical_gpu_name
from coastline.sdk.library.llm_names import kavier_llm_name
from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.recommendation import Prediction
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.base import BasePredictor

logger = logging.getLogger(__name__)


def _unsupported_detail(error: KeyError, error_key: str) -> str:
    """Kavier's own message for an unknown model or GPU (it lists the catalog), else the catalog."""
    if isinstance(error, UnknownSpecError):
        return str(error)
    return (
        f"Unsupported {error_key}. Kavier knows {len(_KAVIER_LLMS)} models ({', '.join(sorted(_KAVIER_LLMS))}) "
        f"and {len(_KAVIER_GPUS)} GPUs ({', '.join(sorted(_KAVIER_GPUS))})."
    )


def _error_prediction(workload: WorkloadSpec, total_gpus: int, metadata: Dict[str, Any]) -> Prediction:
    """Build a null Kavier Prediction carrying error metadata."""
    return Prediction(
        gpus_per_node=workload.gpus_per_node or 1,
        number_of_nodes=workload.number_of_nodes or 1,
        total_gpus=total_gpus,
        predicted_throughput=None,
        predicted_runtime_seconds=None,
        predicted_power=None,
        metadata=metadata,
    )


class KavierPredictor(BasePredictor):
    """Analytical throughput and power predictor using Kavier's physics simulator.

    Returns tokens/sec and per-GPU watts for models and GPUs in Kavier's catalog.
    predicted_runtime_seconds is always None: Kavier computes ``train_runtime`` only when given a
    job size (``total_tokens``, or ``epochs`` x ``dataset_tokens``), and the row built here has
    none, so Kavier would return 0.0.
    """

    #: Closed-form arithmetic at ~2.6 us per prediction; a worker dispatch would cost far more
    #: than the work, so this predictor runs inline.
    EXPENSIVE = False

    def __init__(self):
        if not KAVIER_AVAILABLE:
            logger.warning(f"Kavier not available: {_import_error}")
        else:
            logger.info("KavierPredictor initialized successfully")

    def get_name(self) -> str:
        return "Kavier Physics-Based"

    def predict(self, workload: WorkloadSpec, context: SystemContext) -> Optional[Prediction]:
        """Predict throughput and power: None for invalid inputs, an error Prediction for unsupported configs."""
        if not KAVIER_AVAILABLE:
            logger.debug("Kavier not available, returning error prediction")
            return _error_prediction(
                workload,
                workload.total_gpus,
                {
                    "predictor": "kavier",
                    "error": "not_available",
                    "error_detail": f"Kavier simulator not available: {_import_error}",
                },
            )

        try:
            # kavier.training.performance computes total GPUs as num_gpus x num_nodes, so it gets
            # the per-node count and the node count; total_gpus is used for validation and metadata.
            total_gpus = workload.total_gpus
            num_nodes = workload.number_of_nodes or 1

            if total_gpus <= 0:
                logger.warning(f"Invalid GPU count: {total_gpus}")
                return None
            if workload.batch_size <= 0:
                logger.warning(f"Invalid batch size: {workload.batch_size}")
                return None
            if workload.tokens_per_sample <= 0:
                logger.warning(f"Invalid tokens_per_sample: {workload.tokens_per_sample}")
                return None

            logger.debug(
                f"Simulating: model={workload.llm_model}, gpu={workload.gpu_model}, "
                f"fine_tuning_method={workload.fine_tuning_method}, batch={workload.batch_size}, "
                f"tokens={workload.tokens_per_sample}, gpus={total_gpus}"
            )

            # num_gpus is per node. Derive it from the validated total so it is a positive int even
            # when gpus_per_node is unset (total // nodes == gpus_per_node for grid candidates).
            per_node = max(1, total_gpus // num_nodes)
            row = {
                # The catalog key ('Llama-3-8B', 'llama3.2-3b') for a lowercased name or an HF id;
                # an unknown name goes through unchanged so Kavier's error names it.
                "model": kavier_llm_name(workload.llm_model) or workload.llm_model,
                "gpu": canonical_gpu_name(workload.gpu_model),
                "method": workload.fine_tuning_method,
                "seq_len": workload.tokens_per_sample,
                "batch_size": workload.batch_size,
                "num_gpus": per_node,
                "num_nodes": num_nodes,
            }
            result = _kavier_training.performance(row).iloc[0].to_dict()

            throughput = result.get("train_tokens_per_second")
            power = result.get("gpu_power_watts")
            step_time = result.get("step_time_ms")  # the verb does not export it, so None

            if throughput is None or throughput <= 0:
                logger.warning(f"Invalid throughput from Kavier: {throughput}")
                return None

            metadata: Dict[str, Any] = {
                "predictor": "kavier",
                "model_used": "physics_based",
                "runtime_semantics": "step_time_only",  # only per-step timing is available
            }
            if step_time is not None:
                metadata["step_time_ms"] = step_time
            if "gpu_compute_utilization" in result:
                metadata["gpu_compute_utilization"] = result["gpu_compute_utilization"]
            if "gpu_memory_utilization" in result:
                metadata["gpu_memory_utilization"] = result["gpu_memory_utilization"]

            power_str = f"{power:.1f}W" if power else "N/A"
            logger.info(f"Kavier prediction: {throughput:.1f} tokens/sec, power={power_str}")

            return Prediction(
                gpus_per_node=workload.gpus_per_node or 1,
                number_of_nodes=workload.number_of_nodes or 1,
                total_gpus=workload.total_gpus,
                predicted_throughput=throughput,
                # Kavier returns a total runtime only when given a job size; the row
                # above supplies none, so there is no runtime to report here.
                predicted_runtime_seconds=None,
                predicted_power=power,
                metadata=metadata,
            )

        except KeyError as e:
            # Model, GPU or method not in Kavier's library.
            error_key = str(e).strip("'\"")
            logger.debug(f"Kavier KeyError (unsupported config): {e}")
            return _error_prediction(
                workload,
                total_gpus,
                {
                    "predictor": "kavier",
                    "error": "unsupported_config",
                    "error_detail": _unsupported_detail(e, error_key),
                    "unsupported_key": error_key,
                },
            )

        except ValueError as e:
            logger.warning(f"Kavier ValueError: {e}")
            return _error_prediction(
                workload,
                total_gpus,
                {
                    "predictor": "kavier",
                    "error": "invalid_input",
                    "error_detail": str(e),
                },
            )

        except ImportError as e:
            logger.error(f"Kavier ImportError: {e}")
            return _error_prediction(
                workload,
                workload.total_gpus,
                {
                    "predictor": "kavier",
                    "error": "import_error",
                    "error_detail": f"Kavier dependencies not available: {e}",
                },
            )

        except Exception as e:
            logger.error(f"Kavier unexpected error: {e}", exc_info=True)
            return _error_prediction(
                workload,
                workload.total_gpus,
                {
                    "predictor": "kavier",
                    "error": "simulation_error",
                    "error_detail": str(e),
                },
            )
