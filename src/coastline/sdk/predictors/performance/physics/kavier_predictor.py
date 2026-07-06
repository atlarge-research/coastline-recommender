"""Physics-based predictor using Kavier simulator."""

import logging
from typing import Any, Dict, Optional

# Kavier is installed separately (pip install "kavier>=0.4,<0.5"). We use its
# PUBLIC training API (kavier.training.performance) rather than the internal sim
# engine, so a kavier internal refactor can't break us. If kavier is absent,
# KAVIER_AVAILABLE is False and the predictor surfaces the error at predict time.
try:
    from kavier import training as _kavier_training

    KAVIER_AVAILABLE = True
except ImportError as e:
    KAVIER_AVAILABLE = False
    _import_error = str(e)

from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.recommendation import Prediction
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.base import BasePredictor

logger = logging.getLogger(__name__)


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
    """Analytical throughput+power predictor using Kavier's physics simulator.

    Returns tokens/sec and per-GPU watts for calibrated (model, GPU) pairs, else None.
    predicted_runtime_seconds is always None (Kavier yields per-step time, not job runtime).
    """

    def __init__(self):
        if not KAVIER_AVAILABLE:
            logger.warning(f"Kavier not available: {_import_error}")
        else:
            logger.info("KavierPredictor initialized successfully")

    def get_name(self) -> str:
        return "Kavier Physics-Based"

    def predict(self, workload: WorkloadSpec, context: SystemContext) -> Optional[Prediction]:
        """Predict throughput/power. Returns None on invalid inputs; error Prediction for unsupported configs."""
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
            # kavier.training.performance derives total GPUs as num_gpus × num_nodes,
            # so we feed it the per-node count (gpus_per_node) + number_of_nodes —
            # equivalent to the old direct call's total_gpus (verified identical, single-
            # and multi-node). total_gpus is still used for validation + metadata.
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

            # num_gpus is PER-NODE (the verb multiplies by num_nodes). Derive it from
            # the validated total so it is always a positive int even when the workload
            # left gpus_per_node unset (total // nodes == gpus_per_node for grid candidates).
            per_node = max(1, total_gpus // num_nodes)
            row = {
                "model": workload.llm_model,
                "gpu": workload.gpu_model,
                "method": workload.fine_tuning_method,
                "seq_len": workload.tokens_per_sample,
                "batch_size": workload.batch_size,
                "num_gpus": per_node,
                "num_nodes": num_nodes,
            }
            result = _kavier_training.performance(row).iloc[0].to_dict()

            throughput = result.get("train_tokens_per_second")
            power = result.get("gpu_power_watts")
            step_time = result.get("step_time_ms")  # not exported by the verb -> None

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
                predicted_runtime_seconds=None,  # Kavier gives per-step time, not total runtime
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
                    "error_detail": (
                        f"Unsupported {error_key}. Kavier supports: granite-3-8b, granite-3.3-8b, "
                        "llama3.2-3b, mistral-7b-v0.1 (models); L40S, NVIDIA-A100-80GB-PCIe, "
                        "NVIDIA-A100-SXM4-80GB, NVIDIA-H100-PCIe (GPUs); full, gptq-lora, lora (methods)."
                    ),
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
