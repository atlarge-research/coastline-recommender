"""Energy predictor using Kavier's power estimation."""

import logging
from typing import Optional

from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.recommendation import Prediction
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.base import BasePredictor
from coastline.sdk.predictors.performance.physics import KavierPredictor

logger = logging.getLogger(__name__)


class KavierPowerPredictor(BasePredictor):
    """Power predictor built on KavierPredictor; reports per-GPU watts (MSE power model)."""

    # When the throughput prediction already carries a power (Kavier's does), the pipeline uses
    # it and skips this predictor, so a candidate costs one engine call.
    WRAPS_THROUGHPUT_ENGINE = True

    def __init__(self):
        self.kavier = KavierPredictor()
        logger.info("KavierPowerPredictor initialized")

    def get_name(self) -> str:
        return "Kavier Power Estimator"

    def predict(self, workload: WorkloadSpec, context: SystemContext) -> Optional[Prediction]:
        """Predict per-GPU power in watts, or None if Kavier cannot predict this config.

        When Kavier gives a reason (an unknown model, GPU or method), its result is returned with
        no power and the reason in ``metadata['error_detail']``.
        """
        kavier_prediction = self.kavier.predict(workload, context)
        if kavier_prediction is None:
            logger.debug(f"Kavier returned None for {workload.llm_model}")
            return None

        power = kavier_prediction.predicted_power
        if power is None or power <= 0:
            logger.debug(f"No valid power from Kavier: {power}")
            if kavier_prediction.metadata.get("error_detail"):
                return kavier_prediction.model_copy(
                    update={
                        "predicted_power": None,
                        "metadata": {**kavier_prediction.metadata, "predictor": "kavier_power"},
                    }
                )
            return None

        return Prediction(
            gpus_per_node=kavier_prediction.gpus_per_node,
            number_of_nodes=kavier_prediction.number_of_nodes,
            total_gpus=kavier_prediction.total_gpus,
            predicted_throughput=kavier_prediction.predicted_throughput,
            predicted_runtime_seconds=kavier_prediction.predicted_runtime_seconds,
            predicted_power=power,
            metadata={
                **kavier_prediction.metadata,
                "predictor": "kavier_power",
                "power_model": "mse",
            },
        )
