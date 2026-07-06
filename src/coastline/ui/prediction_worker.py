"""Run one predictor in an isolated process.

Several native ML runtimes (catboost + xgboost + lightgbm + torch) in one process crash
on macOS; a subprocess per model keeps exactly one native runtime per process.
"""

from __future__ import annotations

import os
from typing import Any


def run_one(payload: dict[str, Any]) -> dict[str, Any]:
    """Predict one configuration with one model. Safe to run in a spawned child."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    from coastline.sdk.models.context import SystemContext
    from coastline.sdk.models.workload import WorkloadSpec
    from coastline.sdk.policies import PolicyFactory

    model_id = payload["model_id"]
    entry: dict[str, Any] = {"model": model_id, "label": payload.get("label", model_id)}

    total_gpus = payload["gpus_per_node"] * payload["number_of_nodes"]
    total_tokens = payload["total_tokens"]

    try:
        workload = WorkloadSpec(
            llm_model=payload["llm_model"],
            fine_tuning_method=payload["fine_tuning_method"],
            gpu_model=payload["gpu_model"],
            tokens_per_sample=payload["tokens_per_sample"],
            batch_size=payload["batch_size"],
            gpus_per_node=payload["gpus_per_node"],
            number_of_nodes=payload["number_of_nodes"],
        )
        context = SystemContext.for_gpus(
            [payload["gpu_model"]],
            max_gpus=total_gpus,
            gpus_per_node=payload["gpus_per_node"],
            max_nodes=payload["number_of_nodes"],
        )
        predictor = PolicyFactory.throughput_predictor({"performance": model_id})
        pred = predictor.predict(workload, context)
    except Exception as exc:  # bad payload / unknown GPU / predictor unavailable
        return {**entry, "available": False, "error": str(exc)[:200]}

    if pred is None or not pred.predicted_throughput or pred.predicted_throughput <= 0:
        return {**entry, "available": False}

    throughput = float(pred.predicted_throughput)
    # EST. TIME: derive from predicted throughput + total_tokens (same as recommender) so
    # it's apples-to-apples across predictors; a predictor's own predicted_runtime_seconds
    # refers to a different historical dataset size.
    if throughput > 0 and total_tokens > 0:
        runtime = total_tokens / throughput
    elif pred.predicted_runtime_seconds:
        runtime = float(pred.predicted_runtime_seconds)
    else:
        runtime = None
    power = float(pred.predicted_power) if pred.predicted_power else None
    cluster_power = power * total_gpus if power is not None else None
    energy_kwh = (
        (cluster_power * runtime) / 3_600_000.0 if (cluster_power is not None and runtime is not None) else None
    )
    entry.update(
        {
            "available": True,
            "predicted_throughput": throughput,
            "predicted_runtime_seconds": runtime,
            "power_watts": power,
            "energy_kwh": energy_kwh,
        }
    )
    return entry


if __name__ == "__main__":
    # CLI: read one JSON payload on stdin, emit the JSON result on stdout.
    # Library/import chatter (e.g. the MPS banner) is sent to stderr so it can't
    # corrupt the JSON on stdout.
    import json
    import sys

    _payload = json.load(sys.stdin)
    _real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        _result = run_one(_payload)
    finally:
        sys.stdout = _real_stdout
    json.dump(_result, sys.stdout)
