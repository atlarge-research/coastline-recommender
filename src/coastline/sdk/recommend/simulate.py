"""Predict one given configuration: the recommender's simulate step without ranking.

``recommend`` sweeps a grid and ranks it. ``simulate_one`` runs the same feasibility check and
throughput and power predictors on a single configuration chosen by the caller, and reports the
raw numbers.

There is no score: the policy scores (``power_score``, ``throughput_score``) are min-max
normalized over the candidate grid (``sdk/pipeline/selection.py``) and are always 1.0 for a
single configuration. ``coastline explain`` shows scores.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from coastline.sdk.constants import EnergyBackend, FeasibilityMode
from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.workload import WorkloadSpec

SECONDS_PER_HOUR = 3600.0
WH_PER_KWH = 1000.0


def _finite(value: Optional[float]) -> Optional[float]:
    """Return the value only if it is a usable finite number."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def simulate_one(
    workload: WorkloadSpec,
    context: SystemContext,
    *,
    predictor: str = "kavier",
    energy: str = EnergyBackend.KAVIER_POWER.value,
    feasibility: str = FeasibilityMode.AUTOCONF.value,
    total_tokens: int = 0,
) -> dict[str, Any]:
    """Simulate one configuration; return a flat dict of predicted quantities.

    ``error`` is ``None`` on success and a short reason otherwise; every numeric key is
    ``None`` when it could not be derived. ``predicted_runtime_seconds`` and ``energy_kwh``
    need ``total_tokens`` (the Kavier engine returns per-step time, never total runtime).
    """
    # Imported here to keep `import coastline` light; the predictor backends are heavy.
    from coastline.sdk.pipeline.feasibility import create_feasibility_checker
    from coastline.sdk.pipeline.workflow import measured_power
    from coastline.sdk.policies import PolicyFactory, normalize_predictor

    # Same validation as the facade and batch API: an unknown name raises instead of running the
    # `intelligent` default under the mistyped name.
    predictor = normalize_predictor(predictor)

    gpus_per_node = workload.gpus_per_node or 1
    number_of_nodes = workload.number_of_nodes or 1
    total_gpus = gpus_per_node * number_of_nodes

    result: dict[str, Any] = {
        "llm_model": workload.llm_model,
        "fine_tuning_method": workload.fine_tuning_method,
        "gpu_model": workload.gpu_model,
        "tokens_per_sample": workload.tokens_per_sample,
        "batch_size": workload.batch_size,
        "gpus_per_node": gpus_per_node,
        "number_of_nodes": number_of_nodes,
        "total_gpus": total_gpus,
        "predictor": predictor,
        "energy_backend": energy,
        "feasibility_mode": feasibility,
        "feasible": None,
        "feasibility_metadata": {},
        "predicted_throughput": None,
        "predicted_power_watts": None,
        "cluster_power_watts": None,
        "predicted_runtime_seconds": None,
        "runtime_source": None,
        "energy_kwh": None,
        "tokens_per_watt": None,
        "error": None,
    }

    # 1. Feasibility. `none` skips the check. `autoconf` raises when the model is unavailable
    #    (unless COASTLINE_ALLOW_RULES_FALLBACK=1); that configuration error propagates.
    checker = create_feasibility_checker({"feasibility": feasibility})
    feasible, feasibility_metadata = checker.is_feasible(workload)
    result["feasible"] = feasible
    result["feasibility_metadata"] = feasibility_metadata
    if not feasible:
        result["error"] = "infeasible"
        return result

    # 2. Throughput, from the predictor PolicyFactory resolves. Kavier returns power from the
    #    same call.
    prediction = PolicyFactory.throughput_predictor({"performance": predictor}).predict(workload, context)
    if prediction is None:
        result["error"] = "no prediction"
        return result

    # Kavier reports a failure as a Prediction with error metadata and null numbers, so the
    # metadata is checked too.
    reported_error = (prediction.metadata or {}).get("error")
    if reported_error:
        result["error"] = str(reported_error)
        return result

    throughput = _finite(prediction.predicted_throughput)
    if not throughput or throughput <= 0:
        result["error"] = "no usable throughput"
        return result
    result["predicted_throughput"] = throughput

    # 3. Power. Kavier returned it above; other predictors use the power predictor, and a cache
    #    hit without a power from it uses the power measured in its run, as
    #    pipeline.workflow.simulate_one does.
    power = _finite(prediction.predicted_power)
    if power is None or power <= 0:
        power_prediction = PolicyFactory.power_predictor({"energy": energy}).predict(workload, context)
        power = _finite(power_prediction.predicted_power) if power_prediction is not None else None
    if power is None or power <= 0:
        power = measured_power(prediction)
    if power is not None and power > 0:
        result["predicted_power_watts"] = power
        result["cluster_power_watts"] = power * total_gpus
        result["tokens_per_watt"] = throughput / power

    # 4. Runtime and energy need the dataset size, as Kavier reports only per-step time. Without
    #    --total-tokens a cache or ML predictor may return a runtime, but it is the runtime of the
    #    past run it matched. runtime_source records where the number comes from.
    if total_tokens > 0:
        runtime = total_tokens / throughput
        result["runtime_source"] = "total_tokens"
    else:
        runtime = _finite(prediction.predicted_runtime_seconds)
        result["runtime_source"] = "predictor_history" if runtime else None
    if runtime:
        result["predicted_runtime_seconds"] = runtime
        cluster_power = result["cluster_power_watts"]
        if cluster_power:
            result["energy_kwh"] = (cluster_power * runtime) / SECONDS_PER_HOUR / WH_PER_KWH

    return result
