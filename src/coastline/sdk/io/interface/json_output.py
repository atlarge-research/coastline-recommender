"""JSON output handler for recommendations."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from coastline.sdk.models.recommendation import Recommendation


def recommendation_payload(
    recommendation: Recommendation,
    include_metadata: bool = True,
    rationale: Optional[str] = None,
) -> dict:
    """The JSON-serialisable dict for one recommendation (shared by file + stdout output)."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "total_gpus": recommendation.total_gpus,
            "gpus_per_node": recommendation.gpus_per_node,
            "workers": recommendation.number_of_nodes,
        },
        "strategy": recommendation.strategy,
        "performance": {
            "throughput_tokens_per_sec": recommendation.predicted_throughput,
        },
    }
    if rationale:
        data["rationale"] = rationale

    # Gate on presence, not truthiness: a legitimately-measured 0.0 W is still a value to emit.
    if recommendation.metadata.get("predicted_power_watts") is not None:
        data["energy"] = {
            "power_watts": recommendation.metadata["predicted_power_watts"],
            "efficiency_tokens_per_watt": recommendation.metadata.get("tokens_per_watt", 0),
        }

    if include_metadata:
        data["metadata"] = recommendation.metadata
    return data


def save_recommendation_to_json(
    recommendation: Recommendation,
    filepath: Union[str, Path],
    include_metadata: bool = True,
    rationale: Optional[str] = None,
) -> None:
    """Save a recommendation to JSON, optionally with its metadata and a one-line
    rationale ('why this config')."""
    data = recommendation_payload(recommendation, include_metadata=include_metadata, rationale=rationale)
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def save_batch_recommendations(recommendations: List[Recommendation], filepath: Union[str, Path]) -> None:
    """Save multiple ranked recommendations to a JSON file."""
    data = {"timestamp": datetime.now().isoformat(), "count": len(recommendations), "recommendations": []}

    for i, rec in enumerate(recommendations, 1):
        rec_data = {
            "rank": i,
            "total_gpus": rec.total_gpus,
            "gpus_per_node": rec.gpus_per_node,
            "workers": rec.number_of_nodes,
            "strategy": rec.strategy,
            "throughput": rec.predicted_throughput,
        }

        if rec.metadata.get("predicted_power_watts") is not None:  # presence, not truthiness (0.0 W is valid)
            rec_data["power_watts"] = rec.metadata["predicted_power_watts"]
            rec_data["efficiency"] = rec.metadata.get("tokens_per_watt", 0)

        data["recommendations"].append(rec_data)

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
