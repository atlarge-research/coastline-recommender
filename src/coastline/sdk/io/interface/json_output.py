"""JSON output handler for recommendations."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from coastline.sdk.models.recommendation import Recommendation, round_floats_for_display


def recommendation_payload(
    recommendation: Recommendation,
    include_metadata: bool = True,
    rationale: Optional[str] = None,
) -> dict[str, Any]:
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
    # Presentation layer: round float noise (watts, throughput, scores) to 2
    # decimals for the emitted JSON. round_floats_for_display copies, so the
    # source Recommendation's metadata is left untouched at full precision.
    rounded: dict[str, Any] = round_floats_for_display(data)
    return rounded


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
