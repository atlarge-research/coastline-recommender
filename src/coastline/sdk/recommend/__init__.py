"""Public entry points over the pipeline and policy engine. All three run the same engine:

    Coastline      configured recommender: choose a predictor once, call it per workload
    recommend      DataFrame, list of dicts or dict in, DataFrame of ranked configs out
    recommend_csv  CSV in, CSV out

The names are loaded lazily from their modules (PEP 562), so importing this package is cheap.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

_FACADE_EXPORTS = {
    "Coastline": "coastline.sdk.recommend.facade",
    "WorkloadInput": "coastline.sdk.recommend.facade",
    "recommend": "coastline.sdk.recommend.batch_api",
    "recommend_csv": "coastline.sdk.recommend.batch_csv",
}

__all__ = ["Coastline", "WorkloadInput", "recommend", "recommend_csv"]

if TYPE_CHECKING:
    from coastline.sdk.recommend.batch_api import recommend as recommend
    from coastline.sdk.recommend.batch_csv import recommend_csv as recommend_csv
    from coastline.sdk.recommend.facade import (
        Coastline as Coastline,
    )
    from coastline.sdk.recommend.facade import (
        WorkloadInput as WorkloadInput,
    )


def __getattr__(name: str) -> Any:
    target = _FACADE_EXPORTS.get(name)
    if target is not None:
        value = getattr(importlib.import_module(target), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
