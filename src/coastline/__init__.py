"""Coastline — context-aware GPU/datacenter configuration recommender for LLM fine-tuning.

A bare ``import coastline`` stays light: the public verbs/classes are resolved lazily
from :mod:`coastline.sdk.recommend` on first access (PEP 562), so pandas / kavier / the
predictor backends are not imported until you actually call one. The module is also
callable — ``coastline(throughput_estim=...)`` returns a configured :class:`Coastline`.
"""

from __future__ import annotations

import importlib
import sys as _sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version
from typing import TYPE_CHECKING, Any

try:
    __version__ = _version("coastline-recommender")
except PackageNotFoundError:  # pragma: no cover - source tree without installed metadata
    __version__ = "0.0.0+unknown"

# Public name -> the module it is resolved from (all live under coastline.sdk.recommend).
_LAZY_ATTRS = {
    "recommend": "coastline.sdk.recommend",
    "recommend_csv": "coastline.sdk.recommend",
    "Coastline": "coastline.sdk.recommend",
    "WorkloadInput": "coastline.sdk.recommend",
}

__all__ = ["Coastline", "WorkloadInput", "recommend", "recommend_csv", "__version__"]

if TYPE_CHECKING:
    from coastline.sdk.recommend import (
        Coastline as Coastline,
    )
    from coastline.sdk.recommend import (
        WorkloadInput as WorkloadInput,
    )
    from coastline.sdk.recommend import (
        recommend as recommend,
    )
    from coastline.sdk.recommend import (
        recommend_csv as recommend_csv,
    )


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is not None:
        value = getattr(importlib.import_module(target), name)
        globals()[name] = value  # cache so subsequent access skips __getattr__
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set([*__all__, *globals().keys()]))


class _CallableModule(_sys.modules[__name__].__class__):
    """Make ``coastline(throughput_estim=...)`` return a configured Coastline.

    Subclassing the module type preserves PEP 562 ``__getattr__`` (the lazy attrs above)
    while adding ``__call__``.
    """

    def __call__(self, throughput_estim: str = "kavier", **kwargs: Any) -> "Coastline":
        from coastline.sdk.recommend import Coastline

        return Coastline(throughput_estim=throughput_estim, **kwargs)


_sys.modules[__name__].__class__ = _CallableModule
