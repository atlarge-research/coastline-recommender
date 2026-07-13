"""Format-adapter boundary: foreign CSV shape ↔ the canonical Coastline schema.

An adapter owns *format* only — it maps a foreign column layout to canonical workload rows
(``to_canonical``) and maps a recommendation result back to the foreign shape
(``from_canonical``). It carries no recommendation *policy*: the grid search, feasibility,
prediction, and any per-row fallback logic live in the recommend core / caller, which only
ever sees canonical rows. Register an adapter by ``name`` so a CLI ``--adapter`` flag or an
API ``format=`` field can select it; ``coastline`` (identity) is the default.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class FormatAdapter(Protocol):
    """Maps one foreign CSV shape to/from the canonical Coastline workload schema."""

    name: str

    def to_canonical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Foreign input rows → canonical workload rows (``schema`` column names)."""
        ...

    def from_canonical(self, recommended: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
        """Canonical recommendation rows + the original foreign rows → foreign output shape."""
        ...


_REGISTRY: dict[str, FormatAdapter] = {}


def register(adapter: FormatAdapter) -> FormatAdapter:
    """Register an adapter under its ``name`` (idempotent; last registration wins)."""
    _REGISTRY[adapter.name] = adapter
    return adapter


def get_adapter(name: str) -> FormatAdapter:
    """Resolve a registered adapter by name, raising with the known names on a miss."""
    key = (name or "coastline").strip().lower()
    if key not in _REGISTRY:
        raise ValueError(f"unknown format adapter {name!r}; choose from {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def adapter_names() -> list[str]:
    """The names of all registered adapters."""
    return sorted(_REGISTRY)
