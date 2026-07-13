"""Format adapters: foreign CSV shape ↔ the canonical Coastline schema.

Importing this package registers the built-in adapters (``coastline`` identity, ``ibm_trace``)
so :func:`get_adapter` can resolve them by name.
"""

from __future__ import annotations

from coastline.sdk.io.adapters import coastline, ibm_trace  # noqa: F401 — register on import
from coastline.sdk.io.adapters.base import FormatAdapter, adapter_names, get_adapter, register

__all__ = ["FormatAdapter", "adapter_names", "get_adapter", "register"]
