"""Sysadmin-declared cluster infrastructure; read-only at runtime."""

from __future__ import annotations

import logging
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "total_gpus": 64,
    "max_nodes": 8,
    "max_gpus_per_node": 8,
    "gpu_models": ["NVIDIA-A100-SXM4-80GB"],
}


class Infrastructure(BaseModel):
    """Cluster capacity advertised to the user and enforced server-side."""

    total_gpus: int = Field(..., ge=1, description="Cluster-wide GPU budget")
    max_nodes: int = Field(..., ge=1, description="Maximum number of nodes available")
    max_gpus_per_node: int = Field(..., ge=1, description="Maximum GPUs per node")
    gpu_models: list[str] = Field(..., min_length=1, description="GPU types physically present in the cluster")


def _config_path() -> Path:
    """Infrastructure YAML path; honors INFRASTRUCTURE_CONFIG, else <repo>/coastline/config."""
    override = os.environ.get("INFRASTRUCTURE_CONFIG")
    if override:
        return Path(override)
    # src/coastline/sdk/io/infrastructure.py -> parents[4] == the coastline repo root (holds config/).
    return Path(__file__).resolve().parents[4] / "config" / "coastline_functionality" / "infrastructure.yaml"


@lru_cache(maxsize=1)
def load_infrastructure() -> Infrastructure:
    """Load infrastructure config; falls back to built-in defaults with a warning if missing."""
    path = _config_path()
    if path.is_file():
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return Infrastructure(**data)
        except Exception as exc:
            logger.warning("Could not parse %s (%s); using built-in defaults", path, exc)
    else:
        logger.warning("Infrastructure config not found at %s; using built-in defaults", path)
    return Infrastructure(**_DEFAULTS)


def resolve_cluster_caps(cluster_gpus: Optional[int] = None, node_gpus: Optional[int] = None) -> tuple[int, int, int]:
    """Resolve the cluster GPU caps as ``(total_gpus, gpus_per_node, max_nodes)``.

    The cluster size is sysadmin-declared in ``infrastructure.yaml`` — it is deliberately NOT read
    from the workload trace (a trace must never carry cluster topology). The optional ``cluster_gpus``
    / ``node_gpus`` arguments (from a ``--cluster-gpus`` / ``--node-gpus`` CLI flag) override the
    declared totals; when ``cluster_gpus`` is given, ``max_nodes`` is derived from it, otherwise the
    file's declared ``max_nodes`` is used. The returned triple feeds ``SystemContext`` so the grid
    never proposes a layout larger than the cluster.
    """
    infra = load_infrastructure()
    total = int(cluster_gpus) if cluster_gpus else infra.total_gpus
    if total < 1:
        raise ValueError(f"cluster GPUs must be >= 1, got {total}")
    per_node = int(node_gpus) if node_gpus else infra.max_gpus_per_node
    per_node = max(1, min(per_node, total))
    max_nodes = max(1, math.ceil(total / per_node)) if cluster_gpus else infra.max_nodes
    return total, per_node, max_nodes
