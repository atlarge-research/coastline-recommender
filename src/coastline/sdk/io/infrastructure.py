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
    """Infrastructure YAML path: INFRASTRUCTURE_CONFIG if set, else the repo's config/ directory."""
    override = os.environ.get("INFRASTRUCTURE_CONFIG")
    if override:
        return Path(override)
    # parents[4] of this file is the repo root, which holds config/.
    return Path(__file__).resolve().parents[4] / "config" / "coastline_functionality" / "infrastructure.yaml"


@lru_cache(maxsize=1)
def load_infrastructure() -> Infrastructure:
    """Load the infrastructure config; fall back to the built-in defaults if it is missing."""
    path = _config_path()
    if path.is_file():
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return Infrastructure(**data)
        except Exception as exc:
            logger.warning("Could not parse %s (%s); using built-in defaults", path, exc)
    else:
        # An installed wheel has no config/ directory; only a missing override is a user error.
        log = logger.warning if os.environ.get("INFRASTRUCTURE_CONFIG") else logger.info
        log("Infrastructure config not found at %s; using built-in defaults", path)
    return Infrastructure(**_DEFAULTS)


def resolve_cluster_caps(cluster_gpus: Optional[int] = None, node_gpus: Optional[int] = None) -> tuple[int, int, int]:
    """Resolve the cluster GPU caps as ``(total_gpus, gpus_per_node, max_nodes)``.

    The cluster size comes from ``infrastructure.yaml``, written by the sysadmin, and is never read
    from the workload trace. ``cluster_gpus`` and ``node_gpus`` (the ``--cluster-gpus`` and
    ``--node-gpus`` flags) override it; with ``cluster_gpus`` set, ``max_nodes`` is derived from
    it, else the file's ``max_nodes`` is used. The result feeds ``SystemContext``, so the grid stays
    within the cluster. A value below 1 raises ValueError.
    """
    infra = load_infrastructure()
    total = int(cluster_gpus) if cluster_gpus is not None else infra.total_gpus
    if total < 1:
        raise ValueError(f"cluster GPUs must be >= 1, got {total}")
    per_node = int(node_gpus) if node_gpus is not None else infra.max_gpus_per_node
    if per_node < 1:
        raise ValueError(f"GPUs per node must be >= 1, got {per_node}")
    per_node = min(per_node, total)
    max_nodes = max(1, math.ceil(total / per_node)) if cluster_gpus is not None else infra.max_nodes
    return total, per_node, max_nodes
