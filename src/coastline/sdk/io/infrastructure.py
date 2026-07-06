"""Sysadmin-declared cluster infrastructure; read-only at runtime."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

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
