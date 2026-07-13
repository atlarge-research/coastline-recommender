"""Load strategy YAML for CLI and map orchestrator → predictors (same rules as API)."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

# The one default recommendation policy — multi_objective/balanced — shared by every surface
# (CLI, facade/API, UI). Used only when no config file is found.
_DEFAULT_STRATEGY_CONFIG: dict[str, Any] = {
    "strategy": {"name": "multi_objective", "preset": "balanced"},
    "predictors": {
        "performance": "intelligent",
        "energy": "kavier_power",
        "feasibility": "autoconf",
    },
    "grid": {
        "batch_sizes": [4, 8, 16, 32, 64],
        "total_gpus": [1, 2, 4, 8, 16],
        "top_k": 5,
    },
}

# The single canonical recommendation-policy config file. Every door falls back to this one
# ``experiment.yaml`` (there is no separate ``default.yaml``/``config.yaml`` any more); an env
# override lets a deployment point elsewhere. Repo root: io/ -> sdk/ -> coastline/ -> src/ -> repo.
_CONFIG_ENV_KEYS = ("EXPERIMENT_CONFIG", "STRATEGY_CONFIG", "CONFIG_FILE")
_CANONICAL_CONFIG = (
    Path(__file__).resolve().parents[4] / "config" / "coastline_functionality" / "experiment.yaml"
)


def default_experiment_path() -> Path:
    """The one recommendation-policy config every surface resolves to when none is given.
    An env override (``EXPERIMENT_CONFIG`` / ``STRATEGY_CONFIG`` / ``CONFIG_FILE``) wins; else the
    repo's ``experiment.yaml``. The path may not exist (stripped wheel) — callers then fall back to
    the built-in :data:`_DEFAULT_STRATEGY_CONFIG`."""
    for key in _CONFIG_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            return Path(value)
    return _CANONICAL_CONFIG


def _merge_dict(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_strategy_config(path: str | Path, *, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a strategy dict from experiment YAML (orchestrator section supported).

    ``default`` overrides the base config merged under the file (the UI passes its own
    multi_objective default; the CLI uses the built-in min_gpu default). This is the
    single source of the legacy ``orchestrator:`` -> ``predictors:`` translation — the
    CLI and the web UI both route through it, so the mapping can never diverge.
    """
    config = copy.deepcopy(default if default is not None else _DEFAULT_STRATEGY_CONFIG)
    path = Path(path)
    if not path.is_file():
        return config

    with open(path, encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    for section in ("strategy", "predictors", "grid", "output"):
        if section in loaded:
            if isinstance(loaded[section], dict):
                config[section] = _merge_dict(config.get(section, {}), loaded[section])
            else:
                config[section] = loaded[section]

    # Legacy orchestrator block is translated only when the config has no explicit
    # predictors section (mirrors api/main.py): a modern predictors block always
    # wins over a leftover orchestrator block.
    if "orchestrator" in loaded and "predictors" not in loaded:
        orch = loaded["orchestrator"] or {}
        # Translate legacy orchestrator predictor names; unknown names pass through.
        _LEGACY_PERF = {
            "cache_first": "intelligent",
            "physics": "kavier",
            "physics_driven": "kavier",
            "ensemble": "intelligent",
        }
        perf = orch.get("predictor", "intelligent")
        perf = _LEGACY_PERF.get(perf, perf)
        config["predictors"] = _merge_dict(
            config.get("predictors", {}),
            {
                "performance": perf,
                "energy": orch.get("energy", "kavier_power"),
                "feasibility": orch.get("feasibility", "autoconf"),
            },
        )

    return config
