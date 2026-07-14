"""Resolve + load the recommendation-policy YAML (strategy / predictors / grid)."""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# The one built-in recommendation-policy default (multi_objective/balanced), loaded from the
# bundled ``default_experiment.yaml`` so every surface (CLI, facade/API, UI) shares a single
# source instead of parallel hardcoded dicts. Used only when no config file is found.
_BUILTIN_DEFAULT_PATH = Path(__file__).parent / "default_experiment.yaml"


@lru_cache(maxsize=1)
def _load_builtin_default() -> dict[str, Any]:
    with open(_BUILTIN_DEFAULT_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def builtin_default_config() -> dict[str, Any]:
    """The one built-in recommendation-policy default (``strategy``/``predictors``/``grid``),
    loaded from the bundled ``default_experiment.yaml``. Every surface falls back to this when no
    config file is present. Returns a fresh deep copy — callers may mutate it freely."""
    return copy.deepcopy(_load_builtin_default())


# Module-level default dict (the deep copy every door merges under). Kept as a name for the
# surfaces + tests that reference it; its content is the bundled YAML, never a second literal.
_DEFAULT_STRATEGY_CONFIG: dict[str, Any] = builtin_default_config()

# The single canonical recommendation-policy config file. Every door falls back to this one
# ``experiment.yaml`` (there is no separate ``default.yaml``/``config.yaml`` any more); the
# ``EXPERIMENT_CONFIG`` env var lets a deployment point elsewhere. Repo root: io/ -> sdk/ ->
# coastline/ -> src/ -> repo.
_CONFIG_ENV_KEY = "EXPERIMENT_CONFIG"
_CANONICAL_CONFIG = Path(__file__).resolve().parents[4] / "config" / "coastline_functionality" / "experiment.yaml"


def default_experiment_path() -> Path:
    """The one recommendation-policy config every surface resolves to when none is given.
    The ``EXPERIMENT_CONFIG`` env var wins; else the repo's ``experiment.yaml``. The path may not
    exist (stripped wheel) — callers then fall back to :func:`builtin_default_config`."""
    override = os.environ.get(_CONFIG_ENV_KEY)
    return Path(override) if override else _CANONICAL_CONFIG


def _merge_dict(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_strategy_config(path: str | Path, *, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a recommendation-policy YAML merged under a base default (``strategy``/``predictors``/``grid``).

    ``default`` overrides the base config merged under the file; when omitted, the built-in
    default is used. An absent file returns the base default unchanged.
    """
    config = copy.deepcopy(default if default is not None else _DEFAULT_STRATEGY_CONFIG)
    path = Path(path)
    if not path.is_file():
        return config

    with open(path, encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    for section in ("strategy", "predictors", "grid"):
        if section in loaded:
            if isinstance(loaded[section], dict):
                config[section] = _merge_dict(config.get(section, {}), loaded[section])
            else:
                config[section] = loaded[section]

    return config
