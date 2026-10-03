"""Default config resolution in ``PolicyFactory.load_config`` and ``create_strategy``.

Without a path, load_config reads the first existing default candidate (normally
config/coastline_functionality/experiment.yaml, or the file named by EXPERIMENT_CONFIG) and
otherwise returns a built-in default config. An explicit path is loaded as given, and a missing
file raises. The create_strategy tests use Kavier and rules feasibility, so no ML model is loaded.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from coastline.sdk.policies import (
    _BUILTIN_DEFAULT_CONFIG,
    _REPO_ROOT,
    PolicyFactory,
)
from coastline.sdk.policies.min_gpu import MinGPUStrategy
from coastline.sdk.policies.multi_objective import MultiObjectiveStrategy

# Kavier throughput and power with rules feasibility, so building a strategy loads no ML model.
_KAVIER_PREDICTORS = {
    "performance": "kavier",
    "energy": "kavier_power",
    "feasibility": "rules",
}


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# load_config() with an explicit path
class TestLoadConfigExplicitPath:
    def test_explicit_valid_path_is_loaded_unchanged(self, tmp_path):
        """An existing config file passed by path is parsed and returned unchanged."""
        payload = {
            "strategy": {"name": "min_gpu", "preset": "balanced"},
            "grid": {"batch_sizes": [4], "total_gpus": [1, 2], "top_k": 3},
        }
        cfg_file = _write_yaml(tmp_path / "custom.yaml", payload)

        loaded = PolicyFactory.load_config(str(cfg_file))
        assert loaded == payload

    def test_explicit_missing_path_still_raises_file_not_found(self, tmp_path):
        """A missing path raises FileNotFoundError; only a call without a path falls back."""
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError):
            PolicyFactory.load_config(str(missing))


# load_config() without a path: the default candidates, then the built-in config
class TestLoadConfigDefaultResolution:
    def test_no_arg_uses_the_canonical_experiment_yaml(self):
        """Without a path, load_config returns the contents of experiment.yaml unchanged."""
        experiment = _REPO_ROOT / "config" / "coastline_functionality" / "experiment.yaml"
        assert experiment.is_file(), f"expected {experiment} to exist"

        config = PolicyFactory.load_config()
        expected = yaml.safe_load(experiment.read_text(encoding="utf-8"))
        assert config == expected
        assert config["strategy"]["name"] == "multi_objective"

    def test_falls_back_to_default_yaml_when_experiment_absent(self, tmp_path, monkeypatch):
        """When only ``default.yaml`` exists, it is used (next in candidate order)."""
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        payload = {
            "strategy": {"name": "min_gpu", "preset": "balanced"},
            "grid": {"total_gpus": [1, 2, 4], "batch_sizes": [4]},
        }
        _write_yaml(cfg_dir / "default.yaml", payload)
        assert not (cfg_dir / "experiment.yaml").exists()

        # experiment.yaml is missing from these candidates, so default.yaml is used.
        monkeypatch.setattr(
            PolicyFactory,
            "_default_config_candidates",
            staticmethod(
                lambda: [
                    cfg_dir / "experiment.yaml",  # missing
                    cfg_dir / "default.yaml",  # present
                ]
            ),
        )
        config = PolicyFactory.load_config()
        assert config == payload
        assert config["strategy"]["name"] == "min_gpu"

    def test_built_in_default_when_no_files_exist(self, tmp_path, monkeypatch):
        """With no default file, load_config returns the built-in default config."""
        monkeypatch.setattr(
            PolicyFactory,
            "_default_config_candidates",
            staticmethod(
                lambda: [
                    tmp_path / "missing-experiment.yaml",
                    tmp_path / "missing-default.yaml",
                ]
            ),
        )
        config = PolicyFactory.load_config()
        assert config == _BUILTIN_DEFAULT_CONFIG
        # A deep copy, so callers cannot change the module constant.
        assert config is not _BUILTIN_DEFAULT_CONFIG
        # Changing a nested value leaves the constant alone (a shallow copy would share the
        # nested dict).
        original_name = _BUILTIN_DEFAULT_CONFIG["strategy"]["name"]
        config["strategy"]["name"] = "mutated-by-caller"
        assert _BUILTIN_DEFAULT_CONFIG["strategy"]["name"] == original_name
        # A second call does not see the change.
        assert PolicyFactory.load_config()["strategy"]["name"] == original_name

    def test_default_candidates_point_at_the_canonical_experiment_yaml(self):
        """The default candidates include config/coastline_functionality/experiment.yaml."""
        candidates = [Path(p) for p in PolicyFactory._default_config_candidates()]
        experiment = _REPO_ROOT / "config" / "coastline_functionality" / "experiment.yaml"
        assert experiment in candidates


# create_strategy() without a config
class TestCreateStrategyNoConfig:
    def _patch_candidates(self, monkeypatch, tmp_path, strategy_name, preset="balanced"):
        """Make the default lookup find a temporary config that uses Kavier."""
        cfg_file = tmp_path / "default.yaml"
        cfg_file.write_text(
            textwrap.dedent(
                f"""
                strategy:
                  name: {strategy_name}
                  preset: {preset}
                predictors:
                  performance: kavier
                  energy: kavier_power
                  feasibility: rules
                grid:
                  batch_sizes: [4]
                  total_gpus: [1, 2]
                  top_k: 3
                """
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            PolicyFactory,
            "_default_config_candidates",
            staticmethod(lambda: [tmp_path / "experiment.yaml", cfg_file]),
        )

    def test_create_strategy_no_config_builds_from_default_file(self, tmp_path, monkeypatch):
        """create_strategy() without a config uses the default file and its ``strategy.name``."""
        self._patch_candidates(monkeypatch, tmp_path, "min_gpu")
        strat = PolicyFactory.create_strategy()
        assert isinstance(strat, MinGPUStrategy)
        assert strat.get_name() == "min_gpu"

    @pytest.mark.parametrize("preset", ["balanced", "energy", "performance"])
    def test_create_strategy_no_config_respects_default_strategy_name(self, tmp_path, monkeypatch, preset):
        """A default file naming multi_objective gives a MultiObjectiveStrategy named
        multi_objective_<preset>."""
        self._patch_candidates(monkeypatch, tmp_path, "multi_objective", preset=preset)
        strat = PolicyFactory.create_strategy()
        assert isinstance(strat, MultiObjectiveStrategy)
        assert strat.get_name() == f"multi_objective_{preset}"

    def test_explicit_config_still_takes_precedence_over_defaults(self, tmp_path, monkeypatch):
        """A config passed to create_strategy() is used without looking up the default candidates."""

        def _boom():  # pragma: no cover - must never be called
            raise AssertionError("default candidate lookup should not run")

        monkeypatch.setattr(PolicyFactory, "_default_config_candidates", staticmethod(_boom))
        explicit = {
            "strategy": {"name": "min_gpu"},
            "predictors": dict(_KAVIER_PREDICTORS),
            "grid": {"batch_sizes": [4], "total_gpus": [1, 2], "top_k": 3},
        }
        strat = PolicyFactory.create_strategy(config=explicit)
        assert isinstance(strat, MinGPUStrategy)
