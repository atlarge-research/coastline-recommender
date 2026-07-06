"""
Regression tests for ``PolicyFactory.load_config`` / ``create_strategy`` default
config resolution in ``coastline/recommendation_policies/__init__.py``.

The bug being guarded: ``load_config`` defaulted to a package-relative
``recommender/config/experiment.yaml`` that does not exist, so calling
``PolicyFactory.create_strategy()`` with no ``config`` argument raised a bare
``FileNotFoundError`` at ``open()`` time.

The fix makes the *no-config* path robust:
  - try a list of default config files (package-relative ``experiment.yaml``,
    then the repo-root ``config/coastline_functionality/experiment.yaml`` and
    ``config/coastline_functionality/default.yaml``),
  - and fall back to a built-in default config when none of them exist —
    instead of crashing.

Behaviour when an *explicit* config path is passed is unchanged (the file is
loaded directly and errors propagate), which these tests also pin.

The end-to-end ``create_strategy()`` checks use a Kavier-only predictor config so
the factory never loads the heavy ML artifacts (xgboost/catboost) or Kavier
physics that could segfault the host — mirroring ``test_strategies.py``.

Run:
  cd <repo> && PYTHONPATH=coastline:coastline/common:kavier/src \
    DATA_DIR=./trace-archive .venv/bin/python -m pytest \
    coastline/tests/test_policy_factory_config.py -q
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

# A predictor config that keeps the factory off the real ML models / physics:
# Kavier throughput + Kavier power + rules-only feasibility (same trick the
# strategy tests use to build recommendation_policies without loading xgboost/catboost).
_KAVIER_PREDICTORS = {
    "performance": "kavier",
    "energy": "kavier_power",
    "feasibility": "rules",
}


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# ===========================================================================
# load_config() — explicit path: behaviour MUST be identical to before
# ===========================================================================
class TestLoadConfigExplicitPath:
    def test_explicit_valid_path_is_loaded_verbatim(self, tmp_path):
        """An explicit, existing config file is parsed and returned unchanged."""
        payload = {
            "strategy": {"name": "min_gpu", "preset": "balanced"},
            "grid": {"batch_sizes": [4], "total_gpus": [1, 2], "top_k": 3},
        }
        cfg_file = _write_yaml(tmp_path / "custom.yaml", payload)

        loaded = PolicyFactory.load_config(str(cfg_file))
        assert loaded == payload

    def test_explicit_missing_path_still_raises_file_not_found(self, tmp_path):
        """The explicit-path contract is unchanged: a bad path still raises.

        Only the *no-argument* (default) path is made robust by the fix; passing
        a wrong explicit path is a caller error and must surface as before.
        """
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError):
            PolicyFactory.load_config(str(missing))


# ===========================================================================
# load_config() — no path: default resolution + built-in fallback (the fix)
# ===========================================================================
class TestLoadConfigDefaultResolution:
    def test_no_arg_uses_first_existing_default_file(self):
        """With the real repo layout, the first existing default file wins.

        ``config/coastline_functionality/experiment.yaml`` precedes
        ``config/coastline_functionality/default.yaml`` in the candidate order,
        and declares ``multi_objective`` (default.yaml declares ``min_gpu``), so
        this also pins the ordering. It also covers the core FileNotFoundError
        regression: this call is the no-arg path that used to crash.
        """
        experiment = _REPO_ROOT / "config" / "coastline_functionality" / "experiment.yaml"
        default = _REPO_ROOT / "config" / "coastline_functionality" / "default.yaml"
        assert experiment.is_file(), f"expected {experiment} to exist"
        assert default.is_file(), f"expected {default} to exist"

        config = PolicyFactory.load_config()
        expected = yaml.safe_load(experiment.read_text(encoding="utf-8"))
        # Oracle: independent re-read of experiment.yaml. Returned verbatim (no
        # translation) AND it is experiment.yaml, not default.yaml — the two files
        # declare different strategies, so the strategy name discriminates which
        # file won the ordering race.
        assert config == expected
        assert config["strategy"]["name"] == "multi_objective"
        # Cross-check: default.yaml (the LATER candidate) declares min_gpu, so if
        # ordering were reversed this would have surfaced instead.
        default_cfg = yaml.safe_load(default.read_text(encoding="utf-8"))
        assert default_cfg["strategy"]["name"] == "min_gpu"
        assert config["strategy"]["name"] != default_cfg["strategy"]["name"]

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

        # Point the candidate list at this fake repo root: experiment.yaml is
        # absent, so default.yaml must be picked.
        monkeypatch.setattr(
            PolicyFactory,
            "_default_config_candidates",
            staticmethod(
                lambda: [
                    cfg_dir / "experiment.yaml",  # missing
                    cfg_dir / "default.yaml",  # present -> used
                ]
            ),
        )
        config = PolicyFactory.load_config()
        assert config == payload
        assert config["strategy"]["name"] == "min_gpu"

    def test_built_in_default_when_no_files_exist(self, tmp_path, monkeypatch):
        """No default file anywhere -> built-in default config, NOT a crash."""
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
        # A fresh DEEP copy is returned so callers can't mutate the module constant.
        assert config is not _BUILTIN_DEFAULT_CONFIG
        # Invariant (independent of ==): mutating a NESTED value in the returned
        # config must not bleed into the shared module constant. A shallow copy
        # would share the nested dict and fail this.
        original_name = _BUILTIN_DEFAULT_CONFIG["strategy"]["name"]
        config["strategy"]["name"] = "mutated-by-caller"
        assert _BUILTIN_DEFAULT_CONFIG["strategy"]["name"] == original_name
        # And a second call is unaffected by the first caller's mutation.
        assert PolicyFactory.load_config()["strategy"]["name"] == original_name

    def test_default_candidates_include_repo_top_level_config(self):
        """The repo's top-level config/ dir is among the default candidates.

        This is the directory that actually holds experiment.yaml/default.yaml,
        so its presence in the list is what makes the no-arg path resolve.
        """
        candidates = [Path(p) for p in PolicyFactory._default_config_candidates()]
        experiment = _REPO_ROOT / "config" / "coastline_functionality" / "experiment.yaml"
        default = _REPO_ROOT / "config" / "coastline_functionality" / "default.yaml"
        assert default in candidates
        assert experiment in candidates
        # Precedence contract: experiment.yaml must be tried BEFORE default.yaml.
        # This is the ordering that test_no_arg_uses_first_existing_default_file
        # relies on; pin it here independently of which files happen to exist.
        assert candidates.index(experiment) < candidates.index(default)


# ===========================================================================
# create_strategy() — the end-to-end symptom: no-arg must build a strategy
# ===========================================================================
class TestCreateStrategyNoConfig:
    def _patch_candidates(self, monkeypatch, tmp_path, strategy_name, preset="balanced"):
        """Make the no-arg default resolution land on a Kavier-only temp config."""
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
        """create_strategy() with no config no longer raises FileNotFoundError.

        It resolves a default config and honours its ``strategy.name``.
        """
        self._patch_candidates(monkeypatch, tmp_path, "min_gpu")
        strat = PolicyFactory.create_strategy()
        assert isinstance(strat, MinGPUStrategy)
        assert strat.get_name() == "min_gpu"

    @pytest.mark.parametrize("preset", ["balanced", "energy", "performance"])
    def test_create_strategy_no_config_respects_default_strategy_name(self, tmp_path, monkeypatch, preset):
        """A default file declaring multi_objective yields a multi-objective strategy.

        Oracle: get_name() is composed as ``multi_objective_{preset}`` (see
        MultiObjectiveStrategy.get_name). Varying the preset varies the behavior
        (the name suffix), not just a constant — a bug that hard-coded the suffix
        or dropped the preset would make the energy/performance cases fail.
        """
        self._patch_candidates(monkeypatch, tmp_path, "multi_objective", preset=preset)
        strat = PolicyFactory.create_strategy()
        assert isinstance(strat, MultiObjectiveStrategy)
        assert strat.get_name() == f"multi_objective_{preset}"

    def test_explicit_config_still_takes_precedence_over_defaults(self, tmp_path, monkeypatch):
        """An explicit config passed to create_strategy() bypasses default lookup.

        Behaviour with an explicit valid config is identical to before: the
        default-candidate resolution must not even be consulted.
        """

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
