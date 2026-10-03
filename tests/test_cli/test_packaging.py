"""Packaging checks: the uv-built wheel ships the whole ``coastline`` package (source, bundled
data and templates) without the large model artifacts and dev files, and every declared console
script imports.

The wheel build runs only with ``COASTLINE_RUN_PACKAGING_TESTS=1`` (it calls ``uv build``). The
entry-point import checks are fast and always run; they catch a renamed or moved CLI target
without a build.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import tomllib
import zipfile
from pathlib import Path

import pytest

# Repo root, two levels above tests/test_cli/.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# The wheel build stays out of the normal suite unless this is set.
_RUN_BUILD = os.environ.get("COASTLINE_RUN_PACKAGING_TESTS") == "1"


def _declared_console_scripts() -> dict[str, str]:
    """The ``[project.scripts]`` table from pyproject.toml.

    The targets are read from the file the build backend uses, so a target renamed only in
    pyproject.toml is still checked.
    """
    with (_REPO_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh).get("project", {}).get("scripts", {})


_DECLARED_SCRIPTS = _declared_console_scripts()


def test_documented_public_commands_are_declared_as_console_scripts():
    """The two documented commands, ``coastline`` (the CLI dispatcher) and ``coastline-ui``
    (the FastAPI dashboard), are declared as console scripts.

    A subset check: an added script passes, a missing one fails.
    """
    assert {"coastline", "coastline-ui"} <= set(_DECLARED_SCRIPTS), (
        f"missing documented command(s); declared scripts = {sorted(_DECLARED_SCRIPTS)}"
    )


@pytest.mark.parametrize("script_name", sorted(_DECLARED_SCRIPTS))
def test_console_script_target_resolves_to_a_callable(script_name):
    """Every declared console-script ``module:attr`` target imports to a callable.

    A mistyped, moved or deleted target fails here with ImportError or AttributeError, before
    a user runs the installed command.
    """
    target = _DECLARED_SCRIPTS[script_name]  # e.g. "coastline.cli:main"
    assert target.count(":") == 1, f"malformed entry-point spec: {target!r}"
    module_path, _, attr = target.partition(":")

    module = importlib.import_module(module_path)
    resolved = module
    for part in attr.split("."):  # entry-point object-refs may be dotted
        resolved = getattr(resolved, part)
    assert callable(resolved), f"{script_name} target {target!r} is not callable"


@pytest.mark.skipif(
    not _RUN_BUILD,
    reason="wheel-build packaging test is opt-in (set COASTLINE_RUN_PACKAGING_TESTS=1)",
)
def test_wheel_ships_package_not_heavy_artifacts(tmp_path):
    """`uv build --wheel` ships the src/coastline tree (code, bundled data, templates and the
    small model pickles) and leaves out the large model pickles, tests and dev tooling."""
    out_dir = tmp_path / "wheelhouse"
    proc = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(out_dir)],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"uv build failed:\n{proc.stdout}\n{proc.stderr}"

    wheels = list(out_dir.glob("coastline*.whl"))  # distribution is coastline_recommender-*; import pkg is coastline
    assert len(wheels) == 1, f"expected exactly one coastline wheel, got {wheels}"

    with zipfile.ZipFile(wheels[0]) as zf:
        names = zf.namelist()

    # The whole package ships (facade, engine, cli, ui) under one root, with no PYTHONPATH.
    assert "coastline/__init__.py" in names
    assert "coastline/sdk/recommend/facade.py" in names
    assert "coastline/sdk/models/workload.py" in names
    assert "coastline/cli/main.py" in names
    assert "coastline/ui/app.py" in names
    assert "coastline/py.typed" in names

    # Bundled data and FastAPI templates ship with the tree (uv_build).
    assert "coastline/sdk/io/data/sample_raw_trace.csv" in names
    assert "coastline/sdk/io/data/run_database.csv" in names  # what `lookup: default` reads
    assert "coastline/ui/templates/index.html" in names

    # The parametric models ship in the wheel, so the [ml] extra can serve them; the large or
    # instance-based ones (tabpfn, random_forest, knn, gaussian_process, svr) do not.
    _portfolio = "coastline/sdk/predictors/performance/data_driven/portfolio/"
    for stem in ("catboost", "xgboost", "lightgbm", "bayesian_ridge"):
        assert f"{_portfolio}{stem}.pkl" in names, f"{stem} model not bundled"
    assert any(n.startswith(f"{_portfolio}deep_learning/") for n in names)
    # Their predictor code ships; their trained artifacts do not.
    for excluded in ("tabpfn", "random_forest", "knn", "gaussian_process", "svr"):
        assert not [n for n in names if n.endswith(f"/{excluded}.pkl")], (
            f"{excluded} model artifact must not ship in the public wheel"
        )
    assert not [n for n in names if "/tests/" in n], "wheel must not ship test packages"

    # Dev-only tooling and repo-root artifact dirs never make it into the package tree.
    for stray in ("dev/", "benchmark/", "models/"):
        assert not [n for n in names if n.startswith(stray)], f"wheel must not ship {stray}"

    # LICENSE is carried in the wheel metadata directory.
    assert any(n.endswith("/LICENSE") or n.endswith(".dist-info/licenses/LICENSE") for n in names), (
        f"LICENSE not bundled; sample names: {names[:20]}"
    )
