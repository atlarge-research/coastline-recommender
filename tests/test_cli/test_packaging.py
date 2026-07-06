"""Packaging guards: the uv-built wheel ships the whole ``coastline`` package (source +
bundled data + templates), excludes the heavy/dev bits, and every declared console-script
entry point actually imports.

The wheel build is opt-in: set ``COASTLINE_RUN_PACKAGING_TESTS=1`` to run it (it shells out
to ``uv build``). The entry-point import checks are fast and always run — they catch a
renamed/moved CLI target without building anything.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import tomllib
import zipfile
from pathlib import Path

import pytest

# Repo root: tests/test_cli/ -> tests/ -> <repo>.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Opt-in toggle: the wheel build stays out of the normal fast suite.
_RUN_BUILD = os.environ.get("COASTLINE_RUN_PACKAGING_TESTS") == "1"


def _declared_console_scripts() -> dict[str, str]:
    """The ``[project.scripts]`` table straight out of pyproject.toml.

    Deriving the targets from pyproject (rather than a hand-maintained copy) is what
    makes the import checks below an *independent* oracle: a script whose target is
    renamed in pyproject alone is caught here, because pyproject is the source of truth
    the packaging backend actually consumes.
    """
    with (_REPO_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh).get("project", {}).get("scripts", {})


_DECLARED_SCRIPTS = _declared_console_scripts()


def test_documented_public_commands_are_declared_as_console_scripts():
    """The two documented public commands ship as console scripts.

    Oracle (independent of the pyproject table under test): the deployment surface is
    the single ``coastline`` dispatcher (facade/batch/recommend) plus ``coastline-ui``
    (the FastAPI dashboard launched by ``make gui``). Both MUST be installed on the
    PATH by a `pip install coastline`; a subset check, so adding a new script doesn't
    spuriously fail, but dropping either goes red.
    """
    assert {"coastline", "coastline-ui"} <= set(_DECLARED_SCRIPTS), (
        f"missing documented command(s); declared scripts = {sorted(_DECLARED_SCRIPTS)}"
    )


@pytest.mark.parametrize("script_name", sorted(_DECLARED_SCRIPTS))
def test_console_script_target_resolves_to_a_callable(script_name):
    """Every declared console-script ``module:attr`` target imports to a callable.

    Falsification: a target renamed/moved in pyproject (e.g. ``coastline.cli:main`` ->
    a typo, or a deleted attribute) raises ImportError/AttributeError, or points at a
    non-callable, and this goes red — that is exactly the broken-entry-point bug a user
    would hit the first time they invoke the installed command.
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
    """`uv build --wheel` ships the whole src/coastline tree (code + bundled data +
    templates), and excludes model pickles, tests, and dev tooling."""
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

    # The whole single package ships (facade, engine, cli, ui) — one root, no PYTHONPATH.
    assert "coastline/__init__.py" in names
    assert "coastline/sdk/recommend/facade.py" in names
    assert "coastline/sdk/models/workload.py" in names
    assert "coastline/cli/main.py" in names
    assert "coastline/ui/app.py" in names
    assert "coastline/py.typed" in names

    # Bundled data + FastAPI templates ride along automatically (uv_build ships the tree).
    assert "coastline/sdk/io/data/sample_raw_trace.csv" in names
    assert "coastline/ui/templates/index.html" in names

    # The parametric model subset ships bundled in the wheel (so pip install coastline[ml] serves
    # them); the large/instance-based ones (tabpfn, random_forest, knn, gaussian_process, svr) do NOT.
    _portfolio = "coastline/sdk/predictors/performance/data_driven/portfolio/"
    for stem in ("catboost", "xgboost", "lightgbm", "bayesian_ridge"):
        assert f"{_portfolio}performance_{stem}_featv3.pkl" in names, f"{stem} model not bundled"
    assert any(n.startswith(f"{_portfolio}performance_deep_learning_featv3/") for n in names)
    # (predictor code for these ships; only their trained artifacts must not).
    for excluded in ("tabpfn", "random_forest", "knn", "gaussian_process", "svr"):
        assert not [n for n in names if f"performance_{excluded}_featv3" in n], (
            f"{excluded} model artifact must NOT ship in the public wheel"
        )
    assert not [n for n in names if "/tests/" in n], "wheel must not ship test packages"

    # Dev-only tooling and repo-root artifact dirs never make it into the package tree.
    for stray in ("dev/", "benchmark/", "models/"):
        assert not [n for n in names if n.startswith(stray)], f"wheel must not ship {stray}"

    # LICENSE is carried in the wheel metadata directory.
    assert any(n.endswith("/LICENSE") or n.endswith(".dist-info/licenses/LICENSE") for n in names), (
        f"LICENSE not bundled; sample names: {names[:20]}"
    )
