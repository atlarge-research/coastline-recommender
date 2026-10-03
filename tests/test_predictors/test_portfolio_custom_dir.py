"""portfolio/custom/ holds only models a user tuned locally.

The resolver tries custom/ before the packaged portfolio, so a file committed there would replace
the bundled model, such as the thesis TabPFN (2.1% MdAPE), in every checkout. The git checks read
the repository, so they skip outside a git checkout (an sdist or an installed wheel).
"""

import shutil
import subprocess

import pytest

import coastline.sdk.predictors.performance.data_driven.ml_common as ml_common

_CUSTOM = ml_common._BUNDLED_PORTFOLIO_DIR / "custom"


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=_CUSTOM.parent, capture_output=True, text=True)


def _in_git_checkout() -> bool:
    return shutil.which("git") is not None and _git("rev-parse", "--is-inside-work-tree").stdout.strip() == "true"


needs_git = pytest.mark.skipif(not _in_git_checkout(), reason="not a git checkout")


@needs_git
def test_no_model_file_is_committed_under_custom():
    tracked = _git("ls-files", "--", str(_CUSTOM)).stdout.split()
    assert tracked == []


@needs_git
def test_custom_dir_is_git_ignored():
    # --no-index checks the ignore rules alone, whether or not the path is tracked.
    probe = _CUSTOM / "tabpfn.pkl"
    assert _git("check-ignore", "--no-index", "-q", str(probe)).returncode == 0


def test_tabpfn_resolves_to_the_packaged_model_without_a_local_tune(monkeypatch, tmp_path):
    # PORTFOLIO_DIR with no custom/ beneath it: the packaged portfolio/tabpfn.pkl is served.
    monkeypatch.setattr(ml_common, "PORTFOLIO_DIR", tmp_path)
    assert ml_common.performance_trained_model_path("tabpfn") == ml_common._BUNDLED_PORTFOLIO_DIR / "tabpfn.pkl"


def test_a_local_tune_still_takes_precedence(monkeypatch, tmp_path):
    tuned = tmp_path / "custom" / "tabpfn.pkl"
    tuned.parent.mkdir()
    tuned.write_bytes(b"x")
    monkeypatch.setattr(ml_common, "PORTFOLIO_DIR", tmp_path)
    assert ml_common.custom_models_dir() == tuned.parent
    assert ml_common.performance_trained_model_path("tabpfn") == tuned
