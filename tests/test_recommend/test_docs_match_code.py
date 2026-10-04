"""The CHANGELOG, the docs and the dashboard help say what the code does.

- The configs the CHANGELOG names as leaving top_k unset do leave it unset.
- The dashboard help says min_gpu returns one configuration unless the config sets top_k.
- The CHANGELOG names the explain column rename and the max_gpus check.
- The docs name every skip of the min-GPU loop, also a GPU count that needs too many nodes.
"""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

import pytest
import yaml

import coastline
from coastline.sdk.constants import SelectionPolicy
from coastline.sdk.pipeline.grid import grid_config_from_dict

_REPO = Path(__file__).resolve().parents[2]
_CONFIG_DIRS = (_REPO / "config", _REPO / "src" / "coastline" / "sdk" / "io")


def _unreleased() -> str:
    text = (_REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    return text.split("## [Unreleased]", 1)[1].split("\n## [", 1)[0]


def _bullets() -> list[str]:
    """The [Unreleased] bullets, each on one line."""
    return [" ".join(item.split()) for item in re.split(r"\n- ", _unreleased())[1:]]


def _config(name: str) -> Path:
    found = [path for folder in _CONFIG_DIRS for path in folder.rglob(name)]
    assert len(found) == 1, (name, found)
    return found[0]


def test_the_configs_named_as_leaving_top_k_unset_leave_it_unset():
    bullet = next(item for item in _bullets() if "`top_k` unset" in item)
    names = re.findall(r"`([\w.-]+\.yaml)`", bullet)
    assert len(names) >= 2, bullet
    assert "bundled configs" not in bullet
    for name in names:
        grid = grid_config_from_dict(yaml.safe_load(_config(name).read_text(encoding="utf-8")))
        assert grid.top_k is None, name
        assert grid.top_k_for(SelectionPolicy.MIN_GPU) == 1


def test_the_dashboard_help_conditions_one_configuration_on_top_k():
    page = (resources.files("coastline.ui") / "templates" / "index.html").read_text(encoding="utf-8")
    help_text = " ".join(page.split())
    assert "Minimum GPUs returns one configuration unless the config sets top_k" in help_text
    assert "Minimum GPUs returns one configuration and" not in help_text


def test_the_changelog_names_the_explain_column_rename():
    bullet = next(item for item in _bullets() if "`r_score`" in item)
    for name in ("`p_score`", "`t_score`", "`e_score`", "min_gpu"):
        assert name in bullet, name


def test_the_changelog_names_the_max_gpus_check():
    assert "max_gpus must be >= 1" in _unreleased()
    out = coastline.recommend(
        [
            {
                "llm_model": "mistral-7b-v0.1",
                "fine_tuning_method": "lora",
                "gpu_model": "NVIDIA-A100-SXM4-80GB",
                "tokens_per_sample": 1024,
                "batch_size": 8,
                "max_gpus": 0,
            }
        ],
        predictor="kavier",
        feasibility="rules",
    )
    assert out.iloc[0]["error"] == "max_gpus must be >= 1, got 0"


@pytest.mark.parametrize("skip", ["does not divide the total batch", "less than one sample per device", "more nodes"])
def test_the_docs_name_each_skip_of_the_min_gpu_loop(skip):
    docs = (_REPO / "docs" / "recommendation.md").read_text(encoding="utf-8")
    section = " ".join(docs.split("## Min-GPU", 1)[1].split())
    assert skip in section
