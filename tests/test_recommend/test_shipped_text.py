"""Shipped text reads in plain words.

- No thesis LaTeX label (such as a tab:, lst: or eq: label) in the package, the docs, the configs
  or the tests: a reader of the package cannot look one up.
- The min_gpu error counts GPUs in the singular when it checked 1 GPU only.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

import coastline.sdk.policies as policies
from coastline.sdk.exceptions import NoFeasibleGPUCountError
from coastline.sdk.models.workload import WorkloadSpec

_REPO = Path(__file__).resolve().parents[2]
_LABEL = re.compile(r"\b(?:tab|lst|eq|fig|sec|alg|chap):[A-Za-z]")
_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".html", ".js", ".txt", ".toml"}


def _shipped_files() -> list[Path]:
    roots = [_REPO / "src" / "coastline", _REPO / "docs", _REPO / "config", _REPO / "tests"]
    files = [path for root in roots for path in root.rglob("*") if path.suffix in _SUFFIXES]
    files += [_REPO / "README.md", _REPO / "CHANGELOG.md", _REPO / "pyproject.toml"]
    return [path for path in files if path.is_file() and "__pycache__" not in path.parts]


def test_no_latex_labels_in_shipped_text():
    found = [
        f"{path.relative_to(_REPO)}:{number}: {line.strip()}"
        for path in _shipped_files()
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1)
        if _LABEL.search(line)
    ]
    assert not found, "\n".join(found)


@pytest.mark.parametrize(
    "counts,text",
    [([1], "on 1 GPU."), ([1, 2], "on 1 and 2 GPUs."), ([1, 2, 4, 8], "on 1, 2, 4 and 8 GPUs.")],
)
def test_the_min_gpu_error_counts_gpus(counts, text):
    error = NoFeasibleGPUCountError("min_gpu", 7, counts)
    assert f"a total batch of 7 fails the feasibility check {text}" in str(error)
    assert "1 GPUs" not in str(error)


def test_the_dashboard_says_1_gpu_for_an_odd_batch(monkeypatch):
    from fastapi.testclient import TestClient

    from coastline.ui.app import app

    class Never:
        def is_feasible(self, workload: WorkloadSpec) -> tuple[bool, dict[str, Any]]:
            return False, {}

    monkeypatch.setattr(policies, "create_feasibility_checker", lambda predictor_config: Never())
    body = {
        "llm_model": "mistral-7b-v0.1",
        "fine_tuning_method": "lora",
        "gpu_model": "NVIDIA-A100-SXM4-80GB",
        "tokens_per_sample": 1024,
        "batch_size": 7,
        "prediction_model": "kavier",
        "strategy": "min_gpu",
        "total_gpus": 16,
    }
    with TestClient(app) as client:
        resp = client.post("/api/recommend", json=body)

    assert resp.status_code == 404, resp.text
    detail = resp.json()["detail"]
    # An odd total divides over 1 GPU only.
    assert "a total batch of 7 fails the feasibility check on 1 GPU." in detail
    assert "1 GPUs" not in detail
