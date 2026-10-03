"""build_6model_curated reads and writes the trace archive that sits beside the coastline repository."""

from __future__ import annotations

from pathlib import Path

from .. import build_6model_curated as B


def test_the_profiling_dataset_is_the_one_beside_the_repository():
    # This file is dev/trainer/tests/<name>, so parents[3] is the repository root.
    umbrella = Path(__file__).resolve().parents[3].parent
    assert B.PROFILING_DIR == umbrella / "trace-archive" / "profiling-dataset"
