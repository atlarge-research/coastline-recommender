"""recommend-trace keeps every recommended layout within the cluster's node width.

The node width comes from --node-gpus, else from infrastructure.yaml's max_gpus_per_node, the same
source as the GPU budget. Runs pin Kavier and the `rules` checker under the scale-out
`performance` goal, which is the goal most likely to ask for wide nodes.
"""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from coastline.cli import main
from coastline.sdk.io import infrastructure
from coastline.sdk.trace.recommend import recommend_trace

_ROW = {
    "metadata.model_name": "mistral-7b-v0.1",
    "metadata.method": "lora",
    "resources.gpu_model": "NVIDIA-A100-SXM4-80GB",
    "metadata.tokens_per_sample": 1024,
    "metadata.batch_size": 16,
    "resources.num_gpus_per_node": 8,
    "resources.num_nodes": 1,
    "metadata.output.extrapolated_duration": 3600.0,
}


def _trace(tmp_path, n: int = 2) -> str:
    path = tmp_path / "trace.csv"
    pd.DataFrame([dict(_ROW) for _ in range(n)]).to_csv(path, index=False)
    return str(path)


def _layouts(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    return pd.to_numeric(df["resources.num_gpus_per_node"]), pd.to_numeric(df["resources.num_nodes"])


@pytest.fixture
def four_gpu_nodes(tmp_path, monkeypatch):
    """infrastructure.yaml for 16 GPUs on 4-GPU nodes (the cached load is cleared around the test)."""
    path = tmp_path / "infrastructure.yaml"
    path.write_text(
        yaml.safe_dump(
            {"total_gpus": 16, "max_nodes": 4, "max_gpus_per_node": 4, "gpu_models": ["NVIDIA-A100-SXM4-80GB"]}
        )
    )
    monkeypatch.setenv("INFRASTRUCTURE_CONFIG", str(path))
    infrastructure.load_infrastructure.cache_clear()
    yield
    infrastructure.load_infrastructure.cache_clear()


def test_node_gpus_bounds_every_recommended_layout(tmp_path) -> None:
    df = recommend_trace(
        _trace(tmp_path),
        str(tmp_path / "out.csv"),
        goal="performance",
        feasibility="rules",
        cluster_gpus=16,
        node_gpus=4,
    )

    gpn, nodes = _layouts(df)
    assert df["metadata.recommendation_note"].isna().all()  # every row was recommended
    assert (gpn <= 4).all()
    assert (gpn * nodes <= 16).all()


def test_the_infrastructure_node_width_bounds_the_trace(tmp_path, four_gpu_nodes) -> None:
    df = recommend_trace(_trace(tmp_path), str(tmp_path / "out.csv"), goal="performance", feasibility="rules")

    gpn, nodes = _layouts(df)
    assert (gpn <= 4).all()
    assert (gpn * nodes <= 16).all()


def test_the_cli_node_gpus_flag_bounds_the_trace(tmp_path, capsys) -> None:
    out = tmp_path / "out.csv"
    main(
        [
            "recommend-trace",
            "--input",
            _trace(tmp_path),
            "--output",
            str(out),
            "--goal",
            "performance",
            "--feasibility",
            "rules",
            "--workers",
            "1",
            "--cluster-gpus",
            "16",
            "--node-gpus",
            "4",
        ]
    )

    gpn, nodes = _layouts(pd.read_csv(out))
    assert (gpn <= 4).all()
    assert (gpn * nodes <= 16).all()
