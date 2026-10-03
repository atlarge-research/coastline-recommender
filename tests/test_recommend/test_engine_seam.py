"""The engine entry points: RecommendRequest, build_strategy, execute_strategy and run_request.

run_pipeline (used by batch_api and the interactive path) gives the same result as a
RecommendRequest built by hand, and one strategy from build_strategy can serve many rows,
which batch_csv relies on.
"""

from __future__ import annotations

from coastline.sdk.recommend import engine


def _kavier_answers() -> dict:
    """Answers for the Kavier predictor with rules feasibility, so no ML model or AutoConf is needed."""
    answers = engine.defaults(engine.resolve_options())
    answers["predictor"] = "kavier"
    answers["llm_model"] = "mistral-7b-v0.1"
    answers["gpu_model"] = "NVIDIA-A100-SXM4-80GB"
    answers["feasibility"] = "rules"
    return answers


def _rec_key(rec) -> tuple:
    """The fields of a Recommendation that two runs are compared on."""
    return (
        rec.total_gpus,
        rec.gpus_per_node,
        rec.number_of_nodes,
        rec.predicted_throughput,
        (rec.metadata or {}).get("batch_size"),
    )


def test_run_request_matches_run_pipeline():
    """run_request on a RecommendRequest built by hand returns the same recommendations and
    metadata as run_pipeline."""
    answers = _kavier_answers()

    recs_wrapper, meta_wrapper = engine.run_pipeline(answers, top_k=3)

    config, strategy_name, preset = engine.build_config(answers, top_k=3, feasibility="rules")
    request = engine.RecommendRequest(
        workload=engine.build_workload(answers),
        context=engine.build_context(answers),
        config=config,
        strategy_name=strategy_name,
        preset=preset,
        total_tokens=int(answers["dataset_size"] * answers["epochs"] * answers["tokens_per_sample"]),
    )
    recs_seam, meta_seam = engine.run_request(request)

    assert [_rec_key(r) for r in recs_seam] == [_rec_key(r) for r in recs_wrapper]
    assert meta_seam["strategy_name"] == meta_wrapper["strategy_name"]
    assert meta_seam["preset"] == meta_wrapper["preset"]
    assert meta_seam["predictor"] == meta_wrapper["predictor"] == "kavier"
    assert meta_seam["total_tokens"] == meta_wrapper["total_tokens"]


def test_build_strategy_is_reusable_across_rows():
    """One strategy from build_strategy, run twice by execute_strategy on the same inputs,
    gives the same recommendations."""
    answers = _kavier_answers()
    config, strategy_name, preset = engine.build_config(answers, top_k=3, feasibility="rules")
    workload = engine.build_workload(answers)
    context = engine.build_context(answers)

    strategy = engine.build_strategy(config, strategy_name, preset)

    recs_a, _ = engine.execute_strategy(
        strategy,
        workload,
        context,
        strategy_name=strategy_name,
        preset=preset,
        grid=config["grid"],
        predictor="kavier",
    )
    recs_b, _ = engine.execute_strategy(
        strategy,
        workload,
        context,
        strategy_name=strategy_name,
        preset=preset,
        grid=config["grid"],
        predictor="kavier",
    )

    assert recs_a, "expected at least one feasible recommendation"
    assert [_rec_key(r) for r in recs_a] == [_rec_key(r) for r in recs_b]


def test_run_pipeline_signature_unchanged():
    """run_pipeline returns (recommendations, metadata), with the metadata keys its callers read."""
    recs, meta = engine.run_pipeline(_kavier_answers(), top_k=2)
    assert isinstance(recs, list)
    assert {"strategy_name", "preset", "predictor", "elapsed_s", "grid", "workload", "total_tokens"} <= set(meta)
