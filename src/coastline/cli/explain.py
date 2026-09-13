"""`coastline explain` — show WHY the recommender picked what it picked.

Renders the ranked candidates with the score components the policy actually used, so the
recommendation stops being a bare number. Reads what the pipeline already computed; it does not
recompute or re-rank anything.
"""

from __future__ import annotations

import sys
from typing import Any, Optional, Sequence

from coastline.cli._shared import FriendlyParser
from coastline.sdk.constants import PRESET_WEIGHTS


def _build_parser() -> FriendlyParser:
    p = FriendlyParser(
        prog="coastline explain",
        description="Explain a recommendation: the ranked candidates, each one's score "
        "components, the weights applied, and why the winner won.",
        example="coastline explain --model mistral-7b-v0.1 --method lora "
        "--gpu-model NVIDIA-A100-SXM4-80GB --tokens 1024 --batch-size 16 --preset balanced",
    )
    p.add_argument("--model", required=True, help="LLM model name (Kavier catalog spelling).")
    p.add_argument("--method", required=True, help="Fine-tuning method: full | lora | qlora.")
    p.add_argument("--gpu-model", required=True, help="GPU model, e.g. NVIDIA-A100-SXM4-80GB.")
    p.add_argument("--tokens", type=int, required=True, help="Tokens per sample (sequence length).")
    p.add_argument("--batch-size", type=int, required=True, help="Per-device batch size.")
    p.add_argument(
        "--strategy",
        default="multi_objective",
        choices=["multi_objective", "min_gpu"],
        help="Recommendation policy (default: multi_objective).",
    )
    # choices= matters here: without it a typo silently falls through the strategy's
    # unconditional balanced fallback and explains a policy the user did not ask for.
    p.add_argument(
        "--preset",
        default="balanced",
        choices=sorted(PRESET_WEIGHTS),
        help="Weight preset for multi_objective (the -frontier variants rank only over the "
        "non-dominated Pareto frontier). Ignored by min_gpu.",
    )
    p.add_argument("--max-gpus", type=int, default=8, help="Largest GPU count to consider (default: 8).")
    p.add_argument("--top-k", type=int, default=5, help="How many ranked candidates to show (default: 5).")
    p.add_argument(
        "--predictor",
        default="kavier",
        help="Throughput predictor: kavier (default, analytical physics) | intelligent | cache "
        "| a named data-driven model.",
    )
    p.add_argument(
        "--feasibility",
        default="autoconf",
        help="Feasibility checker: autoconf (default, real OOM check via AutoConf) "
        "| rules (divisibility-only, works without AutoConf) | none.",
    )
    return p


def _fmt(value: Optional[float], width: int, decimals: int) -> str:
    """Fixed-width number, or a dash when the pipeline did not produce one."""
    return "-".rjust(width) if value is None else f"{value:{width}.{decimals}f}"


def _render(recs: list, args: Any) -> str:
    meta = recs[0].metadata or {}
    # selection_policy is a SelectionPolicy enum; str() would render "SelectionPolicy.BALANCED".
    raw_policy = meta.get("selection_policy", args.strategy)
    policy = str(getattr(raw_policy, "value", raw_policy))
    is_min_gpu = policy == "min_gpu"

    lines = [
        f"workload  {args.model} / {args.method} / {args.gpu_model} / {args.tokens} tok, batch {args.batch_size}",
    ]
    if is_min_gpu:
        # min_gpu never scores: workflow.py replaces combined_score with a 1/total_gpus ordering
        # proxy and leaves preset/alpha/beta off the metadata entirely.
        lines.append("policy    min_gpu  (fewest GPUs among feasible candidates; no weighted score)")
    else:
        alpha, beta = meta.get("alpha"), meta.get("beta")
        weights = "" if alpha is None or beta is None else f"  (alpha={alpha:.2f} power, beta={beta:.2f} time)"
        lines.append(f"policy    {policy}  preset={meta.get('preset', args.preset)}{weights}")
    # min_gpu has no weighted score: workflow.py overwrites combined_score with a 1/total_gpus
    # ordering proxy, so showing it under a "combined" heading would be a lie. Drop the column.
    header = "rank  gpus  batch   thr(tok/s)     P(W)  p_score  t_score"
    lines += ["", header if is_min_gpu else header + "  combined"]

    for index, rec in enumerate(recs, start=1):
        rec_meta = rec.metadata or {}
        layout = f"{rec.gpus_per_node}x{rec.number_of_nodes}"
        row = (
            f"{index:>4}  {layout:>4}  {str(rec_meta.get('batch_size', '-')):>5}  "
            f"{_fmt(rec.predicted_throughput, 10, 1)}  "
            f"{_fmt(rec_meta.get('predicted_power_watts'), 7, 1)}  "
            f"{_fmt(rec_meta.get('power_score'), 7, 2)}  "
            f"{_fmt(rec_meta.get('throughput_score'), 7, 2)}"
        )
        if not is_min_gpu:
            row += f"  {_fmt(rec_meta.get('combined_score'), 8, 3)}"
        lines.append(row)

    from coastline.sdk.recommend import engine

    top = recs[0]
    lines += [
        "",
        f"winner    {top.total_gpus} GPU(s), {top.gpus_per_node} per node on {top.number_of_nodes} node(s)",
        f"why       {engine.recommendation_rationale(recs, {'preset': meta.get('preset'), 'strategy_name': policy})}",
    ]
    if not is_min_gpu and meta.get("alpha") is not None:
        # Show the weighted sum the policy actually evaluated: alpha*power + beta*throughput.
        power_score, throughput_score = meta.get("power_score"), meta.get("throughput_score")
        if power_score is not None and throughput_score is not None:
            alpha, beta = meta["alpha"], meta["beta"]
            lines.append(
                f"score     {alpha:.2f} x {power_score:.2f} (power) + {beta:.2f} x "
                f"{throughput_score:.2f} (time) = {alpha * power_score + beta * throughput_score:.3f}"
            )
    feasibility = meta.get("feasibility") or {}
    lines.append(f"feas      {args.feasibility}: {feasibility.get('reason', 'feasible')}")

    if not is_min_gpu:
        scores = [(r.metadata or {}).get("combined_score") for r in recs]
        ranked = [s for s in scores if s is not None]
        if ranked != sorted(ranked, reverse=True):
            # selection.py breaks near-ties (within 0.01) toward higher throughput, so the
            # displayed order is deliberately not combined-score order. Say so.
            lines.append(
                "tie-break scores within 0.01 are reordered by throughput, so `combined` "
                "is not monotonic down this table."
            )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    from coastline.sdk.models.context import SystemContext
    from coastline.sdk.recommend.facade import Coastline

    context = SystemContext.for_gpus([args.gpu_model], max_gpus=args.max_gpus)
    recs = Coastline(predictor=args.predictor, feasibility=args.feasibility).recommend(
        {
            "llm_model": args.model,
            "fine_tuning_method": args.method,
            "gpu_model": args.gpu_model,
            "tokens_per_sample": args.tokens,
            "batch_size": args.batch_size,
        },
        context=context,
        strategy=args.strategy,
        preset=args.preset,
        top_k=args.top_k,
        max_gpus=args.max_gpus,
    )

    if not recs:
        print("No feasible configuration in the search space; nothing to explain.", file=sys.stderr)
        sys.exit(1)
    print(_render(recs, args))


if __name__ == "__main__":
    main()
