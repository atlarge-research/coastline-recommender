"""Recommendation-quality harness: top-1 hit rate, regret, and Spearman rho vs. measured throughput."""

from __future__ import annotations

import numpy as np
import pandas as pd

from coastline.sdk.io.options_loader import _default_options_path
from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.policies import PolicyFactory

_THR = "dataset_tokens_per_second"
_WORKLOAD_KEYS = ["model_name", "gpu_model", "method", "tokens_per_sample"]
_CONFIG_KEYS = ["batch_size", "number_gpus", "number_nodes"]


def _load_trace() -> pd.DataFrame:
    df = pd.read_csv(_default_options_path())
    if "is_valid" in df.columns:
        df = df[df["is_valid"] == 1.0]
    return df[df[_THR].notna() & (df[_THR] > 0)].copy()


def _predicted_throughput(predictor, row: pd.Series) -> float | None:
    gpn, nn = int(row["number_gpus"]), int(row["number_nodes"])
    workload = WorkloadSpec(
        llm_model=str(row["model_name"]),
        fine_tuning_method=str(row["method"]),
        gpu_model=str(row["gpu_model"]),
        tokens_per_sample=int(row["tokens_per_sample"]),
        batch_size=int(row["batch_size"]),
        gpus_per_node=gpn,
        number_of_nodes=nn,
    )
    context = SystemContext.for_gpus([str(row["gpu_model"])], max_gpus=gpn * nn, gpus_per_node=gpn, max_nodes=nn)
    pred = predictor.predict(workload, context)
    if pred is None or not pred.predicted_throughput or pred.predicted_throughput <= 0:
        return None
    return float(pred.predicted_throughput)


def _spearman(pred: np.ndarray, meas: np.ndarray) -> float:
    """Rank correlation (corrcoef of ranks); nan if either side is constant."""
    rp, rm = pred.argsort().argsort().astype(float), meas.argsort().argsort().astype(float)
    if rp.std() == 0 or rm.std() == 0:
        return float("nan")
    return float(np.corrcoef(rp, rm)[0, 1])


def evaluate_predictor(df: pd.DataFrame, predictor_key: str) -> dict:
    predictor = PolicyFactory.throughput_predictor({"performance": predictor_key})
    hits, regrets, spearmans, n_eval = 0, [], [], 0
    for _, grp in df.groupby(_WORKLOAD_KEYS):
        configs = grp.drop_duplicates(subset=_CONFIG_KEYS)
        if len(configs) < 2:
            continue
        preds = [_predicted_throughput(predictor, row) for _, row in configs.iterrows()]
        keep = [(p, float(m)) for p, m in zip(preds, configs[_THR]) if p is not None]
        if len(keep) < 2:  # need >=2 ranked configs to score a recommendation
            continue
        pv = np.array([p for p, _ in keep])
        mv = np.array([m for _, m in keep])
        i_pred, i_meas = int(pv.argmax()), int(mv.argmax())
        hits += int(i_pred == i_meas)
        regrets.append((mv[i_meas] - mv[i_pred]) / mv[i_meas])
        spearmans.append(_spearman(pv, mv))
        n_eval += 1
    sp = [s for s in spearmans if not np.isnan(s)]
    return {
        "predictor": predictor_key,
        "workloads": n_eval,
        "top1_hit_rate": hits / n_eval if n_eval else float("nan"),
        "median_regret": float(np.median(regrets)) if regrets else float("nan"),
        "p90_regret": float(np.percentile(regrets, 90)) if regrets else float("nan"),
        "mean_spearman": float(np.mean(sp)) if sp else float("nan"),
    }


def main() -> None:
    import logging

    logging.disable(logging.WARNING)
    df = _load_trace()
    n_workloads = sum(len(g.drop_duplicates(subset=_CONFIG_KEYS)) >= 2 for _, g in df.groupby(_WORKLOAD_KEYS))
    print(f"Recommendation quality over {len(df)} measured configs · {n_workloads} workloads (>=2 configs each)\n")
    res = pd.DataFrame([evaluate_predictor(df, k) for k in ("kavier", "cache", "intelligent")])
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))
    print(
        "\ntop1_hit_rate = best-predicted config == measured-best   |   regret = throughput fraction lost (0 = perfect)"
    )
    print("spearman = predicted-vs-measured rank correlation (1 = perfect order)")
    print("cache is in-distribution-perfect by construction (exact-match sanity); kavier is the analytical signal.")


if __name__ == "__main__":
    main()
