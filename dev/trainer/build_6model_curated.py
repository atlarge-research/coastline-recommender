#!/usr/bin/env python
"""Build the 6-model curated training set.

The canonical dense-4 curated set (mistral-7b, granite-3.3-8b, llama3.2-3b,
granite-3-8b) unchanged, plus the two granite-3.1 models (granite-3.1-2b,
granite-3.1-8b-instruct) taken as the valid, <=8-GPU rows from the raw trace
under the SAME validity filter the curated set encodes
(is_valid==1, dataset_tokens_per_second>0, train_runtime>0) and reindexed to
the curated schema.

The downstream 70-15-15 split (SEED=42) in common.py then produces the shared
train/val/test used by both the TabPFN retrain and the Kavier 85%
recalibration, with the test 15% held out for per-model accuracy comparison.

Output: PROFILING_DIR/curated_trace_6models.csv (OUT below).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
# coastline_recommender/predictors/performance/data_driven/trainer/ -> the umbrella dir that
# holds the shared trace-archive (one level above the coastline repo root).
REPO = HERE.parents[8]
PROFILING_DIR = REPO / "trace-archive" / "profiling-dataset"
CURATED = PROFILING_DIR / "curated_trace.csv"
RAW = PROFILING_DIR / "raw_trace.csv"
OUT = PROFILING_DIR / "curated_trace_6models.csv"

ADDED = ["granite-3.1-2b", "granite-3.1-8b-instruct"]


def main() -> None:
    cur = pd.read_csv(CURATED, low_memory=False)
    raw = pd.read_csv(RAW, low_memory=False)

    missing = [c for c in cur.columns if c not in raw.columns]
    if missing:
        print(f"WARNING: curated columns absent from raw (filled NaN): {missing}")

    valid = raw[(raw.is_valid == 1.0) & (raw.dataset_tokens_per_second > 0) & (raw.train_runtime > 0)].copy()
    valid["tot"] = (pd.to_numeric(valid.number_gpus) * pd.to_numeric(valid.number_nodes)).astype(int)
    add = valid[(valid.model_name.isin(ADDED)) & (valid.tot <= 8)].drop(columns=["tot"])
    add = add.reindex(columns=cur.columns)  # keep the canonical curated schema

    out = pd.concat([cur, add], ignore_index=True)
    out.to_csv(OUT, index=False)

    print(f"curated dense-4 rows : {len(cur)}")
    print(f"added granite-3.1    : {len(add)}")
    print(f"6-model total        : {len(out)} -> {OUT.relative_to(REPO)}")
    print("per-model rows:")
    for m, n in out.model_name.value_counts().items():
        print(f"  {m:28s} {n}")


if __name__ == "__main__":
    main()
