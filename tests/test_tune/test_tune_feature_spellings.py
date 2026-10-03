"""A tuned model sees the same feature values at tuning time and at prediction time.

WorkloadSpec lowercases the method and maps GPU aliases to Kavier's names before prediction, so
tuning stores the same spellings. A model tuned on 'LoRA' or 'A100-SXM4-80GB' rows would
otherwise see unknown categories at prediction time.
"""

import pandas as pd

from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.predictors.performance.data_driven.ml_common import workload_to_ml_feature_row
from coastline.sdk.predictors.performance.data_driven.tune import _feature_frame, validate_dataset

_ROW = {
    "model_name": "mistralai/Mistral-7B-v0.1",
    "method": "LoRA",
    "gpu_model": "A100-SXM4-80GB",
    "number_nodes": 1,
    "number_gpus": 4,
    "tokens_per_sample": 2048,
    "batch_size": 8,
    "dataset_tokens_per_second": 5000.0,
    "train_runtime": 600.0,
    "torch_dtype": "bfloat16",
    "enable_roce": 0,
}


def test_tuning_features_match_the_prediction_features_for_the_same_run():
    clean, _ = validate_dataset(pd.DataFrame([_ROW]))
    X, cat_features, num_features = _feature_frame(clean)
    tuned = X.iloc[0].to_dict()

    served = workload_to_ml_feature_row(
        WorkloadSpec(
            llm_model=_ROW["model_name"],
            fine_tuning_method=_ROW["method"],
            gpu_model=_ROW["gpu_model"],
            tokens_per_sample=_ROW["tokens_per_sample"],
            batch_size=_ROW["batch_size"],
            gpus_per_node=_ROW["number_gpus"],
            number_of_nodes=_ROW["number_nodes"],
            torch_dtype=_ROW["torch_dtype"],
            enable_roce=bool(_ROW["enable_roce"]),
        )
    )

    assert {c: tuned[c] for c in cat_features} == {c: str(served[c]) for c in cat_features}
    assert {c: float(tuned[c]) for c in num_features} == {c: float(served[c]) for c in num_features}
    assert tuned["method"] == "lora" and tuned["gpu_model"] == "NVIDIA-A100-SXM4-80GB"
