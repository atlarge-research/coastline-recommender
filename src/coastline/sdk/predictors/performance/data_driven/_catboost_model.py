"""The picklable CatBoost wrapper the trained artifact resolves to.

Lives in the shipped package (not the dev trainer) so the committed pickle unpickles without
the training code on the path. ``catboost_predictor`` aliases the artifact's legacy module path
(``trainer.train_performance_catboost``) to this class.
"""

from __future__ import annotations

import numpy as np


class _DualOutputCatBoost:
    """Holds one CatBoostRegressor per target; ``predict`` returns columns [throughput, runtime]
    in log space — the multi-output contract shared with every trainer and the inference path."""

    def __init__(self, throughput_model, runtime_model):
        self.throughput_model = throughput_model
        self.runtime_model = runtime_model
        self.estimators_ = [throughput_model, runtime_model]

    def predict(self, X):
        yt = np.asarray(self.throughput_model.predict(X)).reshape(-1)
        yr = np.asarray(self.runtime_model.predict(X)).reshape(-1)
        return np.column_stack([yt, yr])

    def get_best_iteration(self):
        return self.throughput_model.get_best_iteration()

    def get_feature_importance(self):
        return np.mean(
            [self.throughput_model.get_feature_importance(), self.runtime_model.get_feature_importance()],
            axis=0,
        )
