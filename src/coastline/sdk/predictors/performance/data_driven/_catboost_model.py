"""Picklable CatBoost wrapper; the trained catboost pickle holds an instance of this class.

It lives in the installed package so the bundled pickle loads without the training code.
``sklearn_portfolio`` maps the module path stored in the pickle
(``trainer.train_performance_catboost``) to this class when it loads the catboost model.
"""

from __future__ import annotations

import numpy as np


class _DualOutputCatBoost:
    """One CatBoostRegressor per target. ``predict`` returns columns [throughput, runtime] in log
    space, the multi-output layout the trainers and the inference path share."""

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
