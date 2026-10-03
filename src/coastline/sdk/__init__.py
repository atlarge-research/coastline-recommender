"""Coastline SDK: the recommender engine, usable without the CLI or UI.

Submodules:
    recommend   entry points: one workload, a batch DataFrame, CSV to CSV
    pipeline    grid, feasibility, prediction and ranking
    predictors  performance (physics, retrieval, data-driven, composite), energy, feasibility
    policies    min_gpu and multi_objective; PolicyFactory maps predictor names to predictors
    models      WorkloadSpec, SystemContext, Prediction, Recommendation
    library     GPU and LLM hardware specs
    trace       recommend, plot
    io          config and option loaders, JSON output, infrastructure

Importing this package loads no heavy backend (torch, catboost, xgboost, lightgbm, tabpfn,
ado/AutoConf). Each loads when a data-driven predictor or the OOM feasibility check is selected.
"""
