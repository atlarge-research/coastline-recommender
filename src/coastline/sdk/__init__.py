"""Coastline SDK — the recommender engine, importable without any CLI/UI code.

Submodules:
    recommend   verb facade: single workload · batch DataFrame · CSV→CSV
    pipeline    grid → feasibility → predict → rank orchestrator
    predictors  performance (physics/retrieval/data-driven/composite) · energy · feasibility
    policies    min_gpu · multi_objective  (PolicyFactory — the single predictor resolver)
    models      WorkloadSpec · SystemContext · Prediction · Recommendation
    library     GPU/LLM hardware specs
    trace       recommend · plot
    io          config/options loaders · json/html run artifacts · infrastructure

Import-light by contract: this package imports no heavy backend (torch, catboost,
xgboost, lightgbm, tabpfn, ado/AutoConf) at import time — those load lazily only when a
data-driven predictor or the OOM-feasibility check is actually selected.
"""
