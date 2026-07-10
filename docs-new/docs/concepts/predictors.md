# Predictors

Three independent axes, each selected by name. Defaults work out of the box.

## Throughput

| Name | What it does |
|---|---|
| `intelligent` *(default)* | Cache first, physics second: return the measured result if this exact workload was profiled before, else fall through to Kavier. |
| `kavier` | Analytical physics — models the GPU roofline, step time, and communication. No training data needed. |
| `cache` | Exact-match lookup only; returns nothing on a miss. |
| `catboost`, `xgboost`, `lightgbm`, `tabpfn`, `random_forest`, … | Trained ML models (`[ml]` extra). Backends load lazily — only the model you name is imported. |

Measured accuracy (median absolute percentage error, 15% holdout):

| Predictor | Throughput MdAPE |
|---|---|
| `intelligent` | cache hit 0% → Kavier 6.2% |
| `tabpfn` | 2.1% |
| `xgboost` | 7.2% |
| `catboost` | 8.4% |

!!! tip
    Tune a model on your own measured runs with [`coastline tune`](../guides/tune.md) — a model
    trained on your hardware beats the generic numbers above.

## Energy

| Name | What it does |
|---|---|
| `kavier_power` *(default)* | Analytical per-GPU watts, computed alongside throughput in one pass. |
| `opendc` | Full datacenter simulation via [OpenDC](https://opendc.org). Set `OPENDC_BIN_PATH`. |

## Feasibility

| Name | What it does |
|---|---|
| `autoconf` *(default)* | Divisibility rules plus an OOM classifier — configurations predicted to run out of memory are dropped. Needs the `[autoconf]` extra. |
| `rules` | Divisibility rules only. No install requirements, no OOM protection. |
| `none` | Accept everything (testing only). |

Without the `[autoconf]` extra, the default fails closed: pick `rules` explicitly or set
`COASTLINE_ALLOW_RULES_FALLBACK=1`.
