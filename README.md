# Coastline

[![CI](https://github.com/atlarge-research/coastline-recommender/actions/workflows/ci.yml/badge.svg)](https://github.com/atlarge-research/coastline-recommender/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/atlarge-research/coastline-recommender/branch/main/graph/badge.svg)](https://codecov.io/gh/atlarge-research/coastline-recommender)
[![PyPI](https://img.shields.io/pypi/v/coastline-recommender.svg)](https://pypi.org/project/coastline-recommender/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Context-aware recommender for **GPU / datacenter configurations** for LLM fine-tuning: given a
workload it grid-searches configs, filters infeasible ones, predicts **throughput + power**, and
ranks them on a performance↔energy score. Throughput comes from **Kavier** (analytical physics) or
a data-driven model (TabPFN, CatBoost, …); energy from Kavier-power; feasibility from IBM
**AutoConf**.

**Accuracy** (throughput MdAPE, 15% holdout): default `intelligent` = cache hit (0%) → Kavier
(6.2%). ML predictors (bring your own trained artifacts): TabPFN 2.1%, XGBoost 7.2%, CatBoost 8.4%.

📖 **Full documentation:** run `uv run --group docs mkdocs serve` — Overview · Getting started ·
Architecture · per-component pages (pipeline, predictors, energy, feasibility, policies, library) ·
Contributing.

## Install

```bash
pip install coastline-recommender                 # core engine + AutoConf OOM-feasibility safeguard
pip install "coastline-recommender[ml]"           # + heavy ML backends (TabPFN, XGBoost, …)
```

From a checkout (uv-native): `uv sync`, then `uv run coastline …`.

## Use it

```python
import coastline

rec = coastline(predictor="kavier")            # or "intelligent", "tabpfn", a model name
results = rec({"llm_model": "mistral-7b-v0.1", "fine_tuning_method": "lora",
               "gpu_model": "NVIDIA-A100-SXM4-80GB", "tokens_per_sample": 1024, "batch_size": 32},
              total_gpus=[1, 2, 4, 8], preset="balanced")
print(results[0])                                      # best-ranked Recommendation

df = coastline.recommend(batch_df, predictor="kavier", goal="balanced", max_gpus=8)  # batch → DataFrame
```

One `coastline` command (five subcommands) plus the dashboard:

```bash
coastline recommend-job --interactive                                               # guided REPL
coastline recommend-job --config config/coastline_functionality/experiment.yaml        # one job → recommendation.json
coastline recommend-job --config config.yaml --input workloads.csv --output recs.csv # batch CSV → CSV
coastline recommend-trace --input trace.csv --output enriched.csv --visual           # annotate + plot a trace
coastline simulate --model mistral-7b-v0.1 --method lora --gpu-model NVIDIA-A100-SXM4-80GB \
    --tokens 1024 --batch-size 16 --gpus-per-node 4                                 # predict ONE config
coastline explain --model mistral-7b-v0.1 --method lora --gpu-model NVIDIA-A100-SXM4-80GB \
    --tokens 1024 --batch-size 16 --preset balanced                                 # why that config won
coastline utils tune --data runs.csv --model tabpfn                                  # tune | trace-to-runs | plot-trace
coastline-ui                                                                        # FastAPI dashboard :8000
```

Run the full API tour with `uv run python docs/usage.py` (reproduced in the
[getting-started guide](docs/guides/getting-started.md)); see `config/` for sample configs.

## Structure

One installable package under `src/coastline`:

| Surface | Role |
|---|---|
| `coastline.cli` | the single `coastline` command (argparse dispatch) |
| `coastline.ui`  | the FastAPI dashboard (`coastline-ui`) |
| `coastline.sdk` | the engine: `recommend · pipeline · predictors · policies · models · library · trace · io` |

The `sdk` is import-light: `import coastline` pulls no heavy backend until a predictor needs it.
Dev-only tooling (`benchmark/`, the ML `trainer/`) lives under `dev/` and is excluded from the
wheel; trained model pickles under `models/` are never shipped (regenerate via the trainer). The
ado experiment plugin lives in IBM's `ado` repository, under `plugins/custom_experiments/coastline`.

## Develop

```bash
uv sync --extra ml                                    # + heavy native ML backends
uv run --all-extras pytest                            # main suite
uv run --all-extras pytest dev/trainer/tests          # trainer suite (own invocation)
uv run --all-extras pytest dev/benchmark/tests        # benchmark suite (own invocation)
uv run --all-extras pytest -m ml_isolated -p no:cacheprovider   # native-ML tests (own process)
uv run ruff check . && uv run mypy
uv run --all-extras pytest --cov                      # …with a coverage report (CI adds --cov-report=xml for Codecov)
uv run --group docs mkdocs serve                      # serve the docs at http://127.0.0.1:8000
```

Run `uv run pre-commit install` once per clone and the ruff gates (plus whitespace hygiene) run on
every commit; CI runs the same hooks with `uv run pre-commit run --all-files`.

## External dependencies (not vendored)

- **Kavier** — analytical throughput/power engine; PyPI dependency (`kavier>=0.5.2,<0.6`).
- **AutoConf** — OOM-feasibility safeguard (`ado-autoconf`); ships by default in the core install.

## License

MIT — see [LICENSE](LICENSE).
