# Coastline

Context-aware recommender for **GPU / datacenter configurations** for LLM fine-tuning: given a
workload it grid-searches configs, filters infeasible ones, predicts **throughput + power**, and
ranks them on a performance↔energy score. Throughput comes from **Kavier** (analytical physics) or
a data-driven model (TabPFN, CatBoost, …); energy from Kavier-power or the **OpenDC** simulator;
feasibility from IBM **AutoConf**.

**Accuracy** (throughput MdAPE, 15% holdout): default `intelligent` = cache hit (0%) → Kavier
(6.2%). ML predictors (bring your own trained artifacts): TabPFN 2.1%, XGBoost 7.2%, CatBoost 8.4%.

📖 **Full documentation:** run `make docs` (or `uv run mkdocs serve`) — Overview · Getting started ·
Architecture · per-component pages (pipeline, predictors, energy, feasibility, policies, library) ·
Contributing.

## Install

```bash
pip install coastline-recommender                 # core engine (Kavier physics path)
pip install "coastline-recommender[autoconf]"     # + AutoConf OOM-feasibility safeguard
pip install "coastline-recommender[ml]"           # + heavy ML backends (TabPFN, XGBoost, …)
```

From a checkout (uv-native): `uv sync --extra autoconf` (or `make install`), then `uv run coastline …`.

## Use it

```python
import coastline

rec = coastline(throughput_estim="kavier")            # or "intelligent", "tabpfn", a model name
results = rec({"llm_model": "mistral-7b-v0.1", "fine_tuning_method": "lora",
               "gpu_model": "NVIDIA-A100-SXM4-80GB", "tokens_per_sample": 1024, "batch_size": 32},
              total_gpus=[1, 2, 4, 8], preset="balanced")
print(results[0])                                      # best-ranked Recommendation

df = coastline.recommend(batch_df, predictor="kavier", goal="balanced", max_gpus=8)  # batch → DataFrame
```

One `coastline` command (five subcommands) plus the dashboard:

```bash
coastline recommend --config config.yaml --input workloads.csv --output recs.csv  # batch CSV → CSV
coastline run --config config/coastline_functionality/config.yaml                 # → recommendation.json
coastline recommend-trace --input trace.csv --output enriched.csv                    # annotate a trace
coastline plot-trace --input enriched.csv --output timeline.pdf                   # visualise ([plot] extra)
coastline interactive                                                            # guided REPL
coastline-ui                                                                     # FastAPI dashboard :8000
```

Run the full API tour with `make demo` (`docs/usage.py`, reproduced in the
[getting-started guide](docs/getting-started.md)); see `config/` for sample configs.

## Structure

One installable package under `src/coastline`:

| Surface | Role |
|---|---|
| `coastline.cli` | the single `coastline` command (argparse dispatch) |
| `coastline.ui`  | the FastAPI dashboard (`coastline-ui`) |
| `coastline.sdk` | the engine: `recommend · pipeline · predictors · policies · models · library · trace · io` |

The `sdk` is import-light: `import coastline` pulls no heavy backend until a predictor needs it.
Dev-only tooling (`benchmark/`, the ML `trainer/`, the `ado_plugin/`) lives under `dev/` and is
excluded from the wheel; trained model pickles under `models/` are never shipped (regenerate via the
trainer). See [Architecture](docs/architecture.md).

## Develop

```bash
uv sync --extra ml --extra autoconf   # or: make install-ml
make test                             # main suite + trainer + benchmark
make test-ml                          # native-ML predictor tests (own process)
uv run ruff check . && uv run mypy
make docs                             # serve the docs at http://127.0.0.1:8000
```

## External dependencies (not vendored)

- **Kavier** — analytical throughput/power engine; PyPI dependency (`kavier>=0.5,<0.6`).
- **AutoConf** — OOM-feasibility safeguard (IBM `ado-autoconf`); install via the `[autoconf]` extra.
- **OpenDC** — optional Java simulator for the `energy: opendc` path; set `OPENDC_BIN_PATH`.

## License

MIT — see [LICENSE](LICENSE).
