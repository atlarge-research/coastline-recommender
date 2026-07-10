# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Coastline is a context-aware recommender for **GPU / datacenter configurations** for LLM fine-tuning. Given a workload (LLM, PEFT method, tokens, batch size, GPU model), it grid-searches configurations, filters infeasible ones, predicts **throughput + power**, and ranks them on a performance↔energy score. Throughput comes from **Kavier** (analytical physics) or a data-driven ML model; energy from Kavier-power or the OpenDC simulator.

## Commands

```bash
make install        # uv sync --extra autoconf (core + dev group + the OOM feasibility checker)
make install-ml     # + the heavy native ML backends (torch, xgboost, lightgbm, catboost, tabpfn)
make test           # routine suite: main tests, then the co-located trainer + dev benchmark tests
make test-ml        # native-ML predictor tests (own process — several backends crash if co-loaded)
make test-ado       # ado experiment-plugin tests (skips cleanly if ado core absent)
make lint / format / typecheck   # ruff check . / ruff format . / mypy
make recommend                       # config-driven engine run (default config/coastline_functionality/config.yaml)
make recommend config=path/to.yaml   # same, custom config
make cli            # interactive guided recommender (terminal REPL)
make gui            # FastAPI dashboard on http://127.0.0.1:8000
make demo           # runnable API tour (docs/usage.py)
make recq           # recommendation quality vs the ground-truth trace
make build / docs   # build the wheel / serve the MkDocs site
```

**`make test` runs the ML predictor tests separately.** The data-driven ML predictor tests (`tests/test_predictors/test_ml_predictors.py`, marked `ml_isolated` and deselected by default) load several native backends that each bundle `libomp` and can crash when co-loaded in one interpreter, so they run in their own process as `test-ml`. `KMP_DUPLICATE_LIB_OK=TRUE` must be set before any native ML lib imports — `tests/conftest.py`, `ui/app.py`, and `ui/prediction_worker.py` all do this.

**Running a single test.** The suite is uv-native — no `PYTHONPATH` juggling. The `autoconf` OOM checker is optional; set `COASTLINE_ALLOW_RULES_FALLBACK=1` to let tests fall back to divisibility-only feasibility when it is absent:

```bash
COASTLINE_ALLOW_RULES_FALLBACK=1 uv run --all-extras pytest tests/test_pipeline/test_integration.py::test_name -q
```

## Architecture

One installable package, `src/coastline` (uv-native, `build-backend = "uv_build"`); `dev/` (benchmark, trainer, ado_plugin) and `models/` are dev/research tooling excluded from the wheel (see `[tool.uv.build-backend]` wheel-exclude in `pyproject.toml`).

| Layer | Role |
|---|---|
| `coastline.cli` | The single `coastline` dispatcher: `recommend` / `run` / `enrich-trace` / `plot-trace` / `interactive` |
| `coastline.ui` | FastAPI dashboard + REST (`coastline-ui`) — wizard UI, background prediction worker + queue |
| `coastline.sdk` | The engine, import-light: `recommend` (facade + batch) · `pipeline` · `predictors` · `policies` · `models` · `library` · `trace` · `io` |

### Everything routes through PolicyFactory

`src/coastline/sdk/policies/__init__.py` → `PolicyFactory` is the central resolver. Whatever the entry point (Python facade, batch CSV, config-driven CLI, FastAPI), a config dict with `strategy` / `predictors` / `grid` sections flows into `PolicyFactory.create_strategy()`, which builds one of two strategies:

- `multi_objective` — ranks on a weighted throughput/energy score (`alpha`/`beta`, or a `preset`: balanced/performance/energy)
- `min_gpu` — fewest GPUs among feasible configs

Both strategies wrap the **same** `GridWorkflowPipeline` (`src/coastline/sdk/pipeline/workflow.py`). Its `recommend()` is the core loop:

```
generate_candidates (grid.py)  →  feasibility_checker.is_feasible  →  throughput_predictor.predict
    →  power_predictor.predict  →  normalize_candidates + rank_candidates (selection.py)  →  Recommendation[]
```

`PolicyFactory.throughput_predictor()` is the **single source of truth** for resolving a predictor name → predictor object. The pipeline's `_create_throughput_predictor()` deliberately delegates back to it (an earlier duplicate silently collapsed every named model to CatBoost) — do not reintroduce a parallel resolver.

### The three predictor axes

The config `predictors:` block selects one of each (`src/coastline/sdk/predictors/`):

- **performance** (throughput): `"intelligent"` (default) = `CacheThenPhysicsPredictor` in `performance/composite.py` — exact cache hit of a real past run, else Kavier physics; `"kavier"` physics-only; `"cache"` retrieval-only; or a named ML model (`catboost`, `xgboost`, `lightgbm`, `tabpfn`, `random_forest`, …) resolved lazily by `_build_named_ml_predictor` so unused ML runtimes aren't imported.
- **energy** (power): `"kavier_power"` (default) or `"opendc"` (needs `OPENDC_BIN_PATH`). Kavier returns power alongside throughput in one engine call — the pipeline reuses it when the power predictor sets `WRAPS_THROUGHPUT_ENGINE`, avoiding a second call.
- **feasibility**: `"autoconf"` (default, OOM-aware) or `"rules"` (divisibility only). See gotcha below.

### Config is the interface

Runs are driven by YAML (`config/coastline_functionality/`, `config/batch_config.yaml`). The `strategy` / `predictors` / `grid` schema is canonical; a legacy `orchestrator:` block is auto-translated **only when no modern `predictors:` block is present** — this translation lives in `src/coastline/sdk/io/run_config.py` and is reused by both the CLI and the API. Two config-driven safeguards: `predictors.feasibility: autoconf` (OOM) and `strategy.max_slowdown` / `runtime_guard_k` (never recommend a config slower than N× the fastest feasible one).

### UI service (`src/coastline/ui/`)

FastAPI app (`ui/app.py`) serving the wizard UI + REST. Long predictions run through a background worker + queue (`ui/prediction_worker.py`, `ui/workload_queue.py`) rather than blocking the request.

## Non-obvious gotchas

- **AutoConf must be installed via the extra, not `pip install autoconf`.** The bare PyPI name `autoconf` is an UNRELATED package. The OOM checker is `ado-autoconf`, pulled in by `coastline[autoconf]`. Without it the recommender **refuses by default**; `COASTLINE_ALLOW_RULES_FALLBACK=1` degrades to divisibility-only feasibility (no OOM check). Use `feasibility: rules` if you genuinely don't want the OOM check.
- **`scikit-learn` is pinned to exactly `1.7.2`** to match the serialized model pickles (no version skew). **`pandas` is pinned `<3`** (pandas 3 breaks xgboost 3.1.3 feature-name checks; ado-autoconf also pins `<3`). Do not loosen these casually.
- **The parametric ML models ship in the wheel; the large ones do not.** Five models (catboost, xgboost, lightgbm, bayesian_ridge, deep_learning) are bundled under `src/coastline/sdk/predictors/performance/data_driven/portfolio/` and ship in git + the wheel + Docker. The large/instance-based ones (tabpfn, random_forest via Git LFS, knn, gaussian_process, svr) stay at repo-root `models/` for a dev checkout, found via `PORTFOLIO_DIR`. The default Kavier physics path needs no pickles.
- **Kavier is a real PyPI dependency** (`kavier>=0.5,<0.6`), not vendored. Coastline imports its public API — the top-level `kavier.training` verb plus `kavier.sdk.{library,io,training}` engines. For Kavier development use an editable sibling checkout: `uv pip install -e ../kavier`. The benchmark calibration tooling reads `../kavier/src/...` directly.
- **Dev superproject layout.** Some tooling assumes coastline sits beside optional siblings: `../kavier` (source), `../opendc` (`OPENDC_BIN_PATH` default), `../ado` (ADO autoconf source for `dev/ado_plugin/`). None of this applies to wheel installs.
- The `coastline` package's `__init__.py` reassigns its module class so `coastline(throughput_estim=...)` is callable and returns a configured `Coastline` — that's why `import coastline; coastline(...)` works.

## Entry points at a glance

- `import coastline` — Python facade (`sdk/recommend/facade.py`): single workloads or batch DataFrames, in-process. `coastline.recommend(batch)` → DataFrame; `coastline(...).recommend(workload)` → `list[Recommendation]`.
- `coastline recommend` / `coastline.recommend_csv()` — production batch CSV→CSV with config-declared safeguards.
- `coastline run --config …` (`make recommend`) — config-driven engine run → JSON to stdout (write an artifact via `--output-dir` or `OUTPUT_DIR`).
- `coastline enrich-trace` / `coastline plot-trace` — add Coastline predictions to a fine-tuning trace CSV, then visualise ([plot] extra).
- `coastline-ui` (`make gui`) — the FastAPI dashboard.
