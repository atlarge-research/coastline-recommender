# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Coastline is a context-aware recommender for **GPU / datacenter configurations** for LLM fine-tuning. Given a workload (LLM, PEFT method, tokens, batch size, GPU model), it grid-searches configurations, filters infeasible ones, predicts **throughput + power**, and ranks them on a performance↔energy score. Throughput comes from **Kavier** (analytical physics) or a data-driven ML model; energy from Kavier-power.

## Commands

All tooling is plain uv — there is no Makefile.

```bash
uv sync                                          # install (core + dev group; the AutoConf OOM checker ships in core)
uv sync --extra ml                               # + the heavy native ML backends (torch, xgboost, lightgbm, catboost, tabpfn)
uv run --all-extras pytest                       # main suite
uv run --all-extras pytest dev/trainer/tests     # co-located trainer tests (own invocation — see below)
uv run --all-extras pytest dev/benchmark/tests   # dev benchmark tests (own invocation — see below)
uv run --all-extras pytest -m ml_isolated -p no:cacheprovider   # native-ML predictor tests (own process)
uv run --project dev/ado_plugin pytest dev/ado_plugin           # ado experiment-plugin tests (needs IBM's ado core)
uv run ruff check . / uv run ruff format . / uv run mypy        # lint / format / typecheck
uv run coastline recommend-job --config config/coastline_functionality/experiment.yaml   # config-driven engine run
uv run coastline recommend-job --interactive     # interactive guided recommender (terminal REPL)
uv run coastline-ui                              # FastAPI dashboard on http://127.0.0.1:8000
uv run python docs/usage.py                      # runnable API tour
PYTHONPATH=dev uv run python -m benchmark.recommendation_quality   # recommendation quality vs the ground-truth trace
uv build --wheel                                 # build the wheel
uv run --group docs mkdocs serve                 # serve the MkDocs site
```

**Three test suites run in separate pytest processes.** The main suite, `dev/trainer/tests`, and `dev/benchmark/tests` are separate invocations — the dev suites clash when co-loaded in one process (they import the `trainer`/`benchmark` packages by name; `[tool.pytest.ini_options] pythonpath = ["dev"]` resolves them). The data-driven ML predictor tests (`tests/test_predictors/test_ml_predictors.py`, marked `ml_isolated` and deselected by default) load several native backends that each bundle `libomp` and can crash when co-loaded in one interpreter, so they too run in their own process (command above). `KMP_DUPLICATE_LIB_OK=TRUE` must be set before any native ML lib imports — `tests/conftest.py`, `ui/app.py`, and `ui/prediction_worker.py` all do this.

**Running a single test.** The suite is uv-native — no `PYTHONPATH` juggling. The `autoconf` OOM checker is optional; set `COASTLINE_ALLOW_RULES_FALLBACK=1` to let tests fall back to divisibility-only feasibility when it is absent:

```bash
COASTLINE_ALLOW_RULES_FALLBACK=1 uv run --all-extras pytest tests/test_pipeline/test_integration.py::test_name -q
```

## Architecture

One installable package, `src/coastline` (uv-native, `build-backend = "uv_build"`); `dev/` (benchmark, trainer, ado_plugin) is dev/research tooling excluded from the wheel (see `[tool.uv.build-backend]` wheel-exclude in `pyproject.toml`). Bundled ML models live in `src/coastline/sdk/predictors/performance/data_driven/portfolio/`; the wheel selectively excludes the 5 heavy ones + `custom/`.

| Layer | Role |
|---|---|
| `coastline.cli` | The single `coastline` dispatcher, three verbs: `recommend-job` (`--interactive` \| `--config` \| `--input/--output` CSV) / `recommend-trace` / `utils` (`tune` \| `trace-to-runs` \| `plot-trace`). Each verb is a thin adapter over the SDK. |
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
- **energy** (power): `"kavier_power"` (the only backend). Kavier returns power alongside throughput in one engine call — the pipeline reuses it when the power predictor sets `WRAPS_THROUGHPUT_ENGINE`, avoiding a second call.
- **feasibility**: `"autoconf"` (default, OOM-aware) or `"rules"` (divisibility only). See gotcha below.

### Config is the interface

Runs are driven by YAML (`config/coastline_functionality/`, `config/batch_config.yaml`). The `strategy` / `predictors` / `grid` schema is canonical; a legacy `orchestrator:` block is auto-translated **only when no modern `predictors:` block is present** — this translation lives in `src/coastline/sdk/io/run_config.py` and is reused by both the CLI and the API. Two config-driven safeguards: `predictors.feasibility: autoconf` (OOM) and `strategy.max_slowdown` / `runtime_guard_k` (never recommend a config slower than N× the fastest feasible one).

### UI service (`src/coastline/ui/`)

FastAPI app (`ui/app.py`) serving the wizard UI + REST. Long predictions run through a background worker + queue (`ui/prediction_worker.py`, `ui/workload_queue.py`) rather than blocking the request.

## Non-obvious gotchas

- **AutoConf (`ado-autoconf`) is a core dependency — it ships by default.** The bare PyPI name `autoconf` is an UNRELATED package. The `[autoconf]` extra still exists as a deprecated empty no-op for backward compat. In stripped environments without it the recommender **refuses by default**; `COASTLINE_ALLOW_RULES_FALLBACK=1` degrades to divisibility-only feasibility (no OOM check). Use `feasibility: rules` if you genuinely don't want the OOM check.
- **`scikit-learn` is pinned to exactly `1.7.2`** to match the serialized model pickles (no version skew). **`pandas` is pinned `<3`** (pandas 3 breaks xgboost 3.1.3 feature-name checks; ado-autoconf also pins `<3`). Do not loosen these casually.
- **All 10 bundled models live in one home; only the parametric ones ship in the wheel.** Artifacts are named plainly (`tabpfn.pkl`, `catboost.pkl`, …; the legacy `performance_<stem>_featv3.pkl` spelling still resolves). The single home is `src/coastline/sdk/predictors/performance/data_driven/portfolio/` — all 10 models (tabpfn + random_forest via Git LFS). The wheel ships the 5 parametric ones (catboost, xgboost, lightgbm, bayesian_ridge, deep_learning); the 5 heavy/instance-based ones (tabpfn, random_forest, gaussian_process, svr, knn) are `wheel-exclude`d in `pyproject.toml`. `models/custom/` holds user-tuned artifacts (`coastline utils tune` writes there; also where `dev/trainer` regenerates). Resolution precedence: `custom/` > flat `PORTFOLIO_DIR` > packaged `portfolio/`. The default Kavier physics path needs no pickles.
- **Kavier is a real PyPI dependency** (`kavier>=0.5,<0.6`), not vendored. Coastline imports its public API — the top-level `kavier.training` verb plus `kavier.sdk.{library,io,training}` engines. For Kavier development use an editable sibling checkout: `uv pip install -e ../kavier`. The benchmark calibration tooling reads `../kavier/src/...` directly.
- **Dev superproject layout.** Some tooling assumes coastline sits beside optional siblings: `../kavier` (source), `../ado` (ADO autoconf source for `dev/ado_plugin/`). None of this applies to wheel installs.
- The `coastline` package's `__init__.py` reassigns its module class so `coastline(throughput_estim=...)` is callable and returns a configured `Coastline` — that's why `import coastline; coastline(...)` works.

## Entry points at a glance

- `import coastline` — Python facade (`sdk/recommend/facade.py`): single workloads or batch DataFrames, in-process. `coastline.recommend(batch)` → DataFrame; `coastline(...).recommend(workload)` → `list[Recommendation]`.
- `coastline recommend-job` — the one job verb, three modes: `--interactive` (guided REPL), `--config …` (one declared job → `recommendation.json`/stdout; artifact via `--output-dir`/`OUTPUT_DIR`), and `--config … --input workloads.csv --output recs.csv` (batch CSV→CSV, = `coastline.recommend_csv()`, config-declared safeguards).
- `coastline recommend-trace` — add Coastline predictions to a fine-tuning trace CSV (`--visual` also renders the cluster timeline; [plot] extra).
- `coastline utils` — `tune` (train a predictor, [ml]) / `trace-to-runs` (trace → flat measured-runs) / `plot-trace` (visualise a recommended trace, [plot]).
- `coastline-ui` — the FastAPI dashboard.

Every door (facade / CLI / UI) routes through the single `sdk/recommend/engine.py` seam (`RecommendRequest` → `run_request`); every CSV shape resolves through the one canonical schema (`sdk/io/schema.py`) + format adapters (`sdk/io/adapters/`: `coastline` identity, `ibm_trace`).

## Documentation (`docs/`)

Flat numbered pages (`1_introduction.md` … `7_terminology.md`) in the style of the uv docs. The writing and structure guidelines live in `docs/docs.md` — follow them exactly. Radu's authoring preferences:

- Radu leaves inline `TODO:` comments in the pages; resolve each TODO in place, then delete the TODO line. Keep unresolved cross-references as empty `[text](...)` links.
- One canonical term per thing: **recommendation policy** (never "strategy" in prose; the YAML key stays literally `strategy:`), **preset** (a named α/β weight pair — α weighs energy, β weighs performance in `combined_score = α·power_score + β·throughput_score`), **workload**, **simulation models** (the predictors), **feasibility checker** (IBM AutoConf).
- Number sections and subsections on every page (`## 1.`, `### 1.1`) and keep explicit `{ #anchor }` tags so cross-page links survive renumbering. Open each page with a numbered table of contents, one line of explication per entry.
- Every shown command displays input AND output. Outputs are captured from real runs, never invented or hand-edited; display floats are rounded to 2 decimals at the source (`Recommendation.__str__`, the JSON payload), so capture after rounding.
- Reference specs follow the template: anchored heading, one prose paragraph, then a `| Field | Type | Required | Description |` table.
- Images live in `docs/media/` with descriptive snake_case names; thesis figures under `docs/thesis/floats/` are reusable when current.
- When documenting an input format, ship a ready-to-run sample (e.g. `config/coastline_functionality/sample_workloads.csv`) and run the command against the sample first.
- Never document: `make` commands (the project is pure uv), `energy: opendc` (removed), or CO2 prediction (exists in kavier, not exposed through Coastline).
