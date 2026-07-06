# Contributing

## Get in touch {#touch}

Coastline is developed at [AtLarge Research](https://atlarge-research.com) (VU Amsterdam) with IBM
Research. Open an issue or PR at
[github.com/atlarge-research/coastline-recommender](https://github.com/atlarge-research/coastline-recommender).

## Development environment {#dev-env}

The project is uv-native — one package under `src/coastline`, no `PYTHONPATH`.

```bash
uv sync --extra ml --extra autoconf   # core + dev + ML backends + AutoConf   (or: make install-ml)
uv run coastline --help
```

## Checks before you push {#checks}

```bash
make test          # main suite (tests/) + trainer tests + benchmark tests
make test-ml       # native-ML predictor tests (own process)
uv run ruff check . && uv run ruff format --check .
uv run mypy        # strict on the gated packages (coastline.sdk.models, cli, ui)
```

The default `uv run pytest` collects `tests/` (per-domain subpackages mirroring `sdk`). The
data-driven ML tests carry the `ml_isolated` marker and run in their own process — several native
backends bundle `libomp` and can crash when co-loaded.

## Where the code lives {#where}

Add logic under `sdk/`, argument parsing under `cli/`, HTTP routes under `ui/`. `sdk/` must never
import `cli`/`ui` and must stay import-light (no heavy backend at module top level). Each component
page has a **How to contribute** section pointing at its exact source + test files:
[pipeline](components/pipeline.md#contribute) ·
[performance](components/performance.md#contribute) ·
[energy](components/energy.md#contribute) ·
[feasibility](components/feasibility.md#contribute) ·
[policies](components/policies.md#contribute) ·
[library](components/library.md#contribute).

## Adding a predictor (the common case) {#add-predictor}

1. Implement `BasePredictor` (`sdk/predictors/base.py`): `predict(workload, context) -> Prediction`.
2. Register the name in `PolicyFactory` (`sdk/policies/__init__.py`) — for a data-driven model, add it
   to `_build_named_ml_predictor`'s lazy `importlib` map so its heavy backend never imports eagerly.
3. Add a test under `tests/test_predictors/`; mark it `ml_isolated` if it loads a native backend.

## Integrations {#integrations}

- **Kavier** — the physics engine, used through its *public* API (`kavier.training.performance`;
  `kavier.sdk.library` spec tables). The `kavier.sdk.{io,training}` engines import lazily only where no
  public verb exists; a guard test (`tests/test_recommend/test_kavier_public_api_guard.py`) fails if a
  module-level internal import reappears. Pinned `kavier>=0.5,<0.6`.
- **IBM Ado** — Coastline ships an ado experiment plugin under `dev/ado_plugin/` (distribution
  `ado-coastline`), exposing `coastline_recommender` and `coastline_min_gpu_recommender` experiments.
  Coastline is the successor to **AutoConf**, itself an ado plugin.
