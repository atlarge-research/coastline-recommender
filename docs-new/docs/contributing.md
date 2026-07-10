# Contributing

Coastline is developed at [AtLarge Research](https://atlarge-research.com) (VU Amsterdam) with IBM
Research. Open an issue or PR at
[atlarge-research/coastline-recommender](https://github.com/atlarge-research/coastline-recommender).

## Set up

The project is uv-native — one package under `src/coastline`, no `PYTHONPATH`.

```bash
uv sync --extra ml --extra autoconf   # or: make install-ml
uv run coastline --help
```

## Checks before you push

```bash
make test          # main suite + trainer + benchmark tests
make test-ml       # native-ML predictor tests (separate process)
make lint          # ruff check
make typecheck     # mypy
```

The ML predictor tests run in their own process (`make test-ml`) — several native backends bundle
`libomp` and crash when co-loaded.

## Where code lives

| Directory | Contents |
|---|---|
| `src/coastline/sdk` | The engine: recommend, pipeline, predictors, policies, models, library |
| `src/coastline/cli` | Argument parsing for the `coastline` command |
| `src/coastline/ui` | The FastAPI dashboard |
| `dev/` | Research tooling (benchmark, trainer, ado plugin) — excluded from the wheel |

Rule of thumb: logic goes in `sdk/`, which must never import `cli`/`ui` and must stay import-light
(no heavy backend at module top level).

## Adding a predictor

1. Implement `BasePredictor` (`sdk/predictors/base.py`): `predict(workload, context) -> Prediction`.
2. Register the name in `PolicyFactory` (`sdk/policies/__init__.py`); data-driven models go in the
   lazy import map so their backend never loads eagerly.
3. Add a test under `tests/test_predictors/` — mark it `ml_isolated` if it loads a native backend.

## Integrations

- **Kavier** — the analytical physics engine, a PyPI dependency used through its public API.
- **AutoConf** — IBM's OOM classifier (`ado-autoconf`), behind the `[autoconf]` extra.
- **OpenDC** — optional datacenter simulator for the `energy: opendc` path (`OPENDC_BIN_PATH`).
- **IBM Ado** — Coastline ships an Ado experiment plugin under `dev/ado_plugin/`.
