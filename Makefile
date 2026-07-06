# Coastline — context-aware GPU/datacenter configuration recommender (uv-native).
#
# uv manages the venv and the entrypoints; no PYTHONPATH juggling. Kavier (the
# analytical throughput/power engine) is a declared dependency. The OpenDC energy
# path is optional and gated behind OPENDC_BIN_PATH.

export KMP_DUPLICATE_LIB_OK := TRUE
# Optional: point at a built OpenDC runner to enable the `energy: opendc` path.
OPENDC_BIN_PATH ?= ../opendc/bin/OpenDCExperimentRunner/bin/OpenDCExperimentRunner
export OPENDC_BIN_PATH
# Trainer/benchmark dev tools live under dev/ and import the `benchmark` package by name.
DEV_PYTHONPATH := dev

.PHONY: install install-ml test test-ml test-trainer test-bench test-ado \
        recommend gui cli demo recq lint format typecheck build docs docs-build clean

## Install coastline into a managed venv (core + dev group). Adds the autoconf extra
## so the default `feasibility: autoconf` OOM check is available.
install:
	uv sync --extra autoconf

## Also install the heavy ML backends needed by the data-driven predictors.
install-ml:
	uv sync --extra ml --extra autoconf

## Run the routine suite: the main package tests, then the co-located trainer tests and
## the dev benchmark tests in their own invocations (they exercise package-internal /
## dev-only code outside `testpaths`). The native-ML predictor tests are `test-ml`.
test: test-trainer test-bench
	uv run --all-extras pytest

## Dev trainer tests (the `trainer` package resolves with dev/ on the path).
test-trainer:
	PYTHONPATH=$(DEV_PYTHONPATH) uv run --all-extras pytest dev/trainer/tests

## Dev benchmark tests (the `benchmark` package resolves with dev/ on the path).
test-bench:
	PYTHONPATH=$(DEV_PYTHONPATH) uv run --all-extras pytest dev/benchmark/tests

## Native-ML predictor tests — run in their own process (several native backends can crash
## if co-loaded). NOTE: on some hosts the xgboost/lightgbm featv3 pickles segfault on
## unpickle under Python 3.13; regenerate them with the trainer if needed.
test-ml:
	uv run --all-extras pytest -m ml_isolated -p no:cacheprovider

## ADO experiment-plugin tests (dev/ado_plugin). Needs IBM's ado core; skips when absent.
test-ado:
	@if uv run python -c "import orchestrator" >/dev/null 2>&1; then \
		uv run --project dev/ado_plugin pytest dev/ado_plugin; \
	else \
		echo "test-ado: SKIPPED — ado core not installed ('import orchestrator' failed)."; \
	fi

## Produce a recommendation from a config file (config-driven engine run).
config ?= config/coastline_functionality/config.yaml
recommend:
	uv run coastline run --config $(config)

## Serve the FastAPI dashboard (http://127.0.0.1:8000).
gui:
	uv run coastline-ui

## Launch the interactive guided recommender (terminal REPL).
cli:
	uv run coastline interactive

## Run the runnable API tour (docs/usage.py — facade, batch DataFrame, recommend_csv).
demo:
	uv run python docs/usage.py

## Measure recommendation quality (ranking vs the ground-truth trace).
recq:
	PYTHONPATH=$(DEV_PYTHONPATH) uv run python -m benchmark.recommendation_quality

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy

## Build the distributable wheel (source + data + templates, no PYTHONPATH).
build:
	uv build --wheel

## Serve the MkDocs documentation site locally (http://127.0.0.1:8000).
docs:
	uv run --group docs mkdocs serve

## Build the static documentation site (strict: fails on any broken link).
docs-build:
	uv run --group docs mkdocs build --strict

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} + ; rm -rf .pytest_cache .ruff_cache dist
