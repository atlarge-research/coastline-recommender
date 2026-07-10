# CLI commands

One `coastline` command with six subcommands, plus the `coastline-ui` dashboard.
`coastline <command> --help` shows full usage; `coastline --version` prints the version.

## `coastline recommend`

Batch: CSV of workloads in, CSV of recommendations out. Strategy, predictors, and safeguards come
from the config.

```bash
coastline recommend --config config/batch_config.yaml --input workloads.csv --output recs.csv
```

| Flag | Required | Meaning |
|---|---|---|
| `--config` | yes | Config YAML (`strategy` / `predictors` / `grid`) |
| `--input` | yes | Input workloads CSV |
| `--output` | yes | Output recommendations CSV |

## `coastline run`

One config-driven experiment → recommendation JSON.

```bash
coastline run --config config/coastline_functionality/config.yaml
```

| Flag | Default | Meaning |
|---|---|---|
| `--config` | `$CONFIG_FILE`, else `config/coastline_functionality/default.yaml` | Config YAML |
| `--input` | – | JSON with `workload` and `context` keys, overriding the config's workload |
| `--output-dir` | – | Write `recommendation.json` here; **without it, JSON prints to stdout** |

Env: `OUTPUT_DIR` (root for per-run subdirectories), `RUN_ID` (defaults to a UTC timestamp).

## `coastline recommend-trace`

Recommend a configuration for every job in a fine-tuning trace CSV.

```bash
coastline recommend-trace --input trace.csv --output recommended.csv --method kavier
```

| Flag | Default | Meaning |
|---|---|---|
| `--input` | required | Input trace CSV |
| `--output` | required | Output (recommended) trace CSV |
| `--goal` | `min_gpu` | `min_gpu` \| `performance` |
| `--method` | `kavier` | Duration estimator: `kavier` \| `tabpfn` \| `xgb` |
| `--feasibility` | `autoconf` | `autoconf` \| `rules` |
| `--lookup` | – | Measured-runs CSV, or `default` for the bundled lookup DB |
| `--cluster-gpus` | `16` | Total cluster GPUs |
| `--node-gpus` | `8` | GPUs per node |
| `--visual` | off | Also render the cluster timeline |
| `--visual-output` | `--output` + `.pdf` | Path for the `--visual` figure |

## `coastline plot-trace`

Plot a recommended trace's cluster timeline (GPUs in use + jobs queued). Needs the `[plot]` extra.

```bash
coastline plot-trace --input recommended.csv --output timeline.pdf
```

| Flag | Default | Meaning |
|---|---|---|
| `--input` | required | Recommended trace CSV (from `recommend-trace`) |
| `--output` | required | Output figure path |
| `--method` | `kavier` | Which estimate column to schedule with |
| `--cluster-gpus` | `16` | Total cluster GPUs |
| `--node-gpus` | `8` | GPUs per node |

## `coastline interactive`

Guided, keyboard-driven recommender in the terminal.

```bash
coastline interactive
```

| Flag | Default | Meaning |
|---|---|---|
| `--top-k`, `-k` | `5` | Configurations to rank (1–20) |
| `--save` | – | Write the top recommendation to this JSON file |
| `--verbose`, `-v` | off | Show engine logs |
| `--no-interactive` | – | One-shot run with defaults (also auto-selected when stdin isn't a TTY) |

## `coastline tune`

Tune a data-driven predictor on your measured runs. Needs the `[ml]` extra.
See the [tuning guide](../guides/tune.md).

```bash
coastline tune --data runs.csv --model tabpfn
```

| Flag | Default | Meaning |
|---|---|---|
| `--data` | required | Measured-runs CSV |
| `--model` | `tabpfn` | Model to tune |
| `--train-percentage` | `1.0` | `< 1.0` holds out a test split and reports MdAPE |
| `--output` | auto | Artifact path override |
| `--seed` | `42` | Split seed |
| `--format` | – | Print the dataset contract and exit |

## `coastline-ui`

The FastAPI dashboard — no flags, configured by environment:

```bash
coastline-ui        # http://127.0.0.1:8000
```

| Variable | Default |
|---|---|
| `COASTLINE_UI_HOST` | `127.0.0.1` |
| `COASTLINE_UI_PORT` | `8000` |

## Make targets (from a checkout)

| Target | Runs |
|---|---|
| `make install` / `make install-ml` | `uv sync --extra autoconf` (+ `--extra ml`) |
| `make recommend [config=…]` | `coastline run --config …` |
| `make cli` | `coastline interactive` |
| `make gui` | `coastline-ui` |
| `make demo` | the runnable API tour (`docs/usage.py`) |
| `make test` / `make test-ml` | test suites |
| `make lint` / `make format` / `make typecheck` | ruff / ruff format / mypy |
| `make docs` / `make docs-build` | serve / strict-build this site |

!!! note "Deprecated"
    `coastline enrich-trace` still works as an alias for `recommend-trace`.
