# Installation

Coastline requires Python 3.13+. The core install is light: the analytical **Kavier** physics
path works out of the box — no ML backends, no model pickles.

=== "pip"

    ```bash
    pip install coastline-recommender                 # core engine
    pip install "coastline-recommender[autoconf]"     # + OOM feasibility check (recommended)
    pip install "coastline-recommender[ml]"           # + ML predictors (TabPFN, XGBoost, …)
    pip install "coastline-recommender[plot]"         # + trace plotting
    ```

=== "uv (from a checkout)"

    ```bash
    uv sync --extra autoconf      # core + dev tools + OOM feasibility check
    uv run coastline --help
    ```

## Which extras do I need?

| Extra | Adds | Needed for |
|---|---|---|
| *(none)* | Kavier physics engine | Everything on the default path |
| `autoconf` | AutoConf OOM classifier | The default feasibility check |
| `ml` | Torch, XGBoost, TabPFN, … | Named ML predictors, `coastline tune` |
| `plot` | Matplotlib | `coastline plot-trace` |
| `opendc` | OpenDC bindings | The `energy: opendc` simulator path |

!!! warning "AutoConf is the default feasibility check"
    Without the `[autoconf]` extra the recommender **refuses to run** by default, to avoid
    recommending configs that OOM. Either install it, set `feasibility: rules` (divisibility-only
    checks), or export `COASTLINE_ALLOW_RULES_FALLBACK=1`.

## Next

Make your [first recommendation](first-recommendation.md).
