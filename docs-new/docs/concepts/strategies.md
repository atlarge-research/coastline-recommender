# Ranking strategies

After prediction, feasible candidates are scored and ordered. Two strategies exist.

## `multi_objective` — the throughput↔energy trade-off

Predictions are min-max normalized across the candidate grid, then scored:

**score = α · energy + β · throughput**

You rarely set α/β directly — pick a preset:

| Preset | α (energy) | β (throughput) |
|---|---|---|
| `performance` | 0.2 | 0.8 |
| `balanced` *(default)* | 0.5 | 0.5 |
| `energy` | 0.8 | 0.2 |

Explicit `alpha`/`beta` override the preset; setting only one derives the other as its complement.
Each preset also has a `-frontier` variant (e.g. `balanced-frontier`) that normalizes over the
Pareto-optimal candidates only.

## `min_gpu` — the frugal strategy

Ignores weights entirely: pick the feasible configuration with the fewest total GPUs, breaking
ties by higher throughput. Returns a single recommendation.

## Goals map to strategies

The `goal` knob on every surface is shorthand:

| `goal` | Strategy | Preset |
|---|---|---|
| `performance` | `multi_objective` | `performance` |
| `balanced` | `multi_objective` | `balanced` |
| `energy` | `multi_objective` | `energy` |
| `min_gpu` | `min_gpu` | – |

## The runtime guard

`max_slowdown: N` (config) or `max_slowdown=N` (Python) filters the ranking: any candidate more
than N× slower than the fastest feasible one is dropped **before** scoring — an energy-leaning
preset can never hand you a pathologically slow config.
