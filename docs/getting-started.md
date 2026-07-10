# Getting started

## Install

Coastline is uv-native. The default install serves the analytical **Kavier** physics path — no
heavy ML backends, no trained pickles.

=== "PyPI"

    ```bash
    pip install coastline-recommender                 # core engine (Kavier physics path)
    pip install "coastline-recommender[autoconf]"     # + AutoConf OOM-feasibility safeguard
    pip install "coastline-recommender[ml]"           # + heavy ML backends (TabPFN, XGBoost, …)
    ```

=== "From a checkout (uv)"

    ```bash
    uv sync --extra autoconf              # managed venv: core + dev + AutoConf OOM check
    uv run coastline --help
    ```

Kavier (the analytical throughput/power engine) is a first-class dependency and is pulled
automatically. AutoConf is the memory-safety OOM check; without it the recommender refuses by
default — set `COASTLINE_ALLOW_RULES_FALLBACK=1` to degrade to divisibility-only rules, or pass
`feasibility: rules`.

!!! tip "Which install do I need?"
    Nothing beyond the core for the Kavier physics path. Add `[ml]` only to serve a *named*
    data-driven predictor, and `[autoconf]` for the real OOM feasibility check.

## Your first recommendation — Python

```python
import coastline

advisor = coastline(predictor="kavier")             # or "intelligent", "tabpfn", a model name
results = advisor.recommend(
    {"model": "mistral-7b-v0.1", "method": "lora",
     "gpu_model": "NVIDIA-A100-SXM4-80GB", "tokens_per_sample": 1024, "batch_size": 32},
    goal="performance", max_gpus=8,                 # balanced | performance | energy | min_gpu
)
print(results[0])                                    # best-ranked Recommendation
```

The first argument is a `WorkloadSpec`, a dict of its fields (friendly aliases like `model`/`gpu`
accepted), or a CSV path. `goal` is the shared, discoverable knob; for finer control set
`strategy`/`preset` or explicit `alpha`/`beta` (see [Ranking policies](components/policies.md)).

## Batch — DataFrame in, DataFrame out

```python
import coastline, pandas as pd

df = pd.DataFrame([
    {"model": "mistral-7b-v0.1", "gpu_model": "NVIDIA-A100-SXM4-80GB",
     "tokens_per_sample": 1024, "batch_size": 16},
])
out = coastline.recommend(df, predictor="kavier", goal="balanced", max_gpus=8)
# input rows + ranked config + throughput_tok_s / runtime_s / energy_wh / feasible / rationale
```

Per-row columns override the batch kwargs; one bad row yields `feasible=False` without failing
the rest.

## CLI

One `coastline` command, six subcommands:

```bash
coastline recommend --config config.yaml --input workloads.csv --output recs.csv  # batch CSV → CSV
coastline run       --config config/coastline_functionality/config.yaml           # → recommendation JSON
coastline recommend-trace --input trace.csv --output enriched.csv --method kavier     # annotate a trace
coastline plot-trace   --input enriched.csv --output timeline.pdf              # visualise ([plot] extra)
coastline tune      --data runs.csv --model tabpfn --train-percentage 1.0          # tune on your own runs ([ml] extra)
coastline interactive                                                             # guided REPL
```

### Tune on your own measurements

`coastline tune` fits TabPFN on any measured-runs CSV. `--train-percentage 1.0`
(default) uses every valid row — no holdout; lower values hold out a test split and
report MdAPE for both targets. `coastline tune --format` prints the dataset
contract; a CSV that doesn't meet it fails loudly listing what's missing, and
quality problems (too few rows, one config, models unknown to Kavier's library)
print as *"Tuning may have produced poor results because valid datasets should have
these properties: …"*. The tuned model is picked up immediately by
`--method tabpfn` / `predictors.performance: tabpfn`.

The FastAPI dashboard is the second entrypoint:

```bash
coastline-ui          # http://127.0.0.1:8000  (dashboard + REST API)
```

## Config — the interface

Runs are driven by YAML with `strategy` / `predictors` / `grid` sections plus safeguards:

```yaml
strategy:
  name: multi_objective        # or min_gpu
  preset: balanced             # balanced | performance | energy
  max_slowdown: 3              # never recommend a config slower than 3× the fastest feasible one
predictors:
  performance: intelligent     # kavier | cache | intelligent | <ml model name>
  energy: kavier_power         # kavier_power | opendc
  feasibility: autoconf        # autoconf | rules
grid:
  total_gpus: [1, 2, 4, 8, 16]
  batch_sizes: [4, 8, 16, 32, 64]
  top_k: 5
```

Two config-driven safeguards: `predictors.feasibility: autoconf` (OOM-aware) and
`strategy.max_slowdown` (a runtime SLO). Next: the [Architecture](architecture.md).

## The full example, runnable

`docs/usage.py` is the complete tour — the callable facade, the batch DataFrame, and
`recommend_csv` — run end-to-end in CI so it never drifts. Run it with `uv run python docs/usage.py`:

```python
--8<-- "docs/usage.py"
```

## Run the tests

```bash
make install-ml            # core + dev + ml + autoconf
make test                  # main suite + trainer + benchmark
make test-ml               # native-ML predictor tests (own process)
```
