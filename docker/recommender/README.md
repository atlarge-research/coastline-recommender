# Pre-trained Coastline recommender image

A self-contained Docker image that is **trained on a dataset at build time** and then serves
recommendations for any experiment at run time. Separate from the top-level `../../Dockerfile`
(which is a lean, general recommender with no baked data).

- **Build step** — on the dataset you pass in:
  1. trains Coastline's ML models — `coastline utils tune` → `tabpfn.pkl` (in-context) **and** `xgboost.pkl`
     (the best non-ICL model),
  2. calibrates Kavier, Coastline's physics dependency (`kavier calibrate` → `calibration.json`),
  3. wires the same dataset as Coastline's recommendation lookup/source.
- **Run step** — entrypoint is the Coastline CLI. Given an *experiment* trace (and, optionally, a
  *dataset* to recommend against), it emits a **recommendations CSV** (the patched experiment) and a
  **visualization PDF** (the cluster timeline).

Both Coastline **and** Kavier are installed as freshly built wheels — no source is mounted. The
Kavier wheel is built from a sibling checkout passed as a BuildKit build context named `kavier`,
because the released PyPI kavier lacks `kavier.sdk.cluster` (needed by the timeline plot).

## Build

The dataset is a **build parameter** (never hardcoded). Run from the repo root (the build context)
and pass both the Kavier source context and the dataset:

```bash
docker build -f docker/recommender/Dockerfile \
  --build-context kavier=../kavier \
  --build-arg DATASET_CSV=demo/baseline_15_baseline_shrink12_0805_0806.csv \
  -t coastline-trained .
```

- `--build-context kavier=../kavier` — the sibling Kavier checkout; its wheel is built and installed.
- `--build-arg DATASET_CSV=<path>` — the dataset, relative to the build context. Omitting it fails
  the build fast with guidance rather than baking a default.

The dataset may instead be mounted from any host path with another build context (then change the
trainer-stage `COPY ${DATASET_CSV} ...` to `COPY --from=dataset <file> /work/dataset.csv`):

```bash
docker build -f docker/recommender/Dockerfile \
  --build-context kavier=../kavier --build-context dataset=/abs/host/dir \
  -t coastline-trained .
```

The image ships the full ML stack (`[ml]` + `[plot]`), so it is multi-GB and the build installs
torch/tabpfn.

> The bundled `demo/baseline_15_baseline_shrink12_0805_0806.csv` is a thin 15-row dataset: `tune`
> warns (it likes ≥20 rows) and Kavier calibration falls back toward its shipped defaults where it
> cannot fit. Both still succeed — expect the warnings.

## Run

Entrypoint = the Coastline CLI. Default action is `recommend-trace --visual`.

```bash
# Validation run — uses the embedded dataset as BOTH the experiment and the lookup source.
docker run --rm -v "$PWD/out:/out" coastline-trained
# -> out/recommendations.csv, out/recommendations.pdf

# Your own experiment (trace CSV) and dataset (trace or flat measured-runs CSV).
docker run --rm -v "$PWD/in:/in" -v "$PWD/out:/out" \
  coastline-trained /in/experiment.csv /in/dataset.csv
```

### Options (simplified interface)

```
entrypoint [EXPERIMENT_CSV] [DATASET_CSV] [--method M] [--goal G] [--outdir DIR] [--feasibility F]
```

| Arg | Default | Meaning |
|---|---|---|
| `EXPERIMENT_CSV` | embedded dataset | fine-tuning trace to recommend for |
| `DATASET_CSV` | embedded lookup | measured-runs to recommend against (a trace is auto-converted) |
| `--method` | `intelligent` | `intelligent` (lookup → calibrated Kavier) · `kavier` · `tabpfn` · `xgboost` · `cache` |
| `--goal` | `min_gpu` | `min_gpu` · `performance` |
| `--outdir` | `/out` | where the CSV + PDF are written |
| `--feasibility` | `autoconf` | `autoconf` (OOM-aware) · `rules` |
| `--cluster-gpus` | baked `infrastructure.yaml` | total cluster GPUs; recommendations never exceed it (`--node-gpus` sets GPUs/node) |

```bash
# Method / goal overrides
docker run --rm -v "$PWD/out:/out" coastline-trained --method kavier --goal performance
docker run --rm -v "$PWD/out:/out" coastline-trained --method tabpfn        # the tuned ML model
```

### Passthrough

Any other Coastline verb runs directly — the entrypoint forwards to the CLI when the first argument
is a subcommand or a help/version flag:

```bash
docker run --rm coastline-trained --help
docker run --rm -v "$PWD/out:/out" coastline-trained \
  recommend-trace --input /data/dataset.csv --output /out/r.csv --method kavier
```

## What gets baked in

| Path (in image) | Produced by | Used for |
|---|---|---|
| `/data/profiling-dataset/raw_trace.csv` | `coastline utils trace-to-runs` | recommendation lookup source (`DATA_DIR`) |
| `/data/profiling-dataset/curated_trace.csv` | `coastline utils trace-to-runs` | selectable options |
| `/data/calibration.json` | `kavier calibrate` | Kavier calibration (`KAVIER_CALIBRATION`) |
| `/models/custom/tabpfn.pkl` | `coastline utils tune --model tabpfn` | in-context ML predictor (`--method tabpfn`) |
| `/models/custom/xgboost.pkl` | `coastline utils tune --model xgboost` | best non-ICL ML predictor (`--method xgboost`) |
| `/data/dataset.csv` | the build parameter | default experiment (original trace schema) |
