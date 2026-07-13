#!/usr/bin/env bash
# Entrypoint for the pre-trained Coastline recommender image.
#
# Simplified interface — recommend configs for an experiment and visualize the cluster timeline:
#
#   entrypoint.sh [EXPERIMENT_CSV] [DATASET_CSV] [--method M] [--goal G] [--outdir DIR] [--feasibility F]
#
#     EXPERIMENT_CSV  fine-tuning trace to recommend for   (default: the embedded dataset)
#     DATASET_CSV     measured-runs to recommend against    (default: the embedded lookup source);
#                     a trace CSV is auto-converted to the flat lookup schema
#     --method        intelligent (default) | kavier | tabpfn | xgboost | cache
#     --goal          min_gpu (default) | performance
#     --outdir        output directory                      (default: $OUTPUT_DIR, i.e. /out)
#     --feasibility   autoconf (default, OOM-aware) | rules
#     --cluster-gpus  total cluster GPUs — recommendations never exceed it
#                     (default: the baked infrastructure.yaml); --node-gpus sets GPUs/node
#
#   Outputs:  <outdir>/recommendations.csv  and  <outdir>/recommendations.pdf
#
# Passthrough: if the first argument is a Coastline subcommand (or -h/--help/-V), exec `coastline`
# directly, so every CLI verb (run, recommend, plot-trace, tune, trace-to-runs, ...) stays available.
set -euo pipefail

DEFAULT_EXPERIMENT="${COASTLINE_DEFAULT_EXPERIMENT:-/data/dataset.csv}"
DEFAULT_LOOKUP="${COASTLINE_DEFAULT_LOOKUP:-/data/profiling-dataset/raw_trace.csv}"

case "${1:-}" in
    recommend | run | recommend-trace | plot-trace | interactive | tune | trace-to-runs | -h | --help | -V | --version)
        exec coastline "$@"
        ;;
esac

METHOD="intelligent"
GOAL="min_gpu"
OUTDIR="${OUTPUT_DIR:-/out}"
FEASIBILITY="autoconf"
CLUSTER_GPUS=""   # empty -> recommend-trace uses infrastructure.yaml's declared total
NODE_GPUS=""
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --method) METHOD="$2"; shift 2 ;;
        --goal) GOAL="$2"; shift 2 ;;
        --outdir) OUTDIR="$2"; shift 2 ;;
        --feasibility) FEASIBILITY="$2"; shift 2 ;;
        --cluster-gpus) CLUSTER_GPUS="$2"; shift 2 ;;
        --node-gpus) NODE_GPUS="$2"; shift 2 ;;
        --method=*) METHOD="${1#*=}"; shift ;;
        --goal=*) GOAL="${1#*=}"; shift ;;
        --outdir=*) OUTDIR="${1#*=}"; shift ;;
        --feasibility=*) FEASIBILITY="${1#*=}"; shift ;;
        --cluster-gpus=*) CLUSTER_GPUS="${1#*=}"; shift ;;
        --node-gpus=*) NODE_GPUS="${1#*=}"; shift ;;
        --) shift; while [[ $# -gt 0 ]]; do POSITIONAL+=("$1"); shift; done ;;
        -*) echo "entrypoint: unknown option '$1'" >&2; exit 2 ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done


EXPERIMENT="${POSITIONAL[0]:-$DEFAULT_EXPERIMENT}"
DATASET="${POSITIONAL[1]:-}"

mkdir -p "$OUTDIR"

# Resolve the lookup source. A user-supplied dataset is normalised to the flat measured-runs
# schema (idempotent — an already-flat CSV passes through). With none given, use the lookup
# baked into the image at build time.
if [[ -n "$DATASET" ]]; then
    LOOKUP="$OUTDIR/lookup.csv"
    coastline utils trace-to-runs --input "$DATASET" --output "$LOOKUP"
else
    LOOKUP="$DEFAULT_LOOKUP"
fi

echo "==> recommend-trace: experiment='$EXPERIMENT' lookup='$LOOKUP' method='$METHOD' goal='$GOAL'"
# Build the command as one array (never empty) so optional cluster flags append cleanly.
cmd=(coastline recommend-trace
    --input "$EXPERIMENT"
    --output "$OUTDIR/recommendations.csv"
    --lookup "$LOOKUP"
    --method "$METHOD"
    --goal "$GOAL"
    --feasibility "$FEASIBILITY"
    --visual
    --visual-output "$OUTDIR/recommendations.pdf")
[[ -n "$CLUSTER_GPUS" ]] && cmd+=(--cluster-gpus "$CLUSTER_GPUS")
[[ -n "$NODE_GPUS" ]] && cmd+=(--node-gpus "$NODE_GPUS")
exec "${cmd[@]}"
