#!/usr/bin/env bash
# Build-time training for the pre-trained Coastline recommender image.
#
#   Usage: train.sh <dataset_csv> <workdir>
#
# Runs the three BUILD-step operations on the embedded dataset:
#   1. trace -> flat measured-runs CSV      (coastline utils trace-to-runs -> run_database.csv)
#   2. train Coastline's ML model            (coastline utils tune --model tabpfn -> tabpfn.pkl)
#   3. calibrate Kavier on the same dataset  (kavier calibrate -> calibration.json)
#
# Steps 2 and 3 are best-effort on a thin dataset: `tune` warns (<20 rows) but still writes an
# artifact, and if `kavier calibrate` cannot fit a usable table we fall back to Kavier's shipped
# default calibration so KAVIER_CALIBRATION always resolves to a valid file.
set -euo pipefail

DATASET="${1:?usage: train.sh <dataset_csv> <workdir>}"
WORK="${2:?usage: train.sh <dataset_csv> <workdir>}"
RUNS="$WORK/run_database.csv"
CAL="$WORK/calibration.json"
TABPFN="$WORK/models/custom/tabpfn.pkl"
XGBOOST="$WORK/models/custom/xgboost.pkl"

# Guard: fail fast (with guidance) when no real dataset was passed as a build parameter.
if grep -q '__COASTLINE_DATASET_PLACEHOLDER__' "$DATASET" 2>/dev/null; then
    echo "ERROR: no dataset was provided at build time." >&2
    echo "Pass one with:  --build-arg DATASET_CSV=<path within the build context>" >&2
    exit 1
fi

# Copy Kavier's shipped default calibration into $1 (fallback when a fit is not usable).
copy_default_calibration() {
    python - "$1" <<'PY'
import shutil, sys
from importlib.resources import as_file, files

with as_file(files("kavier.sdk.training").joinpath("calibration", "calibration.json")) as src:
    shutil.copy(src, sys.argv[1])
PY
}

echo "==> [1/4] trace -> flat measured-runs schema"
coastline utils trace-to-runs --input "$DATASET" --output "$RUNS"

echo "==> [2/4] train the in-context ML model (tabpfn) on the dataset"
coastline utils tune --data "$RUNS" --model tabpfn --train-percentage 1.0 --output "$TABPFN"

echo "==> [3/4] train the best non-ICL ML model (xgboost) on the dataset"
coastline utils tune --data "$RUNS" --model xgboost --train-percentage 1.0 --output "$XGBOOST"

echo "==> [4/4] calibrate Kavier on the dataset (best-effort)"
MODELS="$(python -c "import pandas as pd; print(','.join(pd.read_csv('$RUNS')['model_name'].dropna().astype(str).unique()))")"
if kavier calibrate "$RUNS" --models "$MODELS" >"$CAL" 2>"$WORK/calibrate.log" \
    && python -c "import json,sys; d=json.load(open('$CAL')); sys.exit(0 if isinstance(d, dict) and d else 1)"; then
    echo "    fitted calibration written to $CAL"
else
    echo "    calibration not usable -> falling back to Kavier's shipped default calibration"
    sed 's/^/      /' "$WORK/calibrate.log" 2>/dev/null || true
    copy_default_calibration "$CAL"
fi

echo "==> training complete:"
ls -la "$RUNS" "$CAL" "$TABPFN" "$XGBOOST"
