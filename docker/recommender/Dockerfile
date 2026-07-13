# syntax=docker/dockerfile:1
# ============================================================================================
# Pre-trained, self-contained Coastline recommender.
#
# Unlike the top-level ./Dockerfile (a lean, general recommender with no baked data), this image
# is TRAINED on a specific measured-runs dataset at BUILD time and then serves recommendations for
# any experiment at RUN time. The dataset is a BUILD PARAMETER — it is never hardcoded here.
#
# Both Coastline AND Kavier are installed as freshly BUILT WHEELS (no source is mounted). The Kavier
# wheel is built from a sibling checkout passed as a BuildKit build context named `kavier` — the
# released PyPI kavier lacks `kavier.sdk.cluster`, which the timeline plot (`--visual`) needs.
#
# BUILD (bakes three trained capabilities into the image):
#     docker build -f docker/recommender/Dockerfile \
#         --build-context kavier=../kavier \
#         --build-arg DATASET_CSV=demo/baseline_15_baseline_shrink12_0805_0806.csv \
#         -t coastline-trained .
#   1. trains Coastline's ML model on the dataset          (coastline utils tune -> tabpfn.pkl)
#   2. calibrates Kavier (the physics dependency)          (kavier calibrate -> calibration.json)
#   3. wires the dataset as the recommendation lookup      ($DATA_DIR/profiling-dataset/raw_trace.csv)
#
#   The dataset may instead be mounted from any host path with another build context (adjust the COPY):
#     docker build ... --build-context dataset=/abs/host/dir ...
#
# RUN (entrypoint = the Coastline CLI; produces a recommendations CSV + a visualization PDF):
#     docker run --rm -v "$PWD/out:/out" coastline-trained                       # default: embedded dataset
#     docker run --rm -v "$PWD/in:/in" -v "$PWD/out:/out" \
#         coastline-trained /in/experiment.csv /in/dataset.csv --method kavier    # your own experiment + dataset
#   Outputs land in the mounted /out: recommendations.csv and recommendations.pdf.
# ============================================================================================

# ---- stage 1a: build the coastline wheel (contains the new `trace-to-runs` command) ----
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS wheel
WORKDIR /src
COPY . .
RUN uv build --wheel --out-dir /dist

# ---- stage 1b: build the kavier wheel from the sibling checkout (build context `kavier`) ----
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS kwheel
WORKDIR /ksrc
COPY --from=kavier . .
RUN uv build --wheel --out-dir /kdist

# ---- stage 2: shared ML base — installs both wheels + the heavy ML/plot stack ONCE ----
# Both the trainer and the final runtime derive from this, so torch/tabpfn are installed a single
# time. The pyarrow override lets kavier (pyarrow>=23) and autogluon (pyarrow<21, pulled by
# ado-autoconf) co-resolve. KMP_DUPLICATE_LIB_OK allows the native ML backends' bundled libomp.
# scipy (needed by `kavier calibrate` in the trainer) arrives transitively via scikit-learn ([ml]).
FROM python:3.13-slim AS mlbase
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv
ENV KMP_DUPLICATE_LIB_OK=TRUE
COPY --from=wheel /dist/*.whl /tmp/cwheel/
COPY --from=kwheel /kdist/*.whl /tmp/kwheel/
RUN echo "pyarrow>=23.0.1" > /tmp/override.txt && \
    uv pip install --system --compile-bytecode --override /tmp/override.txt \
        "$(echo /tmp/cwheel/*.whl)[ml,plot]" "$(echo /tmp/kwheel/*.whl)" && \
    rm -rf /tmp/cwheel /tmp/kwheel /tmp/override.txt

# ---- stage 3: train on the dataset (ML tune + Kavier calibration + flat lookup) ----
FROM mlbase AS trainer
WORKDIR /work
# The dataset is a build PARAMETER, not a hardcoded path. The default is an inert placeholder
# that makes the build fail fast with guidance; pass --build-arg DATASET_CSV=<path-in-context>.
ARG DATASET_CSV=docker/recommender/REQUIRED_DATASET.csv
COPY docker/recommender/train.sh /usr/local/bin/train.sh
COPY ${DATASET_CSV} /work/dataset.csv
RUN bash /usr/local/bin/train.sh /work/dataset.csv /work

# ---- stage 4: final runtime with the trained artifacts baked in ----
FROM mlbase AS runtime
# Wire the trained artifacts + dataset produced by the trainer stage.
#   DATA_DIR            -> the flat dataset is the recommendation lookup/source
#   PORTFOLIO_DIR       -> the tuned tabpfn model (served by --method tabpfn)
#   KAVIER_CALIBRATION  -> the fitted (or shipped-default) Kavier calibration
ENV DATA_DIR=/data \
    PORTFOLIO_DIR=/models \
    KAVIER_CALIBRATION=/data/calibration.json \
    INFRASTRUCTURE_CONFIG=/data/infrastructure.yaml \
    OUTPUT_DIR=/out \
    COASTLINE_ALLOW_RULES_FALLBACK=1
# The declared cluster size (recommendations never exceed it); override per-run with --cluster-gpus.
COPY config/coastline_functionality/infrastructure.yaml /data/infrastructure.yaml
COPY --from=trainer /work/run_database.csv         /data/profiling-dataset/raw_trace.csv
COPY --from=trainer /work/run_database.csv         /data/profiling-dataset/curated_trace.csv
COPY --from=trainer /work/calibration.json          /data/calibration.json
COPY --from=trainer /work/models/custom/tabpfn.pkl  /models/custom/tabpfn.pkl
COPY --from=trainer /work/models/custom/xgboost.pkl /models/custom/xgboost.pkl
COPY --from=trainer /work/dataset.csv               /data/dataset.csv
COPY docker/recommender/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh && \
    useradd -m -u 1000 coastline && \
    mkdir -p /out && chown -R coastline /out /data /models
USER coastline
WORKDIR /out
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD []
