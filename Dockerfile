# Coastline — multi-stage build. Stage 1 builds a wheel with `uv build`; the runtime installs ONLY
# that wheel (compiled to bytecode) — no source tree, no PYTHONPATH. Two final targets multiplex the
# two entrypoints off one shared runtime:
#     docker build --target cli -t coastline:cli .    # -> `coastline` recommender CLI (default)
#     docker build --target ui  -t coastline:ui  .    # -> `coastline-ui` FastAPI dashboard (:8000)
# Run:  docker run --rm coastline:cli --help
#       docker run --rm -p 8000:8000 coastline:ui
#
# The default image is LEAN (Kavier analytical physics path — no ML backends, no pickles). Bake in the
# optional heavy capabilities with the EXTRAS build arg:
#     docker build --target cli --build-arg EXTRAS="[ml]"          -t coastline:cli-ml .
#     docker build --target ui  --build-arg EXTRAS="[autoconf]"    -t coastline:ui-autoconf .
# The trained ML pickles are NOT in the wheel — mount them (or run the trainer) for the [ml] path; the
# OpenDC energy path additionally needs a JRE + the OpenDC runner mounted at OPENDC_BIN_PATH.

# ---- stage 1: build the wheel ----
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS build
WORKDIR /src
COPY . .
RUN uv build --wheel --out-dir /dist

# ---- shared runtime: install ONLY the wheel (+ optional EXTRAS), compiled to bytecode ----
FROM python:3.13-slim AS runtime
# uv gives us reproducible installs and honors the pyarrow override the [autoconf] extra needs
# (kavier pins pyarrow>=23; autogluon caps <21 — the override lets both resolve).
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv
ARG EXTRAS=""
COPY --from=build /dist/*.whl /tmp/
RUN echo "pyarrow>=23.0.1" > /tmp/override.txt && \
    uv pip install --system --compile-bytecode --override /tmp/override.txt "$(echo /tmp/*.whl)${EXTRAS}" && \
    rm -rf /tmp/*.whl /tmp/override.txt
# Allow multiple OpenMP runtimes (native ML backends each bundle libomp).
ENV KMP_DUPLICATE_LIB_OK=TRUE
# Non-root.
RUN useradd -m -u 1000 coastline
USER coastline

# ---- target: recommender CLI (default image) ----
FROM runtime AS cli
ENTRYPOINT ["coastline"]
CMD ["--help"]

# ---- target: FastAPI dashboard ----
FROM runtime AS ui
ENV COASTLINE_UI_HOST=0.0.0.0 \
    COASTLINE_UI_PORT=8000
EXPOSE 8000
ENTRYPOINT ["coastline-ui"]
