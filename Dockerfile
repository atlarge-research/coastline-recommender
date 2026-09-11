# Coastline — multi-stage build. Stage 1 builds a wheel with `uv build`; the runtime installs the
# LOCKED dependency set (from the committed uv.lock) plus ONLY that wheel (compiled to bytecode) —
# no source tree, no PYTHONPATH. Two final targets multiplex the two entrypoints off one shared
# runtime:
#     docker build --target cli -t coastline:cli .    # -> `coastline` recommender CLI (default)
#     docker build --target ui  -t coastline:ui  .    # -> `coastline-ui` FastAPI dashboard (:8000)
# Run:  docker run --rm coastline:cli --help
#       docker run --rm -p 8000:8000 coastline:ui
#
# The default image is LEAN (Kavier analytical physics path — no ML backends, no pickles). Bake in the
# optional heavy capabilities with the EXTRAS build arg:
#     docker build --target cli --build-arg EXTRAS="[ml]"    -t coastline:cli-ml .
#     docker build --target ui  --build-arg EXTRAS="[plot]"  -t coastline:ui-plot .
# The trained ML pickles are NOT in the wheel — mount them (or run the trainer) for the [ml] path.
#
# REPRODUCIBILITY. Two things make this image pinned rather than "whatever PyPI/Docker Hub had today":
#   1. Every base image is pinned by DIGEST, not just by tag. Tags are mutable; digests are not.
#      Refresh a digest with:  docker buildx imagetools inspect <image>:<tag>
#   2. Dependencies are installed with `uv sync --locked` from the COMMITTED uv.lock, so the build
#      uses the exact versions reviewed in git. `--locked` turns a stale lock into a BUILD FAILURE
#      instead of a silent re-resolution against PyPI. Regenerate with `uv lock` and commit it.

# ---- stage 1: build the wheel ----
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim@sha256:531f855bda2c73cd6ef67d56b733b357cea384185b3022bd09f05e002cd144ca AS build
WORKDIR /src
COPY . .
RUN uv build --wheel --out-dir /dist

# ---- shared runtime: locked deps + ONLY the wheel, compiled to bytecode ----
FROM python:3.13-slim@sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285 AS runtime
# uv gives us reproducible installs and honors the pyarrow override the core ado-autoconf
# dependency needs (kavier pins pyarrow>=23; autogluon caps <21 — the override lets both resolve).
# That override is already baked into uv.lock's [manifest], so no override file is needed here.
COPY --from=ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a /uv /usr/local/bin/uv
ARG EXTRAS=""
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH=/opt/venv/bin:$PATH
WORKDIR /build

# 1. Dependencies — straight from the committed lock. `--locked` fails the build if uv.lock no
#    longer matches pyproject.toml (rather than quietly re-resolving); `--no-install-project` keeps
#    the project itself out of this layer so it arrives as the wheel below, not as a source tree.
#    EXTRAS keeps its documented "[ml]" / "[ml,plot]" spelling and is translated to `--extra` flags.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN set -eu; \
    extras=''; \
    for e in $(printf '%s' "${EXTRAS}" | tr -d '[]' | tr ',' ' '); do extras="${extras} --extra ${e}"; done; \
    uv sync --locked --no-dev --no-install-project --python /usr/local/bin/python3.13 ${extras}

# 2. The project itself, as the built wheel only. `--no-deps`: the dependency set is exactly what
#    the lock said — nothing may be re-resolved from PyPI at this point.
COPY --from=build /dist/*.whl /tmp/
WORKDIR /
RUN uv pip install --python /opt/venv/bin/python --no-deps "$(echo /tmp/*.whl)" && \
    rm -rf /tmp/*.whl /build

# Allow multiple OpenMP runtimes (native ML backends each bundle libomp).
ENV KMP_DUPLICATE_LIB_OK=TRUE
# Non-root.
RUN useradd -m -u 1000 coastline
USER coastline
WORKDIR /home/coastline

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
