# Two targets on one runtime:
#     docker build --target cli -t coastline:cli .    # `coastline` (default)
#     docker build --target ui  -t coastline:ui  .    # `coastline-ui` dashboard on :8000
#     docker run --rm coastline:cli --help
#     docker run --rm -p 8000:8000 coastline:ui
#
# The default image has no ML backends. Add extras with EXTRAS, for example:
#     docker build --target cli --build-arg EXTRAS="[ml]" -t coastline:cli-ml .
# The wheel leaves out the large model pickles; mount them and set PORTFOLIO_DIR for the [ml] path.
#
# Base images are pinned by digest (refresh with `docker buildx imagetools inspect <image>:<tag>`),
# and dependencies come from the committed uv.lock (`uv sync --locked`).

# Stage 1: build the wheel.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim@sha256:531f855bda2c73cd6ef67d56b733b357cea384185b3022bd09f05e002cd144ca AS build
WORKDIR /src
COPY . .
RUN uv build --wheel --out-dir /dist

# Runtime: the locked dependencies, then the wheel.
FROM python:3.13-slim@sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285 AS runtime
COPY --from=ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a /uv /usr/local/bin/uv
ARG EXTRAS=""
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH=/opt/venv/bin:$PATH
WORKDIR /build

# EXTRAS="[ml,plot]" becomes `--extra ml --extra plot`.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN set -eu; \
    extras=''; \
    for e in $(printf '%s' "${EXTRAS}" | tr -d '[]' | tr ',' ' '); do extras="${extras} --extra ${e}"; done; \
    uv sync --locked --no-dev --no-install-project --python /usr/local/bin/python3.13 ${extras}

# --no-deps: every dependency comes from the lock above.
COPY --from=build /dist/*.whl /tmp/
WORKDIR /
RUN uv pip install --python /opt/venv/bin/python --no-deps "$(echo /tmp/*.whl)" && \
    rm -rf /tmp/*.whl /build

# The native ML backends each bundle libomp.
ENV KMP_DUPLICATE_LIB_OK=TRUE
RUN useradd -m -u 1000 coastline
USER coastline
WORKDIR /home/coastline
# Models tuned with `coastline utils tune` go to PORTFOLIO_DIR/custom; the installed package is
# root-owned, so point it at the user's home. The bundled models still load from the package.
ENV PORTFOLIO_DIR=/home/coastline/models

FROM runtime AS cli
ENTRYPOINT ["coastline"]
CMD ["--help"]

FROM runtime AS ui
ENV COASTLINE_UI_HOST=0.0.0.0 \
    COASTLINE_UI_PORT=8000
EXPOSE 8000
ENTRYPOINT ["coastline-ui"]
