# ─── Global Arguments ─────────────────────────────────────────────────────────
# This allows Renovate to track the uv version easily.
ARG UV_VERSION=0.6.5

# ─── Stage 0: Base ────────────────────────────────────────────────────────────
FROM python:3.14-slim AS base

# Install common runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-privileged user once
RUN useradd --uid 1001 --create-home --shell /sbin/nologin app
WORKDIR /app

# ─── Stage 1: Model Fetcher ──────────────────────────────────────────────────
FROM base AS model_fetcher
ARG UV_VERSION
RUN pip install --no-cache-dir uv==${UV_VERSION}

ENV HF_HUB_DISABLE_IMPLICIT_TOKEN=1
RUN uv pip install --system huggingface_hub
RUN --mount=type=cache,target=/root/.cache/huggingface \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-small-en-v1.5', cache_dir='/root/.cache/huggingface', local_dir='/model_cache', token=False, local_dir_use_symlinks=False)" && \
    ls -l /model_cache/config.json

# ─── Stage 2: Builder ─────────────────────────────────────────────────────────
FROM base AS builder
ARG UV_VERSION
RUN pip install --no-cache-dir uv==${UV_VERSION}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

ENV HF_HOME=/hf_cache \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    EMBED_MODEL=/hf_cache

COPY --chown=app:app pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_LINK_MODE=copy uv sync --frozen --no-dev --no-install-project

COPY --chown=app:app data/ ./data/
COPY --chown=app:app agnav/ ./agnav/
COPY --chown=app:app scripts/ ./scripts/
COPY --chown=app:app prompts/ ./prompts/
COPY --chown=app:app app.py conftest.py ./

# ─── Stage 2.5: Test Builder ─────────────────────────────────────────────────
FROM builder AS test_builder

COPY --from=model_fetcher /model_cache /hf_cache
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_LINK_MODE=copy uv sync --frozen --no-install-project
COPY tests/ ./tests/
RUN chown -R app:app /app

# ─── Stage 3: Runtime ─────────────────────────────────────────────────────────
FROM base AS runner

ENV HF_HOME=/hf_cache \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    EMBED_MODEL=/hf_cache \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder --chown=app:app /app /app
COPY --from=model_fetcher --chown=app:app /model_cache /hf_cache

RUN mkdir -p /app/.pdf_cache && chown app:app /app/.pdf_cache

USER app
RUN --mount=type=cache,target=/app/.pdf_cache_mount,uid=1001,gid=1001 \
    mkdir -p /app/.pdf_cache && \
    cp -r /app/.pdf_cache_mount/* /app/.pdf_cache/ 2>/dev/null || true && \
    PATH="/app/.venv/bin:$PATH" python scripts/build_index.py && \
    cp -r /app/.pdf_cache/* /app/.pdf_cache_mount/ 2>/dev/null || true

ARG VERSION="Dev mode"
ARG REPO_URL="https://github.com/MinionTech/vexilon"
ENV AGNAV_VERSION=$VERSION
ENV AGNAV_REPO_URL=$REPO_URL

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=30s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860')" || exit 1
CMD ["python", "app.py"]
