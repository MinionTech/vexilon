# ─── Stage 0: External Binaries ──────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:0.11.3 AS uv_source

# ─── Stage 1: Model Fetcher ──────────────────────────────────────────────────
# This stage only re-runs if the model name changes.
FROM python:3.12-slim AS model_fetcher

# Prevent auth attempts for public models
ENV HF_HUB_DISABLE_IMPLICIT_TOKEN=1
COPY --from=uv_source /uv /usr/local/bin/uv

# Install huggingface_hub
RUN uv pip install --system huggingface_hub

# Fetch model directly into /model_cache using the Python API.
# This avoids shell PATH issues and wildcard copy bloat.
# We use token=False to prevent auth attempts and satisfy security scanners.
RUN --mount=type=cache,target=/root/.cache/huggingface \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-small-en-v1.5', cache_dir='/root/.cache/huggingface', local_dir='/model_cache', token=False, local_dir_use_symlinks=False)" && \
    ls -l /model_cache/config.json # Verify download succeeded

# ─── Stage 2: Builder ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# ── Environment Configuration ────────────────────────────────────────────────
ENV HF_HOME=/hf_cache \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    EMBED_MODEL=/hf_cache

COPY --from=uv_source /uv /usr/local/bin/uv
WORKDIR /app

# 1. Install dependencies
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_LINK_MODE=copy uv sync --frozen --no-dev --no-install-project

# 2. Copy application code
COPY data/ ./data/
COPY agnav/ ./agnav/
COPY scripts/ ./scripts/
COPY prompts/ ./prompts/
COPY app.py conftest.py ./

# ─── Stage 2.5: Test Builder ─────────────────────────────────────────────────
# This stage adds dev dependencies and test suite for the 'tests' service.
FROM builder AS test_builder

# Copy model from model_fetcher so tests can load it
COPY --from=model_fetcher /model_cache /hf_cache

RUN --mount=type=cache,target=/root/.cache/uv \
    UV_LINK_MODE=copy uv sync --frozen --no-install-project

COPY tests/ ./tests/

# ─── Stage 3: Runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runner

# 1. Runtime system deps and setup
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* && \
    useradd --uid 1000 --create-home --shell /sbin/nologin agnav

WORKDIR /app

# 2. Environment Setup
ENV HF_HOME=/hf_cache \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    EMBED_MODEL=/hf_cache \
    PATH="/app/.venv/bin:$PATH"

# 3. Copy artifacts from builder
COPY --from=builder --chown=agnav:agnav /app /app
COPY --from=model_fetcher --chown=agnav:agnav /model_cache /hf_cache

# 4. Prepare data cache and build index
RUN mkdir -p /app/.pdf_cache && chown agnav:agnav /app/.pdf_cache
USER agnav

# Build index (utilizing build cache mount for persistent index)
RUN --mount=type=cache,target=/app/.pdf_cache_mount,uid=1001,gid=1001 \
    mkdir -p /app/.pdf_cache && \
    cp -r /app/.pdf_cache_mount/* /app/.pdf_cache/ 2>/dev/null || true && \
    PATH="/app/.venv/bin:$PATH" python scripts/build_index.py && \
    cp -r /app/.pdf_cache/* /app/.pdf_cache_mount/ 2>/dev/null || true

# ── Final Environment ────────────────────────────────────────────────────────
ARG VERSION="Dev mode"
ARG REPO_URL="https://github.com/MinionTech/vexilon"
ENV AGNAV_VERSION=$VERSION
ENV AGNAV_REPO_URL=$REPO_URL

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=30s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860')" || exit 1

CMD ["/app/scripts/startup.sh"]
