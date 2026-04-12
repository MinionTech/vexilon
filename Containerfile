# ─── Stage 0: External Binaries ──────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:0.11.3 AS uv_source

# ─── Stage 1: Model Fetcher ──────────────────────────────────────────────────
# This stage only re-runs if the model name changes.
FROM python:3.14.3-slim AS model_fetcher

COPY --from=uv_source /uv /usr/local/bin/uv

# Install huggingface_hub
RUN uv pip install --system huggingface_hub

# Fetch model directly into /model_cache using the Python API.
# This avoids shell PATH issues and wildcard copy bloat.
# We use token=False to prevent auth attempts and satisfy security scanners.
RUN --mount=type=cache,target=/root/.cache/huggingface \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-small-en-v1.5', cache_dir='/root/.cache/huggingface', local_dir='/model_cache', token=False)"

# ─── Stage 2: Builder ─────────────────────────────────────────────────────────
FROM python:3.14.3-slim AS builder

COPY --from=uv_source /uv /usr/local/bin/uv
WORKDIR /app

# 1. Install dependencies
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_LINK_MODE=copy uv sync --frozen --no-dev --no-install-project

COPY data/ ./data/
COPY vexilon/ ./vexilon/
COPY scripts/ ./scripts/
# Data and indexing engine code copied — Indexing prerequisites complete.

COPY prompts/ ./prompts/
COPY app.py style.css conftest.py ./


# ─── Stage 2.5: Test Builder ─────────────────────────────────────────────────
# This stage adds dev dependencies and test suite for the 'tests' service.
FROM builder AS test_builder
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_LINK_MODE=copy uv sync --frozen --no-install-project

COPY tests/ ./tests/


# ─── Stage 3: Runtime ─────────────────────────────────────────────────────────
FROM python:3.14.3-slim AS runner

# 1. Runtime system deps and setup
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* && \
    useradd --uid 1001 --create-home --shell /sbin/nologin vexilon

WORKDIR /app

# ── Environment Configuration ────────────────────────────────────────────────
ENV HF_HOME=/app/hf_cache \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    EMBED_MODEL=/app/hf_cache \
    PATH="/app/.venv/bin:$PATH"

# 2. Copy the prepared environment and code from builder
COPY --from=builder --chown=vexilon:vexilon /app /app
COPY --from=model_fetcher --chown=vexilon:vexilon /model_cache /app/hf_cache

# 3. Build index
# Create persistent cache directory
RUN mkdir -p /app/.pdf_cache && chown vexilon:vexilon /app/.pdf_cache

USER vexilon
RUN --mount=type=cache,target=/app/.pdf_cache_mount,uid=1001,gid=1001 \
    mkdir -p /app/.pdf_cache && \
    cp -r /app/.pdf_cache_mount/* /app/.pdf_cache/ 2>/dev/null || true && \
    PATH="/app/.venv/bin:$PATH" python scripts/build_index.py && \
    cp -r /app/.pdf_cache/* /app/.pdf_cache_mount/ 2>/dev/null || true

# ── Final Environment ────────────────────────────────────────────────────────
ARG VERSION
ENV VEXILON_VERSION=$VERSION

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860')" || exit 1

CMD ["/app/scripts/startup.sh"]
