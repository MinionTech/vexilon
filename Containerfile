# ─── Stage 0: Base ────────────────────────────────────────────────────────────
FROM python:3.14-slim AS base

# Silence Hugging Face nag messages globally
ENV HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    HF_HOME=/hf_cache \
    EMBED_MODEL=/hf_cache

# Install common runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-privileged user once (UID 1001 is standard for this repo)
RUN useradd --uid 1001 --create-home --shell /sbin/nologin app
WORKDIR /app

# ─── Stage 1: Model Fetcher ──────────────────────────────────────────────────
FROM base AS model_fetcher

# Extract uv version from pyproject.toml to stay in sync with Renovate
COPY pyproject.toml .
RUN pip install --no-cache-dir uv==$(grep -oP 'uv==\K[\d.]+' pyproject.toml)

RUN uv pip install --system huggingface_hub
# Download model directly into the consolidated HF_HOME path.
RUN --mount=type=cache,target=/root/.cache/huggingface \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-small-en-v1.5', cache_dir='/root/.cache/huggingface', local_dir='/hf_cache', token=False, local_dir_use_symlinks=False)" && \
    ls -l /hf_cache/config.json

# ─── Stage 2: Builder ─────────────────────────────────────────────────────────
FROM base AS builder

# Extract uv version from pyproject.toml
COPY pyproject.toml .
RUN pip install --no-cache-dir uv==$(grep -oP 'uv==\K[\d.]+' pyproject.toml)

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Enforce offline mode for builds and tests
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

# 1. Install dependencies
# We copy pyproject.toml and uv.lock as root.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_LINK_MODE=copy uv sync --frozen --no-dev --no-install-project

# Copy source code and data as root (read-only for the app user later)
COPY data/ ./data/
COPY agnav/ ./agnav/
COPY scripts/ ./scripts/
COPY prompts/ ./prompts/
COPY app.py conftest.py ./

# ─── Stage 2.5: Test Builder ─────────────────────────────────────────────────
FROM builder AS test_builder
COPY --from=model_fetcher /hf_cache /hf_cache
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_LINK_MODE=copy uv sync --frozen --no-install-project
COPY tests/ ./tests/
RUN mkdir -p /app/reports /app/.pytest_cache && chown -R 1001:1001 /app/reports /app/.pytest_cache

# ─── Stage 3: Runtime ─────────────────────────────────────────────────────────
FROM base AS runner

# Enforce offline mode for production runtime
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    PATH="/app/.venv/bin:$PATH"

# Copy everything as root (read-only for the application user)
COPY --from=builder /app /app
COPY --from=model_fetcher /hf_cache /hf_cache

# Only create and chown (by UID) the specific directories that MUST be writable
RUN mkdir -p /app/.pdf_cache && chown 1001:1001 /app/.pdf_cache

USER 1001
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
