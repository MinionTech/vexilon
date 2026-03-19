# ─── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.14.3-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies into a virtualenv
# This creates a standalone /app/.venv directory
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Pre-download the embedding model into a persistent cache
RUN --mount=type=cache,target=/root/.cache/huggingface \
    HF_HOME=/app/hf_cache \
    uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# ─── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.14.3-slim AS runner

# Build provenance
ARG VERSION
ENV VEXILON_VERSION=$VERSION

# Runtime system deps only (libgomp for FAISS, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create our non-root user
RUN useradd --uid 1001 --no-create-home --shell /sbin/nologin vexilon
WORKDIR /app

# 1. Copy the virtualenv and model cache from the builder
# We use --chown to ensure the runner user owns these files immediately
COPY --from=builder --chown=1001:1001 /app/.venv /app/.venv
COPY --from=builder --chown=1001:1001 /app/hf_cache /app/hf_cache

# 2. Copy application code and PDF assets (excluding ignored caches)
COPY --chown=1001:1001 data/ ./data/
COPY --chown=1001:1001 app.py ./

# Bake the build timestamp into a file after code is copied
RUN TZ="America/Vancouver" date +"%Y-%m-%d %H:%M %Z" > /app/build_version.txt && chown 1001:1001 /app/build_version.txt

# The index is provided by the host volume or first-run local bootstrap
# (Removed force_rebuild=True from image layers to prevent 404 errors)

# ─── Final Environment ────────────────────────────────────────────────────────
ENV HF_HOME=/app/hf_cache \
    TRANSFORMERS_OFFLINE=1 \
    PATH="/app/.venv/bin:$PATH"

EXPOSE 7860

CMD ["python", "app.py"]
