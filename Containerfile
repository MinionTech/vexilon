# ─── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.14-slim AS base

# Install system deps: libgomp1 for FAISS, curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv — pin version for reproducible builds; Renovate will keep this current
COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /usr/local/bin/uv

# Create non-root user early
RUN useradd --uid 1001 --no-create-home --shell /sbin/nologin vexilon

WORKDIR /app
RUN chown 1001:1001 /app

# Switch to non-root for the rest of the build to avoid cache-busting 'chown -R' at the end
USER 1001

# ─── Dependencies ─────────────────────────────────────────────────────────────
COPY --chown=1001:1001 pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# ─── Model Download (Owned by 1001) ───────────────────────────────────────────
RUN HF_HOME=/app/hf_cache \
    uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
    && echo "[build] Embedding model cached."

# ─── Runtime env ──────────────────────────────────────────────────────────────
ENV HF_HOME=/app/hf_cache \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    UV_CACHE_DIR=/tmp/uv-cache

# ─── App layer: Volatile files last ───────────────────────────────────────────
COPY --chown=1001:1001 pdf_cache/ ./pdf_cache/
COPY --chown=1001:1001 app.py ./

EXPOSE 7860
CMD ["uv", "run", "python", "app.py"]
