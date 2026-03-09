# ─── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.14-slim AS base

# Install system deps: libgomp1 for FAISS, curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv — pin version for reproducible builds; Renovate will keep this current
COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /usr/local/bin/uv

WORKDIR /app

# Install Python deps in a separate layer so code changes don't bust the cache.
# uv reads pyproject.toml + uv.lock; torch is routed to the CPU-only wheel
# index via [tool.uv.sources] — no separate pip step needed.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# ─── Pre-download embedding model — must happen before TRANSFORMERS_OFFLINE is set ───
# Downloads ~90 MB to /tmp/hf_cache so cold starts never hit the network.
RUN HF_HOME=/tmp/hf_cache \
    uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
    && echo "[build] Embedding model cached."

# ─── Runtime env: go offline now that the model is baked in ──────────────────
# TRANSFORMERS_OFFLINE=1 suppresses HF Hub network checks and cache-miss writes.
# HF_HOME must match the path used during the download above.
ENV HF_HOME=/tmp/hf_cache \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    UV_CACHE_DIR=/tmp/uv-cache

# ─── App layer ────────────────────────────────────────────────────────────────
COPY app.py ./
COPY pdf_cache/ ./pdf_cache/

# ─── Non-root user ────────────────────────────────────────────────────────────
RUN useradd --uid 1001 --no-create-home --shell /sbin/nologin vexilon \
    && chown -R 1001:1001 /app

USER 1001

# Gradio listens on 7860 by default
EXPOSE 7860

CMD ["uv", "run", "python", "app.py"]
