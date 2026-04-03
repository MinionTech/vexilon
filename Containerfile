# ─── Stage 0: External Binaries ──────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:0.11.3 AS uv_source

# ─── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.14.3-slim AS builder

COPY --from=uv_source /uv /usr/local/bin/uv
WORKDIR /app

# Install dependencies into a virtualenv
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_LINK_MODE=copy uv sync --frozen --no-dev --no-install-project

# Pre-download the embedding model into a persistent cache
RUN HF_HOME=/app/hf_cache HF_HUB_DISABLE_IMPLICIT_TOKEN=1 UV_LINK_MODE=copy uv run python -c \
    "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# ─── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.14.3-slim AS runner

# 1. Runtime system deps and setup (runs once, cached forever)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* && \
    useradd --uid 1001 --create-home --shell /sbin/nologin vexilon

WORKDIR /app

# ── Environment Configuration ────────────────────────────────────────────────
# Set these early so they are active during the build-time indexing step.
ENV HF_HOME=/app/hf_cache \
    TRANSFORMERS_OFFLINE=1 \
    PATH="/app/.venv/bin:$PATH"

# 2. Copy the virtualenv and model cache from the builder
COPY --from=builder --chown=vexilon:vexilon /app/.venv /app/.venv
COPY --from=builder /app/hf_cache /app/hf_cache

# 3. Copy only what is needed for indexing (expensive step)
COPY --chown=vexilon:vexilon data/ ./data/
COPY --chown=vexilon:vexilon src/ ./src/
COPY --chown=vexilon:vexilon scripts/build_index.py ./scripts/
RUN mkdir -p /app/.pdf_cache && chown vexilon:vexilon /app/.pdf_cache

USER vexilon
RUN PATH="/app/.venv/bin:$PATH" python scripts/build_index.py

# 4. Copy the remaining scripts and application code
# (Changes here won't trigger a re-index)
COPY --chown=vexilon:vexilon scripts/ ./scripts/
COPY --chown=vexilon:vexilon prompts/ ./prompts/
COPY --chown=vexilon:vexilon app.py style.css ./

# ── Final Environment ────────────────────────────────────────────────────────
# Build provenance (move to end to avoid cache busts on every commit)
ARG VERSION
ENV VEXILON_VERSION=$VERSION

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860')" || exit 1

CMD ["/app/scripts/startup.sh"]
