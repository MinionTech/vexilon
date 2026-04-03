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

# Build provenance
ARG VERSION
ENV VEXILON_VERSION=$VERSION

# Runtime system deps only (libgomp for FAISS, Python-native healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create our non-root user
RUN useradd --uid 1001 --no-create-home --shell /sbin/nologin vexilon
WORKDIR /app

# 1. Copy the virtualenv and model cache from the builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/hf_cache /app/hf_cache

# 2. Copy data, internal libraries, and indexing scripts
COPY data/ ./data/
COPY docs/ ./docs/
COPY src/ ./src/
COPY scripts/ ./scripts/
RUN chmod +x /app/scripts/*.sh

# ─── Final Environment ────────────────────────────────────────────────────────
# Set PATH before any RUN steps that invoke Python so they use the venv.
ENV HF_HOME=/app/hf_cache \
    TRANSFORMERS_OFFLINE=1 \
    PATH="/app/.venv/bin:$PATH"

# 3. Pre-build the FAISS index at image build time.
# Pre-building here allows Docker to cache this expensive layer 
# if the source data (data/) and indexing logic (src/) have not changed.
RUN mkdir -p /app/.pdf_cache && chown 1001:1001 /app/.pdf_cache
USER 1001
RUN python scripts/build_index.py

# 4. Copy application UI and Prompts
# Changes to prompts/ or app.py will NOT trigger a re-index.
COPY prompts/ ./prompts/
COPY app.py style.css ./

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860')" || exit 1

CMD ["/app/scripts/startup.sh"]
