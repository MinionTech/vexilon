# ─── Stage 0: Base ────────────────────────────────────────────────────────────
FROM python:3.14-slim AS base

# Silence Hugging Face nag messages globally
ENV HF_HOME=/hf_cache \
    EMBED_MODEL=/model \
    CHAINLIT_FILES_DIR=/tmp/chainlit_files

# Install common runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libjpeg62-turbo \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-privileged user once (UID 1000 is standard for Hugging Face Spaces)
RUN useradd --uid 1000 --create-home --shell /sbin/nologin app
WORKDIR /app

# Shared Runtime Configuration
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=10 \
  CMD curl -f http://localhost:7860/ || exit 1

# ─── Stage 1: Model Fetcher ──────────────────────────────────────────────────
FROM base AS model_fetcher

# Extract uv version from app/pyproject.toml to stay in sync with Renovate
COPY app/pyproject.toml .
RUN pip install --no-cache-dir uv==$(grep -oP 'uv==\K[\d.]+' pyproject.toml)

RUN uv pip install --system --extra-index-url https://download.pytorch.org/whl/cpu torch sentence-transformers
RUN --mount=type=cache,target=/root/.cache/hf_v4 \
    python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('BAAI/bge-small-en-v1.5', cache_folder='/root/.cache/hf_v4'); model.save('/model')" && \
    ls -l /model/modules.json

# ─── Stage 2: Builder (Dependencies, Indexing, and Source) ────────────────────
FROM base AS builder

# Extract uv version from app/pyproject.toml
COPY app/pyproject.toml .
RUN pip install --no-cache-dir uv==$(grep -oP 'uv==\K[\d.]+' pyproject.toml)

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies (Cached unless uv.lock changes)
COPY app/pyproject.toml app/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    HF_HUB_OFFLINE=1 UV_LINK_MODE=copy uv sync --frozen --no-dev --no-install-project

# ─── Stage 2.1: Test Builder (Unit Tests - Lightweight) ──────────────────────
FROM builder AS test_builder
COPY --from=model_fetcher /model /model
# Layer dev dependencies on top of the production venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

COPY app/ ./
COPY app/data/labour_law/ ./public/docs/labour_law/
COPY PRIVACY.md ./public/docs/

# Prepare directories for testing and ensure permissions
RUN mkdir -p /app/reports /app/.pytest_cache /hf_cache && \
    chown -R 1000:1000 /app/reports /app/.pytest_cache /hf_cache

# ─── Stage 2.2: Indexed Builder (Production Indexing) ────────────────────────
FROM builder AS indexed_builder

# Model and FAISS Index (Cached unless data/ or scripts change)
COPY --from=model_fetcher /model /model
COPY app/data/ ./data/
COPY app/indexing.py ./
COPY app/scripts/build_index.py ./scripts/

RUN --mount=type=cache,target=/app/.pdf_cache_mount \
    mkdir -p /app/.pdf_cache && \
    cp -r /app/.pdf_cache_mount/* /app/.pdf_cache/ 2>/dev/null || true && \
    TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PATH="/app/.venv/bin:$PATH" python scripts/build_index.py && \
    cp -r /app/.pdf_cache/* /app/.pdf_cache_mount/ 2>/dev/null || true

# (Source code will be copied in leaf stages to maximize cache hits)

# ─── Stage 2.5: Functional Builder (Dev/Test Source) ──────────────────────────
FROM indexed_builder AS functional_builder

# Layer dev dependencies on top of the production venv (Cached unless uv.lock changes)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Copy source files in optimal cache order:
# 1. Static files first (least frequent changes)
# 2. Config files next
# 3. Application code last (most frequent changes)
COPY PRIVACY.md ./public/docs/
COPY app/.chainlit/ ./.chainlit/
COPY app/public/ ./public/
COPY app/chainlit.md ./
COPY app/data/labour_law/ ./public/docs/labour_law/

# Then source code (code changes often; trigger only code rebuilds)
COPY app/main.py ./
COPY app/indexing.py ./
COPY app/patches.py ./
COPY app/conftest.py ./
COPY app/prompts/ ./prompts/
COPY app/scripts/ ./scripts/
COPY app/tests/ ./tests/

# Prepare directories for testing and Chainlit runtime, ensure permissions.
# CHAINLIT_FILES_DIR points at /tmp (set in runner stage ENV) so we don't
# need to create /app/.files here. Keep /app/reports and /app/.pytest_cache
# writable for tests; /hf_cache for HF model cache.
RUN mkdir -p /app/reports /app/.pytest_cache /hf_cache /app/.files /app/.pdf_cache && \
    chown -R 1000:1000 /app/reports /app/.pytest_cache /hf_cache /app/.files /app/.pdf_cache

# ─── Stage 3: Runtime ─────────────────────────────────────────────────────────
FROM base AS runner

# Use venv path for all subsequent commands
ENV PATH="/app/.venv/bin:$PATH" \
    CHAINLIT_FILES_DIR=/tmp/chainlit_files

# Copy everything from functional_builder (includes venv, source code, index, config)
# Ensure the entire /app directory is owned by the app user to prevent permission errors on runtime workspaces like .chainlit
COPY --chown=app:app --from=functional_builder /app /app
COPY --from=model_fetcher /model /model

# Writable dirs: /tmp is world-writable already (sticky bit), Chainlit will
# create /tmp/chainlit_files at startup. Ensure the Hugging Face cache directory exists.
RUN mkdir -p /hf_cache && \
    chown -R app:app /hf_cache

USER 1000

ARG VERSION="Dev mode"
ARG REPO_URL="https://github.com/MinionTech/vexilon"
ENV AGNAV_VERSION=$VERSION
ENV AGNAV_REPO_URL=$REPO_URL

CMD ["sh", "-c", "TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=0 chainlit run main.py --host 0.0.0.0 --port 7860 --headless"]
