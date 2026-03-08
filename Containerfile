# ─── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.12-slim AS base

# Install system deps: libgomp1 for pypdf, curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps in a separate layer so code changes don't bust the cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── App layer ────────────────────────────────────────────────────────────────
COPY app.py manifest.json ./
COPY pdf_cache/ ./pdf_cache/

# ─── Non-root user ────────────────────────────────────────────────────────────
RUN useradd --uid 1001 --no-create-home --shell /sbin/nologin vexilon \
    && chown -R 1001:1001 /app

USER 1001

# Gradio listens on 7860 by default
EXPOSE 7860

CMD ["python", "app.py"]
