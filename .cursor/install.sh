#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for Agreement Navigator (vexilon).
#
# Runs once to build the environment baseline: toolchain, Python 3.14 venv,
# the pre-computed FAISS index, and the local Ollama dev model. Safe to re-run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$REPO_ROOT/app"
OLLAMA_MODEL_ID="${OLLAMA_MODEL_ID:-tinyllama}"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_HOST

echo "[install] Ensuring system packages (zstd is required by the Ollama installer)..."
if ! command -v zstd >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq zstd
fi

echo "[install] Ensuring uv is installed..."
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "[install] Ensuring Ollama is installed..."
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi
ollama --version

echo "[install] Syncing Python 3.14 virtualenv from the frozen lockfile..."
cd "$APP_DIR"
uv sync --frozen --python 3.14

echo "[install] Building the pre-computed FAISS index (downloads the embedding model on first run, Smart-Refresh skips afterwards)..."
uv run --no-sync python scripts/build_index.py

echo "[install] Pre-pulling the Ollama dev model '${OLLAMA_MODEL_ID}' so it is baked into the snapshot..."
OLLAMA_PID=""
if ! curl -sf -m 2 "http://${OLLAMA_HOST}/" >/dev/null 2>&1; then
  ollama serve >/tmp/ollama-install.log 2>&1 &
  OLLAMA_PID=$!
  for _ in $(seq 1 30); do
    if curl -sf -m 2 "http://${OLLAMA_HOST}/" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
fi
ollama pull "${OLLAMA_MODEL_ID}"
if [ -n "${OLLAMA_PID}" ]; then
  kill "${OLLAMA_PID}" 2>/dev/null || true
  wait "${OLLAMA_PID}" 2>/dev/null || true
fi

echo "[install] Done."
