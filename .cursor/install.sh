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

UV_VERSION="0.12.13"
OLLAMA_VERSION="v0.34.0"

verify_and_install_archive() {
  local url="$1"
  local expected_sha256="$2"
  local archive_path="$3"
  local install_fn="$4"

  curl -fsSL "${url}" -o "${archive_path}"
  echo "${expected_sha256}  ${archive_path}" | sha256sum -c -
  "${install_fn}" "${archive_path}"
}

install_uv_from_archive() {
  local archive_path="$1"
  local install_dir="${HOME}/.local/bin"
  local uv_target

  case "$(uname -m)" in
    x86_64) uv_target="uv-x86_64-unknown-linux-gnu" ;;
    aarch64|arm64) uv_target="uv-aarch64-unknown-linux-gnu" ;;
    *)
      echo "[install] Unsupported architecture for uv: $(uname -m)" >&2
      exit 1
      ;;
  esac

  mkdir -p "${install_dir}"
  tar -xzf "${archive_path}" -C /tmp
  install -m755 "/tmp/${uv_target}/uv" "${install_dir}/uv"
  install -m755 "/tmp/${uv_target}/uvx" "${install_dir}/uvx"
}

install_ollama_from_archive() {
  local archive_path="$1"
  local install_dir="/usr/local"

  zstd -d "${archive_path}" -o /tmp/ollama.tar
  sudo tar -xf /tmp/ollama.tar -C "${install_dir}"
}

cleanup_install_ollama() {
  if [ -n "${OLLAMA_PID:-}" ]; then
    kill "${OLLAMA_PID}" 2>/dev/null || true
    wait "${OLLAMA_PID}" 2>/dev/null || true
    OLLAMA_PID=""
  fi
}

echo "[install] Ensuring system packages (zstd is required by the Ollama installer)..."
if ! command -v zstd >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq zstd
fi

export PATH="$HOME/.local/bin:$PATH"

echo "[install] Ensuring uv ${UV_VERSION} is installed..."
if ! command -v uv >/dev/null 2>&1; then
  case "$(uname -m)" in
    x86_64)
      UV_TARGET="uv-x86_64-unknown-linux-gnu"
      UV_SHA256="745765a3b6e360ad76743599ae5c42e9278c7edf8bbff9fc76d05bf2623a04dd"
      ;;
    aarch64|arm64)
      UV_TARGET="uv-aarch64-unknown-linux-gnu"
      UV_SHA256="2eaa5d94f5db7b3a1a092156b9420459e42ab0217d917fe74a876309cef9b5e9"
      ;;
    *)
      echo "[install] Unsupported architecture for uv: $(uname -m)" >&2
      exit 1
      ;;
  esac
  verify_and_install_archive \
    "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/${UV_TARGET}.tar.gz" \
    "${UV_SHA256}" \
    "/tmp/uv-${UV_VERSION}-${UV_TARGET}.tar.gz" \
    install_uv_from_archive
fi
uv --version

echo "[install] Ensuring Ollama ${OLLAMA_VERSION} is installed..."
if ! command -v ollama >/dev/null 2>&1; then
  case "$(uname -m)" in
    x86_64)
      OLLAMA_ARCHIVE="ollama-linux-amd64.tar.zst"
      OLLAMA_SHA256="cf95886728959aa09910bb34de5cca1cc5a8f68003b5597197d3f2c2d57c0804"
      ;;
    aarch64|arm64)
      OLLAMA_ARCHIVE="ollama-linux-arm64.tar.zst"
      OLLAMA_SHA256="6a9e5b3650c2024d8a78da86b23876f6eea238657a3262d7e5ec0f3688c5d28e"
      ;;
    *)
      echo "[install] Unsupported architecture for Ollama: $(uname -m)" >&2
      exit 1
      ;;
  esac
  verify_and_install_archive \
    "https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/${OLLAMA_ARCHIVE}" \
    "${OLLAMA_SHA256}" \
    "/tmp/${OLLAMA_ARCHIVE}" \
    install_ollama_from_archive
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
  trap cleanup_install_ollama EXIT
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
trap - EXIT
cleanup_install_ollama

echo "[install] Done."
