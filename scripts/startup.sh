#!/bin/bash
set -e

echo "[startup] Running Smart Index Refresh..."
# This will be nearly instant if the manifest matches.
# If it's a first run or files changed, it ensures the cache is current.
python -c "from app import build_index_from_sources; build_index_from_sources()"

echo "[startup] Starting Vexilon App..."
exec python app.py
