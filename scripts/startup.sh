#!/bin/bash
set -e

echo "[startup] Running Smart Index Refresh..."
# This will be nearly instant if the manifest matches.
# If it's a first run or files changed, it ensures the cache is current.
python scripts/build_index.py

echo "[startup] Starting Vexilon App..."
exec python app.py
