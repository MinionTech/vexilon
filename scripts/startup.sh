#!/bin/bash
set -e

echo "[startup] Running Smart Index Refresh..."
# This will be nearly instant if the manifest matches.
# If it's a first run or files changed, it ensures the cache is current.
python -c "from app import build_index_from_pdfs; build_index_from_pdfs()"

echo "[startup] Starting Vexilon App..."
exec python app.py &

# Wait for the Gradio server to be ready
for i in {1..30}; do
    if curl -s http://localhost:7860 > /dev/null 2>&1; then
        echo ""
        echo "=================================================="
        echo "  VEXILON IS READY! 🚀"
        echo "  Visit: http://localhost:7860"
        echo "=================================================="
        echo ""
        break
    fi
    sleep 1
done
