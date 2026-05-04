#!/bin/bash
set -euo pipefail

echo "🚀 Starting Functional Test Suite..."

# 1. Run Integration Tests
# These verify the RAG pipeline logic (indexing + retrieval) 
# with citation verification enabled.
echo "📋 [1/2] Running Integration Tests..."
export VERIFY_ENABLED=true
export OLLAMA_HOST=ollama
uv run --no-sync python -m pytest tests/integration/ -v

# 2. Run E2E Smoke Test
# This verifies the actual production build artifact.
echo "🔥 [2/2] Running E2E Smoke Test..."
uv run --no-sync python scripts/smoke_e2e.py

echo "✅ All Functional Tests Passed!"
