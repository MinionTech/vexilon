"""
tests/integration/test_app_flow.py — Full-flow integration test

Verifies that the app can start up, load the PDF, index it, and run a RAG query
using the real embedding model but a mocked Anthropic API.
"""

import pytest
import app
from pathlib import Path

@pytest.mark.asyncio
async def test_full_rag_flow_integration(monkeypatch, mock_anthropic, tmp_path):
    """
    Tests the system from PDF loading to streaming response.
    Uses the real PDF and real embedding model.
    """
    # 1. Setup: Ensure we use the real PDF and a temporary index path to avoid clobbering prod
    test_pdf = Path("data/labour_law/bcgeu_19th_main_agreement.pdf")
    if not test_pdf.exists():
        pytest.skip(f"Agreement PDF missing at {test_pdf}; cannot run full integration test.")

    # Redirect pdf_cache to a temp dir so save_index() doesn't fail on missing directory
    cache_dir = tmp_path / "pdf_cache"
    cache_dir.mkdir()
    monkeypatch.setattr(app, "PDF_CACHE_DIR", cache_dir)
    monkeypatch.setattr(app, "INDEX_PATH", cache_dir / "index.faiss")
    monkeypatch.setattr(app, "CHUNKS_PATH", cache_dir / "chunks.json")

    # Mock the anthropic client globally for the app
    monkeypatch.setattr(app, "get_anthropic", lambda: mock_anthropic)
    
    # 2. Startup: This builds the index in memory (slow but thorough)
    # We use force_rebuild=True to ensure we test the parsing/indexing logic
    app.startup(force_rebuild=True)
    
    assert app._index is not None
    assert len(app._chunks) > 0
    
    # 3. Query: Run a real RAG query
    # This will:
    # - Run condense_query (mocked Claude)
    # - Run search_index (REAL FAISS + REAL Embeddings)
    # - Run rag_stream (mocked Claude)
    message = "What are the rules for overtime?"
    history = []
    
    tokens = []
    async for chunk in app.rag_stream(message, history):
        tokens.append(chunk)
    
    # 4. Assertions
    full_response = "".join(tokens)
    assert "Mocked response content" in full_response
    assert app._index.ntotal > 0
    
    # Check that search actually found something
    query = "overtime rate"
    results = app.search_index(app._index, app._chunks, query, top_k=1)
    assert len(results) == 1
    assert "overtime" in results[0]["text"].lower()
