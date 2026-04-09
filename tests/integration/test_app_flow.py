"""
tests/integration/test_app_flow.py — Full-flow integration test

Verifies that the app can start up, load the PDF, index it, and run a RAG query
using the real embedding model but a mocked Anthropic API.
"""

import pytest
import app
from pathlib import Path
import vexilon.indexing as indexing

@pytest.mark.asyncio
async def test_full_rag_flow_integration(monkeypatch, mock_anthropic, tmp_path):
    """
    Tests the system from Markdown loading to streaming response.
    Uses the real MD agreement and real embedding model.
    """
    # 1. Setup: Use a smaller document for isolation to save memory/time in CI
    source_md = Path("data/labour_law/04_jurisprudence/Nexus Test and Off-Duty Conduct.md")
    if not source_md.exists():
        pytest.skip(f"Agreement Markdown missing at {source_md}; cannot run full integration test.")

    # Create a minimal test knowledge base in a temporary directory
    test_data_dir = tmp_path / "data/labour_law"
    test_data_dir.mkdir(parents=True)
    import shutil
    shutil.copy(source_md, test_data_dir / source_md.name)

    # Redirect app paths to the temp dir
    cache_dir = tmp_path / "pdf_cache"
    cache_dir.mkdir()
    monkeypatch.setattr(indexing, "LABOUR_LAW_DIR", test_data_dir)
    monkeypatch.setattr(app, "TESTS_DIR", test_data_dir / "tests")
    monkeypatch.setattr(indexing, "PDF_CACHE_DIR", cache_dir)
    monkeypatch.setattr(indexing, "INDEX_PATH", cache_dir / "index.faiss")
    monkeypatch.setattr(indexing, "CHUNKS_PATH", cache_dir / "chunks.json")
    monkeypatch.setattr(indexing, "MANIFEST_PATH", cache_dir / "manifest.json")
    monkeypatch.setattr(indexing, "INTEGRITY_PATH", cache_dir / "integrity.json")

    # Mock the anthropic client globally for the app
    monkeypatch.setattr(app, "get_anthropic", lambda: mock_anthropic)
    
    # 2. Startup: This builds the index in memory (slow but thorough)
    app.startup(force_rebuild=True)
    
    assert app._index is not None
    assert len(app._chunks) > 0
    
    # 3. Query: Run a real RAG query
    message = "Tell me about the nexus test?"
    history = []
    
    tokens = []
    async for text_chunk, context_chunk in app.rag_stream(message, history):
        if text_chunk:
            tokens.append(text_chunk)
    
    # 4. Assertions
    full_response = "".join(tokens)
    assert "Mocked response content" in full_response
    assert app._index.ntotal > 0
    
    # Check that search actually found something
    query = "millhaven factors"
    results = indexing.search_index(app._index, app._chunks, query, top_k=1)
    assert len(results) == 1
    assert "millhaven" in results[0]["text"].lower()
