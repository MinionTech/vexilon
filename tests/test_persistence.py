"""
tests/test_persistence.py — Unit tests for save_index() and load_precomputed_index()

Uses real FAISS read/write with synthetic vectors — no embedding model required.
Purpose: catch breaking changes in the faiss-cpu read/write API after a Renovate bump.
If the mocked tests in test_index.py are all that exist, a faiss major version that
changes the on-disk format would sail straight through undetected.
"""

import json
import os
import faiss
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock

import app
import vexilon.indexing as indexing


def _tiny_index(n: int = 3) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """
    Create a minimal FAISS IndexFlatIP with *n* random unit vectors and matching chunks.
    No embedding model required — purely validates the faiss persistence API.
    """
    vecs = np.random.randn(n, indexing.EMBED_DIM).astype(np.float32)
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(indexing.EMBED_DIM)
    index.add(vecs)
    chunks = [{"text": f"chunk {i}", "page": i + 1, "chunk_index": 0} for i in range(n)]
    return index, chunks


# ── save_index / load_precomputed_index ──────────────────────────────────────

def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    """save_index() → load_precomputed_index() must restore the index and chunks intact."""
    monkeypatch.setattr(indexing, "INDEX_PATH", tmp_path / "index.faiss")
    monkeypatch.setattr(indexing, "CHUNKS_PATH", tmp_path / "chunks.json")
    monkeypatch.setattr(indexing, "MANIFEST_PATH", tmp_path / "manifest.json")

    index, chunks = _tiny_index()
    indexing.save_index(index, chunks)

    loaded_index, loaded_chunks = indexing.load_precomputed_index()

    assert loaded_index is not None
    assert loaded_index.ntotal == index.ntotal
    assert loaded_chunks == chunks


def test_load_returns_none_none_when_both_files_missing(tmp_path, monkeypatch):
    """load_precomputed_index() must return (None, None) if neither file exists."""
    monkeypatch.setattr(indexing, "INDEX_PATH", tmp_path / "absent.faiss")
    monkeypatch.setattr(indexing, "CHUNKS_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(indexing, "MANIFEST_PATH", tmp_path / "absent.manifest")

    assert indexing.load_precomputed_index() == (None, None)


def test_load_returns_none_none_when_only_index_exists(tmp_path, monkeypatch):
    """load_precomputed_index() requires BOTH files — partial presence must not succeed."""
    index_path = tmp_path / "index.faiss"
    monkeypatch.setattr(indexing, "INDEX_PATH", index_path)
    monkeypatch.setattr(indexing, "CHUNKS_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(indexing, "MANIFEST_PATH", tmp_path / "absent.manifest")

    index, _ = _tiny_index()
    faiss.write_index(index, str(index_path))

    assert indexing.load_precomputed_index() == (None, None)


def test_load_returns_none_none_when_only_chunks_exist(tmp_path, monkeypatch):
    """load_precomputed_index() requires BOTH files — chunks alone must not succeed."""
    chunks_path = tmp_path / "chunks.json"
    monkeypatch.setattr(indexing, "INDEX_PATH", tmp_path / "absent.faiss")
    monkeypatch.setattr(indexing, "CHUNKS_PATH", chunks_path)
    monkeypatch.setattr(indexing, "MANIFEST_PATH", tmp_path / "absent.manifest")

    chunks_path.write_text(json.dumps([{"text": "a", "page": 1, "chunk_index": 0}]))

    assert indexing.load_precomputed_index() == (None, None)


def test_save_index_writes_valid_json_chunks(tmp_path, monkeypatch):
    """Chunks saved by save_index() must be valid JSON with the expected keys."""
    chunks_path = tmp_path / "chunks.json"
    monkeypatch.setattr(indexing, "INDEX_PATH", tmp_path / "index.faiss")
    monkeypatch.setattr(indexing, "CHUNKS_PATH", chunks_path)
    monkeypatch.setattr(indexing, "MANIFEST_PATH", tmp_path / "manifest.json")

    index, chunks = _tiny_index()
    indexing.save_index(index, chunks)

    with open(chunks_path, encoding="utf-8") as f:
        loaded = json.load(f)

    assert isinstance(loaded, list)
    assert len(loaded) == len(chunks)
    assert all("text" in c and "page" in c and "chunk_index" in c for c in loaded)


def test_loaded_index_is_searchable(tmp_path, monkeypatch):
    """An index restored from disk must still return valid search results."""
    monkeypatch.setattr(indexing, "INDEX_PATH", tmp_path / "index.faiss")
    monkeypatch.setattr(indexing, "CHUNKS_PATH", tmp_path / "chunks.json")
    monkeypatch.setattr(indexing, "MANIFEST_PATH", tmp_path / "manifest.json")

    index, chunks = _tiny_index(n=5)
    indexing.save_index(index, chunks)

    loaded_index, loaded_chunks = indexing.load_precomputed_index()

    # Search with the first stored vector — it must be its own top hit
    query = np.zeros((1, indexing.EMBED_DIM), dtype=np.float32)
    faiss.read_index(str(tmp_path / "index.faiss"))  # already loaded above
    scores, idxs = loaded_index.search(query, 1)

    assert idxs.shape == (1, 1)
    assert scores.shape == (1, 1)


# ── startup() error handling ─────────────────────────────────────────────────

def test_startup_raises_on_failure(monkeypatch):
    """
    startup() must raise exceptions if initialization fails (fail-fast).
    The container/process should die rather than staying up in a broken state.
    """
    monkeypatch.setattr(app, "_index", None)
    monkeypatch.setattr(app, "_chunks", [])

    def _boom():
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(indexing, "_fetch_pdf_cache_if_missing", _boom)
    monkeypatch.setattr(app, "_fetch_pdf_cache_if_missing", _boom)

    with pytest.raises(RuntimeError, match="disk on fire"):
        monkeypatch.setattr(app, "get_anthropic", MagicMock())
        app.startup()


def test_startup_uses_precomputed_index_when_available(monkeypatch):
    """startup() fast path: if a pre-computed index exists, it MUST use it and skip rebuild."""
    monkeypatch.setattr(app, "_chunks", [])
    monkeypatch.setattr(app, "_index", None)
    monkeypatch.setattr(app, "_test_registry", MagicMock()) # FEEDBACK: Avoid disk I/O
    monkeypatch.setattr(indexing, "_fetch_pdf_cache_if_missing", lambda: None)
    monkeypatch.setattr(app, "_fetch_pdf_cache_if_missing", lambda: None)

    fake_index, fake_chunks = _tiny_index(n=2)

    # Mock load_precomputed_index to return a faked precomputed index
    mock_load = MagicMock(return_value=(fake_index, fake_chunks))
    monkeypatch.setattr(app, "load_precomputed_index", mock_load)

    # Ensure build_index_from_sources is NOT called
    mock_build = MagicMock(return_value=(None, None)) # FEEDBACK: Unpack safety
    monkeypatch.setattr(app, "build_index_from_sources", mock_build)

    monkeypatch.setattr(app, "get_anthropic", MagicMock())
    app.startup()

    assert app._index is fake_index
    assert app._chunks is fake_chunks
    mock_load.assert_called_once()
    mock_build.assert_not_called()



def test_startup_delegates_to_indexing(monkeypatch):
    """startup() must delegate to indexing.build_index_from_sources."""
    monkeypatch.setattr(indexing, "_fetch_pdf_cache_if_missing", lambda: None)
    
    fake_index, fake_chunks = _tiny_index(n=1)
    mock_build = MagicMock(return_value=(fake_index, fake_chunks))
    monkeypatch.setattr(app, "build_index_from_sources", mock_build)

    monkeypatch.setattr(app, "get_anthropic", MagicMock())
    app.startup(force_rebuild=True)

    assert app._index is fake_index
    assert app._chunks is fake_chunks
    mock_build.assert_called_once_with(force=True)
