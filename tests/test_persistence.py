"""
tests/test_persistence.py — Unit tests for save_index() and load_precomputed_index()

Uses real FAISS read/write with synthetic vectors — no embedding model required.
Purpose: catch breaking changes in the faiss-cpu read/write API after a Renovate bump.
If the mocked tests in test_index.py are all that exist, a faiss major version that
changes the on-disk format would sail straight through undetected.
"""

import json

import faiss
import numpy as np
import pytest

import app


def _tiny_index(n: int = 3) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """
    Create a minimal FAISS IndexFlatIP with *n* random unit vectors and matching chunks.
    No embedding model required — purely validates the faiss persistence API.
    """
    vecs = np.random.randn(n, app.EMBED_DIM).astype(np.float32)
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(app.EMBED_DIM)
    index.add(vecs)
    chunks = [{"text": f"chunk {i}", "page": i + 1, "chunk_index": 0} for i in range(n)]
    return index, chunks


# ── save_index / load_precomputed_index ──────────────────────────────────────

def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    """save_index() → load_precomputed_index() must restore the index and chunks intact."""
    monkeypatch.setattr(app, "INDEX_PATH", tmp_path / "index.faiss")
    monkeypatch.setattr(app, "CHUNKS_PATH", tmp_path / "chunks.json")

    index, chunks = _tiny_index()
    app.save_index(index, chunks)

    loaded_index, loaded_chunks = app.load_precomputed_index()

    assert loaded_index is not None
    assert loaded_index.ntotal == index.ntotal
    assert loaded_chunks == chunks


def test_load_returns_none_none_when_both_files_missing(tmp_path, monkeypatch):
    """load_precomputed_index() must return (None, None) if neither file exists."""
    monkeypatch.setattr(app, "INDEX_PATH", tmp_path / "absent.faiss")
    monkeypatch.setattr(app, "CHUNKS_PATH", tmp_path / "absent.json")

    assert app.load_precomputed_index() == (None, None)


def test_load_returns_none_none_when_only_index_exists(tmp_path, monkeypatch):
    """load_precomputed_index() requires BOTH files — partial presence must not succeed."""
    index_path = tmp_path / "index.faiss"
    monkeypatch.setattr(app, "INDEX_PATH", index_path)
    monkeypatch.setattr(app, "CHUNKS_PATH", tmp_path / "absent.json")

    index, _ = _tiny_index()
    faiss.write_index(index, str(index_path))

    assert app.load_precomputed_index() == (None, None)


def test_load_returns_none_none_when_only_chunks_exist(tmp_path, monkeypatch):
    """load_precomputed_index() requires BOTH files — chunks alone must not succeed."""
    chunks_path = tmp_path / "chunks.json"
    monkeypatch.setattr(app, "INDEX_PATH", tmp_path / "absent.faiss")
    monkeypatch.setattr(app, "CHUNKS_PATH", chunks_path)

    chunks_path.write_text(json.dumps([{"text": "a", "page": 1, "chunk_index": 0}]))

    assert app.load_precomputed_index() == (None, None)


def test_save_index_writes_valid_json_chunks(tmp_path, monkeypatch):
    """Chunks saved by save_index() must be valid JSON with the expected keys."""
    chunks_path = tmp_path / "chunks.json"
    monkeypatch.setattr(app, "INDEX_PATH", tmp_path / "index.faiss")
    monkeypatch.setattr(app, "CHUNKS_PATH", chunks_path)

    index, chunks = _tiny_index()
    app.save_index(index, chunks)

    with open(chunks_path, encoding="utf-8") as f:
        loaded = json.load(f)

    assert isinstance(loaded, list)
    assert len(loaded) == len(chunks)
    assert all("text" in c and "page" in c and "chunk_index" in c for c in loaded)


def test_loaded_index_is_searchable(tmp_path, monkeypatch):
    """An index restored from disk must still return valid search results."""
    monkeypatch.setattr(app, "INDEX_PATH", tmp_path / "index.faiss")
    monkeypatch.setattr(app, "CHUNKS_PATH", tmp_path / "chunks.json")

    index, chunks = _tiny_index(n=5)
    app.save_index(index, chunks)

    loaded_index, loaded_chunks = app.load_precomputed_index()

    # Search with the first stored vector — it must be its own top hit
    query = np.zeros((1, app.EMBED_DIM), dtype=np.float32)
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

    monkeypatch.setattr(app, "_fetch_pdf_cache_if_missing", _boom)

    with pytest.raises(RuntimeError, match="disk on fire"):
        app.startup()


def test_startup_uses_precomputed_index_when_available(monkeypatch, tmp_path):
    """startup() fast path: if a pre-computed index exists, it MUST use it and skip rebuild."""
    monkeypatch.setattr(app, "_chunks", [])
    monkeypatch.setattr(app, "_fetch_pdf_cache_if_missing", lambda: None)

    fake_index, fake_chunks = _tiny_index(n=2)

    monkeypatch.setattr(app, "load_precomputed_index", lambda: (fake_index, fake_chunks))
    # get_embed_model is called to warm the model; mock it so no download happens
    monkeypatch.setattr(app, "get_embed_model", lambda: None)

    app.startup()

    assert app._index is fake_index
    assert app._chunks is fake_chunks


def test_startup_slow_path_builds_and_saves(monkeypatch, tmp_path):
    """
    startup(force_rebuild=True) must: load PDF → build index → save index,
    and wire up _index and _chunks when there is no pre-computed cache.
    """
    monkeypatch.setattr(app, "_index", None)
    monkeypatch.setattr(app, "_chunks", [])
    monkeypatch.setattr(app, "_fetch_pdf_cache_if_missing", lambda: None)

    fake_chunks = [{"text": "article 1", "page": 1, "chunk_index": 0}]
    fake_index, _ = _tiny_index(n=1)

    # Track whether save_index was called
    save_calls = []

    monkeypatch.setattr(app, "load_pdf_chunks", lambda _path: fake_chunks)
    monkeypatch.setattr(app, "build_index", lambda chunks: fake_index)
    monkeypatch.setattr(app, "save_index", lambda idx, cks: save_calls.append((idx, cks)))


    app.startup(force_rebuild=True)

    assert app._index is fake_index
    assert app._chunks is fake_chunks
    assert len(save_calls) == 1, "save_index must be called exactly once during force_rebuild"


def test_startup_slow_path_skips_precomputed_even_if_present(monkeypatch, tmp_path):
    """
    When force_rebuild=True, startup() must NOT use the pre-computed index,
    even if load_precomputed_index() would succeed.
    """
    monkeypatch.setattr(app, "_index", None)
    monkeypatch.setattr(app, "_chunks", [])
    monkeypatch.setattr(app, "_fetch_pdf_cache_if_missing", lambda: None)

    stale_index, stale_chunks = _tiny_index(n=2)
    fresh_index, fresh_chunks_list = _tiny_index(n=1)
    fresh_chunks = [{"text": "fresh", "page": 1, "chunk_index": 0}]

    # load_precomputed_index would return stale data — force_rebuild must bypass it
    monkeypatch.setattr(app, "load_precomputed_index", lambda: (stale_index, stale_chunks))
    monkeypatch.setattr(app, "load_pdf_chunks", lambda _: fresh_chunks)
    monkeypatch.setattr(app, "build_index", lambda _: fresh_index)
    monkeypatch.setattr(app, "save_index", lambda idx, cks: None)


    app.startup(force_rebuild=True)

    assert app._chunks is fresh_chunks, (
        "force_rebuild=True must use freshly-built chunks, not the pre-computed cache"
    )
