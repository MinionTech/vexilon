"""
tests/test_index.py — Unit tests for build_index() and search_index()

Uses synthetic vectors and a mocked OpenAI client — zero API calls.
"""

import faiss
import numpy as np
import pytest

import app
import vexilon.indexing as indexing


def _make_chunks(n: int) -> list[dict]:
    """Create n fake chunks with distinct text."""
    return [{"text": f"chunk number {i}", "page": i + 1, "chunk_index": 0} for i in range(n)]


def _make_embed_fn(vectors: np.ndarray):
    """Return a replacement for embed_texts() that yields rows from *vectors*."""
    call_count = {"n": 0}

    def _embed(texts: list[str]) -> np.ndarray:
        start = call_count["n"]
        call_count["n"] += len(texts)
        return vectors[start : start + len(texts)].astype(np.float32)

    return _embed


# ── build_index ───────────────────────────────────────────────────────────────

def test_build_index_returns_faiss_index(monkeypatch):
    """build_index should return a FAISS IndexFlatIP with ntotal == len(chunks)."""
    n = 5
    chunks = _make_chunks(n)
    # Use random unit vectors (shape matches EMBED_DIM)
    vecs = np.random.randn(n, indexing.EMBED_DIM).astype(np.float32)
    faiss.normalize_L2(vecs)

    monkeypatch.setattr(indexing, "embed_texts", _make_embed_fn(vecs))
    index = indexing.build_index(chunks)

    assert isinstance(index, faiss.IndexFlatIP)
    assert index.ntotal == n


def test_build_index_normalises_vectors(monkeypatch):
    """After build_index, searching with an identical vector should produce score ≈ 1.0."""
    n = 3
    chunks = _make_chunks(n)
    vecs = np.random.randn(n, indexing.EMBED_DIM).astype(np.float32)
    faiss.normalize_L2(vecs)

    monkeypatch.setattr(indexing, "embed_texts", _make_embed_fn(vecs))
    index = indexing.build_index(chunks)

    query = vecs[0:1].copy()
    scores, _ = index.search(query, 1)
    assert scores[0][0] == pytest.approx(1.0, abs=1e-5)


# ── search_index ──────────────────────────────────────────────────────────────

def test_search_index_returns_top_k(monkeypatch):
    """search_index should return exactly top_k results (when index has >= top_k vectors)."""
    n = 10
    top_k = 3
    chunks = _make_chunks(n)
    vecs = np.random.randn(n, indexing.EMBED_DIM).astype(np.float32)
    faiss.normalize_L2(vecs)

    monkeypatch.setattr(indexing, "embed_texts", _make_embed_fn(vecs))
    index = indexing.build_index(chunks)

    # For search, embed_texts is called with the single query string
    query_vec = vecs[0:1].copy()

    def _embed_search(texts):
        return query_vec.copy()

    monkeypatch.setattr(indexing, "embed_texts", _embed_search)
    results = indexing.search_index(index, chunks, "any query", top_k=top_k)

    assert len(results) == top_k


def test_search_index_finds_most_similar(monkeypatch):
    """The top result should be the chunk whose vector is closest to the query."""
    n = 4
    chunks = _make_chunks(n)
    vecs = np.random.randn(n, indexing.EMBED_DIM).astype(np.float32)
    faiss.normalize_L2(vecs)

    monkeypatch.setattr(indexing, "embed_texts", _make_embed_fn(vecs))
    index = indexing.build_index(chunks)

    # Query is identical to chunk 2 — it must be the top hit
    target_vec = vecs[2:3].copy()
    monkeypatch.setattr(indexing, "embed_texts", lambda _texts: target_vec.copy())

    results = indexing.search_index(index, chunks, "irrelevant", top_k=1)
    assert results[0] == chunks[2]


# ── pickle → JSON migration ─────────────────────────────────────────────────

def test_save_index_uses_json(tmp_path, monkeypatch):
    """save_index should write chunks as JSON, not pickle."""
    monkeypatch.setattr(indexing, "PDF_CACHE_DIR", tmp_path)
    monkeypatch.setattr(indexing, "INDEX_PATH", tmp_path / "index.faiss")
    monkeypatch.setattr(indexing, "CHUNKS_PATH", tmp_path / "chunks.json")

    chunks = _make_chunks(3)
    vecs = np.random.randn(3, indexing.EMBED_DIM).astype(np.float32)
    faiss.normalize_L2(vecs)
    monkeypatch.setattr(indexing, "embed_texts", _make_embed_fn(vecs))
    index = indexing.build_index(chunks)

    indexing.save_index(index, chunks)

    # Verify chunks.json is valid JSON (not pickle)
    import json
    chunks_file = tmp_path / "chunks.json"
    assert chunks_file.exists()
    with open(chunks_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert len(loaded) == 3
    assert loaded[0]["text"] == "chunk number 0"


def test_load_precomputed_index_from_json(tmp_path, monkeypatch):
    """load_precomputed_index should load chunks from JSON."""
    import json
    monkeypatch.setattr(indexing, "PDF_CACHE_DIR", tmp_path)
    monkeypatch.setattr(indexing, "INDEX_PATH", tmp_path / "index.faiss")
    monkeypatch.setattr(indexing, "CHUNKS_PATH", tmp_path / "chunks.json")

    chunks = _make_chunks(3)
    vecs = np.random.randn(3, indexing.EMBED_DIM).astype(np.float32)
    faiss.normalize_L2(vecs)
    monkeypatch.setattr(indexing, "embed_texts", _make_embed_fn(vecs))
    index = indexing.build_index(chunks)

    # Save as JSON
    with open(tmp_path / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    faiss.write_index(index, str(tmp_path / "index.faiss"))

    # Load and verify
    loaded_index, loaded_chunks = indexing.load_precomputed_index()
    assert loaded_index.ntotal == 3
    assert len(loaded_chunks) == 3
    assert loaded_chunks[0]["text"] == "chunk number 0"


def test_load_precomputed_index_migrates_legacy_pkl(tmp_path, monkeypatch):
    """load_precomputed_index should migrate legacy .pkl to JSON and delete the .pkl."""
    import pickle
    monkeypatch.setattr(indexing, "PDF_CACHE_DIR", tmp_path)
    monkeypatch.setattr(indexing, "INDEX_PATH", tmp_path / "index.faiss")
    monkeypatch.setattr(indexing, "CHUNKS_PATH", tmp_path / "chunks.json")

    chunks = _make_chunks(3)
    vecs = np.random.randn(3, indexing.EMBED_DIM).astype(np.float32)
    faiss.normalize_L2(vecs)
    monkeypatch.setattr(indexing, "embed_texts", _make_embed_fn(vecs))
    index = indexing.build_index(chunks)

    # Save as legacy pickle
    pkl_path = tmp_path / "chunks.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(chunks, f)
    faiss.write_index(index, str(tmp_path / "index.faiss"))

    # JSON doesn't exist yet
    assert not (tmp_path / "chunks.json").exists()

    # Load should migrate
    loaded_index, loaded_chunks = indexing.load_precomputed_index()
    assert loaded_index.ntotal == 3
    assert len(loaded_chunks) == 3

    # JSON should now exist, pickle should be deleted
    assert (tmp_path / "chunks.json").exists()
    assert not pkl_path.exists()


def test_fetch_downloads_json_not_pkl(tmp_path, monkeypatch):
    """_fetch_pdf_cache_if_missing should request chunks.json, not chunks.pkl."""
    import urllib.request
    monkeypatch.setattr(indexing, "PDF_CACHE_DIR", tmp_path)
    monkeypatch.setattr(indexing, "INDEX_PATH", tmp_path / "index.faiss")
    monkeypatch.setattr(indexing, "CHUNKS_PATH", tmp_path / "chunks.json")

    requested_urls = []
    def mock_urlretrieve(url, dest):
        requested_urls.append(url)
        # Create a dummy file so the function doesn't loop
        Path(dest).write_text("[]")

    monkeypatch.setattr(urllib.request, "urlretrieve", mock_urlretrieve)
    monkeypatch.setenv("VEXILON_RAW_URL_BASE", "https://example.com")

    indexing._fetch_pdf_cache_if_missing()

    # Should have requested chunks.json, not chunks.pkl
    assert any("chunks.json" in url for url in requested_urls)
    assert not any("chunks.pkl" in url for url in requested_urls)
