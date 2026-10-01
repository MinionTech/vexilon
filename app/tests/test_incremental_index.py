"""
tests/test_incremental_index.py — Per-document embedding cache.

build_index_from_sources must encode only new or modified documents and must
rebuild a FAISS index whose search results match a cold rebuild.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

import indexing

# app/conftest.py replaces indexing.get_embed_model before each unit test.
_REAL_GET_EMBED_MODEL = indexing.get_embed_model


def _isolate(monkeypatch, data: Path, cache: Path) -> None:
    data.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(indexing, "DATA_DIR", data)
    monkeypatch.setattr(indexing, "CACHE_DIR", cache)
    monkeypatch.setattr(indexing, "PDF_CACHE_DIR", cache)
    monkeypatch.setattr(indexing, "INDEX_PATH", cache / "index.faiss")
    monkeypatch.setattr(indexing, "CHUNKS_PATH", cache / "chunks.json")
    monkeypatch.setattr(indexing, "MANIFEST_PATH", cache / "manifest.json")
    monkeypatch.setattr(indexing, "INTEGRITY_PATH", cache / "integrity.json")


def _install_loaders(monkeypatch, data: Path) -> None:
    def load_md(path: Path) -> list[dict]:
        rel = str(path.relative_to(data))
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [
            {
                "text": line,
                "source": path.stem,
                "path": rel,
                "chunk_index": i,
                "header": "",
            }
            for i, line in enumerate(lines)
        ]

    monkeypatch.setattr(indexing, "load_md_chunks", load_md)
    monkeypatch.setattr(indexing, "load_pdf_chunks", lambda *_args, **_kwargs: [])


def _install_embed(monkeypatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def embed(texts: list[str]) -> np.ndarray:
        calls.append(list(texts))
        rows = []
        for text in texts:
            vec = np.zeros(indexing.EMBED_DIM, dtype=np.float32)
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vec[0] = digest[0] / 255.0
            vec[1] = 0.5
            vec[2] = digest[1] / 255.0
            rows.append(vec)
        return np.vstack(rows)

    monkeypatch.setattr(indexing, "embed_texts", embed)
    return calls


def _reconstruct(index) -> np.ndarray:
    return np.vstack([index.reconstruct(i) for i in range(index.ntotal)])


def _write_corpus(data: Path) -> None:
    (data / "a.md").write_text("alpha clause\n", encoding="utf-8")
    (data / "b.md").write_text("beta one\nbeta two\n", encoding="utf-8")


def _doc_cache(cache: Path, rel: str, source: Path) -> tuple[Path, Path]:
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    identity = indexing._cache_identity(rel, digest)
    return cache / "embeddings" / f"{identity}.npy", cache / "chunks" / f"{identity}.json"


def _drop_built_index(cache: Path) -> None:
    """Remove the FAISS index so the next build must consult the document cache."""
    (cache / "index.faiss").unlink()
    (cache / "chunks.json").unlink()


def test_changed_document_encodes_only_its_chunks(tmp_path, monkeypatch):
    """Editing one file encodes that file's chunks and leaves the other file cached."""
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    _isolate(monkeypatch, data, cache)
    _install_loaders(monkeypatch, data)
    calls = _install_embed(monkeypatch)
    _write_corpus(data)

    index_cold, chunks_cold = indexing.build_index_from_sources()
    assert index_cold is not None
    assert calls == [["alpha clause"], ["beta one", "beta two"]]
    assert [c["text"] for c in chunks_cold] == ["alpha clause", "beta one", "beta two"]

    calls.clear()
    _drop_built_index(cache)
    index_cached, chunks_cached = indexing.build_index_from_sources()
    assert calls == []
    assert chunks_cached == chunks_cold
    np.testing.assert_array_equal(_reconstruct(index_cached), _reconstruct(index_cold))

    old_b_npy, old_b_json = _doc_cache(cache, "b.md", data / "b.md")
    keep = cache / "embeddings" / "README"
    keep.write_text("keep", encoding="utf-8")
    (data / "b.md").write_text("beta changed\n", encoding="utf-8")
    calls.clear()
    index_inc, chunks_inc = indexing.build_index_from_sources()
    assert calls == [["beta changed"]]
    assert [c["text"] for c in chunks_inc] == ["alpha clause", "beta changed"]
    assert not old_b_npy.exists()
    assert not old_b_json.exists()
    assert keep.read_text(encoding="utf-8") == "keep"
    new_b_npy, _new_b_json = _doc_cache(cache, "b.md", data / "b.md")
    assert new_b_npy.is_file()

    for folder_name in ("embeddings", "chunks"):
        for child in (cache / folder_name).iterdir():
            child.unlink()
    (cache / "manifest.json").unlink()
    (cache / "index.faiss").unlink()
    (cache / "chunks.json").unlink()
    calls.clear()
    index_full, chunks_full = indexing.build_index_from_sources()
    assert calls == [["alpha clause"], ["beta changed"]]
    assert chunks_full == chunks_inc

    query = np.zeros((1, indexing.EMBED_DIM), dtype=np.float32)
    query[0, 0] = 1.0
    scores_inc, ids_inc = index_inc.search(query, index_inc.ntotal)
    scores_full, ids_full = index_full.search(query, index_full.ntotal)
    np.testing.assert_array_equal(ids_inc, ids_full)
    np.testing.assert_array_equal(scores_inc, scores_full)
    np.testing.assert_array_equal(_reconstruct(index_inc), _reconstruct(index_full))


def test_manifest_records_cache_references(tmp_path, monkeypatch):
    """manifest.json stores each file hash and the relative embedding and chunk cache paths."""
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    _isolate(monkeypatch, data, cache)
    _install_loaders(monkeypatch, data)
    _install_embed(monkeypatch)
    _write_corpus(data)

    indexing.build_index_from_sources()

    digest = hashlib.sha256((data / "b.md").read_bytes()).hexdigest()
    identity = indexing._cache_identity("b.md", digest)
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["files"]["b.md"]
    assert entry["content_hash"] == digest
    assert entry["embeddings"] == f"embeddings/{identity}.npy"
    assert entry["chunks"] == f"chunks/{identity}.json"
    assert identity != digest
    vectors = np.load(cache / entry["embeddings"], allow_pickle=False)
    payload = json.loads((cache / entry["chunks"]).read_text(encoding="utf-8"))
    assert payload["path"] == "b.md"
    assert payload["content_hash"] == digest
    assert payload["config"] == indexing._index_config()
    assert vectors.dtype == np.float32
    assert vectors.shape == (2, indexing.EMBED_DIM)
    assert [c["text"] for c in payload["chunks"]] == ["beta one", "beta two"]
    assert manifest["config"]["chunk_size"] == indexing.CHUNK_SIZE
    assert manifest["config"]["embed_model"] == indexing._effective_embed_model()
    assert manifest["config"]["max_embed_tokens"] == indexing._effective_max_embed_tokens()
    assert "agnav_max_embed_tokens" not in manifest["config"]
    assert manifest["config"]["embed_dim"] == indexing.EMBED_DIM
    assert manifest["config"]["pipeline_version"] == indexing.CACHE_PIPELINE_VERSION


def test_unchanged_sources_do_not_encode(tmp_path, monkeypatch):
    """A rebuild that drops only the FAISS index reloads the document cache."""
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    _isolate(monkeypatch, data, cache)
    _install_loaders(monkeypatch, data)
    calls = _install_embed(monkeypatch)
    _write_corpus(data)

    indexing.build_index_from_sources()
    calls.clear()
    _drop_built_index(cache)
    index, chunks = indexing.build_index_from_sources()
    assert calls == []
    assert index.ntotal == len(chunks) == 3


def test_config_change_reencodes_every_document(tmp_path, monkeypatch):
    """A chunk-config change ignores on-disk embeddings even when the file hash is unchanged."""
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    _isolate(monkeypatch, data, cache)
    _install_loaders(monkeypatch, data)
    calls = _install_embed(monkeypatch)
    _write_corpus(data)

    indexing.build_index_from_sources()
    calls.clear()
    monkeypatch.setattr(indexing, "CHUNK_SIZE", indexing.CHUNK_SIZE + 1)
    indexing.build_index_from_sources()
    assert calls == [["alpha clause"], ["beta one", "beta two"]]


def test_corrupt_cache_falls_back_to_encode(tmp_path, monkeypatch):
    """An empty embedding file is a cache miss; a valid sibling cache is reused."""
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    _isolate(monkeypatch, data, cache)
    _install_loaders(monkeypatch, data)
    calls = _install_embed(monkeypatch)
    _write_corpus(data)

    indexing.build_index_from_sources()
    b_npy, _b_json = _doc_cache(cache, "b.md", data / "b.md")
    b_npy.write_bytes(b"")
    _drop_built_index(cache)
    calls.clear()
    indexing.build_index_from_sources()
    assert calls == [["beta one", "beta two"]]
    reloaded = np.load(b_npy, allow_pickle=False)
    assert reloaded.shape == (2, indexing.EMBED_DIM)


def test_corrupt_chunk_json_and_wrong_shaped_npy_are_rewritten(tmp_path, monkeypatch):
    """A bad chunk file and a wrong-shaped vector file are re-encoded into a valid cache."""
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    _isolate(monkeypatch, data, cache)
    _install_loaders(monkeypatch, data)
    calls = _install_embed(monkeypatch)
    _write_corpus(data)

    indexing.build_index_from_sources()
    a_npy, a_json = _doc_cache(cache, "a.md", data / "a.md")
    b_npy, b_json = _doc_cache(cache, "b.md", data / "b.md")
    np.save(a_npy, np.zeros((1, 3), dtype=np.float32))
    b_json.write_text('["not-a-chunk"]', encoding="utf-8")
    _drop_built_index(cache)
    calls.clear()
    indexing.build_index_from_sources()
    assert calls == [["alpha clause"], ["beta one", "beta two"]]

    a_vectors = np.load(a_npy, allow_pickle=False)
    a_payload = json.loads(a_json.read_text(encoding="utf-8"))
    assert a_vectors.dtype == np.float32
    assert a_vectors.shape == (1, indexing.EMBED_DIM)
    assert a_payload["chunks"][0]["text"] == "alpha clause"
    assert a_payload["chunks"][0]["path"] == "a.md"

    b_vectors = np.load(b_npy, allow_pickle=False)
    b_payload = json.loads(b_json.read_text(encoding="utf-8"))
    assert b_vectors.shape == (2, indexing.EMBED_DIM)
    assert [c["text"] for c in b_payload["chunks"]] == ["beta one", "beta two"]
    assert b_payload["config"] == indexing._index_config()


def test_token_limit_change_does_not_reuse_vectors(tmp_path, monkeypatch):
    """The live AGNAV_MAX_EMBED_TOKENS value is the only token-limit cache key."""
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    _isolate(monkeypatch, data, cache)
    _install_loaders(monkeypatch, data)
    calls = _install_embed(monkeypatch)
    _write_corpus(data)

    indexing.build_index_from_sources()
    calls.clear()
    new_limit = indexing._effective_max_embed_tokens() + 8
    monkeypatch.setenv("AGNAV_MAX_EMBED_TOKENS", str(new_limit))
    indexing.build_index_from_sources()
    assert calls == [["alpha clause"], ["beta one", "beta two"]]
    assert indexing._index_config()["max_embed_tokens"] == new_limit


def test_token_limit_identity_matches_encoder(monkeypatch):
    """The fingerprint and get_embed_model() use the same call-time token limit."""
    monkeypatch.setenv("AGNAV_MAX_EMBED_TOKENS", "128")
    monkeypatch.setattr(indexing, "MAX_EMBED_TOKENS", 512)
    assert indexing._effective_max_embed_tokens() == 128
    assert indexing._index_config()["max_embed_tokens"] == 128
    assert "agnav_max_embed_tokens" not in indexing._index_config()

    tokenizer = type("Tok", (), {"is_fast": True, "model_max_length": 512})()
    model = type("Model", (), {"max_seq_length": 512, "tokenizer": tokenizer})()
    monkeypatch.setattr(indexing, "_embed_model", model)
    monkeypatch.setattr(indexing, "_loaded_model_name", indexing._effective_embed_model())
    monkeypatch.setattr(indexing, "get_embed_model", _REAL_GET_EMBED_MODEL)
    loaded = indexing.get_embed_model()
    assert loaded.max_seq_length == 128
    assert loaded.tokenizer.model_max_length == 128
    assert indexing._index_config()["max_embed_tokens"] == loaded.max_seq_length


def test_source_path_change_does_not_reuse_vectors(tmp_path, monkeypatch):
    """A rename, or a second path with the same bytes, does not reuse the old chunks."""
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    _isolate(monkeypatch, data, cache)
    _install_loaders(monkeypatch, data)
    calls = _install_embed(monkeypatch)
    (data / "a.md").write_text("same bytes\n", encoding="utf-8")
    (data / "b.md").write_text("same bytes\n", encoding="utf-8")

    _index, chunks = indexing.build_index_from_sources()
    assert calls == [["same bytes"], ["same bytes"]]
    assert [c["path"] for c in chunks] == ["a.md", "b.md"]
    assert [c["source"] for c in chunks] == ["a", "b"]

    calls.clear()
    (data / "a.md").rename(data / "moved.md")
    _index, chunks = indexing.build_index_from_sources()
    assert calls == [["same bytes"]]
    moved = next(c for c in chunks if c["path"] == "moved.md")
    assert moved["source"] == "moved"
    assert moved["text"] == "same bytes"
    assert any(c["path"] == "b.md" for c in chunks)


def test_force_reencodes_every_document(tmp_path, monkeypatch):
    """force=True does not reuse per-document embeddings."""
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    _isolate(monkeypatch, data, cache)
    _install_loaders(monkeypatch, data)
    calls = _install_embed(monkeypatch)
    _write_corpus(data)

    indexing.build_index_from_sources()
    calls.clear()
    indexing.build_index_from_sources(force=True)
    assert calls == [["alpha clause"], ["beta one", "beta two"]]


def test_mismatched_cache_header_is_a_miss(tmp_path, monkeypatch):
    """A file at the expected cache path is ignored when its stamped identity differs."""
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    _isolate(monkeypatch, data, cache)
    _install_loaders(monkeypatch, data)
    calls = _install_embed(monkeypatch)
    _write_corpus(data)

    indexing.build_index_from_sources()
    _b_npy, b_json = _doc_cache(cache, "b.md", data / "b.md")
    payload = json.loads(b_json.read_text(encoding="utf-8"))
    payload["config"] = {**payload["config"], "pipeline_version": "stale"}
    payload["chunks"][0]["path"] = "old/b.md"
    b_json.write_text(json.dumps(payload), encoding="utf-8")
    _drop_built_index(cache)
    calls.clear()
    _index, chunks = indexing.build_index_from_sources()
    assert calls == [["beta one", "beta two"]]
    restored = json.loads(b_json.read_text(encoding="utf-8"))
    assert restored["config"]["pipeline_version"] == indexing.CACHE_PIPELINE_VERSION
    assert restored["chunks"][0]["path"] == "b.md"
    assert all(c["path"] != "old/b.md" for c in chunks)


def test_prune_deletes_stale_cache_temps_only(tmp_path, monkeypatch):
    """Stale mkstemp leftovers are removed. A temp from a live build is kept."""
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    _isolate(monkeypatch, data, cache)
    _install_loaders(monkeypatch, data)
    _install_embed(monkeypatch)
    _write_corpus(data)
    indexing.build_index_from_sources()

    embeddings = cache / "embeddings"
    stale = embeddings / ".cache-stale.npy"
    fresh = embeddings / ".cache-fresh.npy"
    stale.write_bytes(b"old")
    fresh.write_bytes(b"new")
    aged = time.time() - indexing._CACHE_TEMP_MAX_AGE_SECONDS - 60
    os.utime(stale, (aged, aged))

    (data / "b.md").write_text("beta changed\n", encoding="utf-8")
    indexing.build_index_from_sources()
    assert not stale.exists()
    assert fresh.is_file()


def test_cache_write_error_does_not_abort_build(tmp_path, monkeypatch):
    """A cache-write failure still returns the index built from the new vectors."""
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    _isolate(monkeypatch, data, cache)
    _install_loaders(monkeypatch, data)
    _install_embed(monkeypatch)
    _write_corpus(data)

    def _unwritable(*_args, **_kwargs):
        raise ValueError("npy header failed")

    monkeypatch.setattr(indexing, "_atomic_replace", _unwritable)
    index, chunks = indexing.build_index_from_sources()
    assert index is not None
    assert [c["text"] for c in chunks] == ["alpha clause", "beta one", "beta two"]
    assert index.ntotal == 3
