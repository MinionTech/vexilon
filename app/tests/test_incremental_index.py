"""
tests/test_incremental_index.py — Per-document embedding cache.

build_index_from_sources must encode only new or modified documents and must
rebuild a FAISS index whose search results match a cold rebuild.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

import indexing


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
    (cache / "index.faiss").unlink()
    (cache / "chunks.json").unlink()
    index_cached, chunks_cached = indexing.build_index_from_sources(force=True)
    assert calls == []
    assert chunks_cached == chunks_cold
    np.testing.assert_array_equal(_reconstruct(index_cached), _reconstruct(index_cold))

    (data / "b.md").write_text("beta changed\n", encoding="utf-8")
    calls.clear()
    index_inc, chunks_inc = indexing.build_index_from_sources()
    assert calls == [["beta changed"]]
    assert [c["text"] for c in chunks_inc] == ["alpha clause", "beta changed"]

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
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["files"]["b.md"]
    assert entry["content_hash"] == digest
    assert entry["embeddings"] == f"embeddings/{digest}.npy"
    assert entry["chunks"] == f"chunks/{digest}.json"
    vectors = np.load(cache / entry["embeddings"], allow_pickle=False)
    cached_chunks = json.loads((cache / entry["chunks"]).read_text(encoding="utf-8"))
    assert vectors.dtype == np.float32
    assert vectors.shape == (2, indexing.EMBED_DIM)
    assert [c["text"] for c in cached_chunks] == ["beta one", "beta two"]
    assert manifest["config"]["chunk_size"] == indexing.CHUNK_SIZE
    assert manifest["config"]["embed_model"] == indexing.EMBED_MODEL


def test_unchanged_sources_do_not_encode(tmp_path, monkeypatch):
    """A second build with the same sources and config does not call embed_texts."""
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    _isolate(monkeypatch, data, cache)
    _install_loaders(monkeypatch, data)
    calls = _install_embed(monkeypatch)
    _write_corpus(data)

    indexing.build_index_from_sources()
    calls.clear()
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
    """An unreadable embedding file is encoded again; a valid sibling cache is reused."""
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    _isolate(monkeypatch, data, cache)
    _install_loaders(monkeypatch, data)
    calls = _install_embed(monkeypatch)
    _write_corpus(data)

    indexing.build_index_from_sources()
    digest = hashlib.sha256((data / "b.md").read_bytes()).hexdigest()
    (cache / "embeddings" / f"{digest}.npy").write_bytes(b"not-a-numpy-file")
    (cache / "index.faiss").unlink()
    calls.clear()
    indexing.build_index_from_sources()
    assert calls == [["beta one", "beta two"]]
