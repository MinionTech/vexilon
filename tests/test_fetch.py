"""
tests/test_fetch.py — Unit tests for _fetch_pdf_cache_if_missing()

Mocks urllib.request.urlretrieve so no actual network calls are made.
Purpose: verify the HF Spaces download bootstrap logic — if this function
breaks, the entire app fails to start on Hugging Face Spaces.

This function is stdlib-only (urllib) so Renovate won't bump it, but
refactors and logic changes inside the project can break it silently.
"""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

import vexilon.indexing as indexing

# ── Helpers ───────────────────────────────────────────────────────────────────

def _patch_paths(monkeypatch, tmp_path: Path) -> dict:
    """Redirect pdf_cache paths into tmp_path and return them."""
    index_path = tmp_path / "index.faiss"
    chunks_path = tmp_path / "chunks.pkl"

    monkeypatch.setattr(indexing, "INDEX_PATH", index_path)
    monkeypatch.setattr(indexing, "CHUNKS_PATH", chunks_path)
    monkeypatch.setattr(indexing, "PDF_CACHE_DIR", tmp_path)
    return {"index": index_path, "chunks": chunks_path}


# ── No-op when all files present ─────────────────────────────────────────────

def test_no_download_when_all_files_present(monkeypatch, tmp_path):
    """When both cache files exist, urlretrieve must never be called."""
    paths = _patch_paths(monkeypatch, tmp_path)
    for p in paths.values():
        p.write_bytes(b"placeholder")

    with patch("urllib.request.urlretrieve") as mock_retrieve:
        indexing._fetch_pdf_cache_if_missing()

    mock_retrieve.assert_not_called()


# ── Downloads when files are missing ─────────────────────────────────────────

def test_downloads_both_when_cache_dir_empty(monkeypatch, tmp_path):
    """When no files exist, urlretrieve must be called once for each of the 2 assets."""
    _patch_paths(monkeypatch, tmp_path)

    def _fake_retrieve(url, dest):
        Path(dest).write_bytes(b"fake content")

    with patch("urllib.request.urlretrieve", side_effect=_fake_retrieve) as mock_retrieve:
        indexing._fetch_pdf_cache_if_missing()

    assert mock_retrieve.call_count == 2


def test_only_downloads_missing_files(monkeypatch, tmp_path):
    """If only one file is missing, only that file should be downloaded."""
    paths = _patch_paths(monkeypatch, tmp_path)
    paths["index"].write_bytes(b"existing index")
    # chunks is missing

    def _fake_retrieve(url, dest):
        Path(dest).write_bytes(b"downloaded")

    with patch("urllib.request.urlretrieve", side_effect=_fake_retrieve) as mock_retrieve:
        indexing._fetch_pdf_cache_if_missing()

    assert mock_retrieve.call_count == 1
    downloaded_dest = Path(mock_retrieve.call_args[0][1])
    assert downloaded_dest == paths["chunks"]


# ── URL construction ──────────────────────────────────────────────────────────

def test_urls_point_to_github_raw(monkeypatch, tmp_path):
    """Every download URL must start with the GitHub raw base URL."""
    _patch_paths(monkeypatch, tmp_path)

    called_urls = []

    def _fake_retrieve(url, dest):
        called_urls.append(url)
        Path(dest).write_bytes(b"data")

    with patch("urllib.request.urlretrieve", side_effect=_fake_retrieve):
        indexing._fetch_pdf_cache_if_missing()

    assert len(called_urls) == 2
    for url in called_urls:
        assert url.startswith(indexing._GITHUB_RAW_BASE), (
            f"Download URL {url!r} does not start with _GITHUB_RAW_BASE. "
            "Did someone change the base URL accidentally?"
        )


# ── Cache dir creation ────────────────────────────────────────────────────────

def test_creates_cache_dir_when_missing(monkeypatch, tmp_path):
    """If PDF_CACHE_DIR does not exist, it must be created before downloading."""
    # Point cache dir to a subdirectory that doesn't exist yet
    missing_dir = tmp_path / "new_cache_dir"
    monkeypatch.setattr(indexing, "PDF_CACHE_DIR", missing_dir)
    monkeypatch.setattr(indexing, "INDEX_PATH", missing_dir / "index.faiss")
    monkeypatch.setattr(indexing, "CHUNKS_PATH", missing_dir / "chunks.pkl")

    assert not missing_dir.exists()

    def _fake_retrieve(url, dest):
        Path(dest).write_bytes(b"data")

    with patch("urllib.request.urlretrieve", side_effect=_fake_retrieve):
        indexing._fetch_pdf_cache_if_missing()

    assert missing_dir.exists(), "PDF_CACHE_DIR must be created if it doesn't exist"
