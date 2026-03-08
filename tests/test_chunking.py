"""
tests/test_chunking.py — Unit tests for chunk_text()

Pure logic, no API calls, no mocking required.
"""

import tiktoken
import pytest
from app import chunk_text, CHUNK_SIZE, CHUNK_OVERLAP


def test_short_text_produces_one_chunk():
    """Text shorter than CHUNK_SIZE should produce exactly one chunk."""
    text = "Hello, this is a short sentence."
    chunks = chunk_text(text, page_num=1)
    assert len(chunks) == 1
    assert chunks[0]["page"] == 1
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["text"] == text


def test_chunk_metadata_fields():
    """Every chunk must carry text, page, and chunk_index keys."""
    chunks = chunk_text("Some text here.", page_num=7)
    assert all("text" in c and "page" in c and "chunk_index" in c for c in chunks)
    assert all(c["page"] == 7 for c in chunks)


def test_chunk_index_is_sequential():
    """chunk_index values must be 0-based and contiguous."""
    long_text = "word " * (CHUNK_SIZE * 3)  # guaranteed multi-chunk
    chunks = chunk_text(long_text, page_num=1)
    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk["chunk_index"] == i


def test_multi_chunk_text_overlaps():
    """
    With overlap > 0 the tail of chunk N should share tokens with the head of chunk N+1.
    We verify this by asserting the total reconstructed tokens > len(original tokens),
    which is only possible if overlap is actually applied.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    long_text = "word " * (CHUNK_SIZE * 2)
    chunks = chunk_text(long_text, page_num=1)
    assert len(chunks) > 1
    total_tokens = sum(len(enc.encode(c["text"])) for c in chunks)
    original_tokens = len(enc.encode(long_text))
    # With overlap, the sum of chunk tokens exceeds the original
    assert total_tokens > original_tokens


def test_empty_text_produces_no_chunks():
    """Empty string should yield an empty list (no tokens to encode)."""
    chunks = chunk_text("", page_num=1)
    assert chunks == []


def test_chunk_text_preserves_page_num():
    """page_num should be passed through unchanged regardless of value."""
    for page in (1, 42, 999):
        chunks = chunk_text("some content", page_num=page)
        assert all(c["page"] == page for c in chunks)


def test_large_chunk_size_produces_single_chunk(monkeypatch):
    """If CHUNK_SIZE covers the whole text, only one chunk is produced."""
    import app
    monkeypatch.setattr(app, "CHUNK_SIZE", 10_000)
    monkeypatch.setattr(app, "CHUNK_OVERLAP", 0)
    text = "a short sentence."
    chunks = app.chunk_text(text, page_num=1)
    assert len(chunks) == 1
