"""
tests/test_pdf_loader.py — Unit tests for load_pdf_chunks()

Uses a tiny in-memory PDF fixture so no real file is required.
Requires pypdf (already a project dependency).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


import app

@pytest.fixture(autouse=True)
def mock_tokenizer(monkeypatch):
    """
    Mock the tokenizer used by get_embed_model() for all tests in this file.
    """
    mock_tok = MagicMock()
    
    def _mock_tokenize(text, **kwargs):
        words = text.split()
        input_ids = list(range(len(words)))
        offset_mapping = []
        current_idx = 0
        for word in words:
            start = text.find(word, current_idx)
            if start == -1:
                start = current_idx
            end = start + len(word)
            offset_mapping.append((start, end))
            current_idx = end
            
        return MagicMock(input_ids=input_ids, offset_mapping=offset_mapping)
        
    mock_tok.side_effect = _mock_tokenize
    
    mock_embed_model = MagicMock()
    mock_embed_model.tokenizer = mock_tok
    monkeypatch.setattr(app, "get_embed_model", lambda: mock_embed_model)
    return mock_tok


def _make_mock_reader(pages: list[str]):
    """Return a mock PdfReader whose .pages yield the given text strings."""
    mock_pages = []
    for text in pages:
        page = MagicMock()
        page.extract_text.return_value = text
        mock_pages.append(page)

    reader = MagicMock()
    reader.pages = mock_pages
    return reader


def test_load_pdf_chunks_basic(tmp_path, monkeypatch):
    """load_pdf_chunks should return at least one chunk for non-empty content."""
    import app as _app
    from app import load_pdf_chunks

    # Use a tiny CHUNK_SIZE so even short pages produce multiple chunks across the document.
    monkeypatch.setattr(_app, "CHUNK_SIZE", 2)
    monkeypatch.setattr(_app, "CHUNK_OVERLAP", 0)

    dummy_pdf = tmp_path / "test.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 placeholder")  # path just needs to exist

    with patch("pypdf.PdfReader", return_value=_make_mock_reader(["Page one content.", "Page two content."])):
        chunks = load_pdf_chunks(dummy_pdf)

    assert len(chunks) >= 2  # at least one chunk per page
    pages_seen = {c["page"] for c in chunks}
    assert 1 in pages_seen
    assert 2 in pages_seen


def test_load_pdf_chunks_skips_blank_pages(tmp_path):
    """Pages with only whitespace should be skipped entirely."""
    from app import load_pdf_chunks

    dummy_pdf = tmp_path / "test.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 placeholder")

    with patch(
        "pypdf.PdfReader",
        return_value=_make_mock_reader(["Real content here.", "   \n\t  ", "More real content."]),
    ):
        chunks = load_pdf_chunks(dummy_pdf)

    pages_seen = {c["page"] for c in chunks}
    assert 2 not in pages_seen  # blank page 2 must be absent


def test_load_pdf_chunks_page_numbers_are_one_based(tmp_path):
    """Page numbers in chunk metadata must be 1-based, not 0-based."""
    from app import load_pdf_chunks

    dummy_pdf = tmp_path / "test.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 placeholder")

    with patch("pypdf.PdfReader", return_value=_make_mock_reader(["Content on first page."])):
        chunks = load_pdf_chunks(dummy_pdf)

    assert all(c["page"] >= 1 for c in chunks)
    assert chunks[0]["page"] == 1


def test_load_pdf_chunks_extract_text_none_treated_as_empty(tmp_path):
    """If extract_text() returns None the page should be skipped gracefully."""
    from app import load_pdf_chunks

    dummy_pdf = tmp_path / "test.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 placeholder")

    page = MagicMock()
    page.extract_text.return_value = None
    reader = MagicMock()
    reader.pages = [page]

    with patch("pypdf.PdfReader", return_value=reader):
        chunks = load_pdf_chunks(dummy_pdf)

    assert chunks == []


def test_load_pdf_chunks_tracks_article_headers(tmp_path, monkeypatch):
    """load_pdf_chunks should detect and track Article headers to prepend as breadcrumbs."""
    import app as _app
    from app import load_pdf_chunks

    # Small CHUNK_SIZE forces separate chunks per page so page metadata is cleanly isolated.
    monkeypatch.setattr(_app, "CHUNK_SIZE", 2)
    monkeypatch.setattr(_app, "CHUNK_OVERLAP", 0)

    dummy_pdf = tmp_path / "agreement.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4")

    # Mock 3 pages:
    # 1. Start of Article 10
    # 2. Middle of Article 10 (no header on page)
    # 3. Start of Article 11
    page1 = "ARTICLE 10\nDISMISSAL, SUSPENSION AND DISCIPLINE\n10.1 Burden of Proof"
    page2 = "10.2 Right to Grieve Dismissal"
    page3 = "ARTICLE 11\nSENIORITY\n11.1 Seniority Defined"

    with patch("pypdf.PdfReader", return_value=_make_mock_reader([page1, page2, page3])):
        chunks = load_pdf_chunks(dummy_pdf)

    # Check page 1 - should have Article 10 header
    p1_chunks = [c for c in chunks if c["page"] == 1]
    assert len(p1_chunks) > 0, "No chunks found for page 1"
    assert "ARTICLE 10" in p1_chunks[0]["text"]

    # Check page 2 - should STILL have Article 10 header (inherited from page 1)
    p2_chunks = [c for c in chunks if c["page"] == 2]
    assert len(p2_chunks) > 0, "No chunks found for page 2 — chunk size may be too large"
    assert "ARTICLE 10" in p2_chunks[0]["text"]

    # Check page 3 - should have Article 11 header
    p3_chunks = [c for c in chunks if c["page"] == 3]
    assert len(p3_chunks) > 0, "No chunks found for page 3"
    assert "ARTICLE 11" in p3_chunks[0]["text"]
