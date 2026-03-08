"""
tests/test_pdf_loader.py — Unit tests for load_pdf_chunks()

Uses a tiny in-memory PDF fixture so no real file is required.
Requires pypdf (already a project dependency).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


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


def test_load_pdf_chunks_basic(tmp_path):
    """load_pdf_chunks should return chunks for each non-empty page."""
    from app import load_pdf_chunks

    dummy_pdf = tmp_path / "test.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 placeholder")  # path just needs to exist

    with patch("app.PdfReader", return_value=_make_mock_reader(["Page one content.", "Page two content."])):
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
        "app.PdfReader",
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

    with patch("app.PdfReader", return_value=_make_mock_reader(["Content on first page."])):
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

    with patch("app.PdfReader", return_value=reader):
        chunks = load_pdf_chunks(dummy_pdf)

    assert chunks == []
