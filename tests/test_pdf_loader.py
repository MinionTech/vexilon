"""
tests/test_pdf_loader.py — Unit tests for load_pdf_chunks(), _is_toc_or_index_page(),
and _clean_page_text().

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


# ─── Tests for _is_toc_or_index_page() ───────────────────────────────────────

class TestIsTocOrIndexPage:
    """Tests for the navigational-page detection helper."""

    def test_dot_leader_toc_page_detected(self):
        """A page full of dot-leader lines (classic TOC layout) is detected."""
        toc_text = (
            "ARTICLE 8 - GRIEVANCES ....................................... 15\n"
            "8.1 Grievance Procedure ...................................... 15\n"
            "8.2 Step 1 ................................................... 15\n"
            "8.3 Time Limits to Present Initial Grievance ................. 15\n"
            "8.4 Step 2 ................................................... 15\n"
        )
        assert app._is_toc_or_index_page(toc_text) is True

    def test_index_style_page_detected(self):
        """A page with index-style 'Topic .......... NN' lines is detected."""
        index_text = (
            "Abandonment of Position, 10.10 ......... 23\n"
            "Abnormal Working Conditions, 27.13 ...... 72\n"
            "Accommodation, Board and Lodging, 27.15 . 73\n"
            "Active Trades Classifications - Chart 1 .. 128\n"
            "Adjudication, 28.4 ...................... 78\n"
            "Administrative Provisions, 8.8 .......... 16\n"
        )
        assert app._is_toc_or_index_page(index_text) is True

    def test_substantive_article_text_not_detected(self):
        """Actual contract article text is NOT flagged as TOC."""
        article_text = (
            "ARTICLE 8 - GRIEVANCES\n"
            "8.1 Grievance Procedure\n"
            "(a) The Employer and the Union recognize that grievances may arise concerning:\n"
            "(1) differences between the parties respecting the interpretation, application,\n"
            "operation, or any alleged violation of a provision of this agreement;\n"
            "(2) the dismissal, discipline, or suspension of an employee bound by this agreement.\n"
            "(b) The procedure for resolving a grievance shall be the grievance procedure in this article.\n"
        )
        assert app._is_toc_or_index_page(article_text) is False

    def test_empty_page_not_detected(self):
        """Blank or whitespace-only pages return False (handled upstream)."""
        assert app._is_toc_or_index_page("") is False
        assert app._is_toc_or_index_page("   \n\n   ") is False

    def test_few_dot_leaders_not_detected(self):
        """A page with only 1-2 dot-leader lines (e.g., section heading) is not flagged."""
        mixed_text = (
            "ARTICLE 10 - DISMISSAL, SUSPENSION AND DISCIPLINE\n"
            "See also section 8.9 Dismissal or Suspension Grievances ........... 17\n"
            "10.1 Burden of Proof\n"
            "In all cases of discipline, the burden of proof of just cause shall rest with the Employer.\n"
            "10.2 Dismissal\n"
            "A deputy minister or any other person authorized in accordance with the Public Service Act\n"
            "may dismiss any employee for just cause.\n"
        )
        assert app._is_toc_or_index_page(mixed_text) is False


# ─── Tests for _clean_page_text() ────────────────────────────────────────────

class TestCleanPageText:
    """Tests for the URL/artifact cleaning helper."""

    def test_bclaws_url_removed(self):
        """bclaws.gov.bc.ca URLs are stripped from page text."""
        raw = (
            "17/03/2026, 08:44 Employment Standards Act\n"
            "https://www.bclaws.gov.bc.ca/civix/document/id/complete/statreg/00_96113_01 15/81\n"
            "Minimum wage is $16.75 per hour.\n"
        )
        cleaned = app._clean_page_text(raw)
        assert "bclaws.gov.bc.ca" not in cleaned
        assert "Minimum wage is $16.75 per hour." in cleaned

    def test_date_stamp_removed(self):
        """Web-extraction date/time stamps are stripped."""
        raw = (
            "17/03/2026, 08:44 Employment Standards Act\n"
            "The director must develop policies.\n"
        )
        cleaned = app._clean_page_text(raw)
        assert "17/03/2026" not in cleaned
        assert "The director must develop policies." in cleaned

    def test_clean_text_unchanged(self):
        """Text with no artifacts passes through unchanged."""
        clean = "10.1 Burden of Proof\nIn all cases the burden rests with the Employer."
        assert app._clean_page_text(clean) == clean

    def test_multiple_blank_lines_collapsed(self):
        """Three or more consecutive blank lines are collapsed to one."""
        padded = "Line one.\n\n\n\nLine two."
        cleaned = app._clean_page_text(padded)
        assert "\n\n\n" not in cleaned
        assert "Line one." in cleaned
        assert "Line two." in cleaned


# ─── Tests for TOC-page skipping in load_pdf_chunks() ────────────────────────

def test_toc_pages_skipped_in_load_pdf_chunks(tmp_path, monkeypatch):
    """
    load_pdf_chunks() must skip pages that are pure TOC/index content.
    TOC pages mention every article by name — indexing them causes their
    references to crowd out actual contract text in semantic search.
    """
    import app as _app
    from app import load_pdf_chunks

    monkeypatch.setattr(_app, "CHUNK_SIZE", 5)
    monkeypatch.setattr(_app, "CHUNK_OVERLAP", 0)

    dummy_pdf = tmp_path / "agreement.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4")

    toc_page = (
        "ARTICLE 8 - GRIEVANCES ....................................... 15\n"
        "8.1 Grievance Procedure ...................................... 15\n"
        "8.2 Step 1 ................................................... 15\n"
        "8.3 Time Limits .............................................. 16\n"
    )
    content_page = (
        "ARTICLE 8 - GRIEVANCES\n"
        "8.1 Grievance Procedure\n"
        "The procedure for resolving a grievance shall be the grievance procedure in this article.\n"
    )

    with patch("pypdf.PdfReader", return_value=_make_mock_reader([toc_page, content_page])):
        chunks = load_pdf_chunks(dummy_pdf)

    # All chunks must come from the content page (page 2), not the TOC page (page 1)
    pages_seen = {c["page"] for c in chunks}
    assert 1 not in pages_seen, (
        "TOC page (page 1) should have been skipped, but chunks from it were found."
    )
    assert 2 in pages_seen, "Content page (page 2) should have produced chunks."
