import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import vexilon.indexing as indexing
import app

# ─── Tests for _is_toc_or_index_page() (Restored Parity) ──────────────────────

class TestIsTocOrIndexPage:
    """Tests for the navigational-block detection helper in Markdown."""

    def test_dot_leader_toc_page_detected(self):
        """A line with dot-leaders (classic TOC layout) is detected."""
        toc_text = (
            "ARTICLE 8 - GRIEVANCES ....................................... 15\n"
            "8.1 Grievance Procedure ...................................... 15\n"
            "8.2 Step 1 ................................................... 15\n"
        )
        assert indexing._is_toc_or_index_page(toc_text) is True

    def test_index_style_line_detected(self):
        """A page with index-style 'Topic .......... NN' lines is detected."""
        index_text = (
            "Abandonment of Position, 10.10 ......... 23\n"
            "Abnormal Working Conditions, 27.13 ...... 72\n"
            "Accommodation, Board and Lodging, 27.15 . 73\n"
            "Accumulation of Sick Leave, 19.3 ......... 52\n"
            "Acting School Manager, 27.16 ............. 75\n"
            "Additional Paid Holidays, 17.2 ........... 49\n"
        )
        assert indexing._is_toc_or_index_page(index_text) is True

    def test_substantive_article_text_not_detected(self):
        """Actual contract article text is NOT flagged as TOC."""
        article_text = (
            "ARTICLE 8 - GRIEVANCES\n"
            "8.1 Grievance Procedure\n"
            "(a) The Employer and the Union recognize that grievances may arise concerning:\n"
        )
        assert indexing._is_toc_or_index_page(article_text) is False

    def test_empty_line_not_detected(self):
        """Blank or whitespace-only lines return False."""
        assert indexing._is_toc_or_index_page("") is False
        assert indexing._is_toc_or_index_page("   \n\n   ") is False


# ─── Tests for _clean_page_text() (Restored Parity) ───────────────────────

class TestCleanPageText:
    """Tests for the URL/artifact cleaning helper in Markdown."""

    def test_bclaws_url_removed(self):
        """bclaws.gov.bc.ca URLs are stripped from text."""
        raw = (
            "https://www.bclaws.gov.bc.ca/civix/document/id/complete/statreg/00_96113_01 15/81\n"
            "Minimum wage is $16.75 per hour.\n"
        )
        cleaned = indexing._clean_page_text(raw)
        assert "bclaws.gov.bc.ca" not in cleaned
        assert "Minimum wage is $16.75 per hour." in cleaned

    def test_date_stamp_removed(self):
        """Web-extraction date/time stamps are stripped."""
        raw = (
            "17/03/2026, 08:44 Employment Standards Act\n"
            "The director must develop policies.\n"
        )
        cleaned = indexing._clean_page_text(raw)
        assert "17/03/2026" not in cleaned
        assert "The director must develop policies." in cleaned

    def test_clean_text_unchanged(self):
        """Text with no artifacts passes through unchanged."""
        clean = "10.1 Burden of Proof\nIn all cases the burden rests with the Employer."
        assert indexing._clean_page_text(clean) == clean

    def test_multiple_blank_lines_collapsed(self):
        """Three or more consecutive blank lines are collapsed to one."""
        padded = "Line one.\n\n\n\nLine two."
        cleaned = indexing._clean_page_text(padded)
        assert "\n\n\n" not in cleaned


# ─── Tests for load_md_chunks (Adapted from PDF Loader Tests) ──────────────

def test_load_md_chunks_tracks_headers(tmp_path, monkeypatch):
    """load_md_chunks should detect # headers and track them as breadcrumbs in chunks."""
    monkeypatch.setattr(indexing, "CHUNK_SIZE", 5)
    monkeypatch.setattr(indexing, "CHUNK_OVERLAP", 0)

    md_file = tmp_path / "agreement.md"
    md_file.write_text("# ARTICLE 10\n10.1 Content here.\n# ARTICLE 11\n11.1 More content.")

    chunks = indexing.load_md_chunks(md_file)
    
    # Check chunks for Article 10
    art10_chunks = [c for c in chunks if "ARTICLE 10" in c["text"]]
    assert len(art10_chunks) > 0
    assert "ARTICLE 10" in art10_chunks[0]["header"]

    # Check chunks for Article 11
    art11_chunks = [c for c in chunks if "ARTICLE 11" in c["text"]]
    assert len(art11_chunks) > 0
    assert "ARTICLE 11" in art11_chunks[0]["header"]

def test_md_toc_blocks_skipped(tmp_path, monkeypatch):
    """load_md_chunks must skip Markdown blocks that look like TOC dot-leaders."""
    monkeypatch.setattr(indexing, "CHUNK_SIZE", 100)
    monkeypatch.setattr(indexing, "CHUNK_OVERLAP", 25)

    md_file = tmp_path / "agreement.md"
    md_file.write_text("# TOC\nArticle 1 .......... 5\nArticle 2 .......... 10\nArticle 3 .......... 15\n\n# CONTENT\n## Article 1\nActual policy text is here.")

    chunks = indexing.load_md_chunks(md_file)
    
    # The TOC entries should be filtered out by _is_toc_or_index_page
    full_text = " ".join([c["text"] for c in chunks])
    assert "Actual policy text" in full_text
    assert ".........." not in full_text

def test_load_md_chunks_one_based_metadata(tmp_path):
    """Metadata must have page=1 for MD (current baseline)."""
    md_file = tmp_path / "test.md"
    md_file.write_text("Some text.")
    
    chunks = indexing.load_md_chunks(md_file)
    assert chunks[0]["page"] == 1
    assert chunks[0]["source"] == "Test"

def test_load_md_chunks_skips_whitespace_only(tmp_path):
    """MD loader should return empty list for whitespace files."""
    md_file = tmp_path / "empty.md"
    md_file.write_text("   \n\n   ")
    
    chunks = indexing.load_md_chunks(md_file)
    assert len(chunks) == 0

def test_load_md_chunks_handles_missing_file(tmp_path):
    """MD loader should raise FileNotFoundError if file missing."""
    with pytest.raises(FileNotFoundError):
        indexing.load_md_chunks(tmp_path / "missing.md")
