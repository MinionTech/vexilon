import pytest
import fitz
from pathlib import Path
from unittest.mock import MagicMock, patch
import indexing
import main as app

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

def test_load_md_chunks_omits_page_without_matching_pdf(tmp_path):
    """Without a same-stem PDF, chunks must not carry PDF page numbers."""
    md_file = tmp_path / "test.md"
    md_file.write_text("Some text.")

    chunks = indexing.load_md_chunks(md_file)
    assert "page" not in chunks[0]
    assert chunks[0]["source"] == "test"


def test_load_md_chunks_20th_includes_draft_eoe_tag(tmp_path, monkeypatch):
    """20th main agreement chunks must surface DRAFT / E&OE in retrieved text."""
    monkeypatch.setattr(indexing, "CHUNK_SIZE", 50)
    monkeypatch.setattr(indexing, "CHUNK_OVERLAP", 0)

    md_file = tmp_path / "BCGEU_20th_Main_Agreement.md"
    md_file.write_text("# ARTICLE 1\nSample clause text for retrieval.")

    chunks = indexing.load_md_chunks(md_file)
    assert chunks
    assert indexing._DRAFT_EOE_CHUNK_LABEL in chunks[0]["text"]

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


def test_page_map_needle_joins_table_cells():
    """A table row is looked up by its cells in reading order, a separator row by nothing."""
    pages, collapsed = ["a  b"] * 9, ["a b"] * 9
    assert indexing._page_map_needle("| 1 | 42,055.17 | 3,504.60 |", pages, collapsed) == (
        "1 42,055.17 3,504.60",
        collapsed,
        9,
    )
    assert indexing._page_map_needle("| Administrative  Officer | 42,055.17 |", pages, collapsed)[0] == (
        "Administrative Officer 42,055.17"
    )
    assert indexing._page_map_needle("|---|---|---|", pages, collapsed) == ("", collapsed, 9)
    assert indexing._page_map_needle("Plain clause text", pages, collapsed) == ("Plain clause text", pages, 5)


def test_load_md_chunks_maps_table_rows_to_pdf_pages(tmp_path, monkeypatch):
    """Rows of a Markdown table take the page of the PDF that prints their cells."""
    monkeypatch.setattr(indexing, "CHUNK_SIZE", 30)
    monkeypatch.setattr(indexing, "CHUNK_OVERLAP", 0)
    monkeypatch.setattr(indexing, "DATA_DIR", tmp_path)

    pdf = fitz.open()
    pdf.new_page().insert_text((72, 72), "Effective April 6, 2025 the wage scale is as follows.")
    page = pdf.new_page()
    for i, cell in enumerate(["Step", "Annual", "Monthly", "Bi-weekly", "Hourly", "1", "42,055.17", "3,504.60", "1,611.97", "23.0281"]):
        page.insert_text((72, 72 + 14 * i), cell)
    pdf.save(tmp_path / "grid.pdf")

    md_file = tmp_path / "grid.md"
    md_file.write_text(
        "# WAGE SCALE\n"
        "Effective April 6, 2025 the wage scale is as follows.\n\n"
        "| Step | Annual | Monthly | Bi-weekly | Hourly |\n"
        "|---|---|---|---|---|\n"
        "| 1 | 42,055.17 | 3,504.60 | 1,611.97 | 23.0281 |\n"
    )

    chunks = indexing.load_md_chunks(md_file)
    row_chunks = [c for c in chunks if "42,055.17" in c["text"] and "Effective" not in c["text"]]
    assert row_chunks
    assert {c["page"] for c in row_chunks} == {2}
