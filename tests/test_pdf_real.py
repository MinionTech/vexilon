"""
tests/test_pdf_real.py — load_pdf_chunks() with a real (parseable) PDF

Existing test_pdf_loader.py mocks PdfReader entirely, so a pypdf API break
would sail through undetected. These tests write minimal valid PDF bytes to
disk and let pypdf actually parse them — no mocking of the PDF layer.

If pypdf changes PdfReader, .pages, or .extract_text() in a breaking way
after a Renovate bump, these tests catch it.

The minimal PDF fixture encodes a single-page document with visible text
using only standard library (struct + bytes literals) — no additional deps.
"""

from pathlib import Path
import pytest
from unittest.mock import MagicMock
import app
from app import load_pdf_chunks

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


# ── Minimal PDF fixture ───────────────────────────────────────────────────────

def _write_minimal_pdf(path: Path, text: str) -> None:
    """
    Write a minimal but syntactically valid PDF 1.4 file with one page
    containing *text* as a plain-text stream.

    The PDF structure follows the minimum spec required for pypdf to parse
    a page and extract text via extract_text().
    """
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    stream_len = len(stream)

    body = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
        b" /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        + f"4 0 obj\n<< /Length {stream_len} >>\nstream\n".encode()
        + stream
        + b"\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    )

    xref_offset = len(body)
    xref = (
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
    )
    # Compute byte offsets for each object
    offsets = []
    pos = 0
    for line in body.split(b"\n"):
        if line.endswith(b"obj"):
            offsets.append(pos)
        pos += len(line) + 1  # +1 for newline

    # Pad offsets list to exactly 5 entries (objects 1–5)
    while len(offsets) < 5:
        offsets.append(0)
    offsets = offsets[:5]

    xref += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    trailer = (
        f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    ).encode()

    path.write_bytes(body + xref + trailer)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_load_pdf_chunks_with_real_pdf_returns_chunks(tmp_path):
    """
    load_pdf_chunks() must return at least one chunk when given a real parseable PDF.
    Validates that pypdf's PdfReader, .pages, and .extract_text() API still works.
    """
    pdf = tmp_path / "test.pdf"
    _write_minimal_pdf(pdf, "Article 1 says overtime pay is one and a half times regular.")

    chunks = load_pdf_chunks(pdf)

    assert len(chunks) >= 1, (
        "load_pdf_chunks() returned no chunks for a valid single-page PDF. "
        "pypdf's text extraction API may have changed."
    )


def test_load_pdf_chunks_with_real_pdf_page_number_is_one(tmp_path):
    """Page numbers in chunks from a single-page PDF must be 1 (1-based)."""
    pdf = tmp_path / "test.pdf"
    _write_minimal_pdf(pdf, "Vacation leave accrual policy.")

    chunks = load_pdf_chunks(pdf)

    assert all(c["page"] == 1 for c in chunks), (
        f"Expected all chunks to have page=1, got: {[c['page'] for c in chunks]}"
    )


def test_load_pdf_chunks_with_real_pdf_text_is_nonempty(tmp_path):
    """Every chunk returned from a real PDF must have non-empty text."""
    pdf = tmp_path / "test.pdf"
    _write_minimal_pdf(pdf, "Union stewards protect workers rights under the collective agreement.")

    chunks = load_pdf_chunks(pdf)

    assert all(c["text"].strip() for c in chunks), (
        "One or more chunks have empty text — pypdf extract_text() may have broken."
    )


def test_load_pdf_chunks_with_real_pdf_has_required_keys(tmp_path):
    """Every chunk must carry text, page, and chunk_index metadata keys."""
    pdf = tmp_path / "test.pdf"
    _write_minimal_pdf(pdf, "Grievance procedures are outlined in Article 8.")

    chunks = load_pdf_chunks(pdf)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert "text" in chunk
        assert "page" in chunk
        assert "chunk_index" in chunk
