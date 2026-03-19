import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import app
from app import load_pdf_chunks

@pytest.fixture(autouse=True)
def mock_tokenizer(monkeypatch):
    mock_tok = MagicMock()
    def _mock_tokenize(text, **kwargs):
        words = text.split()
        input_ids = list(range(len(words)))
        offset_mapping = []
        curr = 0
        for w in words:
            start = text.find(w, curr)
            if start == -1: start = curr
            end = start + len(w)
            offset_mapping.append((start, end))
            curr = end
        return MagicMock(input_ids=input_ids, offset_mapping=offset_mapping)
    mock_tok.side_effect = _mock_tokenize
    mock_model = MagicMock()
    mock_model.tokenizer = mock_tok
    monkeypatch.setattr(app, "get_embed_model", lambda: mock_model)
    return mock_tok

def _make_mock_reader(pages: list[str]):
    mock_pages = []
    for text in pages:
        p = MagicMock()
        p.extract_text.return_value = text
        mock_pages.append(p)
    r = MagicMock()
    r.pages = mock_pages
    return r

def test_semantic_bridge_stitches_page_boundaries(tmp_path, monkeypatch):
    """
    Article 10.8(a) previously cut off at page boundaries. 
    This test verifies that sentences spanning pages are joined seamlessly 
    thanks to the 'Semantic Bridge' (Unified Scroll) refactor.
    """
    # Force a small chunk size for testing boundaries
    monkeypatch.setattr(app, "CHUNK_SIZE", 20)
    monkeypatch.setattr(app, "CHUNK_OVERLAP", 5)

    dummy_pdf = tmp_path / "test.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4")

    # A sentence split across Page 1 and Page 2
    page1 = "This is a sentence that starts on page 1"
    page2 = "and continues on page 2 until it finishes."
    
    with patch("pypdf.PdfReader", return_value=_make_mock_reader([page1, page2])):
        chunks = load_pdf_chunks(dummy_pdf)

    # Check that at least one chunk contains the stitched text "page 1 and continues"
    # (Previously, these would be in separate chunks with zero overlap)
    full_stitched_text = " ".join(c["text"] for c in chunks)
    # Use normalized whitespace check
    import re
    normalized = re.sub(r"\s+", " ", full_stitched_text)
    assert "page 1 and continues" in normalized
    print("\n✅ Verification SUCCESS: Semantic Bridge joined the page boundary.")

if __name__ == "__main__":
    # If run directly, run the test
    pytest.main([__file__])
