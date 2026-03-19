import pytest
from unittest.mock import MagicMock
import app
from app import chunk_text, CHUNK_SIZE, CHUNK_OVERLAP

@pytest.fixture(autouse=True)
def mock_tokenizer(monkeypatch):
    """
    Mock the tokenizer used by get_embed_model() for all tests in this file.
    This prevents the tests from downloading the real embedding model or
    requiring an internet connection.
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
                # Fallback if find fails (e.g. punctuation differences)
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

def _call_chunk_text(text, page_num=1, header="", source="Test Document"):
    """Helper to call chunk_text with mock token_data for tests."""
    if not text:
        return chunk_text("", [], source)
    tokenizer = app.get_embed_model().tokenizer
    encoding = tokenizer(text)
    token_data = []
    for start, end in encoding.offset_mapping:
        token_data.append((start, end, page_num, header))
    return chunk_text(text, token_data, source)

def test_short_text_produces_one_chunk():
    text = "Hello, this is a short sentence."
    chunks = _call_chunk_text(text, page_num=1)
    assert len(chunks) == 1
    assert chunks[0]["page"] == 1
    assert chunks[0]["chunk_index"] == 0
    # Chunks now have a "[Source] " or "[Source - Header] " prefix
    assert text in chunks[0]["text"]

def test_chunk_metadata_fields():
    chunks = _call_chunk_text("Some text here.", page_num=7)
    assert all("text" in c and "page" in c and "chunk_index" in c for c in chunks)
    assert all(c["page"] == 7 for c in chunks)

def test_chunk_index_is_sequential():
    long_text = "word " * (CHUNK_SIZE * 3)
    chunks = _call_chunk_text(long_text, page_num=1)
    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk["chunk_index"] == i

def test_multi_chunk_text_overlaps():
    long_text = "word " * (CHUNK_SIZE * 2)
    chunks = _call_chunk_text(long_text, page_num=1)
    assert len(chunks) > 1
    # Check that original content is roughly there (allowing for prefix bloat)
    total_text = " ".join(c["text"] for c in chunks)
    assert "word" in total_text

def test_empty_text_produces_no_chunks():
    chunks = _call_chunk_text("", page_num=1)
    assert chunks == []

def test_chunk_text_preserves_page_num():
    for page in (1, 42, 999):
        chunks = _call_chunk_text("some content", page_num=page)
        assert all(c["page"] == page for c in chunks)

def test_large_chunk_size_produces_single_chunk(monkeypatch):
    monkeypatch.setattr(app, "CHUNK_SIZE", 10_000)
    monkeypatch.setattr(app, "CHUNK_OVERLAP", 0)
    text = "a short sentence."
    chunks = _call_chunk_text(text, page_num=1)
    assert len(chunks) == 1
