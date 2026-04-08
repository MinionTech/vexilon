import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import app
import vexilon.indexing as indexing

def test_load_md_chunks_basic(tmp_path, monkeypatch):
    """load_md_chunks should return chunks with text and page=1 (baseline)."""
    # Use a tiny CHUNK_SIZE so even short texts produce multiple chunks.
    monkeypatch.setattr(indexing, "CHUNK_SIZE", 5)
    monkeypatch.setattr(indexing, "CHUNK_OVERLAP", 2)

    md_file = tmp_path / "test.md"
    md_file.write_text("# Article 1\nThis is some test content for the Markdown loader.")

    chunks = indexing.load_md_chunks(md_file)
    
    assert len(chunks) > 0
    for chunk in chunks:
        assert "text" in chunk
        # app title-cases source names from filenames
        assert chunk["source"] == "Test"
        # Currently, Markdown loader defaults to page 1 for all chunks.
        assert chunk["page"] == 1
        assert "chunk_index" in chunk

def test_load_md_chunks_skips_empty_files(tmp_path):
    """Truly empty Markdown files should return zero chunks."""
    md_file = tmp_path / "empty.md"
    md_file.write_text("   \n\n  ") # Only whitespace

    chunks = indexing.load_md_chunks(md_file)
    assert len(chunks) == 0

def test_load_md_chunks_strips_frontmatter_style_headers(tmp_path):
    """MD loader should handle typical document structures."""
    md_file = tmp_path / "doc.md"
    md_file.write_text("---\ntitle: test\n---\n\n# Actual Content")
    
    chunks = indexing.load_md_chunks(md_file)
    assert any("Actual Content" in c["text"] for c in chunks)
