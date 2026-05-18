"""
tests/test_pdf_resolution.py — Unit tests for PDF path resolution logic.
"""

from pathlib import Path
import pytest
import main as app

def test_resolve_pdf_path_already_pdf_exists(tmp_path):
    """If the md_path is already a .pdf and it exists, return it directly."""
    pdf_file = tmp_path / "test_doc.pdf"
    pdf_file.touch()
    
    resolved = app.resolve_pdf_path(pdf_file)
    assert resolved == pdf_file

def test_resolve_pdf_path_already_pdf_not_exists(tmp_path):
    """If the md_path is already a .pdf but does not exist, return it."""
    pdf_file = tmp_path / "does_not_exist.pdf"
    resolved = app.resolve_pdf_path(pdf_file)
    assert resolved == pdf_file

def test_resolve_pdf_path_same_directory(tmp_path):
    """If a .pdf file exists in the same directory as the .md file, return it."""
    md_file = tmp_path / "document.md"
    pdf_file = tmp_path / "document.pdf"
    
    md_file.touch()
    pdf_file.touch()
    
    resolved = app.resolve_pdf_path(md_file)
    assert resolved == pdf_file

def test_resolve_pdf_path_public_docs_exact(tmp_path, monkeypatch):
    """If the PDF exists in public/docs with the exact stem, return it."""
    md_file = tmp_path / "BCGEU_19th_Main_Agreement.md"
    md_file.touch()
    
    public_docs_dir = tmp_path / "public" / "docs"
    public_docs_dir.mkdir(parents=True)
    monkeypatch.setattr(app, "PUBLIC_DOCS_DIR", public_docs_dir)
    
    pdf_file = public_docs_dir / "BCGEU_19th_Main_Agreement.pdf"
    pdf_file.touch()
    
    resolved = app.resolve_pdf_path(md_file)
    assert resolved == pdf_file

def test_resolve_pdf_path_public_docs_prefix(tmp_path, monkeypatch):
    """If a multi-part file exists, it should match the consolidated PDF in public/docs based on the prefix."""
    md_file = tmp_path / "BC_OHS_Regulation_-_Part_01.md"
    md_file.touch()
    
    public_docs_dir = tmp_path / "public" / "docs"
    public_docs_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(app, "PUBLIC_DOCS_DIR", public_docs_dir)
    
    pdf_file = public_docs_dir / "BC_OHS_Regulation.pdf"
    pdf_file.touch()
    
    resolved = app.resolve_pdf_path(md_file)
    assert resolved == pdf_file

def test_resolve_pdf_path_fallback_to_md(tmp_path, monkeypatch):
    """If no PDF is found at all, it should fall back to the original MD path."""
    md_file = tmp_path / "Only_Markdown_Exists.md"
    md_file.touch()
    
    public_docs_dir = tmp_path / "public" / "docs"
    public_docs_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(app, "PUBLIC_DOCS_DIR", public_docs_dir)
    
    resolved = app.resolve_pdf_path(md_file)
    assert resolved == md_file
