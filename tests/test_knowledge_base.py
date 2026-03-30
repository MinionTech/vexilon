import pytest
from pathlib import Path

# The repository root (relative to tests/)
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data" / "labour_law"

def test_pdf_md_parity():
    """
    Ensure every PDF in data/labour_law/ has a corresponding .md file.
    This test serves as a CI quality gate to enforce 'Markdown-First' AI indexing.
    """
    if not DATA_DIR.exists():
        pytest.skip(f"Directory {DATA_DIR} not found. Skipping parity test.")

    # Skip the internal tests/ and cache directories
    skip_dirs = {DATA_DIR / "tests", DATA_DIR / ".pdf_cache"}
    
    pdfs = [
        p for p in DATA_DIR.rglob("*.pdf") 
        if not any(p.is_relative_to(s) for s in skip_dirs)
    ]
    
    missing_md = []
    for pdf in pdfs:
        md_file = pdf.with_suffix(".md")
        if not md_file.exists():
            missing_md.append(f"  - {pdf.relative_to(REPO_ROOT)}")
            
    if missing_md:
        error_msg = (
            "\n❌ KNOWLEDGE BASE INTEGRITY ERROR:\n"
            "The following PDFs are missing a corresponding .md file for RAG retrieval:\n"
        )
        error_msg += "\n".join(missing_md)
        error_msg += "\n\nAction required: Run `python scripts/pdf_to_md.py` on these files and commit the Markdown."
        pytest.fail(error_msg)
