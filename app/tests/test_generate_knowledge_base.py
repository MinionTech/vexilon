import json
import pytest
from pathlib import Path
from unittest.mock import patch

from scripts.generate_knowledge_base import (
    clean_title_from_stem,
    get_base_stem,
    resolve_document_link,
    generate_knowledge_base_markdown,
    update_knowledge_base_files,
    main,
)


def test_clean_title_from_stem():
    assert clean_title_from_stem("BCGEU_20th_Main_Agreement") == "BCGEU 20th Main Agreement"
    assert clean_title_from_stem("Grievance_-_0_-_Instructions") == "Grievance - 0 - Instructions"
    assert clean_title_from_stem("BC_Labour_Relations_Code") == "BC Labour Relations Code"


def test_get_base_stem():
    assert get_base_stem("BC_OHS_Regulation_-_Part_01") == "BC_OHS_Regulation"
    assert get_base_stem("BC_OHS_Regulation_-_Introduction") == "BC_OHS_Regulation"
    assert get_base_stem("BC_Workers_Compensation_Act_-_Part_08") == "BC_Workers_Compensation_Act"
    assert get_base_stem("BC_Labour_Relations_Code") == "BC_Labour_Relations_Code"
    assert get_base_stem("Grievance_-_0_-_Instructions") == "Grievance_-_0_-_Instructions"


def test_resolve_document_link_prefers_pdf(tmp_path):
    public_docs = tmp_path / "public" / "docs"
    public_docs.mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    pdf_file = public_docs / "BC_Labour_Relations_Code.pdf"
    pdf_file.touch()

    link = resolve_document_link("BC_Labour_Relations_Code", None, public_docs, data_dir)
    assert link == "/public/docs/BC_Labour_Relations_Code.pdf"


def test_resolve_document_link_forms_pdf(tmp_path):
    public_docs = tmp_path / "public" / "docs"
    forms_dir = public_docs / "forms"
    forms_dir.mkdir(parents=True)
    data_dir = tmp_path / "data"

    pdf_file = forms_dir / "Grievance_-_0_-_Instructions.pdf"
    pdf_file.touch()

    link = resolve_document_link("Grievance_-_0_-_Instructions", None, public_docs, data_dir)
    assert link == "/public/docs/forms/Grievance_-_0_-_Instructions.pdf"


def test_resolve_document_link_creates_symlink_for_markdown(tmp_path):
    public_docs = tmp_path / "public" / "docs"
    public_docs.mkdir(parents=True)
    data_dir = tmp_path / "data" / "01_primary"
    data_dir.mkdir(parents=True)

    md_file = data_dir / "BCGEU_20th_Main_Agreement.md"
    md_file.write_text("# 20th Agreement", encoding="utf-8")

    link = resolve_document_link(
        "BCGEU_20th_Main_Agreement",
        "01_primary/BCGEU_20th_Main_Agreement.md",
        public_docs,
        tmp_path / "data",
        create_symlinks=True,
    )
    assert link == "/public/docs/BCGEU_20th_Main_Agreement.md"
    symlink_target = public_docs / "BCGEU_20th_Main_Agreement.md"
    assert symlink_target.exists()
    assert symlink_target.is_symlink()


def test_generate_knowledge_base_markdown_structure(tmp_path):
    data_dir = tmp_path / "data"
    public_docs = tmp_path / "public" / "docs"
    primary_dir = data_dir / "01_primary"
    primary_dir.mkdir(parents=True)
    public_docs.mkdir(parents=True)

    (primary_dir / "BCGEU_20th_Main_Agreement.md").write_text("# 20th", encoding="utf-8")

    manifest = {
        "version": "1.0",
        "sources": {
            "01_primary/BCGEU_20th_Main_Agreement.md": {"hash": "abc", "size_bytes": 100}
        },
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    content = generate_knowledge_base_markdown(data_dir=data_dir, public_docs_dir=public_docs)
    assert "## Knowledge Base" in content
    assert "### Primary Authority" in content
    assert "* [BCGEU 20th Main Agreement](/public/docs/BCGEU_20th_Main_Agreement.md)" in content


def test_update_knowledge_base_files_dry_run(tmp_path):
    data_dir = tmp_path / "data"
    public_docs = tmp_path / "public" / "docs"
    app_root = tmp_path / "app"
    data_dir.mkdir(parents=True)
    public_docs.mkdir(parents=True)
    app_root.mkdir(parents=True)

    manifest = {"version": "1.0", "sources": {}}
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # When file doesn't exist, dry_run returns False
    assert not update_knowledge_base_files(data_dir, public_docs, app_root, dry_run=True)

    # When file matches, dry_run returns True
    update_knowledge_base_files(data_dir, public_docs, app_root, dry_run=False)
    assert update_knowledge_base_files(data_dir, public_docs, app_root, dry_run=True)


def test_main_cli_check_flag(monkeypatch):
    with patch("sys.argv", ["generate_knowledge_base.py", "--check"]):
        assert main() == 0
