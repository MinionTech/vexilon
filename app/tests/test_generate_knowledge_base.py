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


def test_resolve_document_link_copies_file_and_preserves_suffix(tmp_path):
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
        create_public_files=True,
    )
    assert link == "/public/docs/BCGEU_20th_Main_Agreement.md"
    published_target = public_docs / "BCGEU_20th_Main_Agreement.md"
    assert published_target.exists()
    assert not published_target.is_symlink()
    assert published_target.read_text(encoding="utf-8") == "# 20th Agreement"


def test_resolve_document_link_refreshes_stale_markdown(tmp_path):
    import os
    import time

    public_docs = tmp_path / "public" / "docs"
    public_docs.mkdir(parents=True)
    data_dir = tmp_path / "data" / "01_primary"
    data_dir.mkdir(parents=True)

    source_file = data_dir / "Test_Doc.md"
    source_file.write_text("# Initial Content", encoding="utf-8")

    published_file = public_docs / "Test_Doc.md"
    published_file.write_text("# Initial Content", encoding="utf-8")

    # Update source with newer content and explicit future mtime
    source_file.write_text("# Updated Content", encoding="utf-8")
    future_time = time.time() + 10
    os.utime(source_file, (future_time, future_time))

    link = resolve_document_link(
        "Test_Doc",
        "01_primary/Test_Doc.md",
        public_docs,
        tmp_path / "data",
        create_public_files=True,
    )
    assert link == "/public/docs/Test_Doc.md"
    assert published_file.read_text(encoding="utf-8") == "# Updated Content"


def test_resolve_document_link_returns_none_when_unresolvable(tmp_path):
    public_docs = tmp_path / "public" / "docs"
    public_docs.mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    link = resolve_document_link("Nonexistent_Document", None, public_docs, data_dir)
    assert link is None


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


def test_generate_knowledge_base_registry_conduct_ordering(tmp_path):
    data_dir = tmp_path / "data"
    public_docs = tmp_path / "public" / "docs"
    data_dir.mkdir(parents=True)
    public_docs.mkdir(parents=True)
    (data_dir / "03_resources").mkdir(parents=True)
    (data_dir / "03_resources" / "Custom_Conduct_Doc.md").write_text("# Conduct", encoding="utf-8")
    (data_dir / "03_resources" / "Alpha_Resource_Doc.md").write_text("# Resource", encoding="utf-8")

    sources_yaml = (
        "sources:\n"
        "  - path: app/data/03_resources/Custom_Conduct_Doc.md\n"
        "    category: conduct\n"
        "  - path: app/data/03_resources/Alpha_Resource_Doc.md\n"
        "    category: resources\n"
    )
    (data_dir / "sources.yaml").write_text(sources_yaml, encoding="utf-8")

    manifest = {
        "version": "1.0",
        "sources": {
            "03_resources/Custom_Conduct_Doc.md": {"hash": "111", "size_bytes": 10},
            "03_resources/Alpha_Resource_Doc.md": {"hash": "222", "size_bytes": 10},
        },
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    content = generate_knowledge_base_markdown(data_dir=data_dir, public_docs_dir=public_docs)
    conduct_pos = content.find("Custom Conduct Doc")
    alpha_pos = content.find("Alpha Resource Doc")
    assert conduct_pos != -1 and alpha_pos != -1
    assert conduct_pos < alpha_pos


def test_generate_knowledge_base_sources_yaml_corrupt_raises(tmp_path):
    data_dir = tmp_path / "data"
    public_docs = tmp_path / "public" / "docs"
    data_dir.mkdir(parents=True)
    public_docs.mkdir(parents=True)
    manifest = {"version": "1.0", "sources": {}}
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (data_dir / "sources.yaml").write_text("invalid: [yaml: string: :", encoding="utf-8")

    with pytest.raises(Exception):
        generate_knowledge_base_markdown(data_dir=data_dir, public_docs_dir=public_docs)


def test_generate_knowledge_base_restricts_unrecognized_markdown(tmp_path):
    data_dir = tmp_path / "data"
    public_docs = tmp_path / "public" / "docs"
    data_dir.mkdir(parents=True)
    public_docs.mkdir(parents=True)

    # Place an arbitrary markdown file in an unrecognized subfolder without manifest/registry
    (data_dir / "99_internal").mkdir(parents=True)
    (data_dir / "99_internal" / "Secret_Notes.md").write_text("# Secret", encoding="utf-8")

    content = generate_knowledge_base_markdown(data_dir=data_dir, public_docs_dir=public_docs)
    assert "Secret Notes" not in content


def test_generate_knowledge_base_manifest_corrupt_raises(tmp_path):
    data_dir = tmp_path / "data"
    public_docs = tmp_path / "public" / "docs"
    data_dir.mkdir(parents=True)
    public_docs.mkdir(parents=True)

    (data_dir / "manifest.json").write_text("{ corrupt json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        generate_knowledge_base_markdown(data_dir=data_dir, public_docs_dir=public_docs)


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


def test_update_knowledge_base_files_broken_symlink_recovery(tmp_path):
    data_dir = tmp_path / "data"
    public_docs = tmp_path / "public" / "docs"
    app_root = tmp_path / "app"
    data_dir.mkdir(parents=True)
    public_docs.mkdir(parents=True)
    app_root.mkdir(parents=True)

    manifest = {"version": "1.0", "sources": {}}
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # Create dangling symlink
    dangling = app_root / "chainlit_en-US.md"
    dangling.symlink_to("nonexistent.md")

    update_knowledge_base_files(data_dir, public_docs, app_root, dry_run=False)
    assert dangling.exists()
    assert dangling.is_symlink()
    assert dangling.resolve() == (app_root / "chainlit.md").resolve()


def test_main_cli_check_flag(tmp_path, monkeypatch):
    app_root = tmp_path / "app"
    data_dir = app_root / "data"
    public_docs = app_root / "public" / "docs"
    data_dir.mkdir(parents=True)
    public_docs.mkdir(parents=True)

    manifest = {"version": "1.0", "sources": {}}
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    import scripts.generate_knowledge_base as gen_mod
    monkeypatch.setattr(gen_mod, "_APP_ROOT", app_root)

    # When out of sync -> returns 1
    with patch("sys.argv", ["generate_knowledge_base.py", "--check"]):
        assert main() == 1

    # Update files
    update_knowledge_base_files(data_dir, public_docs, app_root, dry_run=False)

    # When in sync -> returns 0
    with patch("sys.argv", ["generate_knowledge_base.py", "--check"]):
        assert main() == 0
