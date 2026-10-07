#!/usr/bin/env python3
"""
generate_knowledge_base.py
--------------------------
Dynamically generates the Chainlit Knowledge Base drawer markdown
(app/chainlit.md and app/chainlit_en-US.md) from the active document corpus,
manifest.json, and public documents.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_knowledge_base")

_SCRIPT_DIR = Path(__file__).resolve().parent
_APP_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _APP_ROOT.parent

PREAMBLE = """## Knowledge Base

### Data Privacy (PIPA Compliance)

This [open source](https://github.com/MinionTech/vexilon) tool is designed around the BC [Personal Information Protection Act (PIPA)](https://www.bclaws.gov.bc.ca/civix/document/id/complete/statreg/03063_01).

For more information about our zero-storage model, see our [Privacy Policy](https://github.com/MinionTech/vexilon/blob/main/PRIVACY.md).
"""

# Ordering preferences for canonical sections
STATUTORY_ORDER = [
    "BC_Labour_Relations_Code",
    "BC_OHS_Regulation",
    "BC_Workers_Compensation_Act",
    "BC_Employment_Standards_Act",
    "BC_Human_Rights_Code",
]

POLICY_ORDER = [
    "Gov_BC_Standards_of_Conduct",
    "BC_Criminal_Notification_Procedures",
    "Gov_BC_Social_Media_Guidelines_for_Personal_Use",
    "BC_Social_Media_Guidance_for_Public_Service_Employees",
    "Nexus_Test_and_Off-Duty_Conduct",
    "BCGEU_Steward_Resources",
]

FORMS_ORDER = [
    "Grievance_-_0_-_Instructions",
    "Grievance_-_A_-_Grievor_Case",
    "Grievance_-_B_-_Notify_Designates",
    "Grievance_-_C_-_Steward_Case",
    "BCGEU_Grievance_Form_Guide",
    "Policy_Grievance_Form_Guide",
]


def clean_title_from_stem(stem: str) -> str:
    """Format file stem into human-readable title."""
    return stem.replace("_-_", " - ").replace("_", " ")


def get_base_stem(stem: str) -> str:
    """Collapse multi-part stems (e.g. BC_OHS_Regulation_-_Part_01 -> BC_OHS_Regulation)."""
    if "_-_" in stem:
        prefix = stem.split("_-_")[0]
        # Only collapse if prefix matches known multi-part statutes
        if prefix in ("BC_OHS_Regulation", "BC_Workers_Compensation_Act"):
            return prefix
    return stem


def resolve_document_link(
    base_stem: str,
    source_rel_path: str | None,
    public_docs_dir: Path,
    data_dir: Path,
    create_public_files: bool = True,
) -> str | None:
    """Resolve the best public URL for a document.

    Prefers PDF in public/docs, falls back to MD in public/docs.
    If only a file exists in data_dir, publishes a copy inside public_docs_dir
    so Chainlit can serve it without path-traversal restrictions.
    Returns None if no matching asset can be found.
    """
    # 1. Check for exact PDF in public/docs
    exact_pdf = public_docs_dir / f"{base_stem}.pdf"
    if exact_pdf.exists():
        return f"/public/docs/{base_stem}.pdf"

    # 2. Check for PDF in forms
    forms_pdf = public_docs_dir / "forms" / f"{base_stem}.pdf"
    if forms_pdf.exists():
        return f"/public/docs/forms/{base_stem}.pdf"

    # 3. Check for MD in forms
    forms_md = public_docs_dir / "forms" / f"{base_stem}.md"
    if forms_md.exists():
        return f"/public/docs/forms/{base_stem}.md"

    # 4. Check manifest source file in data_dir and ensure public target is up-to-date
    if source_rel_path:
        source_file = data_dir / source_rel_path
        if source_file.exists():
            suffix = source_file.suffix or ".md"
            target_file = public_docs_dir / f"{base_stem}{suffix}"
            if create_public_files:
                target_file.parent.mkdir(parents=True, exist_ok=True)
                if not target_file.exists() or target_file.stat().st_mtime < source_file.stat().st_mtime:
                    try:
                        shutil.copy2(source_file, target_file)
                        logger.info(f"Published document: {target_file}")
                    except Exception as e:
                        logger.error(f"Could not copy {source_file} to {target_file}: {e}")
                        raise
            if target_file.exists() or not create_public_files:
                return f"/public/docs/{target_file.name}"

    # 5. Check for standalone MD in public/docs
    exact_md = public_docs_dir / f"{base_stem}.md"
    if exact_md.exists():
        return f"/public/docs/{base_stem}.md"

    logger.warning(f"Could not resolve any document for stem: {base_stem}")
    return None


def generate_knowledge_base_markdown(
    data_dir: Path | None = None,
    public_docs_dir: Path | None = None,
    create_public_files: bool = True,
    create_symlinks: bool | None = None,
) -> str:
    """Generate the full chainlit.md content based on active manifest and public documents."""
    if create_symlinks is not None:
        create_public_files = create_symlinks
    data_dir = data_dir or _APP_ROOT / "data"
    public_docs_dir = public_docs_dir or _APP_ROOT / "public" / "docs"

    manifest_path = data_dir / "manifest.json"
    sources_dict: dict[str, dict] = {}

    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
            sources_dict = data.get("sources", {})
    else:
        # Fallback to scanning data_dir only if manifest.json does not exist
        fixtures_dir = data_dir / "test_fixtures"
        for p in data_dir.rglob("*.md"):
            if not p.is_relative_to(fixtures_dir) and not p.name.endswith(".integrity.md"):
                rel = str(p.relative_to(data_dir))
                sources_dict[rel] = {}

    # Categorize items, reading domain categories from sources.yaml when available
    category_map: dict[str, str] = {}
    sources_yaml_path = data_dir / "sources.yaml"
    if sources_yaml_path.exists():
        try:
            import yaml
            ydata = yaml.safe_load(sources_yaml_path.read_text(encoding="utf-8")) or {}
            raw_sources = ydata.get("sources")
            if raw_sources is None:
                raw_sources = []
                raw_sources.extend(ydata.get("public_sources", []))
                raw_sources.extend(ydata.get("manual_sources", []))
            for s in raw_sources:
                cat = s.get("category", "")
                p = s.get("path", "")
                if p and cat:
                    stem = Path(p).stem
                    category_map[stem] = cat.lower()
                    category_map[get_base_stem(stem)] = cat.lower()
        except Exception as e:
            logger.error("Failed to parse registry %s: %s", sources_yaml_path, e)
            raise

    primary_authorities: dict[str, str] = {}  # base_stem -> rel_path
    statutory_items: dict[str, str] = {}
    policy_items: dict[str, str] = {}
    form_items: dict[str, str] = {}

    for rel_path in sources_dict.keys():
        path_obj = Path(rel_path)
        stem = path_obj.stem
        base_stem = get_base_stem(stem)
        parts = path_obj.parts
        cat = category_map.get(base_stem) or category_map.get(stem)

        if cat == "agreement" or ("01_primary" in parts and "main_agreement" in stem.lower()):
            primary_authorities.setdefault(base_stem, rel_path)
        elif cat == "statutory" or "02_statutory" in parts:
            statutory_items.setdefault(base_stem, rel_path)
        elif cat == "forms" or "forms" in parts:
            form_items.setdefault(base_stem, rel_path)
        elif cat or "03_resources" in parts or "04_jurisprudence" in parts:
            policy_items.setdefault(base_stem, rel_path)

    # Also scan public_docs_dir / forms for static form PDFs
    forms_dir = public_docs_dir / "forms"
    if forms_dir.exists():
        for f in forms_dir.glob("*"):
            if f.suffix.lower() in (".pdf", ".md"):
                form_items.setdefault(f.stem, f"forms/{f.name}")

    sections = [PREAMBLE.strip()]

    # 1. Primary Authority
    sections.append("\n### Primary Authority\n")
    # Sort with preference for active Main Agreement
    sorted_primary = sorted(
        primary_authorities.keys(),
        key=lambda s: (0 if "20th" in s else 1, s),
    )
    for stem in sorted_primary:
        link = resolve_document_link(
            stem, primary_authorities.get(stem), public_docs_dir, data_dir, create_public_files
        )
        if link:
            title = clean_title_from_stem(stem)
            sections.append(f"* [{title}]({link})")

    # 2. Legislation & Regulations
    sections.append("\n### Legislation & Regulations\n")
    sorted_statutory = sorted(
        statutory_items.keys(),
        key=lambda s: (0, STATUTORY_ORDER.index(s)) if s in STATUTORY_ORDER else (1, s),
    )
    for stem in sorted_statutory:
        link = resolve_document_link(
            stem, statutory_items.get(stem), public_docs_dir, data_dir, create_public_files
        )
        if link:
            title = clean_title_from_stem(stem)
            sections.append(f"* [{title}]({link})")

    # 3. Policy & Jurisprudence
    sections.append("\n### Policy & Jurisprudence\n")
    sorted_policy = sorted(
        policy_items.keys(),
        key=lambda s: (
            0 if category_map.get(s) == "conduct" else 1,
            POLICY_ORDER.index(s) if s in POLICY_ORDER else 99,
            s,
        ),
    )
    for stem in sorted_policy:
        link = resolve_document_link(
            stem, policy_items.get(stem), public_docs_dir, data_dir, create_public_files
        )
        if link:
            title = clean_title_from_stem(stem)
            sections.append(f"* [{title}]({link})")

    # 4. Forms
    sections.append("\n### Forms\n")
    sorted_forms = sorted(
        form_items.keys(),
        key=lambda s: (0, FORMS_ORDER.index(s)) if s in FORMS_ORDER else (1, s),
    )
    for stem in sorted_forms:
        link = resolve_document_link(
            stem, form_items.get(stem), public_docs_dir, data_dir, create_public_files
        )
        if link:
            title = clean_title_from_stem(stem)
            sections.append(f"* [{title}]({link})")

    sections.append("")
    return "\n".join(sections)


def update_knowledge_base_files(
    data_dir: Path | None = None,
    public_docs_dir: Path | None = None,
    app_root: Path | None = None,
    dry_run: bool = False,
) -> bool:
    """Generate and write chainlit.md and chainlit_en-US.md."""
    app_root = app_root or _APP_ROOT
    chainlit_md = app_root / "chainlit.md"
    chainlit_en_md = app_root / "chainlit_en-US.md"

    content = generate_knowledge_base_markdown(
        data_dir=data_dir,
        public_docs_dir=public_docs_dir,
        create_public_files=not dry_run,
    )

    if dry_run:
        existing = chainlit_md.read_text(encoding="utf-8") if chainlit_md.exists() else ""
        return existing == content

    chainlit_md.write_text(content, encoding="utf-8")

    # Handle chainlit_en-US.md symlink or regular file safely
    if os.path.islink(chainlit_en_md):
        if not chainlit_en_md.exists():
            chainlit_en_md.unlink(missing_ok=True)
            chainlit_en_md.symlink_to("chainlit.md")
    elif chainlit_en_md.exists():
        chainlit_en_md.write_text(content, encoding="utf-8")
    else:
        chainlit_en_md.symlink_to("chainlit.md")

    logger.info(f"Updated {chainlit_md}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Chainlit Knowledge Base drawer markdown.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if chainlit.md is in sync without writing files.",
    )
    args = parser.parse_args()

    if args.check:
        in_sync = update_knowledge_base_files(dry_run=True)
        if not in_sync:
            print("ERROR: chainlit.md is out of sync with manifest/sources. Run generate_knowledge_base.py.")
            return 1
        print("OK: chainlit.md is in sync.")
        return 0

    update_knowledge_base_files(dry_run=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
