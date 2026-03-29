#!/usr/bin/env python3
"""
batch_convert.py
----------------
Loops through the labour law knowledge base and automatically
runs the forensic converter on PDFs. By default, it skips PDFs that
already have a .md sibling. Use --force to recreate them.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

DATA_DIR = Path("data/labour_law")

def main():
    parser = argparse.ArgumentParser(description="Batch convert PDFs to forensic Markdown.")
    parser.add_argument("--force", action="store_true", help="Recreate MD files even if they already exist")
    args = parser.parse_args()

    if not DATA_DIR.exists():
        print(f"Error: {DATA_DIR} not found.")
        sys.exit(1)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set.")
        print("Export your API key before running the batch converter:")
        print("export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    skip_dirs = {DATA_DIR / "tests", DATA_DIR / ".pdf_cache"}
    pdfs = [
        p for p in DATA_DIR.rglob("*.pdf") 
        if not any(p.is_relative_to(s) for s in skip_dirs)
    ]

    if args.force:
        target_pdfs = pdfs
        print("⚠️ FORCE MODE: All PDFs will be re-converted, overwriting existing Markdown files.")
    else:
        target_pdfs = [p for p in pdfs if not p.with_suffix(".md").exists()]

    if not target_pdfs:
        print("✅ Excellent: All PDFs already have matching Markdown files. Nothing to do.")
        sys.exit(0)

    print(f"Found {len(target_pdfs)} PDF(s) to convert:")
    for m in target_pdfs:
        print(f" - {m.name}")

    print("\nStarting batch conversion...")
    print("=" * 66)

    success_count = 0
    converter = Path(__file__).parent / "pdf_to_md.py"

    for i, target_pdf in enumerate(target_pdfs, start=1):
        print(f"\n[{i}/{len(target_pdfs)}] Processing: {target_pdf.name}")
        try:
            subprocess.run([sys.executable, str(converter), str(target_pdf)], check=True)
            success_count += 1
        except subprocess.CalledProcessError:
            print(f"\n[!] Conversion failed for {target_pdf.name}. Stopping batch.")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n[!] Batch interrupted by user.")
            sys.exit(1)

    print("\n" + "=" * 66)
    print(f"Batch Complete: Successfully converted {success_count}/{len(target_pdfs)} document(s).")
    print("Run `pytest tests/test_knowledge_base.py` to verify parity, then commit.")

if __name__ == "__main__":
    main()
