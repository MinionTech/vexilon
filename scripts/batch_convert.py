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
    parser.add_argument("files", nargs="*", help="Optional: Specific PDF files to convert (paths or filenames)")
    parser.add_argument("--force", action="store_true", help="Recreate MD files even if they already exist")
    parser.add_argument("--resume", action="store_true", help="Resume translation from last recorded checkpoint")
    args = parser.parse_args()

    if not DATA_DIR.exists():
        print(f"Error: {DATA_DIR} not found.")
        sys.exit(1)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set.")
        print("Export your API key before running the batch converter:")
        print("export ANTHROPIC_API_KEY=your-key-here")
        sys.exit(1)

    if args.files:
        # Convert user-provided paths into confirmed Path objects
        input_target_paths = [Path(f) for f in args.files]
        target_pdfs = []
        for path in input_target_paths:
            if path.exists():
                target_pdfs.append(path)
            else:
                # Try finding it in the DATA_DIR if just a filename was given
                matches = list(DATA_DIR.rglob(path.name))
                if matches:
                    target_pdfs.extend(matches)
                else:
                    print(f"⚠️ Warning: File '{path}' not found. Skipping.")
        
        if not target_pdfs:
            print("❌ Error: No valid PDF targets found.")
            sys.exit(1)
        
        print(f"Found {len(target_pdfs)} specific PDF(s) to convert.")

    else:
        # Default behavior: Scan for all missing MD files
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

        print(f"Found {len(target_pdfs)} missing PDF(s) to convert.")
    
    for m in target_pdfs:
        print(f" - {m.name}")

    print("\nStarting batch conversion...")
    print("=" * 66)

    success_count = 0
    converter = Path(__file__).parent / "pdf_to_md.py"

    for i, target_pdf in enumerate(target_pdfs, start=1):
        print(f"\n[{i}/{len(target_pdfs)}] Processing: {target_pdf.name}")
        try:
            cmd = [sys.executable, str(converter), str(target_pdf)]
            if args.resume:
                cmd.append("--resume")
            
            subprocess.run(cmd, check=True)
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
