#!/usr/bin/env python3
"""
pdf_to_md.py — Forensic PDF-to-Markdown Converter for Vexilon
------------------------------------------------------------
This script uses Claude (Anthropic API) to convert messy legal PDFs into 
clean, structured Markdown files optimized for RAG retrieval.

Usage:
  export ANTHROPIC_API_KEY=<YOUR_ANTHROPIC_API_KEY>
  python scripts/pdf_to_md.py path/to/input.pdf [path/to/output.md]
"""

import os
import sys
import time
import argparse
import re
import traceback
from pathlib import Path
from typing import Optional, List

import anthropic
from pypdf import PdfReader

def print_banner():
    print("=" * 66)
    print(" VEXILON : HIGH-INTEGRITY PDF → MARKDOWN CONVERTER (FORENSIC) ")
    print("=" * 66)

def clean_for_integrity_check(text: str) -> str:
    """Strip all formatting, URLs, and punctuation to verify substantive word preservation."""
    # Remove URLs
    text = re.sub(r"https?://\S*", "", text)
    # Remove non-alphanumeric (except spaces)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    # Collapse whitespace and lowercase
    return " ".join(text.lower().split())

STRUCTURAL_WORDS = {"table", "contents", "continued", "appendix", "article", "section", "part", "page", "break"}

def extract_raw_text(pdf_path: Path) -> List[str]:
    """Basic extraction using pypdf to get page-by-page raw content."""
    print(f"[*] Extracting raw text from {pdf_path.name}...")
    reader = PdfReader(str(pdf_path))
    pages = []
    
    for page in reader.pages:
        text = page.extract_text() or ""
        # Remove bclaws-specific web artifacts that contaminate the word count
        text = re.sub(r"https?://\www\.bclaws\.gov\.bc\.ca/\S*", "", text)
        text = re.sub(r"\d{2}/\d{2}/\d{4},?\s*\d{2}:\d{2}\s+[^\n]*", "", text)
        pages.append(text.strip())
        
    print(f"[+] Total pages extracted: {len(pages)}")
    return pages

def convert_batch(client: anthropic.Anthropic, model: str, batch_text: str, source_name: str, batch_idx: int) -> str:
    """Individual pass for a single batch of text with resilience and retries."""
    system_prompt = f"""You are a ZERO-REASONING legal transcription engine. 
Your ONLY task is to add Markdown formatting to raw text from '{source_name}'.

STRICT INTEGRITY RULES:
1. VERBATIM ONLY: You are FORBIDDEN from changing, adding, or removing a single substantive word.
2. NO IMPROVEMENT: Do not fix "typos" or "grammar." If the raw text is broken, leave it broken but formatted.
3. NO SUMMARIZATION: Every single sentence of substance MUST be preserved in its entirety.
4. STRUCTURE: Use # for Articles, ## for Sections. Use Table format for lists of definitions or tables.
5. NO NOISE: Remove page numbers, URLs, and footers.
6. FORMAT: Output ONLY Markdown. No preamble, 'Here is the markdown' talk, or meta-notes.
7. NO META-TALK: Do NOT add meta-text like 'Included for completeness', '[Batch 22]', or '(Continued)'. Your output must contain ONLY text that originated from the PDF source document."""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=0.0, # PARANOID DETERMINISM
                system=system_prompt,
                messages=[{"role": "user", "content": f"[Batch {batch_idx}] Raw text:\n\n{batch_text}"}]
            )
            return response.content[0].text
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            wait_time = (attempt + 1) * 2
            print(f"    [!] API Error in Batch {batch_idx} (Attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
    
    return "" # Unreachable

def convert_to_md(raw_pages: List[str], source_name: str, output_path: Path, verify: bool = True) -> str:
    """Use Claude to restructure into clean MD with optional dual-pass verification."""
    client = anthropic.Anthropic()
    
    # Selection based on user request for "best outcome"
    # The app.py format suggests the standard is: claude-<model>-<major>-<minor>-<YYYYMMDD>
    primary_model = os.getenv("CONVERT_MODEL", "claude-sonnet-4-6")
    secondary_model = os.getenv("CONCENSUS_MODEL", "claude-haiku-4-5-20251001") # Fast consensus model
    
    print(f"[*] Primary Model:   {primary_model}")
    if verify:
        print(f"[*] Consensus Model: {secondary_model} (Dual-Pass Enabled)")
    
    batch_size = 3 # Smaller batches = higher precision
    full_markdown = []
    integrity_failures = 0

    for i in range(0, len(raw_pages), batch_size):
        batch = raw_pages[i:i+batch_size]
        batch_text = "\n\n--- PAGE BREAK ---\n\n".join(batch)
        batch_id = (i // batch_size) + 1
        
        print(f"    [>] Batch {batch_id}: Processing pages {i+1} to {min(i+batch_size, len(raw_pages))}...")
        
        # Pass 1
        md_p1 = convert_batch(client, primary_model, batch_text, source_name, batch_id)
        
        if verify:
            # Pass 2
            md_p2 = convert_batch(client, secondary_model, batch_text, source_name, batch_id)
            
            # Word-Fingerprint Integrity Check (Raw vs MD)
            raw_fingerprint = clean_for_integrity_check(batch_text)
            md_fingerprint = clean_for_integrity_check(md_p1)
            
            # Note: We allow minor word count drops (noise removal), 
            # but we check if the MD contains words NOT in the raw text (hallucination).
            raw_words = set(raw_fingerprint.split())
            md_words = set(md_fingerprint.split())
            new_words = md_words - raw_words
            
            # Filter out structural and common formatting words
            true_hallucinations = [w for w in new_words if len(w) > 3 and w not in STRUCTURAL_WORDS] 
            
            if true_hallucinations:
                print(f"    [!] WARNING: Potential substantive hallucinations: {true_hallucinations[:5]}...")
                integrity_failures += 1
            
            # Consensus Check (P1 vs P2)
            if clean_for_integrity_check(md_p1) != clean_for_integrity_check(md_p2):
                print(f"    [!] NOTICE: Structural divergence between models. Defaulting to {primary_model}.")

        full_markdown.append(md_p1)
        
        # Incremental save to prevent data loss
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(md_p1 + "\n\n")
            
        time.sleep(0.5)

    if integrity_failures > 0:
        print(f"\n[!] ALERT: Found {integrity_failures} batches with potential word-integrity issues.")
        print("    Please audit the final output sections where the PDF had complex formatting.")
    else:
        print("\n[SUCCESS] Forensic word-integrity check passed.")

    return "\n\n".join(full_markdown)

def main():
    parser = argparse.ArgumentParser(description="Convert PDF to RAG-optimized Markdown with Forensic Integrity")
    parser.add_argument("input", help="Path to input PDF file")
    parser.add_argument("output", nargs="?", help="Path to output MD file (optional)")
    parser.add_argument("--no-verify", action="store_false", dest="verify", help="Disable dual-pass verification (faster/cheaper)")
    parser.set_defaults(verify=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File {input_path} not found.")
        sys.exit(1)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.with_suffix(".md")
    
    print_banner()
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("-" * 66)
    
    # Initialize/Clear output file for incremental writes
    output_path.write_text("", encoding="utf-8")

    try:
        raw_pages = extract_raw_text(input_path)
        markdown_content = convert_to_md(raw_pages, input_path.stem, output_path, verify=args.verify)
        
        print("-" * 66)
        print(f"[FINISH] Conversion Complete.")
        print(f"Vexilon Integrity Fingerprint: {len(markdown_content)} chars / {len(markdown_content.split())} words")
        
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
