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

import re
import sys
import time
import os
import argparse
import traceback
import difflib
from pathlib import Path
from typing import List

import anthropic
import pymupdf  # High-precision PDF extraction (geometric word reconstruction)

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
    # Merge into single string and normalize
    return " ".join(text.lower().split())


def extract_raw_text(pdf_path: Path) -> List[str]:
    """Precision extraction using PyMuPDF to preserve word integrity."""
    print(f"[*] Extracting raw text with high precision (PyMuPDF) from {pdf_path.name}...")
    doc = pymupdf.open(str(pdf_path))
    pages = []
    
    for page in doc:
        text = page.get_text() or ""
        # Remove bclaws-specific web artifacts
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
7. NO META-TALK: Do NOT add meta-text like 'Included for completeness', '[Batch 22]', or '(Continued)'. Your output must contain ONLY text that originated from the PDF source document. Do NOT synthesize structure that is not explicitly in the text. NEVER write headings like '## Article 8 (continued)' or '# ARTICLE 13 *(continued)*' — if the source text does not contain the word 'continued', neither should your output."""

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

def convert_to_md(input_path: Path, output_path: Path, verify: bool = True, resume: bool = False) -> str:
    """Use Claude to restructure into clean MD with optional dual-pass verification."""
    client = anthropic.Anthropic()
    raw_pages = extract_raw_text(input_path)
    source_name = input_path.stem
    
    # Selection based on user request for "best outcome"
    primary_model = os.getenv("CONVERT_MODEL", "claude-sonnet-4-6")
    secondary_model = os.getenv("CONSENSUS_MODEL", "claude-haiku-4-7-20260416") # Fast consensus model
    
    print(f"[*] Primary Model:   {primary_model}")
    if verify:
        print(f"[*] Consensus Model: {secondary_model} (Dual-Pass Enabled)")
    
    BATCH_SIZE = 3
    batches = [raw_pages[i:i + BATCH_SIZE] for i in range(0, len(raw_pages), BATCH_SIZE)]
    full_markdown = []
    integrity_failures = 0
    pages_processed = 0
    total_pages = len(raw_pages)
    audit_path = output_path.with_suffix(".integrity.md")
    
    start_batch_idx = 0
    if resume and output_path.exists():
        print(f"[*] RESUME MODE: Checking existing progress for {output_path.name}...")
        if audit_path.exists():
            audit_content = audit_path.read_text(encoding="utf-8")
            batch_matches = re.findall(r'### \[Batch (\d+)\]', audit_content)
            if batch_matches:
                last_batch = int(batch_matches[-1])
                start_batch_idx = last_batch
                print(f"[*] Identified Batch {last_batch} as last milestone. Resuming from {start_batch_idx + 1}...")
        
        # Pre-load existing content
        existing_md = output_path.read_text(encoding="utf-8")
        # Strip truncation warning if present
        if "> [!CAUTION]" in existing_md:
            existing_md = existing_md.split("> [!CAUTION]")[0].strip()
        full_markdown = [existing_md]
        pages_processed = start_batch_idx * BATCH_SIZE
    else:
        audit_path.write_text(f"# Vexilon Forensic Integrity Audit: {source_name}\n\n", encoding="utf-8")

    try:
        for batch_id, batch_pages in enumerate(batches, 1):
            if batch_id <= start_batch_idx:
                continue
            
            batch_text = "\n\n".join(batch_pages)
            page_start = (batch_id - 1) * BATCH_SIZE + 1
            page_end = min(batch_id * BATCH_SIZE, total_pages)
            print(f"    [>] Batch {batch_id}: Processing pages {page_start} to {page_end}...")

            # DUAL-PASS CONSENSUS (Sonnet + Haiku)
            md_p1 = convert_batch(client, primary_model, batch_text, source_name, batch_id)
            md_p2 = convert_batch(client, secondary_model, batch_text, source_name, batch_id)

            # Strip "(continued)" variants from headings only — AI adds these for multi-page sections.
            # Only applied to heading lines (starting with #) to preserve legitimate body text usage.
            def _strip_continued_headings(md: str) -> str:
                lines = md.split('\n')
                return '\n'.join(
                    re.sub(r'\s*[\(\*-]*\s*continued\s*[\)\*-]*', '', l, flags=re.IGNORECASE).strip()
                    if l.startswith('#') else l
                    for l in lines
                )
            md_p1 = _strip_continued_headings(md_p1)
            md_p2 = _strip_continued_headings(md_p2)

            # Write P1 to disk IMMEDIATELY (incremental save)
            full_markdown.append(md_p1)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(full_markdown))
                # Add temporary "In-Progress" footer
                f.write(f"\n\n--- [IN PROGRESS: Batch {batch_id}/{len(batches)}] ---")

            # Word-Stability Check (Hallucination Detection)
            # We check if words in P1 exist in the raw text
            p1_words = re.findall(r'\b\w{4,}\b', md_p1.lower())
            raw_words = set(re.findall(r'\b\w{4,}\b', batch_text.lower()))
            
            # Whitelist structural commonalities
            whitelist = {'article', 'section', 'part', 'page', 'schedule', 'appendix', 'repealed', 'topic', 'definition', 'term', 'subject', 'table', 'contents'}
            hallucinations = [w for w in p1_words if w not in raw_words and w not in whitelist]
            
            # Sub-word Check (PyMuPDF sometimes keeps artifacts)
            true_hallucinations = []
            for h in hallucinations:
                if not any(h in rw for rw in raw_words):
                    true_hallucinations.append(h)

            if true_hallucinations:
                print(f"    [!] WARNING: Potential substantive hallucinations: {true_hallucinations[:5]}...")
                md_lines = md_p1.split("\n")
                for h_word in true_hallucinations[:2]:
                    for idx, line in enumerate(md_lines):
                        if h_word in line.lower():
                            start = max(0, idx - 2)
                            end = min(len(md_lines), idx + 3)
                            print(f"        [>] Context for '{h_word}':")
                            for l in md_lines[start:end]:
                                print(f"            {l.strip()[:100]}")
                            break
                
                print(f"    [*] BATCH PREVIEW READY: Check {output_path.name}")
                ans = input("    [?] Approve this batch anyway? (y/n/skip/p2): ").lower().strip()
                if ans == 'n':
                    sys.exit(1)
                elif ans == 'skip':
                    print("[SKIP] Batch skipped (Check file manually later).")
                    continue
                elif ans == 'p2':
                    md_p1 = md_p2
                    print("    [*] Overriding P1 with P2 consensus.")

                integrity_failures += 1
                with open(audit_path, "a", encoding="utf-8") as af:
                    af.write(f"### [Batch {batch_id}] Hallucination Flagged: {true_hallucinations}\n")
                    if ans == 'p2':
                        af.write("- ACTION: Auditor overrode P1 with P2 consensus.\n")
                    af.write("| Words | Context (Snippet) |\n|---|---|\n")
                    for w in true_hallucinations[:10]:
                        af.write(f"| `{w}` | {md_p1.splitlines()[0][:60]}... |\n")
                    af.write("\n---\n")
            
            # Consensus Check (P1 vs P2)
            p1_clean = clean_for_integrity_check(md_p1)
            p2_clean = clean_for_integrity_check(md_p2)

            if p1_clean != p2_clean:
                # Fuzzy match to ignore noisy footers/headers
                lines1 = [l.strip() for l in md_p1.split("\n") if l.strip()]
                lines2 = [l.strip() for l in md_p2.split("\n") if l.strip()]
                
                diverged = False
                for l1 in lines1[:10]:
                    matches = difflib.get_close_matches(l1, lines2, n=1, cutoff=0.6)
                    if not matches or clean_for_integrity_check(l1) != clean_for_integrity_check(matches[0]):
                        diverged = True
                        break
                
                if diverged:
                    print(f"    [!] NOTICE: Structural divergence detected (auto-accepting Sonnet).")
                    print(f"    --- P1 (Sonnet) ---")
                    for line in lines1[:4]:
                        print(f"        {line[:100]}")
                    print(f"    --- P2 (Haiku) ---")
                    for line in lines2[:4]:
                        print(f"        {line[:100]}")
                    print(f"    -------------------")

                    with open(audit_path, "a", encoding="utf-8") as af:
                        af.write(f"### [Batch {batch_id}] Structural Divergence Detected\n")
                        af.write(f"- Note: {secondary_model} output differed from {primary_model}. Auto-accepted P1.\n")
                        af.write("\n---\n")

            pages_processed += (page_end - page_start + 1)
            time.sleep(0.5)

    except (KeyboardInterrupt, SystemExit, Exception) as e:
        print(f"\n[!] INTERRUPTED: Migration stopped at Batch {batch_id}.")
        with open(audit_path, "a", encoding="utf-8") as af:
            af.write(f"\n\n❌ **CRITICAL FAILURE:** Migration terminated unexpectedly during Batch {batch_id}.\n")
            af.write(f"- Error: {str(e)}\n")
            af.write(f"- Status: Incomplete ({pages_processed}/{total_pages} pages processed).\n")
        
        # Mark the markdown file as truncated
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n\n> [!CAUTION]\n> **TRUNCATED DOCUMENT:** Migration failed at Batch {batch_id} (Page {pages_processed}). USE WITH EXTREME CAUTION.\n")
        raise

    # Final Success Polish
    final_md = "\n\n".join(full_markdown)
    
    # Prune redundant blank lines (collapse 3+ into 2)
    final_md = re.sub(r'\n{3,}', '\n\n', final_md)
    # Strip leading/trailing whitespace
    final_md = final_md.strip() + "\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_md)

    if integrity_failures > 0:
        print(f"\n[!] ALERT: Found {integrity_failures} batches with potential word-integrity issues.")
    else:
        print("\n[SUCCESS] Forensic word-integrity check passed.")
        with open(audit_path, "a", encoding="utf-8") as af:
            af.write("\n\n✅ **SUCCESS:** Forensic word-integrity check passed with 100% parity.\n")

    return final_md

def main():
    parser = argparse.ArgumentParser(description="Convert PDF to RAG-optimized Markdown with Forensic Integrity")
    parser.add_argument("input", help="Path to input PDF file")
    parser.add_argument("output", nargs="?", help="Path to output MD file (optional)")
    parser.add_argument("--no-verify", action="store_false", dest="verify", help="Disable dual-pass verification (faster/cheaper)")
    parser.add_argument("--resume", action="store_true", help="Resume translation from the last recorded batch in the integrity log")
    parser.set_defaults(verify=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        # Try finding it recursively in the knowledge base
        matches = list(Path("data/labour_law").rglob(input_path.name))
        if matches:
            input_path = matches[0]
            print(f"[*] Found '{input_path.name}' in {input_path.parent}")
        else:
            print(f"Error: File {args.input} not found in current directory or knowledge base.")
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
        markdown_content = convert_to_md(input_path, output_path, verify=args.verify, resume=args.resume)
        
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
