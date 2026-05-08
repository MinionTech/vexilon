"""
BC Statute & Regulation Processor (HTML to RAG-ready Markdown)

This script automates the ingestion of 'consolidated' statutory documents from bclaws.gov.bc.ca.
It performs the following:
1. Sanitizes raw legislation HTML by stripping boilerplate navigation, ads, and URLs.
2. Identifies 'Part' boundaries within the statute.
3. Partitions the text into manageable Markdown files.
4. Formats headers to ensure high-accuracy RAG (Retrieval-Augmented Generation) performance.

Usage:
  1. Download 'multi' HTML from bclaws to /tmp/raw_wca.html or /tmp/raw_statute.html
  2. Run: python3 scripts/statute_processor.py
"""

import re
import html
import os
from pathlib import Path

# Project-relative paths
ROOT_DIR = Path(__file__).parent.parent
DEST_DIR = ROOT_DIR / "data" / "labour_law" / "02_statutory"

def clean_content(text: str) -> str:
    """Robustly strip HTML and bclaws noise while preserving structure."""
    if not text:
        return ""
    
    # 1. Unescape HTML entities first (e.g., &nbsp;)
    text = html.unescape(text)
    
    # 2. Replace block-level tags with newlines to prevent word jamming
    # Patterns for common BCLaws block elements
    text = re.sub(r'<(p|div|h\d|li|br|tr|table|blockquote)[^>]*>', '\n', text, flags=re.I)
    
    # 3. Strip all remaining tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # 4. Remove bclaws URLs and footers
    text = re.sub(r'https?://www\.bclaws\.gov\.bc\.ca/\S+', '', text)
    
    # 5. Cleanup whitespace
    # Collapse multiple spaces
    text = re.sub(r'[ \t]+', ' ', text)
    # Ensure paragraphs have clean double-newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    # Remove leading/trailing space on each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text.strip()

def process_file(src_path: Path, prefix: str):
    """Generic BCLaws partitioner."""
    if not src_path.exists():
        print(f"Skipping {prefix}: {src_path} not found.")
        return

    print(f"Processing {prefix}...")
    content = src_path.read_text(encoding="utf-8")
    
    # Split by the specific BCLaws Part header class
    # Regex designed to capture the full header tag for use as a title
    chunks = re.split(r'(<p class="part".*?>(?:Part\s+\d+).*?</p>)', content, flags=re.I | re.S)
    
    # Handle Introduction (Chunk 0)
    if len(chunks) > 0:
        intro = clean_content(chunks[0])
        if len(intro) > 100:
            target = DEST_DIR / f"{prefix} - Introduction.md"
            target.write_text(f"# {prefix} - Introduction\n\n{intro}", encoding="utf-8")

    # Handle Parts
    for i in range(1, len(chunks), 2):
        if i + 1 >= len(chunks):
            break
        
        raw_header = chunks[i]
        raw_body = chunks[i+1]
        
        header = clean_content(raw_header)
        body = clean_content(raw_body)
        
        # Extract Part Number for filename sorting
        num_m = re.search(r'Part\s+(\d+)', header, re.I)
        p_num = num_m.group(1) if num_m else str(i // 2 + 1)
        
        filename = f"{prefix} - Part {p_num.zfill(2)}.md"
        target = DEST_DIR / filename
        
        print(f"  -> Writing {filename}...")
        # Prepend a proper header for FAISS/RAG context
        target.write_text(f"# {header}\n\n{body}", encoding="utf-8")

if __name__ == "__main__":
    # Ensure destination exists
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process both known statutes if files are present
    process_file(Path("/tmp/raw_wca.html"), "BC Workers Compensation Act")
    process_file(Path("/tmp/raw_statute.html"), "BC OHS Regulation")
    
    print("\nProcessing complete. Rebuild the index with: python3 -c 'from app import startup; startup(force_rebuild=True)'")
