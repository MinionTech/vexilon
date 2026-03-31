"""
BC Statute & Regulation Processor (HTML to RAG-ready Markdown)

This script automates the ingestion of 'consolidated' statutory documents from bclaws.gov.bc.ca.
It performs the following:
1. Sanitizes raw legislation HTML by stripping boilerplate navigation, ads, and URLs.
2. Identifies 'Part' boundaries within the statute.
3. Partitions the text into manageable Markdown files in data/labour_law/02_statutory/.
4. Formats headers to ensure high-accuracy RAG (Retrieval-Augmented Generation) performance.

Usage:
  1. Download 'multi' HTML from bclaws to /tmp/raw_statute.html
  2. Update 'src_path' in this script.
  3. Run: python3 scripts/statute_processor.py
"""

import re
from pathlib import Path

src_path = Path("/tmp/raw_act.html")
dest_dir = Path("/home/derek/Repos/vexilon/data/labour_law/02_statutory")

def clean_content(text: str) -> str:
    """Strip HTML and bclaws noise."""
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove repeated bclaws boilerplate
    text = re.sub(r'https?://www\.bclaws\.gov\.bc\.ca/\S+', '', text)
    # Collapse whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def process_wca():
    if not src_path.exists():
        print(f"Error: {src_path} not found.")
        return

    content = src_path.read_text(encoding="utf-8")
    
    # Split by the specific BCLaws Part header class to avoid matching citations
    # Example: <p class="part" ...>Part 1 — Preliminary</p>
    chunks = re.split(r'(<p class="part".*?>(?:Part\s+\d+).*?</p>)', content, flags=re.I | re.S)
    
    # chunks[0] is TOC/Preamble, which we can skip or name "Introduction"
    if len(chunks) > 0:
        intro = clean_content(chunks[0])
        if len(intro) > 100:
            (dest_dir / "BC Workers Compensation Act - Introduction.md").write_text(f"# BC Workers Compensation Act - Introduction\n\n{intro}", encoding="utf-8")

    # Re-assemble partitions
    for i in range(1, len(chunks), 2):
        if i + 1 >= len(chunks): break
        
        raw_header = chunks[i]
        raw_body = chunks[i+1]
        
        # Clean both
        header = clean_content(raw_header)
        body = clean_content(raw_body)
        
        # Pull Part Number accurately
        num_m = re.search(r'Part\s+(\d+)', header, re.I)
        if num_m:
            p_num = num_m.group(1)
            filename = f"BC Workers Compensation Act - Part {p_num}.md"
            target = dest_dir / filename
            
            print(f"Writing {filename}...")
            # Prepend a proper header for FAISS
            target.write_text(f"# {header}\n\n{body}", encoding="utf-8")

if __name__ == "__main__":
    process_wca()
    print("Done! Check your folder and run: python3 -c 'from app import startup; startup(force_rebuild=True)'")
