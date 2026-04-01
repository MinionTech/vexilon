#!/usr/bin/env python3
"""
scripts/build_index.py — Standalone FAISS Index Builder for Vexilon
------------------------------------------------------------------
Extracts indexing logic from app.py to allow Docker layer caching.
This script builds the vector index from Markdown sources in data/labour_law
and saves it to .pdf_cache/ for fast runtime loading.
"""

import os
import json
import time
import hashlib
import threading
from pathlib import Path

# --- Configuration (Mirrored from app.py) ---
PDF_CACHE_DIR = Path("./.pdf_cache")
LABOUR_LAW_DIR = Path("./data/labour_law")
INDEX_PATH = PDF_CACHE_DIR / "index.faiss"
CHUNKS_PATH = PDF_CACHE_DIR / "chunks.json"
MANIFEST_PATH = PDF_CACHE_DIR / "manifest.json"

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
MAX_EMBED_TOKENS = int(os.getenv("VEXILON_MAX_EMBED_TOKENS", 4096))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 450))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))

_embed_model = None

def get_embed_model():
    """Load the local embedding model on CPU."""
    global _embed_model
    if _embed_model is None:
        # Stabilize CPU usage
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")

        print(f"[embed] Loading local embedding model '{EMBED_MODEL}'…")
        
        # Ensure we are truly offline to avoid lock-file permission errors
        if os.getenv("TRANSFORMERS_OFFLINE") == "1":
             os.environ["HF_HUB_OFFLINE"] = "1"

        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL, device="cpu")
        _embed_model.max_seq_length = MAX_EMBED_TOKENS
        if hasattr(_embed_model, "tokenizer"):
            _embed_model.tokenizer.model_max_length = MAX_EMBED_TOKENS
        print("[embed] Embedding model ready.")
    return _embed_model

def _get_rag_source_files():
    """Scan LABOUR_LAW_DIR for Markdown files ONLY (Markdown-First RAG)."""
    if not LABOUR_LAW_DIR.exists():
        return []
    tests_dir = LABOUR_LAW_DIR / "tests"
    mds = [
        p for p in LABOUR_LAW_DIR.rglob("*.md") 
        if not p.is_relative_to(tests_dir) and not p.name.endswith(".integrity.md")
    ]
    return sorted(mds, key=lambda p: str(p))

def _get_source_name(stem):
    parts = stem.split("_", 2)
    if len(parts) == 3: return parts[2]
    elif len(parts) == 2: return parts[1]
    return stem.replace("_", " ").title()

def _is_toc_or_index_page(page_text):
    lines = [l.strip() for l in page_text.split("\n") if l.strip()]
    if not lines: return False
    dot_leader_count = sum(1 for l in lines if l.count(".") >= 8 and ".." in l)
    if dot_leader_count >= 3: return True
    import re
    index_line_re = re.compile(r".{10,}\.\s*\d{1,3}\s*$")
    index_count = sum(1 for l in lines if index_line_re.search(l))
    if len(lines) >= 5 and index_count / len(lines) > 0.4: return True
    return False

def chunk_text(full_text, token_data, source_name):
    chunks = []
    if not token_data: return chunks
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    idx = 0
    start = 0
    while start < len(token_data):
        end = min(start + CHUNK_SIZE, len(token_data))
        char_start, _, page_num, header = token_data[start]
        _, char_end, _, _ = token_data[end - 1]
        prefix = f"[{source_name} - {header}] " if header else f"[{source_name}] "
        chunk_text_str = prefix + full_text[char_start:char_end]
        chunks.append({
            "text": chunk_text_str,
            "page": page_num,
            "source": source_name,
            "header": header,
            "chunk_index": idx,
        })
        idx += 1
        start += step
    return chunks

def load_md_chunks(md_path):
    content = md_path.read_text(encoding="utf-8").strip()
    if not content: return []
    source_name = _get_source_name(md_path.stem)
    print(f"[loader] Parsing Markdown '{source_name}'…")

    tokenizer = get_embed_model().tokenizer
    token_metadata = []
    current_header = ""
    lines = content.split("\n")
    
    sections = []
    current_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            if current_lines: sections.append((current_header, current_lines))
            current_header = stripped.lstrip("#").strip().upper()
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines: sections.append((current_header, current_lines))
    
    filtered_lines = []
    for header, section_lines in sections:
        section_text = "\n".join(section_lines)
        if _is_toc_or_index_page(section_text):
            print(f"[loader]   Skipping TOC/index section: {header or '(untitled)'}")
            continue
        filtered_lines.extend(section_lines)
    
    filtered_content = "\n".join(filtered_lines)
    if not filtered_content.strip(): return []
    
    current_header = ""
    char_offset = 0
    for line in filtered_lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            current_header = stripped.lstrip("#").strip().upper()
        
        page_num = 1
        encoding = tokenizer(line, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        for start_off, end_off in encoding.offset_mapping:
            token_metadata.append((char_offset + start_off, char_offset + end_off, page_num, current_header))
        char_offset += len(line) + 1
    return chunk_text(filtered_content, token_metadata, source_name)

def embed_texts(texts):
    import numpy as np
    model = get_embed_model()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.astype(np.float32)

def build_index(chunks):
    import faiss
    import numpy as np
    texts = [c["text"] for c in chunks]
    print(f"[index] Embedding {len(texts)} chunks locally…")
    t0 = time.time()
    vectors = embed_texts(texts)
    print(f"[index] Embeddings complete in {time.time() - t0:.1f}s")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)
    print(f"[index] FAISS index built — {index.ntotal} vectors")
    return index

def save_index(index, chunks):
    import faiss
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    print(f"[index] Saved index → {INDEX_PATH} and chunks → {CHUNKS_PATH}")

def run_build(force=False):
    """Orchestrate the full index build process."""
    all_files = _get_rag_source_files()
    if not all_files:
        print("[build] No source files found to index!")
        return

    current_manifest = {}
    for source_file in all_files:
        hasher = hashlib.sha256()
        with open(source_file, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        current_manifest[source_file.name] = hasher.hexdigest()

    if not force and MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r") as f:
                stored_manifest = json.load(f)
            if (stored_manifest == current_manifest and INDEX_PATH.exists() and CHUNKS_PATH.exists()):
                print("[build] Smart Refresh: No changes detected. Skipping indexing.")
                return
        except Exception:
            pass

    print(f"[build] Scanning Sources in {LABOUR_LAW_DIR}…")
    chunks = []
    for f in all_files:
        if f.suffix.lower() == ".md":
            chunks.extend(load_md_chunks(f))

    print(f"[build] Total {len(chunks)} chunks loaded from {len(all_files)} files.")
    index = build_index(chunks)
    save_index(index, chunks)

    with open(MANIFEST_PATH, "w") as f:
        json.dump(current_manifest, f, indent=2)
    print(f"[build] Indexing complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force rebuild")
    args = parser.parse_args()
    run_build(force=args.force)
