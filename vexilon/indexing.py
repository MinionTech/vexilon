import os
import json
import time
import hashlib
import fitz
import pickle  # Security: Only used for locally-generated, trusted chunk artifacts.
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer

# ─── Configuration ───────────────────────────────────────────────────────────
PDF_CACHE_DIR = Path("./.pdf_cache")
LABOUR_LAW_DIR = Path("./data/labour_law")
INDEX_PATH = PDF_CACHE_DIR / "index.faiss"
CHUNKS_PATH = PDF_CACHE_DIR / "chunks.pkl"
MANIFEST_PATH = PDF_CACHE_DIR / "manifest.json"
_GITHUB_RAW_BASE = os.getenv("VEXILON_RAW_URL_BASE", "https://raw.githubusercontent.com/DerekRoberts/vexilon/main")
INTEGRITY_PATH = PDF_CACHE_DIR / "integrity.json"
SOURCE_MANIFEST_PATH = LABOUR_LAW_DIR / "manifest.json"

class FileIntegrityError(Exception):
    """Raised when source file parsing fails and strict mode is active."""
    pass

# Models
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
MAX_EMBED_TOKENS = int(os.getenv("VEXILON_MAX_EMBED_TOKENS", 4096))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 450))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", 40))


_embed_model: Any = None
_loaded_model_name: str | None = None

def get_embed_model() -> "SentenceTransformer":
    global _embed_model, _loaded_model_name
    
    # Reload model if EMBED_MODEL env var has changed since last initialization
    current_model_name = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    
    if _embed_model is None or _loaded_model_name != current_model_name:
        if os.getenv("HF_SPACE_ID") or os.getenv("EXTERNAL_CI"):
            for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(var, "1")

        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(current_model_name, device="cpu")
        _loaded_model_name = current_model_name
        _embed_model.max_seq_length = MAX_EMBED_TOKENS
        
        if hasattr(_embed_model, "tokenizer"):
            # Vexilon requires 'Fast' tokenizers for reliable character-offset mapping.
            # Most modern models (including BGE) have fast variants.
            if not getattr(_embed_model.tokenizer, "is_fast", False):
                raise RuntimeError(
                    f"Tokenizer for {current_model_name} is NOT a 'Fast' tokenizer. "
                    "Vexilon requires 'Fast' tokenizers for reliable character-offset mapping."
                )
            
            _embed_model.tokenizer.model_max_length = MAX_EMBED_TOKENS
            
        logger.info(f"[embed] Embedding model '{current_model_name}' ready.")
    return _embed_model

def _get_rag_source_files() -> list[Path]:
    if not LABOUR_LAW_DIR.exists():
        return []
    
    tests_dir = LABOUR_LAW_DIR / "tests"
    files = []
    # Targeted glob patterns for better performance
    for pattern in ["*.md", "*.pdf"]:
        for p in LABOUR_LAW_DIR.rglob(pattern):
            # Skip hidden files, tests, and integrity files
            # CRITICAL: Skip any paths that may exist in sibling worktrees if context is shared
            if (not p.name.startswith(".") 
                and ".workspaces" not in p.parts
                and not p.is_relative_to(tests_dir) 
                and not p.name.endswith(".integrity.md")):
                files.append(p)
                
    return sorted(files, key=lambda p: str(p))

def _get_source_name(stem: str) -> str:
    parts = stem.split("_", 2)
    if len(parts) == 3:
        return parts[2]
    elif len(parts) == 2:
        return parts[1]
    return stem.replace("_", " ").title()

def _clean_page_text(text: str) -> str:
    import re
    # Remove bclaws URLs that clog the embedding
    text = re.sub(r"https?://www\.bclaws\.gov\.bc\.ca/\S+", "", text)
    # Remove web-fetch date/time stamps (e.g. 17/03/2026, 08:44 Employment Standards Act)
    text = re.sub(r"\d{2}/\d{2}/\d{4}, \d{2}:\d{2} .*", "", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_toc_or_index_page(page_text: str) -> bool:
    import re
    lines = [l.strip() for l in page_text.split("\n") if l.strip()]
    if not lines:
        return False
    dot_leader_count = sum(1 for l in lines if l.count(".") >= 8 and ".." in l)
    if dot_leader_count >= 3:
        return True
    index_line_re = re.compile(r".{10,}\.\s*\d{1,3}\s*$")
    index_count = sum(1 for l in lines if index_line_re.search(l))
    if len(lines) >= 5 and index_count / len(lines) > 0.4:
        return True
    return False

def chunk_text(full_text: str, token_data: list[tuple[int, int, int, str]], source_name: str) -> list[dict]:
    chunks = []
    if not token_data:
        return chunks
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

def load_md_chunks(md_path: Path) -> list[dict]:
    content = md_path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    source_name = _get_source_name(md_path.stem)
    logger.info(f"[loader] Parsing Markdown '{source_name}'...")
    tokenizer = get_embed_model().tokenizer
    token_metadata = []
    current_header = ""
    lines = content.split("\n")
    
    sections = []
    current_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            if current_lines:
                sections.append((current_header, current_lines))
            current_header = stripped.lstrip("#").strip().upper()
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_header, current_lines))
    
    filtered_lines = []
    for header, section_lines in sections:
        section_text = "\n".join(section_lines)
        if _is_toc_or_index_page(section_text):
            continue
        filtered_lines.extend(section_lines)
    
    filtered_content = "\n".join(filtered_lines)
    if not filtered_content.strip():
        return []
    
    current_header = ""
    char_offset = 0
    for line in filtered_lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            current_header = stripped.lstrip("#").strip().upper()
        
        page_num = 1
        # Vexilon requires 'Fast' tokenizers for reliable character-offset mapping.
        # This replaces the legacy try-except/char-length fallback blocks.
        encoding = tokenizer(
            line,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        mapping = encoding.get("offset_mapping", [])

        for start_off, end_off in mapping:
            token_metadata.append(
                (char_offset + start_off, char_offset + end_off, page_num, current_header)
            )
            
        char_offset += len(line) + 1

    return chunk_text(filtered_content, token_metadata, source_name)

def load_pdf_chunks(pdf_path: Path, strict: bool = False) -> list[dict]:
    source_name = _get_source_name(pdf_path.stem)
    logger.info(f"[loader] Parsing PDF '{source_name}'...")
    
    chunks = []
    try:
        doc = fitz.open(str(pdf_path))
        tokenizer = get_embed_model().tokenizer
        
        full_text = ""
        token_metadata = []
        char_offset = 0
        
        for i, page in enumerate(doc):
            page_text = page.get_text() or ""
            page_text = _clean_page_text(page_text)
            if not page_text.strip() or _is_toc_or_index_page(page_text):
                continue
            
            page_num = i + 1
            full_text += page_text + "\n"
            
            # Vexilon requires 'Fast' tokenizers for reliable character-offset mapping.
            # This replaces the legacy try-except/char-length fallback blocks.
            encoding = tokenizer(
                page_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                truncation=False,
            )
            mapping = encoding.get("offset_mapping", [])

            for start_off, end_off in mapping:
                token_metadata.append(
                    (char_offset + start_off, char_offset + end_off, page_num, "")
                )

            char_offset += len(page_text) + 1
            
        return chunk_text(full_text, token_metadata, source_name)
    except Exception as e:
        if strict:
            raise FileIntegrityError(f"Critical error parsing {pdf_path}: {e}")
        import traceback
        logger.error(f"[loader] CRITICAL: Error reading PDF {pdf_path}:")
        logger.error(traceback.format_exc())
        raise e

def embed_texts(texts: list[str]) -> "np.ndarray":
    import numpy as np
    model = get_embed_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.astype(np.float32)

def search_index(index: "faiss.IndexFlatIP", chunks: list[dict], query: str, top_k: int | None = None) -> list[dict]:
    import faiss
    if top_k is None:
        top_k = SIMILARITY_TOP_K
    
    query_vec = embed_texts([query])

    faiss.normalize_L2(query_vec)
    _scores, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

def search_index_batch(index: "faiss.IndexFlatIP", chunks: list[dict], queries: list[str], top_ks: list[int]) -> list[list[dict]]:
    """
    Search multiple queries in a single embedding pass to reduce CPU overhead.
    Uses FAISS's native batch search for maximum efficiency (#323).
    """
    import faiss
    import numpy as np
    
    if not queries:
        return []

    # 1. Batch Embed (already optimized in SentenceTransformer)
    query_vecs = embed_texts(queries)
    faiss.normalize_L2(query_vecs)
    
    # 2. Batch Search (FAISS native multi-vector search)
    max_k = max(top_ks)
    _scores, all_indices = index.search(query_vecs, max_k)
    
    results = []
    for i, indices in enumerate(all_indices):
        # Truncate to the specific top_k for this query perspective
        k = top_ks[i]
        results.append([chunks[idx] for idx in indices[:k] if 0 <= idx < len(chunks)])
        
    return results

def build_index(chunks: list[dict]) -> "faiss.IndexFlatIP":
    import faiss
    import numpy as np
    texts = [c["text"] for c in chunks]
    logger.info("[index] Indexing... Please expect a wait (this can take 5-10 minutes on CPU).")
    logger.info(f"[index] Embedding {len(texts)} chunks locally...")
    t0 = time.time()
    vectors = embed_texts(texts)
    logger.info(f"[index] Embeddings complete in {time.time() - t0:.1f}s")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)
    return index

def save_index(index: "faiss.IndexFlatIP", chunks: list[dict]) -> None:
    import faiss
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    # Security: Pickle is used here for its high performance with multi-gigabyte chunk sets.
    # Since these artifacts are generated locally during the build stage and baked into
    # the container image, they represent 'Trusted Content' and do not pose a
    # remote code execution risk in this specific context.
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"[index] Saved index to {INDEX_PATH}")

def build_index_from_sources(force: bool = False) -> tuple[Any, Any] | tuple[None, None]:
    """
    Main entry point for index creation.
    NOTE: Manifest hashing is performed here to determine if a re-index is needed.
    In container environments, this is typically called during the build stage.
    """
    all_files = _get_rag_source_files()
    if not all_files:
        logger.warning("[build] No source files found!")
        return None, None

    current_manifest = {}
    for source_file in all_files:
        hasher = hashlib.sha256()
        with open(source_file, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        # Use relative path to avoid clashes with duplicate names in subdirs
        rel_key = str(source_file.relative_to(LABOUR_LAW_DIR))
        current_manifest[rel_key] = hasher.hexdigest()

    if not force and MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r") as f:
                stored_manifest = json.load(f)
            if stored_manifest == current_manifest and INDEX_PATH.exists() and CHUNKS_PATH.exists():
                logger.info("[build] Smart Refresh: No changes detected in sources. Skipping build.")
                return load_precomputed_index()
        except Exception:
            pass

    logger.info(f"[build] Change detected or forced rebuild. Indexing {len(all_files)} files...")
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    chunks = []
    failed_files = []
    for f in all_files:
        try:
            if f.suffix.lower() == ".md":
                chunks.extend(load_md_chunks(f))
            elif f.suffix.lower() == ".pdf":
                file_chunks = load_pdf_chunks(f, strict=os.getenv("VEXILON_STRICT_BUILD", "false").lower() == "true")
                chunks.extend(file_chunks)
        except Exception as e:
            logger.error(f"[build] ERROR: Failed to index {f.name}: {e}")
            failed_files.append(f.name)
            if os.getenv("VEXILON_STRICT_BUILD", "false").lower() == "true":
                raise

    # Save integrity report
    integrity_data = {
        "timestamp": time.time(),
        "failed_files": failed_files,
        "success_count": len(all_files) - len(failed_files),
        "total_count": len(all_files)
    }
    with open(INTEGRITY_PATH, "w") as f:
        json.dump(integrity_data, f, indent=2)

    if failed_files and os.getenv("VEXILON_STRICT_BUILD", "false").lower() == "true":
        raise FileIntegrityError(f"Build failed due to integrity errors in: {', '.join(failed_files)}")
    
    if not chunks:
        logger.error("[build] No chunks found in source files!")
        return None, None

    index = build_index(chunks)
    save_index(index, chunks)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(current_manifest, f, indent=2)
    return index, chunks

def load_precomputed_index() -> tuple[Any, Any] | tuple[None, None]:
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        return None, None
    logger.info(f"[startup] Loading pre-computed index from {INDEX_PATH}...")
    import faiss
    index = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    logger.info(f"[startup] Pre-computed index loaded — {index.ntotal} vectors, {len(chunks)} chunks.")
    return index, chunks

def get_integrity_report() -> dict:
    if INTEGRITY_PATH.exists():
        try:
            with open(INTEGRITY_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _fetch_pdf_cache_if_missing() -> None:
    import urllib.request
    import urllib.error
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if INDEX_PATH.exists() and CHUNKS_PATH.exists():
        return
    base = _GITHUB_RAW_BASE
    urls = {}
    if not INDEX_PATH.exists():
        urls[INDEX_PATH] = f"{base}/.pdf_cache/index.faiss"
    if not CHUNKS_PATH.exists():
        urls[CHUNKS_PATH] = f"{base}/.pdf_cache/chunks.pkl"
    for dest_path, url in urls.items():
        logger.info(f"[fetch] Downloading {dest_path.name} from {url}...")
        try:
            urllib.request.urlretrieve(url, dest_path)
            logger.info(f"[fetch] Saved {dest_path}")
        except (urllib.error.URLError, OSError) as e:
            logger.warning(f"[fetch] Warning: could not fetch {dest_path.name}: {e}. Will build index from source.")
