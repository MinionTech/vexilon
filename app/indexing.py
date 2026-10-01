import os
import json
import time
import hashlib
import fitz
import logging
from pathlib import Path
from functools import lru_cache
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer

# ─── Configuration ───────────────────────────────────────────────────────────
_PKG_ROOT = Path(__file__).parent

def _resolve_data_dir() -> Path:
    if "AGNAV_DATA_DIR" in os.environ:
        return Path(os.environ["AGNAV_DATA_DIR"])
    container_sources = Path("/data/sources")
    if container_sources.exists():
        return container_sources
    pkg_sources = _PKG_ROOT / "data" / "sources"
    if pkg_sources.exists():
        return pkg_sources
    return _PKG_ROOT / "data"

def _resolve_cache_dir() -> Path:
    if "AGNAV_CACHE_DIR" in os.environ:
        return Path(os.environ["AGNAV_CACHE_DIR"])
    container_cache = Path("/data/cache")
    if container_cache.exists():
        return container_cache
    if Path("/data").exists() and os.access("/data", os.W_OK):
        return container_cache
    return _PKG_ROOT / "data" / "cache"

DATA_DIR = _resolve_data_dir()
CACHE_DIR = _resolve_cache_dir()
PDF_CACHE_DIR = CACHE_DIR  # Backward-compatibility alias
INDEX_PATH = CACHE_DIR / "index.faiss"
CHUNKS_PATH = CACHE_DIR / "chunks.json"
MANIFEST_PATH = CACHE_DIR / "manifest.json"
_GITHUB_RAW_BASE = os.getenv("AGNAV_RAW_URL_BASE", "https://raw.githubusercontent.com/MinionTech/vexilon/main")
INTEGRITY_PATH = CACHE_DIR / "integrity.json"


class FileIntegrityError(Exception):
    """Raised when source file parsing fails and strict mode is active."""
    pass

# Models
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
MAX_EMBED_TOKENS = int(os.getenv("AGNAV_MAX_EMBED_TOKENS", 512))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", 40))
# Document score boosting weights
TIER1_BOOST = float(os.getenv("AGNAV_TIER1_BOOST", "1.2"))
TIER3_BOOST = float(os.getenv("AGNAV_TIER3_BOOST", "0.8"))


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
            # Agreement Navigator requires 'Fast' tokenizers for reliable character-offset mapping.
            # Most modern models (including BGE) have fast variants.
            if not getattr(_embed_model.tokenizer, "is_fast", False):
                raise RuntimeError(
                    f"Tokenizer for {current_model_name} is NOT a 'Fast' tokenizer. "
                    "Agreement Navigator requires 'Fast' tokenizers for reliable character-offset mapping."
                )
            
            _embed_model.tokenizer.model_max_length = MAX_EMBED_TOKENS
            
        logger.info(f"[embed] Embedding model '{current_model_name}' ready.")
    return _embed_model

def _get_rag_source_files() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    
    fixtures_dir = DATA_DIR / "test_fixtures"
    files = []
    # Targeted glob patterns for better performance
    for pattern in ["*.md", "*.pdf"]:
        for p in DATA_DIR.rglob(pattern):
            try:
                rel = p.relative_to(DATA_DIR)
            except ValueError:
                rel = p
            # Skip hidden files, tests, integrity files, and cache directories
            # CRITICAL: Skip any paths that may exist in sibling worktrees if context is shared
            if (not p.name.startswith(".") 
                and ".workspaces" not in p.parts
                and not p.is_relative_to(fixtures_dir) 
                and not p.name.endswith(".integrity.md")
                and "cache" not in rel.parts
                and ".pdf_cache" not in rel.parts):
                files.append(p)
                
    return sorted(files, key=lambda p: str(p))

def _get_source_name(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) > 2 and parts[0].isdigit():
        return " ".join(parts[2:])
    return stem.replace("_", " ")

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

_DRAFT_20TH_AGREEMENT_STEM = "BCGEU_20th_Main_Agreement"
_DRAFT_EOE_CHUNK_LABEL = "DRAFT consolidation — E&OE"


def _chunk_text_prefix(source_name: str, header: str, path: str) -> str:
    path_norm = path.replace("\\", "/").lower()
    cite = f"[{source_name} - {header}] " if header else f"[{source_name}] "
    if _DRAFT_20TH_AGREEMENT_STEM.lower() in path_norm or path_norm.endswith(
        f"{_DRAFT_20TH_AGREEMENT_STEM.lower()}.md"
    ):
        return f"[{_DRAFT_EOE_CHUNK_LABEL}] {cite}"
    return cite


def chunk_text(
    full_text: str,
    token_data: list[tuple[int, int, int | None, str]],
    source_name: str,
    path: str = "",
) -> list[dict]:
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
        line_prefix = _chunk_text_prefix(source_name, header, path)
        chunk_text_str = line_prefix + full_text[char_start:char_end]
        chunk: dict[str, Any] = {
            "text": chunk_text_str,
            "source": source_name,
            "header": header,
            "chunk_index": idx,
            "path": path,
        }
        if page_num is not None:
            chunk["page"] = page_num
        chunks.append(chunk)
        idx += 1
        start += step
    return chunks

def _resolve_pdf_path(md_path: Path) -> Path:
    if md_path.suffix.lower() == ".pdf":
        return md_path
    pdf_same_dir = md_path.with_suffix(".pdf")
    if pdf_same_dir.exists():
        return pdf_same_dir
    public_docs = _PKG_ROOT / "public" / "docs"
    exact_pdf = public_docs / f"{md_path.stem}.pdf"
    if exact_pdf.exists():
        return exact_pdf
    if "_-_" in md_path.stem:
        base_stem = md_path.stem.split("_-_")[0]
        prefix_pdf = public_docs / f"{base_stem}.pdf"
        if prefix_pdf.exists():
            return prefix_pdf
    return md_path

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
    
    pdf_path = _resolve_pdf_path(md_path)
    pdf_pages: list[str] = []
    same_stem_pdf = (
        pdf_path.suffix.lower() == ".pdf"
        and pdf_path.exists()
        and pdf_path.stem == md_path.stem
    )
    if same_stem_pdf:
        try:
            with fitz.open(str(pdf_path)) as doc:
                for page in doc:
                    pdf_pages.append(page.get_text().replace("\n", " "))
        except Exception as e:
            logger.warning(f"Could not read PDF for {md_path.name}: {e}")
            
    
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
    current_pdf_page = 0
    
    for line in filtered_lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            current_header = stripped.lstrip("#").strip().upper()
        
        page_num: int | None = None
        if pdf_pages and stripped:
            if len(stripped) > 20:
                found = False
                for p_idx in range(current_pdf_page, min(current_pdf_page + 5, len(pdf_pages))):
                    if stripped in pdf_pages[p_idx]:
                        current_pdf_page = p_idx
                        page_num = current_pdf_page + 1
                        found = True
                        break
                if not found:
                    page_num = current_pdf_page + 1
            else:
                page_num = current_pdf_page + 1
        elif pdf_pages:
            page_num = current_pdf_page + 1

        # Agreement Navigator requires 'Fast' tokenizers for reliable character-offset mapping.
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

    try:
        rel_path = str(md_path.relative_to(DATA_DIR))
    except ValueError:
        rel_path = md_path.name
    return chunk_text(filtered_content, token_metadata, source_name, path=rel_path)

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
            
            # Agreement Navigator requires 'Fast' tokenizers for reliable character-offset mapping.
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
            
        try:
            rel_path = str(pdf_path.relative_to(DATA_DIR))
        except ValueError:
            rel_path = pdf_path.name
        return chunk_text(full_text, token_metadata, source_name, path=rel_path)
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

@lru_cache(maxsize=128)
def get_document_tier_weight(source_name: str, path: str = "") -> float:
    """
    Determine the retrieval boost weight for a document based on its tier.
    - Tier 1: 20th Main Agreement & Standards of Conduct (default 1.2)
    - Tier 3: Statutory and general secondary resources (default 0.8)
    - Tier 2: Core agreements, jurisprudence, forms, etc. (default 1.0)
    """
    # Normalise input paths and source names for robust matching
    path_lower = path.lower().replace("\\", "/")
    source_lower = source_name.lower()

    # Tier 1 checks:
    # 1. Gov BC Standards of Conduct (either via relative path or source name)
    # 2. BCGEU 20th Main Agreement (either via relative path or source name)
    is_standards_of_conduct = (
        "standards_of_conduct" in path_lower 
        or "standards of conduct" in source_lower
    )
    is_main_agreement = (
        "20th_main_agreement" in path_lower
        or "20th main agreement" in source_lower
    )

    if is_standards_of_conduct or is_main_agreement:
        return TIER1_BOOST

    # Tier 3 checks:
    # 1. Statutory regulations: '02_statutory/' folder or source names
    # 2. Other general resources: '03_resources/' folder
    is_statutory = (
        "02_statutory" in path_lower 
        or "statutory" in path_lower
        or "ohs_regulation" in path_lower
        or "workers_compensation" in path_lower
    )
    is_general_resource = (
        "03_resources" in path_lower
        and not is_standards_of_conduct
    )

    if is_statutory or is_general_resource:
        return TIER3_BOOST

    # Tier 2: Default
    return 1.0

def search_index(index: "faiss.IndexFlatIP", chunks: list[dict], query: str, top_k: int | None = None) -> list[dict]:
    if top_k is None:
        top_k = SIMILARITY_TOP_K
    
    # Reuse the batch-based search implementation with score weighting
    results = search_index_batch(index, chunks, [query], [top_k])
    return results[0] if results else []

def search_index_batch(index: "faiss.IndexFlatIP", chunks: list[dict], queries: list[str], top_ks: list[int]) -> list[list[dict]]:
    """
    Search multiple queries in a single embedding pass to reduce CPU overhead.
    Uses FAISS's native batch search for maximum efficiency (#323).
    Applies document-tier score boosting to retrieve context preferentially.
    """
    import faiss
    import numpy as np
    
    if not queries:
        return []

    # 1. Batch Embed (already optimized in SentenceTransformer)
    query_vecs = embed_texts(queries)
    faiss.normalize_L2(query_vecs)
    
    # 2. Determine a larger candidate pool size for re-ranking
    # We retrieve more candidates from FAISS so that top-tier matches 
    # can bubble up through re-ranking.
    max_k = max(top_ks)
    candidate_k = max(max_k * 3, 50)
    # Ensure candidate_k does not exceed total indexed chunks
    candidate_k = min(candidate_k, len(chunks))
    if candidate_k <= 0:
        return [[] for _ in queries]
    
    # 3. Batch Search (FAISS native multi-vector search)
    scores, all_indices = index.search(query_vecs, candidate_k)
    
    results = []
    for i, indices in enumerate(all_indices):
        query_scores = scores[i]
        k = top_ks[i]
        
        # Build candidate chunks with original and weighted scores
        candidates = []
        for idx_in_search, chunk_idx in enumerate(indices):
            if 0 <= chunk_idx < len(chunks):
                chunk = chunks[chunk_idx]
                orig_score = float(query_scores[idx_in_search])
                
                # Determine weight based on tier
                weight = get_document_tier_weight(
                    chunk.get("source", ""),
                    chunk.get("path", "")
                )
                weighted_score = orig_score * weight
                
                candidates.append((chunk, weighted_score))
        
        # Sort candidates by weighted score in descending order
        candidates.sort(key=lambda item: item[1], reverse=True)
        
        # Extract the top-k chunks
        top_k_chunks = [item[0] for item in candidates[:k]]
        results.append(top_k_chunks)
        
    return results

def _index_config() -> dict[str, Any]:
    return {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embed_model": EMBED_MODEL,
        "tiering_version": "v2",
    }


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(65536):
            hasher.update(block)
    return hasher.hexdigest()


def _embedding_cache_file(content_hash: str) -> Path:
    return CACHE_DIR / "embeddings" / f"{content_hash}.npy"


def _chunk_cache_file(content_hash: str) -> Path:
    return CACHE_DIR / "chunks" / f"{content_hash}.json"


def _manifest_file_entry(content_hash: str) -> dict[str, str]:
    """Per-file manifest record. Cache paths are present only when both files exist."""
    entry = {"content_hash": content_hash}
    embeddings = _embedding_cache_file(content_hash)
    chunks = _chunk_cache_file(content_hash)
    if embeddings.is_file() and chunks.is_file():
        entry["embeddings"] = f"embeddings/{content_hash}.npy"
        entry["chunks"] = f"chunks/{content_hash}.json"
    return entry


def _load_document_cache(content_hash: str) -> "tuple[np.ndarray, list[dict]] | None":
    """Load cached float32 embeddings and chunks for one document, or None on miss."""
    import numpy as np
    npy_path = _embedding_cache_file(content_hash)
    json_path = _chunk_cache_file(content_hash)
    if not npy_path.is_file() or not json_path.is_file():
        return None
    try:
        vectors = np.load(npy_path, allow_pickle=False)
        with open(json_path, encoding="utf-8") as f:
            chunks = json.load(f)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        logger.warning(f"[build] Ignoring unreadable cache for {content_hash}: {e}")
        return None
    if not isinstance(chunks, list) or len(chunks) == 0:
        return None
    if getattr(vectors, "ndim", None) != 2:
        return None
    if vectors.shape != (len(chunks), EMBED_DIM):
        logger.warning(
            f"[build] Cache shape {getattr(vectors, 'shape', None)} does not match "
            f"{len(chunks)} chunks of dimension {EMBED_DIM} for {content_hash}"
        )
        return None
    return np.ascontiguousarray(vectors, dtype=np.float32), chunks


def _save_document_cache(content_hash: str, vectors: "np.ndarray", chunks: list[dict]) -> None:
    import numpy as np
    npy_path = _embedding_cache_file(content_hash)
    json_path = _chunk_cache_file(content_hash)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, np.ascontiguousarray(vectors, dtype=np.float32))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)


def build_index_from_vectors(vectors: "np.ndarray") -> "faiss.IndexFlatIP":
    """L2-normalize stacked embeddings and build an inner-product FAISS index."""
    import faiss
    import numpy as np
    matrix = np.ascontiguousarray(vectors, dtype=np.float32)
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(matrix)
    return index


def build_index(chunks: list[dict]) -> "faiss.IndexFlatIP":
    import numpy as np
    texts = [c["text"] for c in chunks]
    logger.info(f"[index] Indexing {len(texts)} chunks (Batched for memory safety)...")
    
    # 2026-05-15: Optimization to prevent OOM on 2GB limits
    batch_size = 64
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        all_vectors.append(embed_texts(batch))
        if (i // batch_size) % 5 == 0:
            logger.info(f"[index] Progress: {min(i + batch_size, len(texts))}/{len(texts)} chunks embedded...")
            
    vectors = np.vstack(all_vectors)
    logger.info("[index] Embeddings complete. Normalizing...")
    return build_index_from_vectors(vectors)

def save_index(index: "faiss.IndexFlatIP", chunks: list[dict]) -> None:
    import faiss
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    logger.info(f"[index] Saved index to {INDEX_PATH}")

def _read_stored_manifest() -> dict | None:
    if not MANIFEST_PATH.exists():
        return None
    try:
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            loaded = json.load(f)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _manifest_for(file_hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "config": _index_config(),
        "files": {
            rel_key: _manifest_file_entry(digest)
            for rel_key, digest in file_hashes.items()
        },
    }


def _chunks_for_source(source_file: Path, strict: bool) -> list[dict]:
    if source_file.suffix.lower() == ".md":
        return load_md_chunks(source_file)
    if source_file.suffix.lower() == ".pdf":
        return load_pdf_chunks(source_file, strict=strict)
    return []


def build_index_from_sources(force: bool = False) -> tuple[Any, Any] | tuple[None, None]:
    """
    Main entry point for index creation.
    NOTE: Manifest hashing is performed here to determine if a re-index is needed.
    Unchanged documents reuse cache/embeddings/<content_hash>.npy and
    cache/chunks/<content_hash>.json. Only new or modified documents are encoded.
    In container environments, this is typically called during the build stage.
    """
    import numpy as np

    all_files = _get_rag_source_files()
    if not all_files:
        logger.warning("[build] No source files found!")
        return None, None

    file_hashes: dict[str, str] = {}
    for source_file in all_files:
        # Use relative path to avoid clashes with duplicate names in subdirs
        rel_key = str(source_file.relative_to(DATA_DIR))
        file_hashes[rel_key] = _hash_file(source_file)

    stored_manifest = _read_stored_manifest()
    current_manifest = _manifest_for(file_hashes)
    if (
        not force
        and stored_manifest == current_manifest
        and INDEX_PATH.exists()
        and CHUNKS_PATH.exists()
    ):
        logger.info("[build] Smart Refresh: No changes detected in sources or config. Skipping build.")
        return load_precomputed_index()

    logger.info(f"[build] Change detected or forced rebuild. Indexing {len(all_files)} files...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    config_ok = (
        isinstance(stored_manifest, dict)
        and stored_manifest.get("config") == _index_config()
    )
    strict = os.getenv("AGNAV_STRICT_BUILD", "false").lower() == "true"
    chunks: list[dict] = []
    vector_blocks: list[Any] = []
    failed_files: list[str] = []
    for source_file in all_files:
        rel_key = str(source_file.relative_to(DATA_DIR))
        digest = file_hashes[rel_key]
        cached = _load_document_cache(digest) if config_ok else None
        if cached is not None:
            doc_vectors, doc_chunks = cached
            logger.info(f"[build] Cache hit for {rel_key} ({len(doc_chunks)} chunks)")
            vector_blocks.append(doc_vectors)
            chunks.extend(doc_chunks)
            continue
        try:
            doc_chunks = _chunks_for_source(source_file, strict)
        except Exception as e:
            logger.error(f"[build] ERROR: Failed to index {source_file.name}: {e}")
            failed_files.append(source_file.name)
            if strict:
                raise
            continue
        if not doc_chunks:
            continue
        logger.info(f"[build] Encoding {rel_key} ({len(doc_chunks)} chunks)")
        doc_vectors = embed_texts([c["text"] for c in doc_chunks])
        doc_vectors = np.ascontiguousarray(doc_vectors, dtype=np.float32)
        if doc_vectors.shape != (len(doc_chunks), EMBED_DIM):
            raise RuntimeError(
                f"Embedding shape {doc_vectors.shape} does not match "
                f"{len(doc_chunks)} chunks of dimension {EMBED_DIM} for {rel_key}"
            )
        _save_document_cache(digest, doc_vectors, doc_chunks)
        vector_blocks.append(doc_vectors)
        chunks.extend(doc_chunks)

    # Save integrity report
    integrity_data = {
        "timestamp": time.time(),
        "failed_files": failed_files,
        "success_count": len(all_files) - len(failed_files),
        "total_count": len(all_files)
    }
    try:
        with open(INTEGRITY_PATH, "w") as f:
            json.dump(integrity_data, f, indent=2)
    except PermissionError as e:
        logger.warning(f"[build] Could not write integrity manifest: {e}")

    if failed_files and strict:
        raise FileIntegrityError(f"Build failed due to integrity errors in: {', '.join(failed_files)}")
    
    if not chunks:
        logger.error("[build] No chunks found in source files!")
        return None, None

    index = build_index_from_vectors(np.vstack(vector_blocks))
    save_index(index, chunks)
    # Cache files exist now, so manifest entries include their relative paths.
    current_manifest = _manifest_for(file_hashes)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(current_manifest, f, indent=2)
    return index, chunks

def load_precomputed_index() -> tuple[Any, Any] | tuple[None, None]:
    # Security: proactively delete legacy .pkl file (RCE risk).
    legacy_pkl = CACHE_DIR / "chunks.pkl"
    if legacy_pkl.exists():
        legacy_pkl.unlink(missing_ok=True)
        logger.warning("[startup] Deleted legacy chunks.pkl (security).")

    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        return None, None
    logger.info(f"[startup] Loading pre-computed index from {INDEX_PATH}...")
    import faiss
    index = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
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
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    if INDEX_PATH.exists() and CHUNKS_PATH.exists():
        return
    base = _GITHUB_RAW_BASE
    urls = {}
    if not INDEX_PATH.exists():
        urls[INDEX_PATH] = f"{base}/data/cache/index.faiss"
    if not CHUNKS_PATH.exists():
        urls[CHUNKS_PATH] = f"{base}/data/cache/chunks.json"
    for dest_path, url in urls.items():
        logger.info(f"[fetch] Downloading {dest_path.name} from {url}...")
        try:
            urllib.request.urlretrieve(url, dest_path)
            logger.info(f"[fetch] Saved {dest_path}")
        except (urllib.error.URLError, OSError) as e:
            # Fallback attempt from legacy path if remote repo still uses .pdf_cache
            if "/data/cache/" in url:
                legacy_url = url.replace("/data/cache/", "/.pdf_cache/")
                try:
                    urllib.request.urlretrieve(legacy_url, dest_path)
                    logger.info(f"[fetch] Saved {dest_path} from legacy URL")
                    continue
                except Exception:
                    pass
            logger.warning(f"[fetch] Warning: could not fetch {dest_path.name}: {e}. Will build index from source.")
