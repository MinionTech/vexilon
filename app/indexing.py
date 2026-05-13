import os
import json
import time
import hashlib
import fitz
import logging
import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer

# ─── Configuration ───────────────────────────────────────────────────────────
_PKG_ROOT = Path(__file__).parent
PDF_CACHE_DIR = Path(os.getenv("AGNAV_CACHE_DIR", _PKG_ROOT / ".pdf_cache"))
LABOUR_LAW_DIR = Path(os.getenv("AGNAV_DATA_DIR", _PKG_ROOT / "data/labour_law"))
INDEX_PATH = PDF_CACHE_DIR / "index.faiss"
CHUNKS_PATH = PDF_CACHE_DIR / "chunks.json"
MANIFEST_PATH = PDF_CACHE_DIR / "manifest.json"

# Models
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
MAX_EMBED_TOKENS = int(os.getenv("AGNAV_MAX_EMBED_TOKENS", 512))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", 40))

_embed_model: Any = None

def get_embed_model() -> "SentenceTransformer":
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL, device="cpu")
        _embed_model.max_seq_length = MAX_EMBED_TOKENS
    return _embed_model

def search_index(index: "faiss.IndexFlatIP", chunks: list[dict], query: str, top_k: int = 10) -> list[dict]:
    model = get_embed_model()
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        if idx != -1:
            results.append(chunks[idx])
    return results

class IndexManager:
    def __init__(self):
        self.pdf_dir = LABOUR_LAW_DIR
        self.cache_dir = PDF_CACHE_DIR
        self.index = None
        self.chunks = []

    async def ensure_index(self):
        # Mocking for now to avoid long build times, but in reality would load/build
        import faiss
        if INDEX_PATH.exists() and CHUNKS_PATH.exists():
            self.index = faiss.read_index(str(INDEX_PATH))
            with open(CHUNKS_PATH, "r") as f:
                self.chunks = json.load(f)
            return {"status": "loaded"}
        else:
            # Empty index fallback
            self.index = faiss.IndexFlatIP(EMBED_DIM)
            self.chunks = []
            return {"status": "empty_fallback", "failed_files": []}

def sanitize_input(text: str) -> tuple[str, bool]:
    # Basic sanitization
    if not text: return "", False
    bad_patterns = ["<script>", "DROP TABLE", "DELETE FROM"]
    for p in bad_patterns:
        if p in text.upper():
            return text, True
    return text.strip(), False

async def get_multi_perspective_context(query: str, history: list[dict]):
    # In a real app, this would use an LLM to generate multiple queries
    return [query], "Context from multi-perspective search.", []

async def rag_review_stream(query: str, history: list[dict], persona: str):
    # Simulated RAG stream
    yield f"Analyzing query '{query}' as **{persona}**...\n\n"
    await asyncio.sleep(0.5)
    yield "Found relevant sections in the 19th Main Agreement regarding discipline and just cause.\n"
    await asyncio.sleep(0.5)
    yield "According to Article 10, the employer must provide written reasons for discipline within 10 days."
