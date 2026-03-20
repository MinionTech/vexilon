"""
app.py — BCGEU Steward Assistant
--------------------------------------------
Tech stack:
  - pypdf                : PDF → pages with page number preservation
  - sentence-transformers: Local CPU embeddings (all-MiniLM-L6-v2, no API key)
  - FAISS                : In-memory vector index (no server process)
  - Anthropic            : Claude (claude-haiku-4-5-20251001) for responses
  - Gradio 6             : Web UI at http://localhost:7860

Quick start:
  1. export ANTHROPIC_API_KEY=sk-ant-...
  2. uv sync
  3. uv run python app.py

Index pre-computation (run once after updating the PDF):
  python -c "from app import startup; startup(force_rebuild=True)"
  # Saves pdf_cache/index.faiss + pdf_cache/chunks.json for fast cold starts.
"""

# ─── Standard Library ────────────────────────────────────────────────────────
import sys
print("[boot] Python started, importing stdlib...", flush=True)
import json
import os
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

# Ensure the HuggingFace model cache is writable and persistent.
# Inside the container (WORKDIR /app), this resolves to /app/hf_cache.
# Locally, it resolves to ./hf_cache in the repo root.
if not os.getenv("HF_HOME"):
    os.environ["HF_HOME"] = str(Path("./hf_cache").absolute())

# ─── Third-party: Deferred Imports ───────────────────────────────────────────
# (numpy, pypdf, anthropic, faiss, sentence_transformers, gradio)
# are imported inside functions to keep startup and test-loading fast.
print("[boot] All boilerplate complete.", flush=True)

# ─── Configuration ───────────────────────────────────────────────────────────
PDF_CACHE_DIR = Path("./pdf_cache")
LABOUR_LAW_DIR = Path("./data/labour_law")
INDEX_PATH = PDF_CACHE_DIR / "index.faiss"
CHUNKS_PATH = PDF_CACHE_DIR / "chunks.json"
MANIFEST_PATH = PDF_CACHE_DIR / "manifest.json"

# Public GitHub raw URL base for labour_law PDFs.
# Used for folder/file links in the UI.
GITHUB_LABOUR_LAW_URL = (
    "https://github.com/DerekRoberts/vexilon/tree/main/data/labour_law"
)

# Raw URL base for downloading pre-computed index from GitHub.
# Used by _fetch_pdf_cache_if_missing() for HF Spaces bootstrap.
_GITHUB_RAW_BASE = "https://raw.githubusercontent.com/DerekRoberts/vexilon/main"

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
CONDENSE_MODEL = os.getenv("CONDENSE_MODEL", "claude-haiku-4-5-20251001")
# Brain: Local Embeddings (Search) + Cloud LLM (Claude)
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5") # 512-token window
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 450))       # Sized for BGE-small
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100)) 
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", 40)) # More context depth

# Memory / Context Condensation
CONDENSE_QUERY_HISTORY_TURNS = int(os.getenv("CONDENSE_QUERY_HISTORY_TURNS", 3))
CONDENSE_QUERY_CONTENT_MAX_LEN = int(os.getenv("CONDENSE_QUERY_CONTENT_MAX_LEN", 200))

def get_vexilon_info():
    """
    1. Get Version (Priority: Env Var -> Baked File -> Fallback)
    2. Get Python and OS metadata
    3. Print a beautiful startup banner
    """
    import platform
    
    # Priority 1: Env Var
    version = os.getenv("VEXILON_VERSION")
    source = "External/CI"
    
    if not version:
        # Priority 2: Baked-in Build File
        try:
            with open("/app/build_version.txt", "r") as f:
                version = f.read().strip()
                source = "Local Build"
        except FileNotFoundError:
            # Priority 3: Fallback (Never dev!)
            version = "unspecified-local"
            source = "fallback"

    py_ver = sys.version.split()[0]
    os_info = platform.system()

    # The Banner
    print("=" * 50)
    print(f" VEXILON VERSION : {version} ({source})")
    print(f" PYTHON VERSION  : {py_ver}")
    print(f" RUNTIME OS      : {os_info}")
    print("=" * 50, flush=True)

    return {
        "ver": version,
        "src": source,
        "py": py_ver,
        "os": os_info
    }


# Initialise version and logging at imports
_info = get_vexilon_info()
VEXILON_VERSION = _info["ver"]
VEXILON_USERNAME = os.getenv("VEXILON_USERNAME", "admin")
VEXILON_PASSWORD = os.getenv("VEXILON_PASSWORD")

# Embedding dimension for all-MiniLM-L6-v2
EMBED_DIM = 384

# ─── Typing ──────────────────────────────────────────────────────────────────
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    import faiss
    import gradio as gr

# ─── Clients ─────────────────────────────────────────────────────────────────
_embed_model: "SentenceTransformer | None" = None
_anthropic_client: "anthropic.AsyncAnthropic | None" = None


def get_embed_model() -> "SentenceTransformer":
    global _embed_model
    if _embed_model is None:
        print(f"[embed] Loading local embedding model '{EMBED_MODEL}'…")
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL)
        # Increase limits to handle full-page tokenization mapping without warnings.
        # We manually chunk to 256 tokens later, so this is just to keep the logs clean.
        _embed_model.max_seq_length = 100000
        if hasattr(_embed_model, "tokenizer"):
            _embed_model.tokenizer.model_max_length = 100000
        print("[embed] Embedding model ready.")
    return _embed_model


def get_anthropic() -> "anthropic.AsyncAnthropic":
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        # Reads ANTHROPIC_API_KEY from environment automatically; raises AuthenticationError if missing
        _anthropic_client = anthropic.AsyncAnthropic()
    return _anthropic_client
def get_knowledge_manifest() -> str:
    """
    Dynamically scan the labour_law directory and build a formatted list for the system prompt.
    Files follow the naming convention: [Index]_[Category]_[Title].pdf
    Example: 1_Primary_BCGEU 19th Main Agreement.pdf
    """
    if not LABOUR_LAW_DIR.exists():
        return "No documents available."

    pdfs = sorted(LABOUR_LAW_DIR.glob("*.pdf"))
    if not pdfs:
        return "No documents available."

    lines = []
    for pdf in pdfs:
        stem = pdf.stem
        # Try to parse our convention: Index_Category_Title
        parts = stem.split("_", 2)
        if len(parts) == 3:
            idx, cat, title = parts
            lines.append(f"{idx}. {title} ({cat})")
        else:
            # Fallback for non-conforming filenames
            lines.append(f"- {stem.replace('_', ' ').title()}")

    return "\n".join(lines)


# ─── System Prompt ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Vexilon, a highly authoritative professional assistant for BCGEU union stewards.

--- HOW YOUR SEARCH WORKS ---
Your library contains the COMPLETE, full text of these documents:
{manifest}

IMPORTANT: For each question you receive, a semantic search retrieves the most relevant \
excerpts from this library. You see a SUBSET of the library per query — not the whole thing. \
Content that does not appear in the excerpts below may still exist in the library; it simply \
was not retrieved for THIS particular question. \
NEVER claim that an Article, section, or document is "missing" or "not in my documents" \
just because it is not in the current excerpts. Instead, say: \
"The specific text was not retrieved for this search. Try asking about [topic] directly."
--------------------------

Rules you must follow without exception:

1. ANSWER FROM EXCERPTS ONLY: Base your answer strictly on the excerpts provided below. \
If the excerpts contain only a reference to a section (e.g., "See Section 10.4") but not the \
actual text, say the specific language was not retrieved for this search and suggest the user \
ask about that section directly. NEVER guess or fabricate contract language.
2. Every claim must be supported by a verbatim quote from the provided excerpts, formatted as a markdown blockquote (> "...") followed by its citation: — [Document Name], Article/Section [X], [Title if available], p. [N]
3. Plain-language explanation comes BEFORE the verbatim quote, not after.
4. ALWAYS prioritize and lead with the Collective Agreement (Main Agreement) as the primary authority.
5. If consecutive sections appear with a gap (e.g., you see 10.1 and 10.3 but not 10.2), note the gap and suggest the user ask about the missing section specifically.
6. Do not predict outcomes or give legal opinions.
7. Tone: professional, forensic, and confident. Do NOT be apologetic about retrieval limitations — the library is comprehensive; the search just needs more specific queries.
8. Cite every relevant clause separately.
9. Maintain conversational continuity. Use the previous conversation context and the provided excerpts.
10. If the search results are contradictory or unclear, flag this ambiguity to the user immediately.
11. Every chunk is tagged with its Article or Appendix name for context.
12. If asked about your capabilities, knowledge gaps, or what documents you have: describe the library manifest above. Do NOT audit or list "missing" articles — you have the complete text of everything listed above.

Response format:

[Plain-language explanation]

> "[Verbatim quote]"
— [Document Name], Article/Section [X], p. [N]
"""

# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(full_text: str, token_data: list[tuple[int, int, int, str]], source_name: str) -> list[dict]:
    """
    Split *full_text* into overlapping token-based chunks across the whole document.
    Uses 'token_data' [(char_start, char_end, page_num, header)] to preserve metadata.
    Returns list of dicts: {text, page, source, header, chunk_index}.
    """
    chunks = []
    if not token_data:
        return chunks
        
    idx = 0
    start = 0
    while start < len(token_data):
        end = min(start + CHUNK_SIZE, len(token_data))
        
        # Get metadata from the first token of the chunk
        char_start, _, page_num, header = token_data[start]
        # End Char index is from the last token of the chunk
        _, char_end, _, _ = token_data[end - 1]
        
        # Contextual Breadcrumb: Prepend source + header 
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
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ─── PDF Loader ───────────────────────────────────────────────────────────────

def _is_toc_or_index_page(page_text: str) -> bool:
    """
    Detect whether a page is a Table of Contents or alphabetical Index page.

    These navigational pages mention every article/clause by name but contain
    no substantive content. Indexing them causes TOC entries to dominate
    semantic search results, drowning out actual contract text.

    Returns True if the page appears to be TOC/index content.
    """
    import re
    lines = [l.strip() for l in page_text.split("\n") if l.strip()]
    if not lines:
        return False

    # Heuristic 1: 3+ dot-leader lines (..........) strongly indicates a TOC page
    dot_leader_count = sum(1 for l in lines if l.count(".") >= 8 and ".." in l)
    if dot_leader_count >= 3:
        return True

    # Heuristic 2: >40% of lines match index-style pattern "Some Text ... NN"
    # e.g. "Abandonment of Position, 10.10 ........ 23"
    index_line_re = re.compile(r".{10,}\.\s*\d{1,3}\s*$")
    index_count = sum(1 for l in lines if index_line_re.search(l))
    if len(lines) >= 5 and index_count / len(lines) > 0.4:
        return True

    return False


def _clean_page_text(page_text: str) -> str:
    """
    Remove noise artifacts injected by web-based PDF extraction.

    BC statute PDFs (ESA, Human Rights Code, Labour Relations Code) were
    exported from bclaws.gov.bc.ca and contain repeated URL lines and
    date-stamps that waste ~30 tokens per chunk and dilute embedding quality.
    """
    import re
    # Remove bclaws.gov.bc.ca URL lines
    page_text = re.sub(
        r"https?://www\.bclaws\.gov\.bc\.ca/\S*",
        "",
        page_text,
    )
    # Remove date/time stamps from web-to-PDF artifacts, e.g.:
    # "17/03/2026, 08:44 Employment Standards Act"
    page_text = re.sub(
        r"\d{2}/\d{2}/\d{4},?\s*\d{2}:\d{2}\s+[A-Z][^\n]*",
        "",
        page_text,
    )
    # Collapse runs of 3+ blank lines down to a single blank line
    page_text = re.sub(r"\n{3,}", "\n\n", page_text)
    return page_text.strip()


def load_pdf_chunks(pdf_path: Path) -> list[dict]:
    """
    Parse the PDF at *pdf_path* into one continuous stream before chunking.
    This bridges page boundaries so sentences that span pages aren't decapitated.

    Navigational pages (Table of Contents, alphabetical Index) are skipped
    so they don't contaminate semantic search results.  URL artifacts from
    web-extracted statute PDFs are also stripped before embedding.
    """
    from pypdf import PdfReader
    import re
    
    reader = PdfReader(str(pdf_path))
    # Parse source name from filenames like "1_Primary_Agreement Name" -> "Agreement Name"
    stem = pdf_path.stem
    parts = stem.split("_", 2)
    source_name = parts[2] if len(parts) == 3 else stem.replace("_", " ").title()
    print(f"[loader] Parsing '{source_name}' ({len(reader.pages)} pages)…")
    
    tokenizer = get_embed_model().tokenizer
    full_text = ""
    token_metadata = [] # List of (char_start, char_end, page_num, header)
    
    current_header = ""
    header_pattern = re.compile(r"^\s*(ARTICLE|APPENDIX)\s+(\d+|[A-Z]+)", re.IGNORECASE)
    skipped_pages = 0

    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        page_text = page.extract_text() or ""
        if not page_text.strip():
            continue

        # Skip pure navigational pages (TOC / alphabetical Index).
        # These mention every article by name but add no substantive content;
        # indexing them causes TOC entries to crowd out real contract text in
        # semantic search results.
        if _is_toc_or_index_page(page_text):
            skipped_pages += 1
            continue

        # Strip web-extraction artifacts (URLs, timestamps) from statute PDFs.
        page_text = _clean_page_text(page_text)
        if not page_text.strip():
            continue
            
        # Update breadcrumb header context
        # Look through more lines - headers can appear anywhere on the page due to
        # complex PDF layouts (two-column, footnotes, etc.)
        page_lines = page_text.split("\n")
        lines_to_check = min(50, len(page_lines))
        for line in page_lines[:lines_to_check]:
            # Skip TOC-style entries and page numbers
            if ".........." in line or (line.strip().endswith(".") and re.search(r"\d+$", line.strip())):
                continue
            match = header_pattern.search(line)
            if match:
                current_header = match.group(0).strip().upper()
                break # Usually one primary header per page

        # Track offsets in the global full_text
        page_offset = len(full_text)
        full_text += page_text + "\n"
        
        # Tokenize this page and record metadata for every token
        encoding = tokenizer(page_text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        for start, end in encoding.offset_mapping:
            token_metadata.append((
                page_offset + start,
                page_offset + end,
                page_num,
                current_header
            ))

    if skipped_pages:
        print(f"[loader] Skipped {skipped_pages} navigational pages (TOC/index) in '{source_name}'.")
            
    return chunk_text(full_text, token_metadata, source_name)


# ─── FAISS Index ──────────────────────────────────────────────────────────────
def embed_texts(texts: list[str]) -> "np.ndarray":
    """Embed a list of texts using the local sentence-transformers model. Returns (N, EMBED_DIM) float32 array."""
    import numpy as np
    model = get_embed_model()
    # encode() handles batching internally; show_progress_bar=False keeps logs clean
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.astype(np.float32)


def build_index(chunks: list[dict]) -> "faiss.IndexFlatIP":
    """
    Embed all chunks and build a FAISS inner-product index.
    Vectors are L2-normalised so inner product == cosine similarity.
    """
    import faiss
    texts = [c["text"] for c in chunks]
    print(f"[index] Embedding {len(texts)} chunks locally (may take 30–90 s on CPU)…")
    t0 = time.time()
    vectors = embed_texts(texts)
    print(f"[index] Embeddings complete in {time.time()-t0:.1f}s")
    # L2-normalise for cosine similarity via inner product
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)
    print(f"[index] FAISS index built — {index.ntotal} vectors")
    return index


def search_index(
    index: "faiss.IndexFlatIP",
    chunks: list[dict],
    query: str,
    top_k: int = SIMILARITY_TOP_K,
) -> list[dict]:
    """Return the top-k most similar chunks for *query*."""
    import faiss
    query_vec = embed_texts([query])  # (1, EMBED_DIM)
    faiss.normalize_L2(query_vec)
    _scores, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


# ─── RAG App State (module-level, built at startup) ──────────────────────────
_chunks: list[dict] = []
_index: "faiss.IndexFlatIP | None" = None


def save_index(index: "faiss.IndexFlatIP", chunks: list[dict]) -> None:
    """Persist the FAISS index and chunk metadata to pdf_cache/ for fast cold starts."""
    import faiss
    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    print(f"[index] Saved index → {INDEX_PATH} and chunks → {CHUNKS_PATH}")


def load_precomputed_index() -> tuple["faiss.IndexFlatIP", list[dict]] | tuple[None, None]:
    """
    Load a pre-computed FAISS index and chunks from disk if both exist.
    Returns (index, chunks) on success, (None, None) if files are missing.
    """
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        return None, None
    print(f"[startup] Loading pre-computed index from {INDEX_PATH}…")
    import faiss
    index = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"[startup] Pre-computed index loaded — {index.ntotal} vectors, {len(chunks)} chunks.")
    return index, chunks


def _fetch_pdf_cache_if_missing() -> None:
    """
    Download pre-computed index and chunks from GitHub raw if not present locally.
    This enables fast cold starts on HuggingFace Spaces where PDFs aren't bundled.
    """
    import urllib.request

    # Ensure cache directory exists
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Check if both files already exist
    if INDEX_PATH.exists() and CHUNKS_PATH.exists():
        return

    # Build URLs for the missing files
    base = _GITHUB_RAW_BASE
    urls = {}
    if not INDEX_PATH.exists():
        urls[INDEX_PATH] = f"{base}/pdf_cache/index.faiss"
    if not CHUNKS_PATH.exists():
        urls[CHUNKS_PATH] = f"{base}/pdf_cache/chunks.json"

    # Download missing files
    for dest_path, url in urls.items():
        print(f"[fetch] Downloading {dest_path.name} from {url}…")
        urllib.request.urlretrieve(url, dest_path)
        print(f"[fetch] Saved {dest_path}")


def build_index_from_pdfs(force: bool = False) -> None:
    """
    Parse all PDFs in LABOUR_LAW_DIR, embed them, and write the pre-built index to
    pdf_cache/index.faiss + pdf_cache/chunks.json.

    Args:
        force (bool): If True, ignores the existing manifest and rebuilds the 
                     index from scratch. Defaults to False.

    SMART REFRESH: This function generates a manifest.json containing MD5 hashes
    of all PDF files. If the current files match the stored manifest (and the 
    index exists), the expensive embedding process is skipped.
    """
    import hashlib
    global _chunks, _index
    
    if not LABOUR_LAW_DIR.exists():
        print(f"[build] {LABOUR_LAW_DIR} does not exist — nothing to index.")
        return

    pdf_files = sorted(LABOUR_LAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print("[build] No PDF files found to index!")
        return

    # Calculate hashes for the current PDF set (chunked to handle large files)
    current_manifest = {}
    for pdf in pdf_files:
        hasher = hashlib.sha256()
        with open(pdf, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        current_manifest[pdf.name] = hasher.hexdigest()

    # Check against stored manifest
    if not force and MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r") as f:
                stored_manifest = json.load(f)
            if stored_manifest == current_manifest and INDEX_PATH.exists() and CHUNKS_PATH.exists():
                print("[build] Smart Refresh: No changes detected in data/labour_law/. Skipping indexing.")
                return
        except Exception as e:
            print(f"[build] Failed to read manifest: {e}. Rebuilding anyway.")

    # Ensure cache directory exists before writing
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[build] Scanning for PDFs in {LABOUR_LAW_DIR}…")
    _chunks = []
    for pdf in pdf_files:
        _chunks.extend(load_pdf_chunks(pdf))

    num_chunks = len(_chunks)
    print(f"[build] Total {num_chunks} chunks loaded from {len(pdf_files)} files.")
    _index = build_index(_chunks)
    save_index(_index, _chunks)
    
    # Save the new manifest
    with open(MANIFEST_PATH, "w") as f:
        json.dump(current_manifest, f, indent=2)
    print(f"[build] Index and manifest written to {PDF_CACHE_DIR}.")
    print("[build] Indexing complete.")


def startup(force_rebuild: bool = False) -> None:
    """
    Initialise the vector index and load document chunks.
    
    If force_rebuild=False (default), we try to load a pre-computed index
    from pdf_cache/. If missing, we fall back to build_index_from_pdfs()
    which embeds documents from scratch.
    """
    global _chunks, _index
    get_anthropic()  # Ping early to catch missing ANTHROPIC_API_KEY

    # Try to fetch pre-computed index from GitHub if not present locally
    # (Needed for HuggingFace Spaces where PDFs aren't bundled)
    _fetch_pdf_cache_if_missing()

    if not force_rebuild:
        index, chunks = load_precomputed_index()
        if index is not None and chunks is not None:
            _index = index
            _chunks = chunks
            # Warm the embedding model so the first query isn't slow
            get_embed_model()
            print("[startup] Ready.")
            return

    # ── Slow path: delegate to the API-key-free build function ────────────
    build_index_from_pdfs(force=force_rebuild)
    print("[startup] Ready.")


# ─── RAG Query ────────────────────────────────────────────────────────────────
async def condense_query(message: str, history: list[dict]) -> str:
    """
    Use Claude to condense conversation history and the latest message into a 
    standalone, search-friendly query.
    """
    if not history:
        return message

    client = get_anthropic()
    
    # Simple history formatting
    context_lines = []
    for turn in history[-CONDENSE_QUERY_HISTORY_TURNS:]:  # Uses configured history context
        role = "User" if turn["role"] == "user" else "Assistant"
        # Truncate content for the condensation prompt.
        # In Gradio 6, content can be a string or a list of blocks.
        raw_content = turn["content"]
        if isinstance(raw_content, list):
            # Extract text from message parts
            text_parts = [
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in raw_content
            ]
            raw_content = "".join(text_parts)
        
        # Ensure it is a string before slicing/concatenating
        raw_content = str(raw_content)
        msg_len = CONDENSE_QUERY_CONTENT_MAX_LEN
        content = raw_content[:msg_len] + ("..." if len(raw_content) > msg_len else "")
        context_lines.append(f"{role}: {content}")
    
    context_str = "\n".join(context_lines)

    prompt = (
        "Given the following conversation history and a follow-up question, "
        "rephrase the question to be a standalone search query that captures "
        "the full intent. If the question is about the bot's own capabilities, "
        "make the query specifically about its documentation and manifest. "
        "Only provide the rephrased query, nothing else.\n\n"
        f"History:\n{context_str}\n\n"
        f"Follow-up question: {message}\n\n"
        "Standalone query:"
    )

    try:
        response = await client.messages.create(
            model=CONDENSE_MODEL,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        condensed = response.content[0].text.strip().strip('"')
        print(f"[rag] Condensed query: '{message}' -> '{condensed}'")
        return condensed
    except Exception as exc:
        # We catch generic Exception here since anthropic is deferredly imported
        print(f"[rag] Query condensation failed: {exc}. Using raw message.")
        return message


async def rag_stream(message: str, history: list[dict]) -> AsyncIterator[str]:
    """
    Retrieve relevant chunks, build the prompt, and stream a response from Claude.
    *history* is a list of {"role": ..., "content": ...} dicts (Gradio messages format).
    Yields text chunks as they arrive from the Anthropic streaming API.
    """
    if _index is None:
        yield "⚠️ The index is not ready yet. Please wait a moment and try again."
        return

    # Rewrite query for RAG if there is history
    query = await condense_query(message, history)
    relevant_chunks = search_index(_index, _chunks, query)

    # Build context block from retrieved chunks
    context_parts = []
    for chunk in relevant_chunks:
        context_parts.append(
            f"[Source: {chunk.get('source', 'Unknown')}, Page: {chunk['page']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # Build message list for Claude: prior history + new user message
    messages = []
    for turn in history:
        if turn["role"] in ("user", "assistant"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": message})

    client = get_anthropic()
    try:
        async with client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            # Two cache breakpoints:
            # 1. Static instructions — identical every request; cached once per session.
            # 2. Dynamic excerpts — changes per query; cached separately.
            # This avoids re-caching the instructions block whenever the excerpts change.
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT.format(manifest=get_knowledge_manifest()),
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": (
                        "--- AGREEMENT EXCERPTS ---\n\n"
                        + context
                        + "\n\n--- END EXCERPTS ---"
                    ),
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            messages=messages,
        ) as stream:
            async for text_chunk in stream.text_stream:
                yield text_chunk
            # Log cache effectiveness so we can verify caching is working.
            final = await stream.get_final_message()
            usage = final.usage
            cache_created = usage.cache_creation_input_tokens or 0
            cache_read = usage.cache_read_input_tokens or 0
            print(
                f"[rag] Tokens — input: {usage.input_tokens}, "
                f"cache_create: {cache_created}, cache_read: {cache_read}, "
                f"output: {usage.output_tokens}"
            )
    except Exception as exc:
        yield f"\n\n⚠️ API error: {exc}"


# ─── Gradio UI ────────────────────────────────────────────────────────────────
EXAMPLE_QUESTIONS = [
    "What are the just cause requirements for discipline?",
    "What rights do stewards have in investigation meetings?",
    "What is the nexus test for off-duty conduct?",
    "Does my employer have a social media policy?",
    "What happens if I'm disciplined for off-duty behavior?",
]

# Disclaimer rendered entirely with inline styles so Gradio theme cannot override text colour.
DISCLAIMER_HTML = (
    '<div style="'
    "background-color:#fff8e1;"
    "border-left:4px solid #f59e0b;"
    "color:#7c4a00;"
    "padding:10px 14px;"
    "border-radius:4px;"
    "font-size:0.85rem;"
    "margin-bottom:12px;"
    '">'
    "This project is not affiliated with the BCGEU. AI-generated responses may contain errors."
    "</div>"
)

ATTRIBUTION_HTML = f"""
<div style='text-align: center; color: #6b7280; font-size: 0.85rem; margin-top: 1rem;'>
    <a href='https://github.com/DerekRoberts/vexilon' target='_blank' style='color: #005691; text-decoration: none;'>View code or contribute on GitHub</a>
    <span style='margin-left: 0.5rem; opacity: 0.7;'>•</span>
    <a href='https://github.com/DerekRoberts/vexilon/pkgs/container/vexilon' target='_blank' style='color: #005691; text-decoration: none;'>{VEXILON_VERSION}</a>
</div>
"""



def build_ui() -> "gr.Blocks":
    """Assemble and return the Gradio Blocks application."""
    import gradio as gr
    with gr.Blocks(title="Collective Agreement Explorer") as demo:

        # ── Header ────────────────────────────────────────────────────────────
        gr.Markdown("## BCGEU Steward Assistant")

        with gr.Accordion("Knowledge Base & Priority", open=False):
            gr.Markdown(f"""
            The **Collective Agreement** is our primary reference. Anything else provides additional context.
            
            All source documents are available for review:
            [📁 Browse Knowledge Base on GitHub]({GITHUB_LABOUR_LAW_URL})
            """)

        # ── Disclaimer (persistent, non-dismissible) ──────────────────────────
        gr.HTML(DISCLAIMER_HTML)

        with gr.Row(visible=True) as chip_row:
            chip_btns = [
                gr.Button(q, size="sm")
                for q in EXAMPLE_QUESTIONS
            ]

        # ── Chat interface ────────────────────────────────────────────────────
        chatbot = gr.Chatbot(
            height=480,
            buttons=["copy"],
            render_markdown=True,
            show_label=False,
        )

        # ── Input row ─────────────────────────────────────────────────────────
        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Ask about the collective agreement…",
                label="",
                lines=2,
                max_lines=6,
                scale=5,
                show_label=False,
                container=False,
            )
            send_btn = gr.Button("Send ➤", scale=1, variant="primary")

        # ── Submit handlers ───────────────────────────────────────────────────
        async def submit(
            message: str, history: list[dict]
        ) -> AsyncIterator[tuple[list[dict], str, dict]]:
            import gradio as gr
            hide = gr.update(visible=False)
            show = gr.update(visible=True)
            if not message.strip():
                yield history, "", show
                return
            prior_history = list(history)
            # Append user turn; seed an empty assistant bubble for streaming.
            # Hide onboarding components on first message.
            history = prior_history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": ""},
            ]
            yield history, "", hide
            # Stream tokens from RAG; accumulate into the assistant bubble
            accumulated = ""
            async for chunk in rag_stream(message, prior_history):
                accumulated += chunk
                history[-1]["content"] = accumulated
                yield history, "", hide

        submit_inputs = [msg_input, chatbot]
        submit_outputs = [chatbot, msg_input, chip_row]

        send_btn.click(fn=submit, inputs=submit_inputs, outputs=submit_outputs)
        msg_input.submit(fn=submit, inputs=submit_inputs, outputs=submit_outputs)

        # ── Chip click handlers — populate input and auto-submit ──────────────
        for chip in chip_btns:
            chip.click(
                fn=lambda q: q,
                inputs=[chip],
                outputs=[msg_input],
            ).then(
                fn=submit,
                inputs=submit_inputs,
                outputs=submit_outputs,
            )

        # ── Attribution Footer ────────────────────────────────────────────────
        gr.HTML(ATTRIBUTION_HTML)

    return demo


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    startup()
    app = build_ui()
    # Enable authentication if a password is set in the environment.
    auth_creds = None
    if VEXILON_PASSWORD:
        auth_creds = (VEXILON_USERNAME, VEXILON_PASSWORD)
        print(f"[startup] Authentication enabled for user '{VEXILON_USERNAME}'")

    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
        allowed_paths=[],
        auth=auth_creds,
    )
