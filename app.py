"""
app.py — Vexilon: BCGEU Agreement Assistant
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
import urllib.request
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

# Ensure the HuggingFace model cache is writable and persistent.
# Move this out of /tmp to avoid being shadowed by the tmpfs mount in compose.yml.
os.environ.setdefault("HF_HOME", "/app/hf_cache")

# ─── Third-party: Deferred Imports ───────────────────────────────────────────
# (numpy, pypdf, anthropic, faiss, sentence_transformers, gradio)
# are imported inside functions to keep startup and test-loading fast.
print("[boot] All boilerplate complete.", flush=True)

# ─── Configuration ───────────────────────────────────────────────────────────
PDF_CACHE_DIR = Path("./pdf_cache")
PDF_PATH = PDF_CACHE_DIR / "main_public_service_19th.pdf"
INDEX_PATH = PDF_CACHE_DIR / "index.faiss"
CHUNKS_PATH = PDF_CACHE_DIR / "chunks.json"

# Public GitHub raw URL base for pdf_cache/ assets.
# Used as a fallback when the app runs in an environment where pdf_cache/
# was not committed (e.g. Hugging Face Spaces — HF rejects binary files in git).
_GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/DerekRoberts/vexilon/main/pdf_cache"
)

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
CONDENSER_MODEL = os.getenv("CONDENSER_MODEL", "claude-haiku-4-5-20251001")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 256))       # tokens per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))  # token overlap
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", 5))

# Memory / Context Condensation
CONDENSE_QUERY_HISTORY_TURNS = int(os.getenv("CONDENSE_QUERY_HISTORY_TURNS", 3))
CONDENSE_QUERY_CONTENT_MAX_LEN = int(os.getenv("CONDENSE_QUERY_CONTENT_MAX_LEN", 200))

# Security / Auth
VEXILON_USERNAME = os.getenv("VEXILON_USERNAME", "admin")
VEXILON_PASSWORD = os.getenv("VEXILON_PASSWORD")
VEXILON_VERSION = os.getenv("VEXILON_VERSION", "dev")

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


# ─── System Prompt ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful assistant for looking up the 19th Main Public Service Agreement. \
You help users navigate the collective agreement for the Social, Information & Health bargaining unit.

Rules you must follow without exception:

1. You may only answer using the provided agreement excerpts. Do not draw on outside knowledge.
2. Every claim must be supported by a verbatim quote from the provided excerpts, formatted as a \
markdown blockquote (> "...") followed by its citation: — Article [X], [Title], p. [N]
3. Plain-language explanation comes BEFORE the verbatim quote, not after.
4. If the excerpts do not address the question, say so clearly: \
"The collective agreement does not appear to address this question in the excerpts I was given."
5. Do not predict outcomes, advise on strategy, or offer legal opinions.
6. Tone: plain language. Your audience has no legal background.
7. If multiple clauses are relevant, quote each one separately with its own citation.
8. Maintain conversational continuity. If the user asks a follow-up question, use the previous \
conversation context and the provided excerpts to provide a coherent answer.

Response format:

[Plain-language explanation]

> "[Verbatim quote from the agreement]"
> — Article [X], [Title], p. [N]

[Optional: "This may also be relevant:" + follow-up suggestion]
"""

# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, page_num: int) -> list[dict]:
    """
    Split *text* into overlapping token-based chunks using the embedding model's tokenizer.
    Returns list of dicts: {text, page, chunk_index}.
    """
    if not text.strip():
        return []
    tokenizer = get_embed_model().tokenizer
    # Ensure the tokenizer doesn't truncate the whole page so we can split it manually
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    tokens = encoding.input_ids
    offsets = encoding.offset_mapping
    chunks = []
    
    if not tokens:
        return chunks
        
    start = 0
    idx = 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        
        # We need the original text that spans these tokens to preserve original case
        chunk_char_start = offsets[start][0]
        chunk_char_end = offsets[end - 1][1]
        
        chunk_text_str = text[chunk_char_start:chunk_char_end]
        
        chunks.append({
            "text": chunk_text_str,
            "page": page_num,
            "chunk_index": idx,
        })
        idx += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ─── PDF Loader ───────────────────────────────────────────────────────────────
def load_pdf_chunks(pdf_path: Path) -> list[dict]:
    """
    Parse the PDF at *pdf_path* and return all chunks with page metadata.
    Page numbers are 1-based (matching the printed page numbers in the PDF).
    """
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    all_chunks = []
    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1  # 1-based
        text = page.extract_text() or ""
        if text.strip():
            all_chunks.extend(chunk_text(text, page_num))
    return all_chunks


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
    Download pdf_cache/ assets from GitHub if they are absent from the local filesystem.

    This is a no-op when running locally (files are already present) and a transparent
    fallback when running on Hugging Face Spaces, where binary files cannot be committed
    to the Space git repo. Files are downloaded from the public GitHub raw URL.
    """
    files = {
        PDF_PATH: f"{_GITHUB_RAW_BASE}/main_public_service_19th.pdf",
        INDEX_PATH: f"{_GITHUB_RAW_BASE}/index.faiss",
        CHUNKS_PATH: f"{_GITHUB_RAW_BASE}/chunks.json",
    }
    missing = [path for path in files if not path.exists()]
    if not missing:
        return
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for path, url in files.items():
        if not path.exists():
            print(f"[startup] Downloading {path.name} from GitHub…")
            urllib.request.urlretrieve(url, path)
            print(f"[startup] {path.name} downloaded ({path.stat().st_size:,} bytes).")


def startup(force_rebuild: bool = False) -> None:
    """
    Load the FAISS index and chunks.

    Fast path (normal operation): loads pre-computed index.faiss + chunks.json from pdf_cache/.
    Slow path (first run or force_rebuild=True): parses the PDF, embeds all chunks, saves to disk.

    On Hugging Face Spaces, pdf_cache/ is not committed to the Space git repo (HF rejects
    binary files). _fetch_pdf_cache_if_missing() downloads the assets from GitHub on first run.

    After updating the agreement PDF, run:
        python -c "from app import startup; startup(force_rebuild=True)"
    """
    global _chunks, _index
    get_anthropic()  # Ping early to catch missing ANTHROPIC_API_KEY
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

    # ── Slow path: build from scratch ────────────────────────────────────
    print(f"[startup] Loading PDF from {PDF_PATH}…")
    _chunks = load_pdf_chunks(PDF_PATH)
    print(f"[startup] {len(_chunks)} chunks loaded from {PDF_PATH.name}")
    _index = build_index(_chunks)
    save_index(_index, _chunks)
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
        "the full intent. Only provide the rephrased query, nothing else.\n\n"
        f"History:\n{context_str}\n\n"
        f"Follow-up question: {message}\n\n"
        "Standalone query:"
    )

    try:
        response = await client.messages.create(
            model=CONDENSER_MODEL,
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
            f"[Page {chunk['page']}]\n{chunk['text']}"
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
                    "text": SYSTEM_PROMPT,
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
    "What are my overtime rights?",
    "What is the probationary period?",
    "Can my employer change my schedule without notice?",
    "What happens if I am disciplined?",
    "Am I entitled to union representation?",
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
    "⚠️ <strong style=\"color:#7c4a00;\">Unofficial Explorer: </strong> This is an independent community project. "
    "It is not affiliated with, endorsed by, or sponsored by the BCGEU. All responses are AI-generated; the"
    "<a href='https://www2.gov.bc.ca/gov/content/careers-myhr/managers-supervisors/employee-labour-relations/conditions-agreements/collective-agreements' target='_blank' style='color:#7c4a00; font-weight:bold;'>official PDF</a>"
    "is the sole authoritative source. Consult your BCGEU representative or a legal advisor as appropriate."
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
        gr.Markdown("## BCGEU Collective Agreement Explorer\n"
                    "*19th Main Agreement; Social, Information & Health*")

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
