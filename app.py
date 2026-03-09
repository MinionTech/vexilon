"""
app.py — Vexilon: BCGEU Agreement Assistant
--------------------------------------------
Tech stack:
  - pypdf                : PDF → pages with page number preservation
  - tiktoken             : Token counting for chunking
  - sentence-transformers: Local CPU embeddings (all-MiniLM-L6-v2, no API key)
  - FAISS                : In-memory vector index (no server process)
  - Anthropic            : Claude (claude-haiku-4-5) for responses
  - Gradio 5             : Web UI at http://localhost:7860

Quick start:
  1. export ANTHROPIC_API_KEY=sk-ant-...
  2. pip install -r requirements.txt
  3. python app.py

Index pre-computation (run once after updating the PDF):
  python -c "from app import startup; startup(force_rebuild=True)"
  # Saves pdf_cache/index.faiss + pdf_cache/chunks.json for fast cold starts.
"""

# ─── Standard Library ────────────────────────────────────────────────────────
import json
import os
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

# Ensure the HuggingFace model cache is writable by the non-root container user.
# The Containerfile sets this too, but HF Spaces and other runtimes may not inherit it.
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")

# ─── Third-party: PDF ────────────────────────────────────────────────────────
from pypdf import PdfReader

# ─── Third-party: Tokenizer ──────────────────────────────────────────────────
import tiktoken

# ─── Third-party: Embeddings (local, no API key) ────────────────────────────
from sentence_transformers import SentenceTransformer

# ─── Third-party: Vector Store (FAISS) ──────────────────────────────────────
import faiss
import numpy as np

# ─── Third-party: LLM (Anthropic) ───────────────────────────────────────────
import anthropic

# ─── Third-party: Gradio UI ──────────────────────────────────────────────────
import gradio as gr

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

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))       # tokens per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100)) # token overlap
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", 5))

# Embedding dimension for all-MiniLM-L6-v2
EMBED_DIM = 384

# ─── Clients ─────────────────────────────────────────────────────────────────
_embed_model: SentenceTransformer | None = None
_anthropic_client: anthropic.Anthropic | None = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        print(f"[embed] Loading local embedding model '{EMBED_MODEL}'…")
        _embed_model = SentenceTransformer(EMBED_MODEL)
        print("[embed] Embedding model ready.")
    return _embed_model


def get_anthropic() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        # Reads ANTHROPIC_API_KEY from environment automatically; raises AuthenticationError if missing
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


# ─── System Prompt ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Vexilon, an assistant for BCGEU union stewards. \
You help stewards understand the 19th Main Public Service Agreement (Social, Information & Health).

Rules you must follow without exception:

1. You may only answer using the provided agreement excerpts. Do not draw on outside knowledge.
2. Every claim must be supported by a verbatim quote from the provided excerpts, formatted as a \
markdown blockquote (> "...") followed by its citation: — Article [X], [Title], p. [N]
3. Plain-language explanation comes BEFORE the verbatim quote, not after.
4. If the excerpts do not address the question, say so clearly: \
"The collective agreement does not appear to address this question in the excerpts I was given."
5. Do not predict outcomes, advise on strategy, or offer legal opinions.
6. Tone: plain language. Your audience is new stewards with no legal background.
7. If multiple clauses are relevant, quote each one separately with its own citation.

Response format:

[Plain-language explanation]

> "[Verbatim quote from the agreement]"
> — Article [X], [Title], p. [N]

[Optional: "This may also be relevant:" + follow-up suggestion]
"""

# ─── Chunking ─────────────────────────────────────────────────────────────────
# cl100k_base is used by text-embedding-3-small.
# Lazy init: tiktoken downloads the BPE vocab (~1 MB) on first call if not cached;
# we init inside startup() where we already have print() context.
_ENCODER: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def chunk_text(text: str, page_num: int) -> list[dict]:
    """
    Split *text* into overlapping token-based chunks.
    Returns list of dicts: {text, page, chunk_index}.
    """
    enc = _get_encoder()
    tokens = enc.encode(text)
    chunks = []
    start = 0
    idx = 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_str = enc.decode(chunk_tokens)
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
    reader = PdfReader(str(pdf_path))
    all_chunks = []
    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1  # 1-based
        text = page.extract_text() or ""
        if text.strip():
            all_chunks.extend(chunk_text(text, page_num))
    return all_chunks


# ─── FAISS Index ──────────────────────────────────────────────────────────────
def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts using the local sentence-transformers model. Returns (N, EMBED_DIM) float32 array."""
    model = get_embed_model()
    # encode() handles batching internally; show_progress_bar=False keeps logs clean
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.astype(np.float32)


def build_index(chunks: list[dict]) -> faiss.IndexFlatIP:
    """
    Embed all chunks and build a FAISS inner-product index.
    Vectors are L2-normalised so inner product == cosine similarity.
    """
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
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    query: str,
    top_k: int = SIMILARITY_TOP_K,
) -> list[dict]:
    """Return the top-k most similar chunks for *query*."""
    query_vec = embed_texts([query])  # (1, EMBED_DIM)
    faiss.normalize_L2(query_vec)
    _scores, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


# ─── RAG App State (module-level, built at startup) ──────────────────────────
_chunks: list[dict] = []
_index: faiss.IndexFlatIP | None = None
_startup_error: str | None = None   # set if startup fails; surfaced in chat UI


def save_index(index: faiss.IndexFlatIP, chunks: list[dict]) -> None:
    """Persist the FAISS index and chunk metadata to pdf_cache/ for fast cold starts."""
    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    print(f"[index] Saved index → {INDEX_PATH} and chunks → {CHUNKS_PATH}")


def load_precomputed_index() -> tuple[faiss.IndexFlatIP, list[dict]] | tuple[None, None]:
    """
    Load a pre-computed FAISS index and chunks from disk if both exist.
    Returns (index, chunks) on success, (None, None) if files are missing.
    """
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        return None, None
    print(f"[startup] Loading pre-computed index from {INDEX_PATH}…")
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
    global _chunks, _index, _startup_error
    try:
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
        print("[startup] Initialising tokeniser (downloads BPE vocab on first run)…")
        _get_encoder()
        print("[startup] Tokeniser ready.")
        print(f"[startup] Loading PDF from {PDF_PATH}…")
        _chunks = load_pdf_chunks(PDF_PATH)
        print(f"[startup] {len(_chunks)} chunks loaded from {PDF_PATH.name}")
        _index = build_index(_chunks)
        save_index(_index, _chunks)
        print("[startup] Ready.")
    except Exception as exc:  # noqa: BLE001
        _startup_error = str(exc)
        print(f"[startup] ⚠️  Startup failed — app will run but queries will fail: {exc}")
        print("[startup] Set ANTHROPIC_API_KEY, then restart.")


# ─── RAG Query ────────────────────────────────────────────────────────────────
def rag_stream(message: str, history: list[dict]) -> Iterator[str]:
    """
    Retrieve relevant chunks, build the prompt, and stream a response from Claude.
    *history* is a list of {"role": ..., "content": ...} dicts (Gradio messages format).
    Yields text chunks as they arrive from the Anthropic streaming API.
    """
    if _startup_error is not None:
        yield (
            "⚠️ **The app failed to start.**\n\n"
            f"```\n{_startup_error}\n```\n\n"
            "Make sure `ANTHROPIC_API_KEY` is set in your "
            "environment, then restart the container:\n\n"
            "```bash\nexport ANTHROPIC_API_KEY=sk-ant-...\n"
            "podman-compose up\n```"
        )
        return
    if _index is None:
        yield "⚠️ The index is not ready yet. Please wait a moment and try again."
        return

    relevant_chunks = search_index(_index, _chunks, message)

    # Build context block from retrieved chunks
    context_parts = []
    for chunk in relevant_chunks:
        context_parts.append(
            f"[Page {chunk['page']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    full_system = (
        SYSTEM_PROMPT
        + "\n\n--- AGREEMENT EXCERPTS ---\n\n"
        + context
        + "\n\n--- END EXCERPTS ---"
    )

    # Build message list for Claude: prior history + new user message
    messages = []
    for turn in history:
        if turn["role"] in ("user", "assistant"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": message})

    client = get_anthropic()
    try:
        with client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=full_system,
            messages=messages,
        ) as stream:
            for text_chunk in stream.text_stream:
                yield text_chunk
    except anthropic.APIError as exc:
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
    "margin-bottom:8px;"
    '">'
    "⚠️ <strong style=\"color:#7c4a00;\">This tool references the collective agreement text only. "
    "It is not legal advice. "
    "Consult your BCGEU staff representative for complex matters.</strong>"
    "</div>"
)



def build_ui() -> gr.Blocks:
    """Assemble and return the Gradio Blocks application."""
    with gr.Blocks(title="Vexilon — BCGEU Agreement Assistant") as demo:

        # ── Header ────────────────────────────────────────────────────────────
        gr.Markdown("## 📋 Vexilon — BCGEU Agreement Assistant\n"
                    "*19th Main Public Service Agreement (Social, Information & Health)*")

        # ── Disclaimer (persistent, non-dismissible) ──────────────────────────
        gr.HTML(DISCLAIMER_HTML)

        # ── Empty-state onboarding (visible until first message) ───────────────
        onboarding_text = gr.HTML(
            "<p>I help BCGEU union stewards look up the 19th Main Public Service Agreement "
            "(Social, Information &amp; Health). Ask a question and I'll give you a "
            "plain-language explanation with verbatim quotes and citations. "
            "I cannot give legal advice or predict how a grievance will be decided.</p>"
            "<p><em>Try one of these questions:</em></p>",
            visible=True,
        )
        with gr.Row(visible=True) as chip_row:
            chip_btns = [
                gr.Button(q, size="sm")
                for q in EXAMPLE_QUESTIONS
            ]

        # ── Chat interface ────────────────────────────────────────────────────
        chatbot = gr.Chatbot(
            type="messages",
            height=480,
            show_copy_button=True,
            render_markdown=True,
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
        def submit(
            message: str, history: list[dict]
        ) -> Iterator[tuple[list[dict], str, dict, dict]]:
            hide = gr.update(visible=False)
            show = gr.update(visible=True)
            if not message.strip():
                yield history, "", show, show
                return
            prior_history = list(history)
            # Append user turn; seed an empty assistant bubble for streaming.
            # Hide both onboarding components on first message.
            history = prior_history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": ""},
            ]
            yield history, "", hide, hide
            # Stream tokens from RAG; accumulate into the assistant bubble
            accumulated = ""
            for chunk in rag_stream(message, prior_history):
                accumulated += chunk
                history[-1]["content"] = accumulated
                yield history, "", hide, hide

        submit_inputs = [msg_input, chatbot]
        submit_outputs = [chatbot, msg_input, onboarding_text, chip_row]

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

    return demo


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    startup()
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
        allowed_paths=[],
    )
