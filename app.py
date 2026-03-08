"""
app.py — Vexilon: BCGEU Agreement Assistant
--------------------------------------------
Tech stack:
  - pypdf     : PDF → pages with page number preservation
  - tiktoken  : Token counting for chunking
  - OpenAI    : text-embedding-3-small embeddings
  - FAISS     : In-memory vector index (no server process)
  - Anthropic : Claude (claude-3-5-haiku-20241022) for responses
  - Gradio 5  : Web UI at http://localhost:7860

Quick start:
  1. export ANTHROPIC_API_KEY=sk-ant-...
     export OPENAI_API_KEY=sk-...
  2. pip install -r requirements.txt
  3. python app.py
"""

# ─── Standard Library ────────────────────────────────────────────────────────
import os
from collections.abc import Iterator
from pathlib import Path

# ─── Third-party: PDF ────────────────────────────────────────────────────────
from pypdf import PdfReader

# ─── Third-party: Tokenizer ──────────────────────────────────────────────────
import tiktoken

# ─── Third-party: Embeddings (OpenAI) ───────────────────────────────────────
from openai import OpenAI

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

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-5-haiku-20241022")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))       # tokens per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100)) # token overlap
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", 5))

# Embedding dimension for text-embedding-3-small
EMBED_DIM = 1536

# ─── Clients ─────────────────────────────────────────────────────────────────
_openai_client: OpenAI | None = None
_anthropic_client: anthropic.Anthropic | None = None


def get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        # Reads OPENAI_API_KEY from environment automatically; raises AuthenticationError if missing
        _openai_client = OpenAI()
    return _openai_client


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
    """Embed a list of texts using text-embedding-3-small. Returns (N, EMBED_DIM) float32 array."""
    client = get_openai()
    # OpenAI allows up to 2048 inputs per call; batch to be safe
    batch_size = 512
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    return np.array(all_embeddings, dtype=np.float32)


def build_index(chunks: list[dict]) -> faiss.IndexFlatIP:
    """
    Embed all chunks and build a FAISS inner-product index.
    Vectors are L2-normalised so inner product == cosine similarity.
    """
    import time
    texts = [c["text"] for c in chunks]
    print(f"[index] Embedding {len(texts)} chunks via OpenAI API (may take 15–60 s)…")
    t0 = time.time()
    vectors = embed_texts(texts)
    print(f"[index] Embeddings received in {time.time()-t0:.1f}s")
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


def startup() -> None:
    """Load PDF, chunk, embed, and build FAISS index. Called once before Gradio starts."""
    global _chunks, _index, _startup_error
    try:
        print("[startup] Initialising tokeniser (downloads BPE vocab on first run)…")
        _get_encoder()
        print("[startup] Tokeniser ready.")
        print(f"[startup] Loading PDF from {PDF_PATH}…")
        _chunks = load_pdf_chunks(PDF_PATH)
        print(f"[startup] {len(_chunks)} chunks loaded from {PDF_PATH.name}")
        _index = build_index(_chunks)
        print("[startup] Ready.")
    except Exception as exc:  # noqa: BLE001
        _startup_error = str(exc)
        print(f"[startup] ⚠️  Startup failed — app will run but queries will fail: {exc}")
        print("[startup] Set OPENAI_API_KEY and ANTHROPIC_API_KEY, then restart.")


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
            "Make sure `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` are set in your "
            "environment, then restart the container:\n\n"
            "```bash\nexport OPENAI_API_KEY=sk-...\n"
            "export ANTHROPIC_API_KEY=sk-ant-...\n"
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

WELCOME_MESSAGE = """**Welcome to Vexilon — BCGEU Agreement Assistant**

I help BCGEU union stewards look up the 19th Main Public Service Agreement \
(Social, Information & Health). Ask me any question about the agreement and I'll \
give you a plain-language explanation with verbatim quotes and citations.

I can only tell you what the agreement says — I cannot give legal advice or predict \
how a grievance will be decided."""

BCGEU_CSS = """
:root {
    --primary: #005691;
    --primary-dark: #003366;
    --accent: #008542;
    --bg: #f5f7fa;
    --border: #cdd5e0;
    --text-primary: #333;
    --text-secondary: #666;
}

/* App background */
.gradio-container {
    background-color: var(--bg) !important;
    max-width: 900px !important;
    margin: 0 auto !important;
    overflow-x: hidden !important;
}

/* Header */
#app-header {
    background-color: var(--primary-dark);
    color: white;
    padding: 16px 20px;
    border-radius: 8px;
    margin-bottom: 8px;
}
#app-header h1 {
    color: white !important;
    font-size: 1.4rem;
    margin: 0;
}
#app-header p {
    color: #c8d8e8 !important;
    font-size: 0.85rem;
    margin: 4px 0 0;
}

/* Disclaimer bar */
#disclaimer {
    background-color: #fff8e1 !important;
    border-left: 4px solid #f59e0b !important;
    color: #7c4a00 !important;
    padding: 10px 14px !important;
    border-radius: 4px !important;
    font-size: 0.85rem !important;
    margin-bottom: 8px !important;
}
#disclaimer strong {
    color: #7c4a00 !important;
}

/* Chatbot */
#chatbot {
    border: 1px solid var(--border);
    border-radius: 8px;
    background-color: white;
}

/* Blockquote rendering in chat messages */
.message-bubble-border blockquote,
.message blockquote {
    border-left: 4px solid var(--primary);
    background-color: #eef4fb;
    padding: 8px 12px;
    margin: 8px 0;
    border-radius: 0 4px 4px 0;
    font-style: italic;
    color: #1a2a3a;
    /* prevent blockquotes from causing horizontal scroll on mobile */
    max-width: 100%;
    overflow-x: auto;
    white-space: pre-wrap;
    overflow-wrap: break-word;
}

/* Empty-state onboarding panel */
#onboarding {
    background-color: white;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 8px;
}
#onboarding p {
    color: var(--text-primary, #333);
    font-size: 0.95rem;
    margin: 0 0 16px;
}
#onboarding .chip-label {
    font-size: 0.8rem;
    color: var(--text-secondary, #666);
    margin-bottom: 8px;
}

/* Example question chips */
.example-chip {
    display: inline-block;
    margin: 4px !important;
}
.example-chip button {
    background-color: white !important;
    border: 1px solid var(--primary) !important;
    color: var(--primary) !important;
    border-radius: 20px !important;
    font-size: 0.85rem !important;
    padding: 6px 14px !important;
    cursor: pointer !important;
    white-space: normal !important;
    text-align: left !important;
    line-height: 1.3 !important;
}
.example-chip button:hover {
    background-color: var(--primary) !important;
    color: white !important;
}

/* Send button */
#send-btn {
    background-color: var(--primary) !important;
    color: white !important;
    min-width: 80px;
}
#send-btn:hover {
    background-color: var(--primary-dark) !important;
}

/* Mobile: ensure input row stacks gracefully on narrow viewports */
@media (max-width: 480px) {
    #app-header h1 {
        font-size: 1.1rem;
    }
    .example-chip button {
        font-size: 0.8rem !important;
        padding: 5px 10px !important;
    }
}
"""


def build_ui() -> gr.Blocks:
    """Assemble and return the Gradio Blocks application."""
    with gr.Blocks(
        title="Vexilon — BCGEU Agreement Assistant",
        css=BCGEU_CSS,
        head=(
            '<link rel="manifest" href="/file=manifest.json">'
            '<meta name="theme-color" content="#005691">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
        ),
    ) as demo:

        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML(
            '<div id="app-header">'
            "<h1>📋 Vexilon — BCGEU Agreement Assistant</h1>"
            "<p>19th Main Public Service Agreement (Social, Information &amp; Health)</p>"
            "</div>"
        )

        # ── Disclaimer (persistent, non-dismissible) ──────────────────────────
        gr.HTML(DISCLAIMER_HTML)

        # ── Empty-state onboarding (visible until first message) ───────────────
        onboarding = gr.Group(elem_id="onboarding", visible=True)
        with onboarding:
            gr.HTML(
                "<p>I help BCGEU union stewards look up the 19th Main Public Service Agreement "
                "(Social, Information &amp; Health). Ask a question and I'll give you a "
                "plain-language explanation with verbatim quotes and citations. "
                "I cannot give legal advice or predict how a grievance will be decided.</p>"
                '<p class="chip-label">Try one of these questions:</p>'
            )
            with gr.Row(elem_classes="chip-row"):
                chip_btns = [
                    gr.Button(q, elem_classes="example-chip", size="sm")
                    for q in EXAMPLE_QUESTIONS
                ]

        # ── Chat interface ────────────────────────────────────────────────────
        chatbot = gr.Chatbot(
            elem_id="chatbot",
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
            send_btn = gr.Button("Send ➤", elem_id="send-btn", scale=1, variant="primary")

        # ── Submit handlers ───────────────────────────────────────────────────
        def submit(
            message: str, history: list[dict]
        ) -> Iterator[tuple[list[dict], str, dict]]:
            if not message.strip():
                yield history, "", gr.update(visible=True)
                return
            prior_history = list(history)
            # Append user turn; seed an empty assistant bubble for streaming.
            # Hide onboarding on first message.
            history = prior_history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": ""},
            ]
            yield history, "", gr.update(visible=False)
            # Stream tokens from RAG; accumulate into the assistant bubble
            accumulated = ""
            for chunk in rag_stream(message, prior_history):
                accumulated += chunk
                history[-1]["content"] = accumulated
                yield history, "", gr.update(visible=False)

        submit_inputs = [msg_input, chatbot]
        submit_outputs = [chatbot, msg_input, onboarding]

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
        allowed_paths=["./manifest.json"],
    )
