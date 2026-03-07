"""
app.py — BCGEU Collective Agreement RAG Chatbot
------------------------------------------------
Tech stack:
  - LlamaIndex  : PDF → chunk → vector index (512 token chunks, 100 overlap)
  - Chroma DB   : local persistent vector store  (./chroma_db)
  - Ollama      : Llama 3.1:8b LLM + nomic-embed-text embeddings (local)
  - Gradio      : Web UI at http://localhost:7860

Quick start:
  1. Install Ollama and pull the required models:
       ollama pull llama3.1:8b
       ollama pull nomic-embed-text
  2. pip install -r requirements.txt
  3. python app.py

Hugging Face Spaces: set OLLAMA_BASE_URL to an external Ollama endpoint or
swap the LLM / embedding provider for an HF-hosted model.
"""

# ─── Standard Library ────────────────────────────────────────────────────────
import os
import re
import json
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

# ─── Third-party: HTTP ───────────────────────────────────────────────────────
import requests

# ─── Third-party: Gradio UI ──────────────────────────────────────────────────
import gradio as gr

# ─── Third-party: Vector Store ───────────────────────────────────────────────
import chromadb

# ─── Third-party: LlamaIndex (v0.10+) ────────────────────────────────────────
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# ─── Configuration ───────────────────────────────────────────────────────────
CHROMA_DIR = Path("./chroma_db")      # Persistent Chroma vector store
PDF_CACHE_DIR = Path("./pdf_cache")   # Cached downloaded PDFs
CHUNK_SIZE = 512                      # Tokens per chunk (nomic-embed-text context limit: 2048)
CHUNK_OVERLAP = 100                   # Overlap between chunks (~20% of chunk size)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")      # Ollama LLM model name
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")    # Ollama embedding model name
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
RESPONSE_PREFIX = "From document only—not legal advice.\n\n"
REQUEST_TIMEOUT = 30                  # HTTP download timeout in seconds
SIMILARITY_TOP_K = 5                  # Number of chunks to retrieve per query

# Ensure storage directories exist at startup
CHROMA_DIR.mkdir(exist_ok=True)
PDF_CACHE_DIR.mkdir(exist_ok=True)

# ─── Agreement Registry ──────────────────────────────────────────────────────
# Display names shown in the dropdown → PDF source URL + Chroma collection name.
# Prefer public direct-PDF URLs (www2.gov.bc.ca); bundled fallbacks live in pdf_cache/.
AGREEMENTS: dict[str, dict] = {
    "19th Main Public Service Agreement (Social, Information & Health)": {
        "url": "https://www2.gov.bc.ca/assets/gov/careers/managers-supervisors/managing-employee-labour-relations/bcgeu_19th_main_agreement_38fa.pdf",
        "collection": "main_public_service_19th",
    },
}

AGREEMENT_NAMES = list(AGREEMENTS.keys())

# ─── BC Contacts (hardcoded) ─────────────────────────────────────────────────
# Keys are lowercase search terms; values are contact info dicts.
# Supports partial substring matching (see lookup_contacts).
BC_CONTACTS: dict[str, dict] = {
    # Health Authorities — HR / Labour Relations
    "health island": {
        "name": "Island Health (VIHA) — Human Resources",
        "phone": "250-519-3500",
        "email": "hr@viha.ca",
        "website": "https://www.islandhealth.ca",
    },
    # "health victoria" is an alias for Island Health (Victoria is on Vancouver Island)
    "health victoria": {
        "name": "Island Health (VIHA) — Human Resources",
        "phone": "250-519-3500",
        "email": "hr@viha.ca",
        "website": "https://www.islandhealth.ca",
    },
    "health northern": {
        "name": "Northern Health — Human Resources",
        "phone": "250-565-2000",
        "email": "hr@northernhealth.ca",
        "website": "https://www.northernhealth.ca",
    },
    "health interior": {
        "name": "Interior Health — Human Resources",
        "phone": "1-800-707-8550",
        "email": "hr@interiorhealth.ca",
        "website": "https://www.interiorhealth.ca",
    },
    "health fraser": {
        "name": "Fraser Health — Human Resources",
        "phone": "604-587-4600",
        "email": "hr@fraserhealth.ca",
        "website": "https://www.fraserhealth.ca",
    },
    "health coastal": {
        "name": "Vancouver Coastal Health — Human Resources",
        "phone": "604-875-4111",
        "website": "https://www.vch.ca",
    },
    "health vancouver coastal": {
        "name": "Vancouver Coastal Health — Human Resources",
        "phone": "604-875-4111",
        "website": "https://www.vch.ca",
    },
    "health providence": {
        "name": "Providence Health Care — Human Resources",
        "phone": "604-682-2344",
        "website": "https://www.providencehealthcare.org",
    },
    "health phsa": {
        "name": "PHSA (Provincial Health Services Authority) — Human Resources",
        "phone": "604-875-2000",
        "website": "https://www.phsa.ca",
    },
    "health first nations": {
        "name": "First Nations Health Authority",
        "phone": "604-693-6500",
        "website": "https://www.fnha.ca",
    },
    # Public Service / Provincial Government
    "bcgeu": {
        "name": "BCGEU — Provincial Office",
        "phone": "604-291-9611",
        "toll_free": "1-800-663-1674",
        "website": "https://www.bcgeu.ca",
    },
    "psc": {
        "name": "BC Public Service Commission",
        "phone": "250-387-5082",
        "website": "https://www2.gov.bc.ca/gov/content/careers-myhr",
    },
    "myhr": {
        "name": "BC Government MyHR (Employee Self-Service)",
        "phone": "250-952-6000",
        "website": "https://www2.gov.bc.ca/gov/content/careers-myhr",
    },
    # Corrections / Justice
    "corrections": {
        "name": "BC Corrections — Labour Relations",
        "phone": "250-387-5041",
        "website": "https://www2.gov.bc.ca/gov/content/justice/criminal-justice/corrections",
    },
    # Community Social Services
    "cssea": {
        "name": "Community Social Services Employers' Association (CSSEA)",
        "phone": "604-942-0505",
        "website": "https://www.cssea.bc.ca",
    },
}

# ─── LlamaIndex Global Settings ──────────────────────────────────────────────
def configure_llm() -> None:
    """Configure LlamaIndex to use local Ollama LLM and embedding model."""
    Settings.llm = Ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=120.0,
    )
    Settings.embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


# ─── Contact Lookup ──────────────────────────────────────────────────────────
def lookup_contacts(query: str) -> Optional[str]:
    """
    Return formatted BC contact info if the query matches any known key.
    Checks whether any BC_CONTACTS key appears as a substring of the query.
    Returns a markdown-formatted string if found, else None.

    Example: 'health victoria' → Island Health HR: 250-519-3500
    """
    q = query.lower().strip()
    for key, contact in BC_CONTACTS.items():
        if key in q:
            lines = [f"📞 **{contact['name']}**"]
            if "phone" in contact:
                lines.append(f"  Phone: {contact['phone']}")
            if "toll_free" in contact:
                lines.append(f"  Toll-free: {contact['toll_free']}")
            if "email" in contact:
                lines.append(f"  Email: {contact['email']}")
            if "website" in contact:
                lines.append(f"  Web: {contact['website']}")
            return "\n".join(lines)
    return None


# ─── PDF Downloader ───────────────────────────────────────────────────────────
def download_pdf(url: str, dest_path: Path) -> bool:
    """
    Download a PDF from *url* to *dest_path*.
    Handles both direct PDF links and HTML pages containing an embedded PDF link.
    Returns True on success, False on any failure.
    """
    try:
        resp = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            stream=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; blabot/1.0)"},
        )
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")

        if "application/pdf" in content_type:
            # Direct PDF — stream to disk
            with open(dest_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    fh.write(chunk)
            return True

        # HTML page — search for an embedded PDF link
        html = resp.text
        patterns = [
            r'href=["\']([^"\']+\.pdf[^"\']*)["\']',
            r'src=["\']([^"\']+\.pdf[^"\']*)["\']',
            r'"url"\s*:\s*"([^"]+\.pdf[^"]*)"',
        ]
        for pattern in patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                pdf_url = match.group(1)
                if not pdf_url.startswith("http"):
                    pdf_url = urljoin(url, pdf_url)
                # Recurse once to download the discovered PDF link
                return download_pdf(pdf_url, dest_path)

        return False  # No PDF found in the page

    except Exception as exc:
        print(f"[download_pdf] Error fetching {url}: {exc}")
        return False


# ─── Index Builder / Loader ──────────────────────────────────────────────────
# In-memory cache: collection_name → query_engine (avoids reloading on every query)
_index_cache: dict[str, object] = {}


def get_query_engine(
    agreement_name: str,
    pdf_path: Optional[Path] = None,
) -> tuple[Optional[object], str]:
    """
    Return (query_engine, status_message) for *agreement_name*.

    Load order:
      1. In-memory cache (fastest)
      2. Existing Chroma collection (fast — ~1 s)
      3. Build from PDF (slow — ~10 s on first load)

    *pdf_path*: path to a manually uploaded PDF; bypasses URL download and
                forces a full re-index of the collection.
    """
    config = AGREEMENTS[agreement_name]
    collection_name = config["collection"]

    # Track whether the caller supplied a manual upload
    is_manual_upload = pdf_path is not None

    # Return from memory cache when no new upload is provided
    if collection_name in _index_cache and not is_manual_upload:
        return _index_cache[collection_name], "✅ Index already loaded — ready to chat"

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Load from existing persisted Chroma collection (skip re-indexing)
    if not is_manual_upload:
        try:
            collection = chroma_client.get_collection(collection_name)
            if collection.count() > 0:
                vector_store = ChromaVectorStore(chroma_collection=collection)
                storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
                index = VectorStoreIndex.from_vector_store(
                    vector_store,
                    storage_context=storage_ctx,
                )
                engine = index.as_query_engine(similarity_top_k=SIMILARITY_TOP_K)
                _index_cache[collection_name] = engine
                return engine, "✅ Loaded from saved index — ready to chat"
        except Exception:
            pass  # Collection doesn't exist yet — fall through to build

    # Resolve PDF path: use upload, cached file, or download fresh
    if pdf_path is None:
        cached_pdf = PDF_CACHE_DIR / f"{collection_name}.pdf"
        if not cached_pdf.exists():
            status_msg = f"⏳ Downloading PDF for '{agreement_name}'…"
            print(status_msg)
            success = download_pdf(config["url"], cached_pdf)
            if not success:
                return None, (
                    "❌ Could not download the PDF automatically.\n"
                    "Please upload the PDF manually using the 'Upload PDF' button below."
                )
        pdf_path = cached_pdf

    # Read and chunk the PDF
    try:
        reader = SimpleDirectoryReader(input_files=[str(pdf_path)])
        documents = reader.load_data()
    except Exception as exc:
        return None, f"❌ Error reading PDF: {exc}"

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents)

    # Build (or rebuild) the Chroma collection.
    # Delete the existing collection when re-indexing from a manual upload so
    # stale vectors from the previous PDF are not mixed in.
    try:
        if is_manual_upload:
            try:
                chroma_client.delete_collection(collection_name)
            except Exception:
                pass
        collection = chroma_client.get_or_create_collection(collection_name)
    except Exception as exc:
        return None, f"❌ Chroma error: {exc}"

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_ctx)
    engine = index.as_query_engine(similarity_top_k=SIMILARITY_TOP_K)
    _index_cache[collection_name] = engine

    return engine, (
        f"✅ Indexed {len(nodes)} chunks from {len(documents)} page(s). "
        "Ready — ask about the agreement."
    )


# ─── Citation Formatter ───────────────────────────────────────────────────────
def format_citations(source_nodes) -> str:
    """
    Format LlamaIndex source nodes as readable page-citation footnotes.
    Each unique (filename, page) pair is listed once.
    """
    if not source_nodes:
        return ""

    seen: set[tuple] = set()
    lines = ["\n\n---\n📄 **Sources:**"]

    for node_with_score in source_nodes:
        meta = node_with_score.node.metadata
        page = meta.get("page_label") or meta.get("page") or "?"
        fname = meta.get("file_name", "document")
        key = (fname, page)
        if key not in seen:
            seen.add(key)
            page_label = f"p.{page}" if page != "?" else "unknown page"
            lines.append(f"  - {fname} — {page_label}")

    return "\n".join(lines)


# ─── Gradio Event Handlers ────────────────────────────────────────────────────
def on_load_agreement(
    agreement_name: str,
    upload,
) -> tuple[str, dict, list]:
    """
    Called when the user clicks 'Load Agreement'.
    Returns (status_text, engine_state_dict, cleared_chat_history).
    """
    pdf_path = Path(upload.name) if upload is not None else None
    engine, status = get_query_engine(agreement_name, pdf_path)
    return status, {"engine": engine}, []


def on_submit(
    message: str,
    history: list,
    agreement_name: str,
    engine_state: dict,
) -> tuple[list, dict, str]:
    """
    Called when the user sends a message.
    Returns (updated_history, unchanged_engine_state, cleared_input).
    """
    if not message.strip():
        return history, engine_state, ""

    engine = engine_state.get("engine")

    # ── 1. BC Contacts lookup (fast, no LLM needed) ────────────────────────
    contact_result = lookup_contacts(message)
    if contact_result:
        response = RESPONSE_PREFIX + "**BC Contacts:**\n" + contact_result
        return history + [(message, response)], engine_state, ""

    # ── 2. Guard: agreement not yet loaded ─────────────────────────────────
    if engine is None:
        response = (
            RESPONSE_PREFIX
            + "⚠️ No agreement loaded yet.\n"
            "Please select an agreement and click **📥 Load Agreement** first."
        )
        return history + [(message, response)], engine_state, ""

    # ── 3. RAG query via LlamaIndex ────────────────────────────────────────
    try:
        result = engine.query(message)
        answer = str(result.response)
        citations = format_citations(result.source_nodes)
        response = RESPONSE_PREFIX + answer + citations
    except Exception as exc:
        response = RESPONSE_PREFIX + f"⚠️ Error querying the document: {exc}"

    return history + [(message, response)], engine_state, ""


# ─── Gradio UI ────────────────────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    """Assemble and return the Gradio Blocks application."""
    configure_llm()

    with gr.Blocks(
        title="BCGEU Collective Agreement Chatbot",
        theme=gr.themes.Soft(),
        head=(
            '<link rel="manifest" href="/file=manifest.json">'
            '<meta name="theme-color" content="#3b82f6">'
        ),
    ) as demo:
        # Shared state: {"engine": <query_engine | None>}
        engine_state = gr.State({"engine": None})

        gr.Markdown(
            """
# 📋 BCGEU Collective Agreement Chatbot
Ask questions about BC Government / BCGEU collective agreements. \
All answers come directly from the selected document.

> ⚠️ **From document only — not legal advice.**
"""
        )

        with gr.Row():
            # ── Left panel: controls ────────────────────────────────────────
            with gr.Column(scale=1, min_width=280):
                agreement_dd = gr.Dropdown(
                    choices=AGREEMENT_NAMES,
                    value=AGREEMENT_NAMES[0],
                    label="Select Agreement",
                    interactive=True,
                )
                pdf_upload = gr.File(
                    label="Upload PDF (optional — overrides URL download)",
                    file_types=[".pdf"],
                    file_count="single",
                )
                load_btn = gr.Button("📥 Load Agreement", variant="primary")
                status_box = gr.Textbox(
                    label="Status",
                    value="Select an agreement and click Load.",
                    interactive=False,
                    lines=4,
                )
                gr.Markdown(
                    """
**Example questions:**
- *Article 12 overtime rules?*
- *What is the probationary period?*
- *Vacation entitlement for new employees?*

**BC Contacts (type to look up):**
- `health victoria` → Island Health HR
- `health northern` → Northern Health HR
- `health fraser` → Fraser Health HR
- `bcgeu` → BCGEU provincial office
"""
                )

            # ── Right panel: chat ───────────────────────────────────────────
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Agreement Chat",
                    height=520,
                    show_copy_button=True,
                    bubble_full_width=False,
                )
                msg_input = gr.Textbox(
                    placeholder="Ask a question about the agreement…",
                    label="Your Question",
                    lines=2,
                    max_lines=6,
                )
                with gr.Row():
                    submit_btn = gr.Button("Send ➤", variant="primary")
                    clear_btn = gr.Button("🗑 Clear Chat")

        # ── Wiring ────────────────────────────────────────────────────────
        load_btn.click(
            fn=on_load_agreement,
            inputs=[agreement_dd, pdf_upload],
            outputs=[status_box, engine_state, chatbot],
            show_progress="minimal",
        )

        submit_btn.click(
            fn=on_submit,
            inputs=[msg_input, chatbot, agreement_dd, engine_state],
            outputs=[chatbot, engine_state, msg_input],
        )
        # Allow submitting with the Enter key as well
        msg_input.submit(
            fn=on_submit,
            inputs=[msg_input, chatbot, agreement_dd, engine_state],
            outputs=[chatbot, engine_state, msg_input],
        )

        clear_btn.click(fn=lambda: [], outputs=[chatbot])

    return demo


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
        allowed_paths=["./manifest.json"],  # Serve PWA manifest
    )
