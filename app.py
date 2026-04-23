# BCGEU Navigator - UI Version: 2026-04-22_13-17
# Integrated RAG Backend + Stabilized Gradio 6 UI
import os
import sys
import re
import html
import time
import logging
import asyncio
import datetime
import textwrap
import urllib.parse
from collections import OrderedDict
from collections.abc import AsyncIterator
from pathlib import Path
import threading
import tempfile
from threading import Lock

import numpy as np
import anthropic
import faiss
import gradio as gr

# ─── Vexilon Imports ────────────────────────────────────────────────────────
from vexilon.indexing import (
    _get_source_name,
    _get_rag_source_files,
    build_index_from_sources,
    get_integrity_report,
    load_precomputed_index,
    search_index_batch,
    _fetch_pdf_cache_if_missing,
    LABOUR_LAW_DIR,
)

# ─── Global State & Config ──────────────────────────────────────────────────
# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
_chunks: list[dict] = []
_index: "faiss.IndexFlatIP | None" = None
INTEGRITY_WARNING: str | None = None

VEXILON_VERSION = os.getenv("VEXILON_VERSION", "Dev mode")
VEXILON_REPO_URL = os.getenv("VEXILON_REPO_URL", "https://github.com/DerekRoberts/vexilon")
GITHUB_LABOUR_LAW_URL = os.getenv(
    "VEXILON_KNOWLEDGE_URL", f"{VEXILON_REPO_URL}/tree/main/data/labour_law"
)

# Models
DEFAULT_MODEL_LLM = os.getenv("VEXILON_DEFAULT_MODEL", "claude-haiku-4-5-20251001")
CLAUDE_MODEL = os.getenv("VEXILON_CLAUDE_MODEL", DEFAULT_MODEL_LLM)
REVIEWER_MODEL = os.getenv("VEXILON_REVIEWER_MODEL", DEFAULT_MODEL_LLM)
CONDENSE_MODEL = os.getenv("VEXILON_CONDENSE_MODEL", DEFAULT_MODEL_LLM)
VERIFY_MODEL = os.getenv("VEXILON_VERIFY_MODEL", DEFAULT_MODEL_LLM)

RAG_MAX_TOKENS = 4096
REVIEWER_MAX_TOKENS = 4096

# ─── Test Registry Logic ────────────────────────────────────────────────────
class TestDoctrine:
    def __init__(self, name: str, keywords: set[str], content: str, file_path: Path):
        self.name = name
        self.keywords = keywords
        self.content = content
        self.file_path = file_path

class TestRegistry:
    def __init__(self):
        self.tests: list[TestDoctrine] = []
        self._lock = Lock()

    def load(self, directory: Path) -> None:
        if not directory.exists(): return
        with self._lock:
            self.tests = []
            for f in directory.glob("*.md"):
                if f.name == "index.md": continue
                try:
                    text = f.read_text(encoding="utf-8")
                    lines = text.split("\n")
                    keywords = set()
                    content_start = 0
                    for i, line in enumerate(lines):
                        if line.startswith("**Keywords:**"):
                            kw_line = line.replace("**Keywords:**", "").strip()
                            keywords = {k.strip().lower() for k in kw_line.split(",") if k.strip()}
                            content_start = i + 1
                            break
                    self.tests.append(TestDoctrine(
                        name=f.stem.replace("_", " ").title(),
                        keywords=keywords,
                        content="\n".join(lines[content_start:]).strip(),
                        file_path=f
                    ))
                except Exception as e:
                    logger.error(f"[registry] Failed to load {f.name}: {e}")

    def find_matches(self, query: str) -> list[TestDoctrine]:
        q_lower = query.lower()
        with self._lock:
            return [test for test in self.tests if any(k in q_lower for k in test.keywords)]

_test_registry = TestRegistry()
TESTS_DIR = LABOUR_LAW_DIR / "tests"

# ─── RAG Pipeline Constants ─────────────────────────────────────────────────
_SIMPLE_KEYWORDS = {"phone", "number", "address", "email", "contact", "list", "who", "are", "you", "hello", "hi"}
_JOKE_KEYWORDS = {"joke", "funny", "nose", "pick", "mad", "angry", "boss", "dumb", "stupid"}
_ALL_SIMPLE_KEYWORDS = _SIMPLE_KEYWORDS | _JOKE_KEYWORDS

UNION_MANDATORY_RULES = """--- MANDATORY OPERATIONAL RULES (UNION) ---
1. ANSWER FROM EXCERPTS ONLY: Base your answer strictly on the provided excerpts.
2. CITATIONS: Every claim MUST be supported by a verbatim quote in a blockquote (> "...") followed by its citation.
3. NO MERIT ASSESSMENT: Do NOT judge the merit or likelihood of success of a grievance.
4. GRIEVANCE FILING: Facilitate the filing process by identifying potential contract violations.
"""

MANAGER_MANDATORY_RULES = """--- MANDATORY OPERATIONAL RULES (MANAGEMENT) ---
1. ANSWER FROM EXCERPTS ONLY: Base your answer strictly on the provided excerpts.
2. CITATIONS: Every claim MUST be supported by a verbatim quote in a blockquote (> "...") followed by its citation.
3. COMPLIANCE AUDIT: Proactively identify operational risks, policy gaps, and compliance failures.
4. NO UNION ADVICE: Do NOT provide guidance on grievance filing or member advocacy.
"""

def get_persona_prompt(mode_name: str) -> str:
    """Return the combined mandatory rules and persona guidelines."""
    if mode_name == "Manage":
        rules = MANAGER_MANDATORY_RULES
        persona = "You are a Senior Strategic Management Consultant focusing on compliance and risk mitigation."
    elif mode_name == "Grieve":
        rules = UNION_MANDATORY_RULES
        persona = "You are a Senior BCGEU Staff Rep acting as a Forensic Auditor to build air-tight grievance cases."
    else:
        rules = UNION_MANDATORY_RULES
        persona = "You are a BCGEU Steward Navigator. Your goal is to find specific clauses and provide literal guidance."
    
    return f"{rules}\n\nROLE: {persona}"

# ─── RAG Pipeline Functions ─────────────────────────────────────────────────
async def condense_query(message: str, history: list[dict]) -> str:
    """Turn the conversation history and new message into a standalone search query."""
    if not history: return message
    client = anthropic.AsyncAnthropic()
    
    # Actually include the history in the messages list (#369)
    messages = []
    for turn in history[-5:]: # Last 5 turns for context
        role = turn["role"] if isinstance(turn, dict) else turn.role
        content = turn["content"] if isinstance(turn, dict) else turn.content
        if isinstance(content, list):
            content = "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in content])
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": f"Condense the above conversation and this new message into a single standalone search query for a RAG system. Return ONLY the search query text: {message}"})
    
    try:
        response = await client.messages.create(
            model=CONDENSE_MODEL,
            max_tokens=100,
            messages=messages,
        )
        return response.content[0].text.strip().strip('"')
    except Exception as e:
        logger.error(f"[rag] Condense failed: {e}")
        return message

async def get_multi_perspective_context(message: str, history: list[dict]) -> tuple[list[str], str]:
    condensed = await condense_query(message, history)
    # Simplified retrieval call
    queries = [condensed]
    all_relevant_chunks = await asyncio.to_thread(search_index_batch, _index, _chunks, queries, [10])
    context = "\n\n".join([f"[Source: {c['source']}]\n{c['text']}" for c in all_relevant_chunks[0]])
    return queries, context

async def rag_review_stream(message: str, history: list[dict], persona_mode: str = "Lookup") -> AsyncIterator[str]:
    if _index is None:
        yield "⚠️ Index not ready."
        return

    queries, context = await get_multi_perspective_context(message, history)
    
    # Simple Draft Step
    client = anthropic.AsyncAnthropic()
    system_prompt = get_persona_prompt(persona_mode)
    prompt = f"Context from Knowledge Base:\n{context}\n\nQuestion: {message}"
    
    async with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for text in stream.text_stream:
            yield text

# ─── UI Utility Functions ───────────────────────────────────────────────────
def _get_download_source_files() -> list[Path]:
    """Scan LABOUR_LAW_DIR for PDF files (human downloads). Excludes tests/."""
    if not LABOUR_LAW_DIR.exists(): return []
    tests_dir = LABOUR_LAW_DIR / "tests"
    pdfs = [p for p in LABOUR_LAW_DIR.rglob("*.pdf") if not p.is_relative_to(tests_dir)]
    return sorted(list(set(pdfs)), key=lambda p: str(p))

def build_pdf_download_links() -> str:
    """Scan labour_law for PDFs and return a Markdown list of download links."""
    files = _get_download_source_files()
    if not files: return ""
    lines = []
    for f in files:
        display_name = f.stem.replace("_", " ").title()
        display_name = display_name.replace("Bcgeu", "BCGEU").replace("Main Agreement", "Agreement")
        display_name = display_name.replace("Bc ", "BC ").replace(" Bc", " BC")
        rel_path = f.relative_to(Path("."))
        encoded_path = urllib.parse.quote(str(rel_path))
        lines.append(f"* [{display_name}](/file={encoded_path})")
    return "\n".join(lines)

def history_to_markdown(history: list[dict]) -> str:
    """Convert chat history to a Markdown string for export."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md = f"# Vexilon Conversation Export - {timestamp}\n\n"
    for turn in (history or []):
        role = (turn["role"] if isinstance(turn, dict) else turn.role).capitalize()
        content = turn["content"] if isinstance(turn, dict) else turn.content
        if isinstance(content, list):
            text_parts = [part.get("text", "") if isinstance(part, dict) else str(part) for part in content]
            content = "".join(text_parts)
        md += f"### {role}\n{content}\n\n"
    return md

def markdown_to_history(file_path: str) -> list[dict]:
    """Parse a Markdown conversation file back into a list of dicts."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    history, current_role, current_content = [], None, []
    for line in lines:
        new_role = None
        if line.startswith("### User"): new_role = "user"
        elif line.startswith("### Assistant"): new_role = "assistant"
        if new_role:
            if current_role:
                history.append({"role": current_role, "content": "\n".join(current_content).strip()})
            current_role, current_content = new_role, []
        elif current_role:
            current_content.append(line.rstrip("\n"))
    if current_role:
        history.append({"role": current_role, "content": "\n".join(current_content).strip()})
    return history

# ─── Gradio App Logic ───────────────────────────────────────────────────────
def startup(force_rebuild: bool = False):
    global _index, _chunks, INTEGRITY_WARNING
    _test_registry.load(TESTS_DIR)
    _fetch_pdf_cache_if_missing()
    _index, _chunks = load_precomputed_index()
    if _index is None or force_rebuild:
        _index, _chunks = build_index_from_sources(force=True)
    if _index is not None:
        report = get_integrity_report()
        if report.get("failed_files"):
            INTEGRITY_WARNING = f"⚠️ Index Incomplete: {len(report['failed_files'])} documents failed."

async def chat_fn(message, history, persona):
    history = history or []
    if not message:
        yield "", history, gr.update()
        return
    
    # 1. Update history with user message
    new_history = history + [{"role": "user", "content": message}]
    yield "", new_history, gr.update(open=False)
    
    # 2. Stream assistant response
    accumulated = ""
    async for chunk in rag_review_stream(message, history, persona):
        accumulated += chunk
        # Update history with current accumulated response
        current_history = new_history + [{"role": "assistant", "content": accumulated}]
        yield "", current_history, gr.update(open=False)

# ─── UI Layout ──────────────────────────────────────────────────────────────
EXAMPLES = [
    "What are the just cause requirements for discipline?",
    "What rights do stewards have in investigation meetings?",
    "What is the nexus test for establishing a link in off-duty conduct cases?",
    "Show me the Harassment Threshold test.",
    "Does my employer have a social media policy?"
]

CLOSE_ACCORDION_JS = """
() => {
    const btn = document.querySelector('#quick-questions-accordion .label-wrap');
    if (btn && btn.classList.contains('open')) {
        btn.click();
    }
}
"""

_CSS = """
footer { display: none !important; }
/* Hanging indent for the Resources & Utilities list items */
#resources-accordion .prose ul {
    list-style-position: outside;
    padding-left: 1.5rem;
}
#resources-accordion .prose li {
    margin-bottom: 0.5rem;
}
"""

if __name__ == "__main__":
    startup()

with gr.Blocks(title="BCGEU Navigator", fill_height=True) as demo:
    with gr.Row():
        gr.HTML("<div style='display: flex; height: 100%; align-items: center;'><h3 style='margin: 0;'>BCGEU Navigator</h3></div>")
        persona = gr.Dropdown(
            choices=["Lookup", "Grieve", "Manage"],
            value="Lookup",
            show_label=False,
            container=False,
            min_width=100,
            interactive=True
        )
    
    chatbot = gr.Chatbot(show_label=False, scale=1, height="70vh", min_height=400)
    
    with gr.Row():
        msg = gr.Textbox(show_label=False, placeholder="Type a message...", container=False, scale=7)
        submit = gr.Button("Send", variant="primary", scale=1)

    with gr.Accordion("Toolbox", open=False, elem_id="steward-toolbox") as toolbox:
        gr.Markdown("### Examples")
        with gr.Row():
            for q in EXAMPLES:
                example_btn = gr.Button(q, size="sm", variant="secondary")
                example_btn.click(
                    chat_fn, 
                    [gr.State(q), chatbot, persona], 
                    [msg, chatbot, toolbox],
                    js=CLOSE_ACCORDION_JS.replace("quick-questions-accordion", "steward-toolbox")
                )

        gr.Markdown("---")
        if INTEGRITY_WARNING:
            gr.Markdown(f"⚠️ {INTEGRITY_WARNING}")
        gr.Markdown(f"### Reference Documents\n{build_pdf_download_links()}")
        gr.Markdown(f"[Browse Full Knowledge Base on GitHub]({GITHUB_LABOUR_LAW_URL})")
        
        gr.Markdown("---")
        gr.Markdown("### Conversation Tools")
        with gr.Row():
            export_btn = gr.DownloadButton("⬇️ Save Conversation", variant="secondary", size="sm")
            import_btn = gr.UploadButton("⬆️ Load Conversation", file_types=[".md"], variant="secondary", size="sm")

    # ── Export / Import Handlers ──────────────────────────────────────────
    def handle_export(history):
        if not history: return None
        md_str = history_to_markdown(history)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        filename = f"vexilon_chat_{timestamp}.md"
        save_path = os.path.join(tempfile.gettempdir(), filename)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(md_str)
        threading.Timer(600, lambda: os.path.exists(save_path) and os.remove(save_path)).start()
        return save_path

    export_btn.click(fn=handle_export, inputs=[chatbot], outputs=[export_btn])

    def handle_import(file):
        if file is None: return gr.update()
        try:
            return markdown_to_history(file.name)
        except Exception:
            logger.error("[ui] Import failed", exc_info=True)
            return gr.update()

    import_btn.upload(fn=handle_import, inputs=[import_btn], outputs=[chatbot])

    gr.HTML(f"""
        <div style="text-align: center; color: #6b7280; font-size: 0.85rem; padding: 10px 0;">
            <a href="{VEXILON_REPO_URL}" target="_blank" style="color: #3b82f6; text-decoration: none;">GitHub</a>
            &nbsp;&nbsp;•&nbsp;&nbsp;
            <a href="{VEXILON_REPO_URL}/blob/main/docs/PRIVACY.md" target="_blank" style="color: #3b82f6; text-decoration: none;">Privacy</a>
            &nbsp;&nbsp;•&nbsp;&nbsp;
            <a href="{VEXILON_REPO_URL}" target="_blank" style="color: #3b82f6; text-decoration: none;">{VEXILON_VERSION[:11]}</a>
        </div>
    """)

    msg.submit(chat_fn, [msg, chatbot, persona], [msg, chatbot, toolbox], js=CLOSE_ACCORDION_JS.replace("quick-questions-accordion", "steward-toolbox"))
    submit.click(chat_fn, [msg, chatbot, persona], [msg, chatbot, toolbox], js=CLOSE_ACCORDION_JS.replace("quick-questions-accordion", "steward-toolbox"))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, css=_CSS)
