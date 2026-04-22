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
logger = logging.getLogger(__name__)
_chunks: list[dict] = []
_index: "faiss.IndexFlatIP | None" = None
INTEGRITY_WARNING: str | None = None

VEXILON_VERSION = os.getenv("VEXILON_VERSION", "Dev mode")
VEXILON_REPO_URL = os.getenv("VEXILON_REPO_URL", "https://github.com/DerekRoberts/vexilon")

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

GLOBAL_MANDATORY_RULES = """--- MANDATORY OPERATIONAL RULES ---
1. ANSWER FROM EXCERPTS ONLY: Base your answer strictly on the provided excerpts.
2. CITATIONS: Every claim MUST be supported by a verbatim quote in a blockquote (> "...") followed by its citation.
3. NO MERIT ASSESSMENT: Do NOT judge the merit or likelihood of success of a grievance.
"""

REVIEWER_SYSTEM_PROMPT = """You are a Senior BCGEU Staff Representative reviewing a junior steward's output for accuracy and completeness.
SCORE: [1-10]
VERIFIED STEPS: [final safe instructions]
ISSUES: [specific errors]
"""

REFINER_SYSTEM_PROMPT = "You are a Senior BCGEU Expert Rep. Synthesize the draft and critique into a final response."

# ─── RAG Pipeline Functions ─────────────────────────────────────────────────
async def condense_query(message: str, history: list[dict]) -> str:
    if not history: return message
    client = anthropic.AsyncAnthropic()
    prompt = f"Condense this history and message into a standalone search query: {message}"
    try:
        response = await client.messages.create(
            model=CONDENSE_MODEL,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip().strip('"')
    except: return message

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
    prompt = f"Mode: {persona_mode}\nContext: {context}\nQuestion: {message}"
    
    async with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=GLOBAL_MANDATORY_RULES,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for text in stream.text_stream:
            yield text

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

_CSS = "footer {display: none !important;}"

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
    
    chatbot = gr.Chatbot(show_label=False, scale=1, min_height=400)
    
    with gr.Row():
        msg = gr.Textbox(show_label=False, placeholder="Type a message...", container=False, scale=7)
        submit = gr.Button("Send", variant="primary", scale=1)

    with gr.Accordion("Quick Questions", open=False, elem_id="quick-questions-accordion") as examples_accordion:
        with gr.Row():
            for q in EXAMPLES:
                example_btn = gr.Button(q, size="sm", variant="secondary")
                example_btn.click(
                    chat_fn, 
                    [gr.State(q), chatbot, persona], 
                    [msg, chatbot, examples_accordion],
                    js=CLOSE_ACCORDION_JS
                )

    gr.HTML(f"""
        <div style="text-align: center; color: #6b7280; font-size: 0.85rem; padding: 10px 0;">
            <a href="{VEXILON_REPO_URL}" target="_blank" style="color: #3b82f6; text-decoration: none;">GitHub</a>
            &nbsp;&nbsp;•&nbsp;&nbsp;
            <a href="{VEXILON_REPO_URL}/blob/main/docs/PRIVACY.md" target="_blank" style="color: #3b82f6; text-decoration: none;">Privacy</a>
            &nbsp;&nbsp;•&nbsp;&nbsp;
            <a href="{VEXILON_REPO_URL}" target="_blank" style="color: #3b82f6; text-decoration: none;">{VEXILON_VERSION}</a>
        </div>
    """)

    msg.submit(chat_fn, [msg, chatbot, persona], [msg, chatbot, examples_accordion], js=CLOSE_ACCORDION_JS)
    submit.click(chat_fn, [msg, chatbot, persona], [msg, chatbot, examples_accordion], js=CLOSE_ACCORDION_JS)

if __name__ == "__main__":
    startup()
    port = int(os.getenv("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, css=_CSS)
