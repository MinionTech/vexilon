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
    get_embed_model,
    EMBED_DIM,
    chunk_text,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
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

MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", 10000))
LOG_SUSPICIOUS_INPUTS = os.getenv("LOG_SUSPICIOUS_INPUTS", "true").lower() == "true"

PROMPT_INJECTION_PATTERNS = [
    re.compile(r, re.IGNORECASE)
    for r in [
        r"ignore\s+.*instructions",
        r"forget\s+.*instructions",
        r"disregard\s+.*rules",
        r"you\s+are\s+now\s+.+\s+instead",
        r"new\s+(system\s+|)prompt:",
        r"#\#\#\s*(system\s+|)instructions",
        r"\[\[SYSTEM\]\]",
        r"override\s+.*instructions",
        r"disable\s+.*safety",
        r"\bjailbreak\b",
        r"developer\s+mode",
        r"sudo\s+mode",
        r"roleplay\s+as",
        r"pretend\s+(you\s+are|to\s+be)",
        r"forget\s+everything\s+above",
        r"discard\s+.*instructions",
    ]
]

def sanitize_input(user_input: str) -> tuple[str, bool]:
    """Check for prompt injection patterns and length limits."""
    if not user_input:
        return user_input, False

    injection_found = False
    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(user_input):
            injection_found = True
            if LOG_SUSPICIOUS_INPUTS:
                logger.warning(f"[security] Prompt injection detected: {pattern.pattern[:100]}...")
            break

    too_long = len(user_input) > MAX_INPUT_LENGTH
    if too_long and LOG_SUSPICIOUS_INPUTS:
        logger.warning(f"[security] Input too long: {len(user_input)}")

    return user_input[:MAX_INPUT_LENGTH], injection_found or too_long

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

# ─── Rate Limiter ───────────────────────────────────────────────────────────
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "100"))

class RateLimiter:
    def __init__(self, max_per_minute: int = 10, max_per_hour: int = 100):
        self.minute_limit = max_per_minute
        self.hour_limit = max_per_hour
        self.requests: dict[str, list[float]] = {}
        self._lock = Lock()

    def _clean_old_requests(self, key: str) -> None:
        now = time.time()
        hour_ago = now - 3600
        if key in self.requests:
            self.requests[key] = [t for t in self.requests[key] if t > hour_ago]
            if not self.requests[key]: del self.requests[key]

    def is_allowed(self, user_id: str = "default") -> tuple[bool, str]:
        with self._lock:
            self._clean_old_requests(user_id)
            now = time.time()
            minute_ago = now - 60
            requests = self.requests.get(user_id, [])
            recent = [t for t in requests if t > minute_ago]
            if len(recent) >= self.minute_limit:
                return False, f"Rate limit exceeded: {self.minute_limit} per minute."
            if len(requests) >= self.hour_limit:
                return False, f"Rate limit exceeded: {self.hour_limit} per hour."
            self.requests.setdefault(user_id, []).append(now)
            return True, ""

_rate_limiter = RateLimiter(RATE_LIMIT_PER_MINUTE, RATE_LIMIT_PER_HOUR)

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

# ─── Verification & Anthropic Helpers ───────────────────────────────────────
VERIFY_ENABLED = os.getenv("VERIFY_ENABLED", "true").lower() == "true"
VERIFY_SYSTEM_PROMPT = """You are a verification assistant. Your job is to verify that the claims made in an AI response are supported by the provided source citations.

For each claim in the response:
1. Check if the quoted text actually supports the claim being made
2. Check if the citation (document name, article/section, page number) is accurate
3. Identify any hallucinations, misquotes, or unsupported claims

Respond in this format:
- VERIFIED: [claim summary] — the quote supports the claim
- DISPUTED: [claim summary] — the quote does NOT support the claim
- UNCERTAIN: [claim summary] — cannot verify due to unclear citation

If all claims are verified, respond with "ALL_CLAIMS_VERIFIED".
If there are disputed claims, list them with explanations."""

_anthropic_client = None
def get_anthropic() -> anthropic.AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.AsyncAnthropic()
    return _anthropic_client

async def verify_response(assistant_response: str, context: str) -> str:
    if not VERIFY_ENABLED: return ""
    client = get_anthropic()
    try:
        verify_resp = await client.messages.create(
            model=VERIFY_MODEL,
            max_tokens=512,
            system=[{"type": "text", "text": VERIFY_SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": f"RESPONSE:\n{assistant_response}\n\nCONTEXT:\n{context}"}],
        )
        return verify_resp.content[0].text
    except Exception as exc:
        return f"⚠️ Verification unavailable: {exc}"

def get_system_prompt(developer_mode: bool = False) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d")
    header = f"--- VEXILON SYSTEM STATE ---\nDATE: {now}\nVERSION: {VEXILON_VERSION}\n----------------------------\n\n"
    content = "You are Vexilon, a professional assistant for BCGEU union stewards.\n\nKnowledge Base:\n{manifest}\n\n{verify_message}"
    return f"{header}{content}"

async def rag_stream(message: str, history: list[dict]) -> AsyncIterator[tuple[str, str]]:
    """Yields (chunk, context) for tests and legacy callers."""
    if _index is None:
        yield "⚠️ Knowledge base not loaded.", ""
        return
    queries, context = await get_multi_perspective_context(message, history)
    yield "", context
    client = get_anthropic()
    async with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=RAG_MAX_TOKENS,
        system=[
            {"type": "text", "text": get_system_prompt().format(manifest="", verify_message=""), "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": f"Context:\n{context}", "cache_control": {"type": "ephemeral"}},
        ],
        messages=history + [{"role": "user", "content": message}],
    ) as stream:
        async for chunk in stream.text_stream:
            yield chunk, ""

# ─── RAG Pipeline Functions ─────────────────────────────────────────────────
async def condense_query(message: str, history: list[dict]) -> str:
    """Turn the conversation history and new message into a standalone search query."""
    if not history: return message
    client = get_anthropic()
    
    history_text = ""
    for turn in history[-5:]:
        role = (turn["role"] if isinstance(turn, dict) else turn.role).capitalize()
        content = turn["content"] if isinstance(turn, dict) else turn.content
        if isinstance(content, list):
            content = "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in content])
        history_text += f"{role}: {content}\n"
    
    prompt = f"CONVERSATION HISTORY:\n{history_text}\nUSER MESSAGE: {message}\n\nTask: Condense into a standalone search query."
    try:
        resp = await client.messages.create(
            model=CONDENSE_MODEL,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text.strip().strip('"')
    except Exception:
        return message

async def generate_perspective_queries(message: str, history: list[dict]) -> list[str]:
    """Generate 3 different search perspectives for a complex query."""
    client = get_anthropic()
    prompt = f"Given this user message: '{message}', generate 3 different standalone search queries that look at this from different angles (e.g. legal, procedural, factual). Return ONLY a JSON list of strings."
    try:
        resp = await client.messages.create(
            model=CONDENSE_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        # More robust extraction using regex to find the JSON block
        text = resp.content[0].text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            import json
            return json.loads(match.group(0))
        return [message]
    except Exception:
        return [message]

async def get_multi_perspective_context(message: str, history: list[dict]) -> tuple[list[str], str]:
    condensed = await condense_query(message, history)
    # Issue #361: Heuristic for complexity
    if len(condensed.split()) > 10:
        queries = await generate_perspective_queries(condensed, history)
    else:
        queries = [condensed]
    
    all_res = search_index_batch(_index, _chunks, queries, [5] * len(queries))
    seen = set()
    context_parts = []
    for res_list in all_res:
        for c in res_list:
            if c["text"] not in seen:
                seen.add(c["text"])
                source = c.get("source", "Unknown")
                page = c.get("page", "?")
                context_parts.append(f"[Source: {source}, Page: {page}]\n{c['text']}")
    return queries, "\n\n".join(context_parts)

async def rag_review_stream(message: str, history: list[dict], persona_mode: str = "Lookup", all_chunks: list[dict] = None) -> AsyncIterator[str]:
    if _index is None:
        yield "⚠️ Index not ready."
        return

    queries, context = await get_multi_perspective_context(message, history)
    query = queries[0]
    
    # ── Audit Logic (Restored) ──────────────────────────────────────────────
    system_prompt = get_persona_prompt(persona_mode)
    if persona_mode in ("Grieve", "Manage"):
        matched_tests = _test_registry.find_matches(message + " " + query)
        for test in matched_tests:
            system_prompt += f"\n\n--- MANDATORY LOGIC CHECK: {test.name.upper()} ---\n"
            system_prompt += f"This case involves potential {test.name}. You MUST follow the EXPLAIN/QUESTION/APPLY/CITE pattern.\n"
            system_prompt += f"CRITERIA:\n{test.content}\n"

    # Simple Draft Step
    client = get_anthropic()
    prompt = f"Context from Knowledge Base:\n{context}\n\nQuestion: {message}"
    
    async with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=[
            {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": f"Context from Knowledge Base:\n{context}", "cache_control": {"type": "ephemeral"}},
        ],
        messages=[{"role": "user", "content": message}],
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

# (Obsolete build_pdf_download_links removed)

def history_to_markdown(history: list[dict]) -> str:
    """Convert chat history to a Markdown string for export."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"# Vexilon Conversation Export - {timestamp}\n"]
    for turn in (history or []):
        role = (turn["role"] if isinstance(turn, dict) else turn.role).capitalize()
        content = turn["content"] if isinstance(turn, dict) else turn.content
        if isinstance(content, list):
            content = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in content])
        lines.append(f"### {role}\n{content}\n")
    return "\n".join(lines)

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

async def chat_fn(message, history, persona, request: gr.Request = None):
    # 0. Rate Limit & Security Check
    user_id = request.client.host if request else "default"
    allowed, rate_msg = _rate_limiter.is_allowed(user_id)
    if not allowed:
        yield gr.update(), history + [{"role": "assistant", "content": rate_msg}], gr.update()
        return

    msg_str = message
    if isinstance(message, list):
        msg_str = "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in message])
    
    sanitized, flagged = sanitize_input(msg_str)
    if flagged:
        yield gr.update(), history + [{"role": "assistant", "content": "⚠️ Input flagged for security review."}], gr.update()
        return

    # Normalization for Gradio 6 / Backend compatibility
    if history:
        history = [
            h if isinstance(h, dict) else {"role": h.role, "content": h.content}
            for h in history
        ]
    history = history or []
    
    if not message:
        yield gr.update(value=""), history, gr.update()
        return
        
    new_history = history + [{"role": "user", "content": sanitized}]
    yield gr.update(value=""), new_history, gr.update(open=False)
    
    # 2. Stream assistant response
    accumulated = ""
    async for chunk in rag_review_stream(sanitized, history, persona):
        accumulated += chunk
        # Update history with current accumulated response
        current_history = new_history + [{"role": "assistant", "content": accumulated}]
        yield gr.update(), current_history, gr.update(open=False)

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
#steward-toolbox .prose ul {
    list-style-position: outside;
    padding-left: 1.5rem;
}
#steward-toolbox .prose li {
    margin-bottom: 0.5rem;
}
/* Aggressive button suppression for Gradio 6.x UI stability */
.message-buttons, .share-button, .undo-button, .retry-button, .copy-button, .clear-button, button[aria-label="Clear"] {
    display: none !important;
}
/* Prevent infinite growth in iframes while maintaining responsiveness */
.is-iframe .gradio-chatbot {
    max-height: 75vh !important;
    height: auto !important;
}
"""

if __name__ == "__main__":
    startup()

def build_ui() -> gr.Blocks:
    """Export the demo object for tests."""
    return demo

_HEAD = """
<script>
    if (window.self !== window.top) {
        document.documentElement.classList.add('is-iframe');
    }
    // Handle Enter key for submission (Issue #118)
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            const textarea = document.querySelector('.gradio-container textarea');
            if (textarea && document.activeElement === textarea) {
                e.preventDefault();
                const sendBtn = document.querySelector('button.primary');
                if (sendBtn) sendBtn.click();
            }
        }
    }, true);
</script>
"""

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
    
    chatbot = gr.Chatbot(
        show_label=False, 
        scale=1, 
        height="70vh", 
        min_height=400, 
        buttons=[]
    )
    
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
        gr.Markdown("### Reference Documents")
        
        # Use native Gradio components for reliable file serving on HF Spaces and Localhost
        files = _get_download_source_files()
        for f in files:
            display_name = f.stem.replace("_", " ").title()
            display_name = display_name.replace("Bcgeu", "BCGEU").replace("Main Agreement", "Agreement")
            display_name = display_name.replace("Bc ", "BC ").replace(" Bc", " BC")
            
            # Restore the relative path improvement for container reliability
            try:
                val = str(f.relative_to(Path.cwd()))
            except ValueError:
                val = str(f.resolve())
                
            gr.DownloadButton(display_name, value=val, size="sm", variant="secondary")
            
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

    _URL_VEXILON_VERSION = urllib.parse.quote(VEXILON_VERSION)
    gr.HTML(f"""
        <div style="text-align: center; color: #6b7280; font-size: 0.85rem; padding: 10px 0;">
            <a href="{VEXILON_REPO_URL}" target="_blank" style="color: #3b82f6; text-decoration: none;">GitHub</a>
            &nbsp;&nbsp;•&nbsp;&nbsp;
            <a href="{VEXILON_REPO_URL}/blob/main/docs/PRIVACY.md" target="_blank" style="color: #3b82f6; text-decoration: none;">Privacy</a>
            &nbsp;&nbsp;•&nbsp;&nbsp;
            <a href="{VEXILON_REPO_URL}/pkgs/container/vexilon/versions?filters%5Bversion_type%5D=tagged&query={_URL_VEXILON_VERSION}" target="_blank" style="color: #3b82f6; text-decoration: none;">{VEXILON_VERSION[:11]}</a>
        </div>
    """)

    msg.submit(chat_fn, [msg, chatbot, persona], [msg, chatbot, toolbox])
    submit.click(chat_fn, [msg, chatbot, persona], [msg, chatbot, toolbox])

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    
    # Restore allowed_paths for local file serving (fixes 404s for PDFs)
    # Using resolve() ensures we handle symlinks and container volume mounts correctly
    allowed_paths = [
        str(LABOUR_LAW_DIR.resolve()), 
        str(Path("docs").resolve()),
        str(Path("data").resolve()),
        os.getcwd()
    ]
    
    # Restore basic auth if configured in environment
    vex_password = os.getenv("VEXILON_PASSWORD")
    auth = None
    if vex_password:
        vex_user = os.getenv("VEXILON_USERNAME", "admin")
        auth = (vex_user, vex_password)
        logger.info(f"[startup] Authentication enabled for user '{vex_user}'")

    demo.launch(
        server_name="0.0.0.0", 
        server_port=port, 
        allowed_paths=allowed_paths,
        auth=auth,
        css=_CSS, 
        head=_HEAD
    )
