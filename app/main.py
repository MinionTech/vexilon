import os
# Force online mode for the API but keep local models offline for speed
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import re
# Agreement Navigator - UI Version: 2026-05-03
# Integrated RAG Backend + Stabilized Gradio 6 UI
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
import openai
from openai import AsyncOpenAI

import faiss
import chainlit as cl
from chainlit.input_widget import Select

# ─── Agnav Imports ────────────────────────────────────────────────────────
from indexing import (
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
# Single Source of Truth for local development models.
OLLAMA_MODEL_ID = "qwen3:14b"
# Allow environment override for CI (e.g. tinyllama for smoke tests)
CURRENT_MODEL_ID = os.getenv("OLLAMA_MODEL_ID", OLLAMA_MODEL_ID)

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

AGNAV_VERSION = os.getenv("AGNAV_VERSION", "Dev mode")
IS_DEV = AGNAV_VERSION == "Dev mode"
AGNAV_REPO_URL = os.getenv("AGNAV_REPO_URL", "https://github.com/MinionTech/vexilon")
GITHUB_LABOUR_LAW_URL = os.getenv(
    "AGNAV_KNOWLEDGE_URL", f"{AGNAV_REPO_URL}/tree/main/data/labour_law"
)

# Models & Providers
def get_llm_provider() -> str:
    # 1. Explicit override (keeps the 'prod' profile working)
    val = os.getenv("AGNAV_LLM_PROVIDER")
    if val:
        return val.lower().strip()

    # 2. Smart Detection based on version
    if IS_DEV:
        return "ollama"  # We're coding locally!
    return "huggingface" # We're in the clouds!

def _get_default_model():
    provider = get_llm_provider()
    # Default to Hugging Face or Ollama
    if provider == "ollama":
        val = os.getenv("OLLAMA_MODEL")
        return val if (val and val.strip()) else CURRENT_MODEL_ID
    return "Qwen/Qwen3-32B"

DEFAULT_MODEL_LLM = os.getenv("AGNAV_DEFAULT_MODEL", _get_default_model())
HF_PROVIDER = os.getenv("AGNAV_HF_PROVIDER", "featherless-ai")
CLAUDE_MODEL = os.getenv("AGNAV_CLAUDE_MODEL", DEFAULT_MODEL_LLM)
REVIEWER_MODEL = os.getenv("AGNAV_REVIEWER_MODEL", DEFAULT_MODEL_LLM)
CONDENSE_MODEL = os.getenv("AGNAV_CONDENSE_MODEL", DEFAULT_MODEL_LLM)
VERIFY_MODEL = os.getenv("AGNAV_VERIFY_MODEL", DEFAULT_MODEL_LLM)

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
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "999999" if IS_DEV else "10"))
RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "999999" if IS_DEV else "100"))

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
2. STRICT CITATIONS: Every claim MUST be supported by a verbatim quote followed by its citation.
   EXAMPLE: > "verbatim text" [Document Name, Page X]
3. STRUCTURE: Use clear headings, bullet points, and numbered lists to organize complex answers.
4. NO MERIT ASSESSMENT: Do NOT judge the merit or likelihood of success of a grievance.
5. GRIEVANCE FILING: Facilitate the filing process by identifying potential contract violations.
"""

MANAGER_MANDATORY_RULES = """--- MANDATORY OPERATIONAL RULES (MANAGEMENT) ---
1. ANSWER FROM EXCERPTS ONLY: Base your answer strictly on the provided excerpts.
2. STRICT CITATIONS: Every claim MUST be supported by a verbatim quote followed by its citation.
   EXAMPLE: > "verbatim text" [Document Name, Page X]
3. STRUCTURE: Use clear headings, bullet points, and numbered lists to organize complex answers.
4. COMPLIANCE AUDIT: Proactively identify operational risks, policy gaps, and compliance failures.
5. INADVERTENT BENEFIT WARNING: If a manager suggests a "Nuclear Option" (Suspension/Firing) for a minor variance, you MUST warn them that skipping Progressive Discipline (Article 14) is a "Low-ROI Strategy" that often results in "Remedial Back-Pay Awards".
6. NO UNION ADVICE: Do NOT provide guidance on grievance filing or member advocacy.
"""

def get_persona_prompt(mode_name: str) -> str:
    """Return the combined mandatory rules and persona guidelines."""
    if mode_name == "Manage":
        rules = MANAGER_MANDATORY_RULES
        persona = "You are a Senior Strategic Management Consultant focusing on compliance and risk mitigation within the Operational Framework."
    elif mode_name == "Grieve":
        rules = UNION_MANDATORY_RULES
        persona = "You are a Senior BCGEU Staff Rep acting as a Forensic Auditor to build air-tight grievance cases."
    else:
        rules = UNION_MANDATORY_RULES
        persona = "You are a BCGEU Steward Navigator. Your goal is to find specific clauses and provide literal guidance."
    
    return f"{rules}\n\nROLE: {persona}"

# ─── Verification & LLM Helpers ───────────────────────────────────────
VERIFY_ENABLED = os.getenv("VERIFY_ENABLED", "false" if IS_DEV else "true").lower() == "true"
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

_llm_client = None
def get_llm_client():
    global _llm_client
    if _llm_client is None:
        provider = get_llm_provider()
        if provider == "huggingface":
            # Use the OpenAI-compatible router endpoint for reliable routing
            _llm_client = AsyncOpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=os.environ.get("HF_TOKEN")
            )
        elif provider == "ollama":
            ollama_host = os.getenv("OLLAMA_HOST", "ollama:11434")
            if "://" not in ollama_host:
                ollama_host = f"http://{ollama_host}"
            _llm_client = AsyncOpenAI(
                base_url=f"{ollama_host.rstrip('/')}/v1",
                api_key="ollama"
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    return _llm_client

def _build_messages(messages: list, system: str | list = None) -> list:
    full_messages = []
    if system:
        if isinstance(system, list):
            system_text = "\n\n".join([b["text"] if isinstance(b, dict) else str(b) for b in system])
        else:
            system_text = system
        full_messages.append({"role": "system", "content": system_text})
    full_messages.extend(messages)
    return full_messages

async def unified_chat_create(model: str, messages: list, system: str | list = None, max_tokens: int = 1024) -> str:
    client = get_llm_client()
    full_messages = _build_messages(messages, system)
    
    # Use the 'model:provider' syntax for the most robust routing on the HF Router
    actual_model = f"{model}:{HF_PROVIDER}" if get_llm_provider() == "huggingface" else model
    kwargs = {"model": actual_model, "max_tokens": max_tokens, "messages": full_messages, "timeout": 300.0}

    resp = await client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content

async def unified_chat_stream(model: str, messages: list, system: str | list = None, max_tokens: int = 2048) -> AsyncIterator[str]:
    client = get_llm_client()
    full_messages = _build_messages(messages, system)
    
    # Use the 'model:provider' syntax for the most robust routing on the HF Router
    actual_model = f"{model}:{HF_PROVIDER}" if get_llm_provider() == "huggingface" else model
    kwargs = {"model": actual_model, "max_tokens": max_tokens, "messages": full_messages, "stream": True, "timeout": 300.0}

    stream = await client.chat.completions.create(**kwargs)
    # Stateful buffer for filtering <think> blocks (handles split-token tags)
    in_think_block = False
    buffer = ""
    start_tag = "<think>"
    end_tag = "</think>"
    
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            buffer += chunk.choices[0].delta.content
            
            while buffer:
                if not in_think_block:
                    # Look for start tag
                    start_idx = buffer.find(start_tag)
                    if start_idx != -1:
                        # Yield everything before the tag
                        if start_idx > 0:
                            yield buffer[:start_idx]
                        in_think_block = True
                        buffer = buffer[start_idx + len(start_tag):] # Skip start tag
                    else:
                        # No complete start tag found. 
                        # But wait! What if we have a partial tag at the end (e.g. '...<thi')?
                        partial_idx = buffer.find("<")
                        if partial_idx != -1 and len(buffer[partial_idx:]) < len(start_tag):
                            # Yield everything before the partial tag and keep the partial in buffer
                            if partial_idx > 0:
                                yield buffer[:partial_idx]
                            buffer = buffer[partial_idx:]
                            break
                        else:
                            # Safe to yield everything
                            yield buffer
                            buffer = ""
                else:
                    # In a think block, look for end tag
                    end_idx = buffer.find(end_tag)
                    if end_idx != -1:
                        in_think_block = False
                        buffer = buffer[end_idx + len(end_tag):] # Skip end tag
                    else:
                        # Still thinking, discard buffer (careful not to discard a partial end tag)
                        partial_end_idx = buffer.find("<")
                        if partial_end_idx != -1 and len(buffer[partial_end_idx:]) < len(end_tag):
                            # Keep potential partial end tag
                            buffer = buffer[partial_end_idx:]
                            break
                        else:
                            buffer = ""
                            break

async def verify_response(assistant_response: str, context: str) -> str:
    if not VERIFY_ENABLED: return ""
    try:
        return await unified_chat_create(
            model=VERIFY_MODEL,
            max_tokens=512,
            system=VERIFY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"RESPONSE:\n{assistant_response}\n\nCONTEXT:\n{context}"}]
        )
    except Exception as exc:
        return f"⚠️ Verification unavailable: {exc}"

def get_system_prompt(developer_mode: bool = False) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d")
    header = f"--- BCGEU NAVIGATOR SYSTEM STATE ---\nDATE: {now}\nVERSION: {AGNAV_VERSION}\n----------------------------\n\n"
    content = "You are BCGEU Navigator, a professional assistant for BCGEU union stewards. IMPORTANT: DO NOT use <think> tags. Provide your answer directly and professionally. ALWAYS cite your sources using the [Source, Page] format provided in the context.\n\nKnowledge Base:\n{manifest}\n\n{verify_message}"
    return f"{header}{content}"

async def rag_stream(message: str, history: list[dict]) -> AsyncIterator[tuple[str, str]]:
    """Yields (chunk, context) for tests and legacy callers."""
    if _index is None:
        yield "⚠️ Knowledge base not loaded.", ""
        return
    try:
        queries, context = await get_multi_perspective_context(message, history)
        
        system = get_system_prompt().format(manifest="", verify_message="") + f"\n\nContext:\n{context}"

        # Cap history to last 2 turns, truncated to reduce prompt size on CPU
        capped = []
        for h in history[-2:]:
            c = h["content"] if isinstance(h["content"], str) else str(h["content"])
            capped.append({"role": h["role"], "content": c[:300] + "..." if len(c) > 300 else c})
        messages = capped + [{"role": "user", "content": message}]
        
        has_yielded_context = False
        async for chunk in unified_chat_stream(
            model=CLAUDE_MODEL,
            max_tokens=RAG_MAX_TOKENS,
            system=system,
            messages=messages
        ):
            if not has_yielded_context:
                yield "", context
                has_yielded_context = True
            yield chunk, ""
    except Exception as exc:
        yield f"⚠️ API error: {exc}", ""

# ─── RAG Pipeline Functions ─────────────────────────────────────────────────
async def condense_query(message: str, history: list[dict]) -> str:
    """Turn the conversation history and new message into a standalone search query."""
    if not history: return message
    
    history_text = ""
    for turn in history[-5:]:
        role = (turn["role"] if isinstance(turn, dict) else turn.role).capitalize()
        content = turn["content"] if isinstance(turn, dict) else turn.content
        if isinstance(content, list):
            content = "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in content])
        history_text += f"{role}: {content}\n"
    
    prompt = f"CONVERSATION HISTORY:\n{history_text}\nUSER MESSAGE: {message}\n\nTask: Condense into a standalone search query."
    try:
        resp_text = await unified_chat_create(
            model=CONDENSE_MODEL,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp_text.strip().strip('"')
    except Exception:
        return message

async def generate_perspective_queries(message: str, history: list[dict]) -> list[str]:
    """Generate 3 different search perspectives for a complex query."""
    prompt = f"Given this user message: '{message}', generate 3 different standalone search queries that look at this from different angles (e.g. legal, procedural, factual). Return ONLY a JSON list of strings."
    try:
        text = await unified_chat_create(
            model=CONDENSE_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        # More robust extraction using regex to find the JSON block
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            import json
            return json.loads(match.group(0))
        return [message]
    except Exception:
        return [message]

async def get_multi_perspective_context(message: str, history: list[dict]) -> tuple[list[str], str]:
    # Optimization: Skip condensation in DEV or if no history exists to save ~30s of latency
    if IS_DEV or not history:
        condensed = message
    else:
        condensed = await condense_query(message, history)

    # Issue #361: Heuristic for complexity - Skip perspectives in DEV for speed
    if (not IS_DEV or os.getenv("AGNAV_FORCE_PERSPECTIVES") == "true") and len(condensed.split()) > 10:
        queries = await generate_perspective_queries(condensed, history)
    else:
        queries = [condensed]
    
    # Optimization: Fewer chunks in dev to speed up inference
    top_k_count = 3 if IS_DEV else 5
    all_res = search_index_batch(_index, _chunks, queries, [top_k_count] * len(queries))
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

async def rag_review_stream(message: str, history: list[dict], persona_mode: str = "Lookup", queries: list[str] = None, context: str = "") -> AsyncIterator[str]:
    try:
        if not context or not queries:
            queries, context = await get_multi_perspective_context(message, history)
        
        base_persona = get_persona_prompt(persona_mode)
        audit_rules = ""
        if persona_mode in ("Grieve", "Manage"):
            matched_tests = _test_registry.find_matches(message + " " + queries[0])
            for test in matched_tests:
                audit_rules += f"\n\n--- MANDATORY LOGIC CHECK: {test.name.upper()} ---\n"
                audit_rules += f"This case involves potential issues related to {test.name}. You MUST follow the EXPLAIN/QUESTION/APPLY/CITE pattern (Explain the principle, ask the relevant Question, Apply it to the facts, and Cite the source).\n"
                audit_rules += f"Criteria:\n{test.content}\n"

        master_rules = get_system_prompt().format(manifest="", verify_message="")
        system = f"{master_rules}\n\n{base_persona}\n\n{audit_rules}\n\n--- KNOWLEDGE BASE CONTEXT ---\n{context}"
        
        # Cap history to last 2 turns, truncated to reduce prompt size on CPU
        capped = []
        for h in history[-2:]:
            c = h["content"] if isinstance(h["content"], str) else str(h["content"])
            capped.append({"role": h["role"], "content": c[:300] + "..." if len(c) > 300 else c})
        messages = capped + [{"role": "user", "content": message}]
        
        async for text in unified_chat_stream(
            model=REVIEWER_MODEL,
            max_tokens=1024,
            system=system,
            messages=messages
        ):
            yield text
    except Exception as exc:
        logger.error(f"[rag] Pipeline error: {exc}", exc_info=True)
        yield f"⚠️ API error: {exc}"

# ─── UI Utility Functions ───────────────────────────────────────────────────
def _get_download_source_files() -> list[Path]:
    """Scan LABOUR_LAW_DIR for PDF files (human downloads). Excludes tests/."""
    if not LABOUR_LAW_DIR.exists(): return []
    tests_dir = LABOUR_LAW_DIR / "tests"
    pdfs = [p for p in LABOUR_LAW_DIR.rglob("*.pdf") if not p.is_relative_to(tests_dir)]
    return sorted(list(set(pdfs)), key=lambda p: str(p))


def startup(force_rebuild: bool = False):
    global _index, _chunks, INTEGRITY_WARNING
    
    # Identify environment
    provider = get_llm_provider()
    model = DEFAULT_MODEL_LLM
    logger.info(f"[startup] AgNav {AGNAV_VERSION} starting...")
    logger.info(f"[startup] Provider: {get_llm_provider()}")
    logger.info(f"[startup] Default Model: {DEFAULT_MODEL_LLM}")
    if get_llm_provider() == "huggingface":
        logger.info(f"[startup] HF Routing: {HF_PROVIDER}")

    _test_registry.load(TESTS_DIR)
    # Ensure cache directory is writable
    import indexing
    indexing.PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        test_file = indexing.PDF_CACHE_DIR / "permissions_test"
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        logger.warning(f"[startup] {indexing.PDF_CACHE_DIR} is not writable: {e}. Indexing may fail.")

    _fetch_pdf_cache_if_missing()
    _index, _chunks = load_precomputed_index()
    if _index is None or force_rebuild:
        _index, _chunks = build_index_from_sources(force=True)
    if _index is not None:
        report = get_integrity_report()
        if report.get("failed_files"):
            INTEGRITY_WARNING = f"⚠️ Index Incomplete: {len(report['failed_files'])} documents failed."

# ─── Chainlit App Logic ─────────────────────────────────────────────────────
@cl.on_chat_start
async def on_chat_start():
    # Ensure index is loaded
    if _index is None:
        startup()
        
    settings = await cl.ChatSettings(
        [
            Select(
                id="Persona",
                label="Persona",
                values=["Lookup", "Grieve", "Manage"],
                initial_index=0,
            )
        ]
    ).send()
    cl.user_session.set("Persona", "Lookup")
    cl.user_session.set("history", [])

    elements = []
    files = _get_download_source_files()
    for f in files:
        display_name = f.stem.replace("_", " ").title()
        display_name = display_name.replace("Bcgeu", "BCGEU").replace("Main Agreement", "Agreement")
        display_name = display_name.replace("Bc ", "BC ").replace(" Bc", " BC")
        elements.append(cl.File(name=display_name, path=str(f.resolve()), display="inline"))
    
    msg_content = "### BCGEU Navigator\nWelcome! Select your persona in settings and ask a question."
    if INTEGRITY_WARNING:
        msg_content += f"\n\n⚠️ {INTEGRITY_WARNING}"
        
    await cl.Message(content=msg_content, elements=elements).send()

@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set("Persona", settings["Persona"])

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Just Cause",
            message="What are the just cause requirements for discipline?",
            icon="/public/favicon.ico",
        ),
        cl.Starter(
            label="Stewards' Rights",
            message="What rights do stewards have in investigation meetings?",
            icon="/public/favicon.ico",
        ),
        cl.Starter(
            label="Off-duty Conduct",
            message="What is the nexus test for establishing a link in off-duty conduct cases?",
            icon="/public/favicon.ico",
        ),
        cl.Starter(
            label="Harassment Test",
            message="Show me the Harassment Threshold test.",
            icon="/public/favicon.ico",
        ),
        cl.Starter(
            label="Social Media",
            message="Does my employer have a social media policy?",
            icon="/public/favicon.ico",
        )
    ]

@cl.on_message
async def on_message(message: cl.Message):
    persona = cl.user_session.get("Persona") or "Lookup"
    history = cl.user_session.get("history") or []
    
    # Rate Limit & Security Check
    user_id = cl.user_session.get("id") or "default"
    allowed, rate_msg = _rate_limiter.is_allowed(user_id)
    if not allowed:
        await cl.Message(content=rate_msg).send()
        return

    sanitized, flagged = sanitize_input(message.content)
    if flagged:
        await cl.Message(content="⚠️ Input flagged for security review.").send()
        return
        
    # Show thinking step
    async with cl.Step(name="Analyzing knowledge base...") as step:
        step.type = "run"
        queries, context = await get_multi_perspective_context(sanitized, history)
        step.output = f"**Searched Perspectives:**\n" + "\n".join([f"- {q}" for q in queries])
        
    # Main response stream
    msg = cl.Message(content="")
    await msg.send()
    
    async for chunk in rag_review_stream(sanitized, history, persona, queries=queries, context=context):
        await msg.stream_token(chunk)
        
    await msg.update()
    
    # Save to session history
    new_history = history + [
        {"role": "user", "content": sanitized},
        {"role": "assistant", "content": msg.content}
    ]
    cl.user_session.set("history", new_history)

if __name__ == "__main__":
    startup()

