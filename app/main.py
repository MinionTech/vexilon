import os
from pathlib import Path
CACHE_DIR = Path(os.getenv("AGNAV_CACHE_DIR", "./.pdf_cache"))
os.environ["CHAINLIT_FILES_DIR"] = "/tmp/chainlit_files"
Path("/tmp/chainlit_files").mkdir(parents=True, exist_ok=True)

# Force online mode for the API but keep local models offline for speed
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import re
import time
# Agreement Navigator - UI Version: 2026-05-10
import logging
import asyncio
import datetime
from collections.abc import AsyncIterator
from pathlib import Path
from threading import Lock

import openai
from openai import AsyncOpenAI

import faiss
import chainlit as cl
import anyio.to_thread
import anyio._backends._asyncio as _anyio_asyncio_backend
import sniffio

from patches import apply_patches
apply_patches()
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

def get_persona_prompt(persona_key: str) -> str:
    """Return the combined mandatory rules and persona guidelines."""
    if persona_key == "Manage":
        rules = MANAGER_MANDATORY_RULES
        persona = "You are a Senior Strategic Management Consultant focusing on compliance and risk mitigation within the Operational Framework. Provide precise, fact-based answers using the provided context."
    elif persona_key == "Grieve":
        rules = UNION_MANDATORY_RULES
        persona = (
            "You are an expert in workplace grievances. Analyze the provided context and history to suggest a strategy.\n"
            "Focus on identifying relevant articles and supporting the member's case. Maintain a professional, supportive, and analytical tone."
        )
    elif persona_key == "Train":
        rules = UNION_MANDATORY_RULES
        persona = (
            "You are an expert in labor relations training. Explain the concepts in the context using a helpful, educational tone.\n"
            "Focus on empowering the user with knowledge and clear explanations."
        )
    elif persona_key == "Audit":
        rules = UNION_MANDATORY_RULES
        persona = "You are a Senior BCGEU Staff Rep acting as a Forensic Auditor to build air-tight grievance cases using the provided context."
    else:
        rules = UNION_MANDATORY_RULES
        persona = (
            "You are a forensic labor law expert. Your goal is to provide precise, fact-based answers using ONLY the provided context.\n"
            "If the information is not in the context, state that you don't know."
        )
    
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
                api_key=os.environ.get("HF_TOKEN"),
                timeout=60.0 # PR Feedback: Added explicit timeout
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
    kwargs = {"model": actual_model, "max_tokens": max_tokens, "messages": full_messages, "timeout": 60.0}

    # timeout is handled by the client initialization or default
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
    
    # Final flush of the buffer if there's remaining content that isn't a think block
    if buffer and not in_think_block:
        yield buffer

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
        queries, context, snippets = await get_multi_perspective_context(message, history)
        
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

async def get_multi_perspective_context(message: str, history: list[dict]) -> tuple[list[str], str, list[dict]]:
    # Optimization: Skip condensation in DEV or if no history exists to save ~30s of latency
    if IS_DEV or not history:
        condensed = message
    else:
        condensed = await condense_query(message, history)

    # Issue #361: Heuristic for complexity - Use perspectives for long/complex queries
    if len(condensed.split()) > 10 or os.getenv("AGNAV_FORCE_PERSPECTIVES") == "true":
        queries = await generate_perspective_queries(condensed, history)
    else:
        queries = [condensed]
    
    # Optimization: Fewer chunks in dev to speed up inference
    top_k_count = 3 if IS_DEV else 5
    # Embedding is CPU-heavy; offload to thread to keep the event loop alive
    all_res = await asyncio.to_thread(search_index_batch, _index, _chunks, queries, [top_k_count] * len(queries))
    seen = set()
    context_parts = []
    unique_snippets = []
    for res_list in all_res:
        for c in res_list:
            if c["text"] not in seen:
                seen.add(c["text"])
                unique_snippets.append(c)
                source = c.get("source", "Unknown")
                page = c.get("page", "?")
                context_parts.append(f"<<< SOURCE: {source} | Page: {page} >>>\n{c['text']}")
    return queries, "\n\n".join(context_parts), unique_snippets

async def rag_review_stream(message: str, history: list[dict], persona_mode: str = "Lookup", context: str | None = None, queries: list[str] | None = None) -> AsyncIterator[str]:
    try:
        if not context or not queries:
            q_new, c_new, s_new = await get_multi_perspective_context(message, history)
            context = context or c_new
            queries = queries or q_new

        # 2. Forensic Analysis & Logic Injection
        # Removed unused variable (was identical to base_persona)
        
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
    """Scan LABOUR_LAW_DIR for PDF and MD files. Excludes tests/."""
    if not LABOUR_LAW_DIR.exists(): return []
    tests_dir = LABOUR_LAW_DIR / "tests"
    files = [p for p in LABOUR_LAW_DIR.rglob("*") if p.suffix.lower() in (".pdf", ".md")
             and not p.is_relative_to(tests_dir)
             and not p.name.endswith(".integrity.md")]
    return sorted(list(set(files)), key=lambda p: str(p))


# ─── App Logic ──────────────────────────────────────────────────────────────
_source_path_map: dict[str, Path] = {}

def startup(force_rebuild: bool = False):
    global _index, _chunks, INTEGRITY_WARNING, _source_path_map
    
    # Identify environment
    provider = get_llm_provider()
    model = DEFAULT_MODEL_LLM
    logger.info(f"[startup] AgNav {AGNAV_VERSION} starting...")
    logger.info(f"[startup] Provider: {provider}")
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
        all_files = _get_rag_source_files()
        _source_path_map = { _get_source_name(p.stem): p for p in all_files }
        report = get_integrity_report()
    
    doc_list = _get_download_source_files()
    logger.info(f"[startup] {len(doc_list)} reference documents found.")

# ─── Chainlit UI ────────────────────────────────────────────────────────────
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Discipline Analysis",
            message="What are the Article 14 (Discipline) requirements for just cause?",
        ),
        cl.Starter(
            label="Grievance Builder",
            message="I need to file a grievance for a member. What steps should I take?",
        ),
        cl.Starter(
            label="Steward Rights",
            message="What are my rights as a steward during an investigation meeting?",
        ),
        cl.Starter(
            label="Nexus Analysis",
            message="How does the nexus test apply to off-duty conduct discipline?",
        ),
    ]

# Module-level startup gate. startup() is sync (it does blocking I/O: PDF
# fetch, FAISS index build/load, registry load). We run it exactly once,
# off the event loop, on the first chat session.
_startup_done = False
_startup_lock = asyncio.Lock()


async def _ensure_startup() -> None:
    global _startup_done
    if _startup_done:
        return
    async with _startup_lock:
        if _startup_done:
            return
        
        msg = None
        try:
            # Attempt to show progress if we're in a chat session
            msg = cl.Message(content="Initializing Knowledge Base... This may take a moment if indexing is required.")
            await msg.send()
        except Exception:
            pass

        await asyncio.to_thread(startup)
        
        if msg:
            try:
                msg.content = "Knowledge Base initialized."
                await msg.update()
            except Exception:
                pass
        
        _startup_done = True


# ── Auth ────────────────────────────────────────────────────────────────────
# Match the previous Gradio behaviour: enable password auth only when the
# AGNAV_PASSWORD env var is set. Chainlit only registers the callback if it
# is decorated, so we gate the decoration itself.
if os.getenv("AGNAV_PASSWORD"):
    _agn_user = os.getenv("AGNAV_USERNAME", "admin")
    _agn_password = os.environ["AGNAV_PASSWORD"]
    logger.info(f"[startup] Authentication enabled for user '{_agn_user}'")

    @cl.password_auth_callback
    async def auth_callback(username: str, password: str) -> "cl.User | None":
        if username == _agn_user and password == _agn_password:
            return cl.User(identifier=username)
        return None


@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set("persona", settings["Persona"])


PERSONAS = ["Lookup", "Grieve", "Audit", "Manage"]
DEFAULT_PERSONA = "Lookup"

EXAMPLES = [
    "What are the Article 14 (Discipline) requirements for just cause?",
    "I need to file a grievance for a member. What steps should I take?",
    "What are my rights as a steward during an investigation meeting?",
]

WELCOME_MSG = """# BCGEU Navigator"""

def get_welcome_actions():
    return [
        cl.Action(name="starter_query", payload={"value": "What are the Article 14 (Discipline) requirements for just cause?"}, label="How do I evaluate a disciplinary action for 'Just Cause'?"),
        cl.Action(name="starter_query", payload={"value": "I need to file a grievance for a member. What steps should I take?"}, label="What are the mandatory steps for filing a formal grievance?"),
        cl.Action(name="starter_query", payload={"value": "What are my rights as a steward during an investigation meeting?"}, label="What are my specific rights as a steward during an investigation?"),
    ]


@cl.set_chat_profiles
async def chat_profiles(user: cl.User):
    return [
        cl.ChatProfile(
            name="Lookup",
            default=True,
            markdown_description="Forensic lookup of labor law excerpts.",
            starters=[cl.Starter(label="Discipline Analysis", message=EXAMPLES[0])],
        ),
        cl.ChatProfile(
            name="Grieve",
            markdown_description="Strategic guidance for grievance filing.",
            starters=[cl.Starter(label="Grievance Builder", message=EXAMPLES[1])],
        ),
        cl.ChatProfile(
            name="Audit",
            markdown_description="Forensic auditing of compliance risks.",
            starters=[cl.Starter(label="Audit Analysis", message=EXAMPLES[2])],
        ),
        cl.ChatProfile(
            name="Manage",
            markdown_description="Strategic management consulting.",
            starters=[cl.Starter(label="Strategy Session", message=EXAMPLES[0])],
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    await _ensure_startup()
    

    # ── Chat Settings (Gear Icon) ─────────────────────────────────────────
    await cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="Persona",
                label="Navigator Persona",
                values=["Lookup", "Grieve", "Audit", "Manage"],
                initial="Lookup",
            ),
        ]
    ).send()
    
    # Initialize session state
    cl.user_session.set("history", [])
    cl.user_session.set("persona", "Lookup")

    # ── Welcome Header ────────────────────────────────────────────────────
    await cl.Message(
        content=WELCOME_MSG, 
        author="System", 
        actions=get_welcome_actions()
    ).send()

    if INTEGRITY_WARNING:
        await cl.Message(content=INTEGRITY_WARNING, author="system").send()


# ── Message handler ────────────────────────────────────────────────────────
def _client_id(message: cl.Message) -> str:
    """Best-effort client identifier for rate limiting.

    Chainlit doesn't expose request headers on cl.Message directly; fall back
    to the session id, which keeps per-user limits sensible without leaking
    real client IPs.
    """
    sid = getattr(cl.user_session, "id", None) or cl.user_session.get("id")
    return str(sid) if sid else "default"



async def on_persona_action(action: cl.Action):
    persona = action.payload.get("value")
    if persona:
        cl.user_session.set("persona", persona)

@cl.action_callback("starter_query")
async def on_action(action: cl.Action):
    query = action.payload.get('value')
    if not query:
        return
    # Show the query the user selected
    await cl.Message(content=f"**Query:** {query}", author="System").send()
    # Trigger the processing
    await on_message(cl.Message(content=query))
    # Remove the buttons to keep it clean
    await action.remove()

@cl.on_message
async def on_message(message: cl.Message) -> None:
    await _ensure_startup()

    msg_str = (message.content or "").strip()
    if not msg_str:
        return

    # Rate limit (per session)
    allowed, rate_msg = _rate_limiter.is_allowed(_client_id(message))
    if not allowed:
        return

    # Prompt-injection / length sanitisation
    sanitized, flagged = sanitize_input(msg_str)
    if flagged:
        return

    # Order of precedence: session['persona'] (settings) -> session['chat_profile'] (profiles) -> DEFAULT_PERSONA
    persona = cl.user_session.get("persona") or cl.user_session.get("chat_profile") or DEFAULT_PERSONA
    history: list[dict] = cl.user_session.get("history") or []

    out = cl.Message(content="")
    await out.send()

    accumulated = ""
    word_count = len(sanitized.split())
    char_count = len(sanitized)
    logger.info(f"[chat] Starting stream for {persona} mode (Words: {word_count}, Chars: {char_count})")
    try:
        queries, context, snippets = await get_multi_perspective_context(sanitized, history)
        
        # Attach sources as inline file chips (PDF preferred, MD fallback)
        # TODO: clickable links blocked by Chainlit only linkifying http/https — see issue #501
        elements = []
        seen_sources = set()
        for s in snippets:
            source_name = s.get("source", "Unknown")
            if source_name not in seen_sources:
                md_path = _source_path_map.get(source_name)
                if md_path:
                    pdf_path = md_path.with_suffix(".pdf")
                    download_path = pdf_path if pdf_path.exists() else md_path
                    elements.append(cl.File(
                        name=f"{source_name} ({download_path.suffix.lstrip('.').upper()})",
                        path=str(download_path),
                        display="inline",
                    ))
                seen_sources.add(source_name)
        out.elements = elements

        async for chunk in rag_review_stream(sanitized, history, persona, context=context, queries=queries):
            if not chunk:
                continue
            accumulated += chunk
            await out.stream_token(chunk)
    except Exception as exc:  # defensive — rag_review_stream already catches
        logger.error(f"[chat] Unexpected error: {exc}", exc_info=True)
        accumulated = f"⚠️ API error: {exc}"
        out.content = accumulated

    await out.update()

    # Async Verification pass as per SPEC.md Section 9
    if VERIFY_ENABLED:
        async def verify_and_update():
            report = await verify_response(accumulated, context)
            if report and "ALL_CLAIMS_VERIFIED" not in report:
                out.content += f"\n\n---\n**Verification Note:**\n{report}"
                await out.update()
        
        cl.run_task(verify_and_update())

    history.append({"role": "user", "content": sanitized})
    history.append({"role": "assistant", "content": accumulated})
    cl.user_session.set("history", history)
    logger.info(f"[chat] Stream completed. Total length: {len(accumulated)}")
