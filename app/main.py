import os
from pathlib import Path
CACHE_DIR = Path(os.getenv("AGNAV_CACHE_DIR", "./.pdf_cache"))
# CHAINLIT_FILES_DIR is set in Containerfile ENV (must be set before
# chainlit imports). Defensive fallback for non-container dev:
os.environ.setdefault("CHAINLIT_FILES_DIR", "/tmp/chainlit_files")
Path(os.environ["CHAINLIT_FILES_DIR"]).mkdir(parents=True, exist_ok=True)

# Force online mode for the API but keep local models offline for speed
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import re
import time
import json
# Agreement Navigator - UI Version: 2026-05-10
import logging
import asyncio
import datetime
import tempfile
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
    DATA_DIR,
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
_background_tasks: set[asyncio.Task] = set()

AGNAV_VERSION = os.getenv("AGNAV_VERSION", "Dev mode")
IS_DEV = "dev" in AGNAV_VERSION.lower()
AGNAV_REPO_URL = os.getenv("AGNAV_REPO_URL", "https://github.com/MinionTech/vexilon")
GITHUB_DATA_URL = os.getenv(
    "AGNAV_KNOWLEDGE_URL", f"{AGNAV_REPO_URL}/tree/main/app/data"
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
TESTS_DIR = DATA_DIR / "test_fixtures"
PUBLIC_DOCS_DIR = Path(__file__).parent / "public" / "docs"

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

# ─── Save/Load Conversation ─────────────────────────────────────────────────
def serialize_conversation(history: list[dict], persona: str) -> str:
    """Serialize conversation history to markdown with JSON metadata.
    
    PIPA Compliance: Metadata and conversation are end-user readable markdown
    (not encrypted, but client-side only). No PII logged on server.
    
    Args:
        history: List of message dicts with 'role' and 'content' keys
        persona: Current persona (Lookup/Grieve/Manage)
    
    Returns:
        Markdown string with YAML front matter and conversation turns
    """
    saved_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    # YAML front matter (end-user readable metadata)
    md = f"""---
saved_at: {saved_at}
persona: {persona}
message_count: {len(history)}
---

# Conversation Export

**Persona:** {persona}  
**Saved:** {saved_at}  
**Messages:** {len(history)}

---

"""
    
    # Convert each turn to readable markdown
    for i, msg in enumerate(history, 1):
        role_label = "👤 You" if msg["role"] == "user" else "🤖 Assistant"
        md += f"## Turn {i}: {role_label}\n\n{msg['content']}\n\n"
    
    # Append JSON payload at end in code block for re-import
    payload = {
        "saved_at": saved_at,
        "persona": persona,
        "messages": history,
    }
    md += "---\n\n<details><summary>Technical Metadata (JSON)</summary>\n\n```json\n"
    md += json.dumps(payload, indent=2)
    md += "\n```\n\n</details>"
    
    return md


def deserialize_conversation(content: str) -> tuple[list[dict], str, str, list[str]]:
    """Deserialize conversation from markdown file (JSON fallback for compatibility).
    
    Extracts JSON metadata from either:
    1. The technical JSON section in markdown export
    2. Raw JSON (for backward compat)
    
    Args:
        content: Markdown file contents or raw JSON string
    
    Returns:
        Tuple of (messages, persona, saved_at timestamp, warnings list)
    
    Raises:
        ValueError: If format is invalid or missing required fields
    """
    warnings = []
    
    # Try to extract JSON from markdown <details> section
    json_matches = re.findall(r'```json\s*\n(.*?)\n\s*```', content, re.DOTALL)
    if json_matches:
        json_str = json_matches[-1]
    else:
        # Fallback: assume raw JSON (for old exports or direct JSON files)
        json_str = content
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse conversation data: {e}")
    
    if not isinstance(data, dict):
        raise ValueError("Invalid conversation file format: root must be an object")
    
    messages = data.get("messages", [])
    persona = data.get("persona", "Lookup")
    saved_at = data.get("saved_at", "Unknown")
    
    if not isinstance(messages, list):
        raise ValueError("Messages must be a list")
    
    # Enforce standard safe string types for persona and saved_at
    persona = str(persona)[:50]
    saved_at = str(saved_at)[:50]
    
    # Limit number of messages to prevent memory exhaustion DoS
    if len(messages) > 100:
        messages = messages[:100]
        warnings.append("Conversation exceeded the limit of 100 turns. Truncated excess historical messages.")
    
    sanitized_messages = []
    truncated_count = 0
    script_stripped = False
    
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            raise ValueError(f"Message turn {i+1} is malformed: must contain 'role' and 'content' keys.")
        
        role = str(msg["role"]).strip()
        if role not in ("user", "assistant"):
            raise ValueError(f"Message turn {i+1} has invalid role: must be 'user' or 'assistant'.")
            
        orig_content = str(msg["content"])
        
        # Enforce max length constraint strictly
        if len(orig_content) > MAX_INPUT_LENGTH:
            content_str = orig_content[:MAX_INPUT_LENGTH]
            truncated_count += 1
        else:
            content_str = orig_content
            
        # Secure HTML/script sanitization pass
        clean_content = re.sub(r'(?i)<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', content_str)
        clean_content = re.sub(r'(?i)\bon\w+\s*=\s*(?:"[^"]*"|\'[^\']*\'|[^\s>]+)', '', clean_content)
        clean_content = re.sub(r'(?i)<iframe\b[^<]*(?:(?!<\/iframe>)<[^<]*)*<\/iframe>', '', clean_content)
        clean_content = re.sub(r'(?i)(href|src)\s*=\s*["\']?\s*javascript:[^"\'>\s]*["\']?', r'\1="#"', clean_content)

        if clean_content != content_str:
            script_stripped = True
            
        sanitized_messages.append({
            "role": role,
            "content": clean_content
        })
        
    if truncated_count > 0:
        warnings.append(f"Safely truncated {truncated_count} messages exceeding the length limit of {MAX_INPUT_LENGTH} characters.")
        
    if script_stripped:
        warnings.append("Security sanitization: Removed potential script injections from conversation history.")
        
    return sanitized_messages, persona, saved_at, warnings

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
            "You are a Senior BCGEU Staff Rep acting as a Forensic Auditor to build air-tight grievance cases. "
            "Analyze the provided context and history to suggest a strategic grievance path, identify contract violations, "
            "and recommend specific evidence to gather. Maintain a supportive, analytical, and professional tone."
        )
    elif persona_key == "Train":
        rules = UNION_MANDATORY_RULES
        persona = (
            "You are an expert in labor relations training. Explain the concepts in the context using a helpful, educational tone.\n"
            "Focus on empowering the user with knowledge and clear explanations."
        )
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
        # More robust extraction using regex to find the JSON block (list or object)
        match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
        if match:
            import json
            parsed = json.loads(match.group(0))
            
            # Handle if the LLM returned a dict with a list inside
            if isinstance(parsed, dict):
                for val in parsed.values():
                    if isinstance(val, list):
                        parsed = val
                        break
            
            if isinstance(parsed, list):
                sanitized_queries = []
                for item in parsed:
                    if isinstance(item, str):
                        sanitized_queries.append(item)
                    elif isinstance(item, dict):
                        # Extract the query string from common keys, or fallback to the first string value
                        q_val = item.get("q") or item.get("query") or item.get("text")
                        if not q_val:
                            for val in item.values():
                                if isinstance(val, str):
                                    q_val = val
                                    break
                        if q_val:
                            sanitized_queries.append(str(q_val))
                    else:
                        sanitized_queries.append(str(item))
                
                final_queries = [q.strip() for q in sanitized_queries if q and isinstance(q, str)]
                if final_queries:
                    return final_queries
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
    """Scan DATA_DIR for PDF and MD files. Excludes test_fixtures/."""
    if not DATA_DIR.exists(): return []
    fixtures_dir = DATA_DIR / "test_fixtures"
    files = [p for p in DATA_DIR.rglob("*") if p.suffix.lower() in (".pdf", ".md")
             and not p.is_relative_to(fixtures_dir)
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

    # Resolve dynamic build SHA (CI environment or local 'dev mode' default)
    build_sha = os.getenv("BUILD_SHA", "dev mode")
    logger.info(f"[startup] Build Integrity: {build_sha}")

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
# Starters are handled by ChatProfiles now.

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
        
        await asyncio.to_thread(startup)
        
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


PERSONAS = ["Lookup", "Grieve", "Manage"]
DEFAULT_PERSONA = "Lookup"
VEXILON_SAVE_SENTINEL = "__VEXILON_SAVE__"

EXAMPLES = [
    "What are the Article 14 (Discipline) requirements for just cause?",
    "What are my rights as a steward during an investigation meeting?",
    "What is the nexus test for establishing a link in off-duty conduct cases?",
    "Show me the Harassment Threshold test.",
    "I need to file a grievance for a member. What steps should I take?",
]
@cl.set_chat_profiles
async def chat_profiles(user: cl.User):
    all_starters = [
        cl.Starter(label="Discipline Just Cause", message=EXAMPLES[0]),
        cl.Starter(label="Steward Rights", message=EXAMPLES[1]),
        cl.Starter(label="Nexus Off-Duty Test", message=EXAMPLES[2]),
        cl.Starter(label="Harassment Threshold", message=EXAMPLES[3]),
        cl.Starter(label="Grievance Builder", message=EXAMPLES[4]),
    ]
    return [
        cl.ChatProfile(
            name="Lookup",
            icon="",
            default=True,
            markdown_description="Forensic lookup of labor law excerpts.",
            starters=all_starters,
        ),
        cl.ChatProfile(
            name="Grieve",
            icon="",
            markdown_description="Strategic guidance and forensic auditing for grievance filing.",
            starters=all_starters,
        ),
        cl.ChatProfile(
            name="Manage",
            icon="",
            markdown_description="Strategic management consulting.",
            starters=all_starters,
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
                values=["Lookup", "Grieve", "Manage"],
                initial="Lookup",
            ),
        ]
    ).send()
    
    # Initialize session state
    cl.user_session.set("history", [])
    cl.user_session.set("persona", "Lookup")



    if INTEGRITY_WARNING:
        await cl.Message(content=INTEGRITY_WARNING, author="system").send()


# ── Message handler ────────────────────────────────────────────────────────
def _client_id() -> str:
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

async def trigger_session_save():
    """Save conversation history to downloadable markdown file (PIPA-compliant)."""
    allowed, rate_msg = _rate_limiter.is_allowed(_client_id())
    if not allowed:
        await cl.Message(content=rate_msg, author="System").send()
        return

    history: list[dict] = cl.user_session.get("history") or []
    persona: str = cl.user_session.get("persona") or "Lookup"
    
    if not history:
        await cl.Message(content="No conversation to save yet.", author="System").send()
        return
    
    try:
        markdown_content = serialize_conversation(history, persona)
        
        # Generate timestamped filename (client-side, user controls final location)
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.md"
        
        msg = cl.Message(
            content=f"Conversation saved as markdown. Click below to download.",
            author="System",
            elements=[cl.File(name=filename, content=markdown_content, display="inline", mime="text/markdown")]
        )
        await msg.send()
        
        logger.info(f"[save] Conversation saved by {_client_id()} ({len(history)} messages)")
    except Exception as e:
        logger.error(f"[save] Failed to save conversation: {e}")
        await cl.Message(content=f"Error saving conversation: {e}", author="System").send()
        if file_path:
            try:
                Path(file_path).unlink(missing_ok=True)
            except OSError:
                pass


@cl.action_callback("save_conversation")
async def on_save_conversation(action: cl.Action):
    """Callback for native Chainlit Action button."""
    await trigger_session_save()


async def trigger_session_load(file_content: str):
    """Load conversation from uploaded markdown or JSON file."""
    allowed, rate_msg = _rate_limiter.is_allowed(_client_id())
    if not allowed:
        await cl.Message(content=rate_msg, author="System").send()
        return
        
    try:
        messages, saved_persona, saved_at, warnings = deserialize_conversation(file_content)
        
        # Append to current history
        current_history: list[dict] = cl.user_session.get("history") or []
        current_history.extend(messages)
        cl.user_session.set("history", current_history)
        
        # If there are warnings, display them clearly to the user
        if warnings:
            warnings_text = "\n".join(f"- {w}" for w in warnings)
            await cl.Message(
                content=f"⚠️ **Upload Notice**\n\n{warnings_text}",
                author="System",
            ).send()
        
        # Display notification with restore marker
        msg = cl.Message(
            content=f"**Restored Conversation** (Persona: {saved_persona}, Saved: {saved_at})\n\nLoaded {len(messages)} messages. These are read-only.",
            author="System",
        )
        msg.metadata = {"restored": True}
        await msg.send()
        
        # Re-render loaded messages as read-only in UI
        for i, loaded_msg in enumerate(messages):
            display_role = "👤 You" if loaded_msg["role"] == "user" else "🤖 Assistant"
            msg_content = loaded_msg["content"]
            msg_obj = cl.Message(content=msg_content, author=display_role)
            msg_obj.metadata = {"restored": True, "readonly": True}
            await msg_obj.send()
        
        logger.info(f"[load] Conversation loaded by {_client_id()} ({len(messages)} messages)")
    except json.JSONDecodeError as e:
        logger.error(f"[load] Invalid JSON in file: {e}")
        await cl.Message(content="Invalid conversation file format. Expected JSON.", author="System").send()
    except ValueError as e:
        logger.error(f"[load] Conversation validation error: {e}")
        await cl.Message(content=f"Conversation file is invalid: {e}", author="System").send()
    except Exception as e:
        logger.error(f"[load] Failed to load conversation: {e}")
        await cl.Message(content=f"Error loading conversation: {e}", author="System").send()


async def _ask_for_session_file() -> None:
    """Background coroutine: prompts for a session file using AskFileMessage.

    Runs detached from the action callback so the UI is never locked.
    contextvars are copied by asyncio.create_task(), preserving the
    Chainlit session context for send() calls.
    """
    try:
        res = await cl.AskFileMessage(
            content="Select your `.md` session backup file to restore this conversation.",
            accept=["text/markdown", "text/plain"],
            max_size_mb=2,
            timeout=120,
        ).send()

        if res:
            file = res[0]
            with open(file.path, "r", encoding="utf-8") as f:
                file_content = f.read()
            await trigger_session_load(file_content)
    except Exception as e:
        logger.error(f"[load] AskFileMessage background task failed: {e}")
        await cl.Message(content="Failed to load session file.", author="System").send()


@cl.action_callback("load_conversation")
async def on_load_conversation(action: cl.Action):
    """Callback for native Chainlit Action button.

    Returns immediately to prevent UI lock; file prompt runs in background.
    """
    allowed, rate_msg = _rate_limiter.is_allowed(_client_id())
    if not allowed:
        await cl.Message(content=rate_msg, author="System").send()
        return

    task = asyncio.create_task(_ask_for_session_file())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

def resolve_pdf_path(md_path: Path) -> Path:
    """Resolve the matching PDF file path for a given Markdown source path."""
    if md_path.suffix.lower() == ".pdf":
        return md_path

    # 1. Try same directory PDF
    pdf_same_dir = md_path.with_suffix(".pdf")
    if pdf_same_dir.exists():
        return pdf_same_dir

    # 2. Try public docs directory
    exact_pdf = PUBLIC_DOCS_DIR / f"{md_path.stem}.pdf"
    if exact_pdf.exists():
        return exact_pdf

    if "_-_" in md_path.stem:
        base_stem = md_path.stem.split("_-_")[0]
        prefix_pdf = PUBLIC_DOCS_DIR / f"{base_stem}.pdf"
        if prefix_pdf.exists():
            return prefix_pdf

    return md_path

@cl.on_message
async def on_message(message: cl.Message) -> None:
    # Internal sentinel: toolbar save button bypasses normal message flow
    if (message.content or "").strip() == VEXILON_SAVE_SENTINEL:
        await trigger_session_save()
        return

    await _ensure_startup()

    # Intercept session file uploads natively
    if message.elements:
        for element in message.elements:
            if element.mime in ["text/markdown", "text/plain", "application/json", "application/octet-stream"] or element.name.lower().endswith((".md", ".json")):
                try:
                    with open(element.path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    await trigger_session_load(file_content)
                    # Cleanly close the execution loop so the UI stop button unlocks
                    await cl.Message(content="✓ Session restored successfully.", author="System").send()
                except Exception as e:
                    logger.error(f"[load] Failed to read uploaded session file: {e}")
                    await cl.Message(content="Failed to read the uploaded session file.", author="System").send()
                return

    msg_str = (message.content or "").strip()
    if not msg_str:
        return

    # Strip action buttons from the previous assistant message to keep thread clean
    prev_msg = cl.user_session.get("last_assistant_message")
    if prev_msg:
        try:
            prev_msg.actions = []
            await prev_msg.update()
        except Exception:
            pass

    # Rate limit (per session)
    allowed, rate_msg = _rate_limiter.is_allowed(_client_id())
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
                    download_path = resolve_pdf_path(md_path)
                    elements.append(cl.File(
                        name=f"{source_name} ({download_path.suffix.lstrip('.').upper()})",
                        path=str(download_path),
                        mime="application/pdf" if download_path.suffix.lower() == ".pdf" else "text/markdown",
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

    out.actions = [
        cl.Action(
            name="save_conversation",
            value="save",
            payload={},
            label="💾 Save Session"
        )
    ]
    await out.update()

    # Async Verification pass as per SPEC.md Section 9
    if VERIFY_ENABLED:
        async def verify_and_update():
            try:
                report = await verify_response(accumulated, context)
                if report and "ALL_CLAIMS_VERIFIED" not in report:
                    out.content += f"\n\n---\n**Verification Note:**\n{report}"
                    await out.update()
            except Exception as e:
                logger.error(f"[chat] Background verification task failed: {e}", exc_info=True)
        
        task = asyncio.create_task(verify_and_update())
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    history.append({"role": "user", "content": sanitized})
    history.append({"role": "assistant", "content": accumulated})
    cl.user_session.set("history", history)
    cl.user_session.set("last_assistant_message", out)
    logger.info(f"[chat] Stream completed. Total length: {len(accumulated)}")


# ─── Custom FastAPI Routes ───────────────────────────────────────────────────
from chainlit.server import app as cl_app
from fastapi.routing import APIRoute

def get_version():
    return {
        "version": AGNAV_VERSION,
        "sha": os.getenv("BUILD_SHA", "dev mode")
    }

# Prepend the API route to bypass Chainlit's catch-all wildcard router
version_route = APIRoute(
    "/api/version",
    endpoint=get_version,
    methods=["GET"],
    include_in_schema=False
)
cl_app.router.routes.insert(0, version_route)
