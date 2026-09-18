import os
import re
import time
import random
import email.utils
import asyncio
import datetime
import logging
import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from threading import Lock

import openai
from openai import AsyncOpenAI
import faiss
import chainlit as cl

from brand import AGNAV_APP_NAME, AGNAV_APP_DESCRIPTION
from core.config import (
    AGNAV_VERSION,
    IS_DEV,
    HF_PROVIDER,
    DEFAULT_HF_MODEL_ID,
    CURRENT_MODEL_ID,
    DEFAULT_MODEL_LLM,
    CLAUDE_MODEL,
    REVIEWER_MODEL,
    CONDENSE_MODEL,
    VERIFY_MODEL,
    RAG_MAX_TOKENS,
    REVIEWER_MAX_TOKENS,
    VERIFY_ENABLED,
    LLM_MAX_RETRIES,
    LLM_RETRY_BASE_DELAY,
    LLM_RETRY_MAX_DELAY,
    HIGH_TRAFFIC_MESSAGE,
    GENERIC_ERROR_MESSAGE,
    TESTS_DIR,
    PUBLIC_DOCS_DIR,
    get_llm_provider,
    _get_active_main,
)
from indexing import (
    DATA_DIR,
    CACHE_DIR,
    _get_source_name,
    _get_rag_source_files,
    build_index_from_sources,
    get_integrity_report,
    load_precomputed_index,
    search_index_batch,
    _fetch_pdf_cache_if_missing,
)

logger = logging.getLogger(__name__)

# ─── Global State ───────────────────────────────────────────────────────────
_chunks: list[dict] = []
_index: "faiss.IndexFlatIP | None" = None
INTEGRITY_WARNING: str | None = None
_source_path_map: dict[str, Path] = {}
_startup_done = False
_startup_lock = asyncio.Lock()

def has_chainlit_context() -> bool:
    try:
        from chainlit.context import get_context
        return get_context() is not None
    except Exception:
        return False

# ─── Prompts & Operational Rules ────────────────────────────────────────────
UNION_MANDATORY_RULES = """--- MANDATORY OPERATIONAL RULES (UNION) ---
1. ANSWER FROM EXCERPTS ONLY: Base your answer strictly on the provided excerpts.
   EXCEPTION: When asked for grievance forms, filing a grievance, or grievance documentation, you MUST provide the official form download links listed in Rule 5 below.
2. STRICT CITATIONS: Every claim MUST be supported by a verbatim quote followed by its citation.
   EXAMPLE: > "verbatim text" [Document Name, Page X]
3. STRUCTURE: Use clear headings, bullet points, and numbered lists to organize complex answers.
4. NO MERIT ASSESSMENT: Do NOT judge the merit or likelihood of success of a grievance.
5. GRIEVANCE FILING & FORMS: Facilitate the filing process by identifying potential contract violations. When asked about grievance forms or filing a grievance, ALWAYS provide a brief forensic analysis of the user's situation and relevant contract provisions FIRST, followed by informing the user that official forms are available and providing these exact links:
   - [Grievance - 0 - Instructions](/public/docs/forms/Grievance_-_0_-_Instructions.pdf)
   - [Grievance - A - Grievor Case](/public/docs/forms/Grievance_-_A_-_Grievor_Case.pdf)
   - [Grievance - B - Notify Designates](/public/docs/forms/Grievance_-_B_-_Notify_Designates.pdf)
   - [Grievance - C - Steward Case](/public/docs/forms/Grievance_-_C_-_Steward_Case.pdf)
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
            "You are a forensic labor law expert. Your goal is to provide precise, fact-based answers using the provided context.\n"
            "If asked for grievance forms or how to file a grievance, assist the user by providing the official form links listed in the rules."
        )

    return f"{rules}\n\nROLE: {persona}"

VERIFY_SYSTEM_PROMPT = """You are a verification assistant. Your job is to verify that the claims made in an AI response are supported by the provided source citations.

For each claim in the response:
1. Check if the quoted text actually supports the claim being made
2. Check if the citation (document name, article/section, page number) is accurate
3. Identify any hallucinations, misquotes, or unsupported claims

NOTE: Static system resources (such as official grievance form links like /public/docs/forms/...) are official application assets provided by the platform. Do NOT flag official form links or document downloads as DISPUTED or unsupported claims.

Respond in this format:
- VERIFIED: [claim summary] — the quote supports the claim
- DISPUTED: [claim summary] — the quote does NOT support the claim
- UNCERTAIN: [claim summary] — cannot verify due to unclear citation

If all claims are verified, respond with "ALL_CLAIMS_VERIFIED".
If there are disputed claims, list them with explanations."""

def get_system_prompt(developer_mode: bool = False) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d")
    header = f"--- {AGNAV_APP_NAME.upper()} SYSTEM STATE ---\nDATE: {now}\nVERSION: {AGNAV_VERSION}\n----------------------------\n\n"
    content = f"You are {AGNAV_APP_NAME}, a professional assistant for union stewards. IMPORTANT: DO NOT use <think> tags. Provide your answer directly and professionally. ALWAYS cite your sources using the [Document Name, Header/Article] format provided in the context.\n\nKnowledge Base:\n{{manifest}}\n\n{{verify_message}}"
    return f"{header}{content}"

# ─── Client Resolution & Routing ─────────────────────────────────────────────
_huggingface_client = None
_ollama_client = None

def get_llm_client(provider: str = None) -> AsyncOpenAI:
    global _huggingface_client, _ollama_client

    if provider is None:
        session_model = None
        if has_chainlit_context():
            session_model = cl.user_session.get("selected_model")

        if session_model and ":" in session_model:
            provider = session_model.split(":", 1)[0]
        else:
            provider = get_llm_provider()

    if provider == "huggingface":
        if _huggingface_client is None:
            token = os.environ.get("HF_TOKEN")
            if not token:
                raise ValueError("Missing HF_TOKEN environment variable for Hugging Face provider.")
            _huggingface_client = AsyncOpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=token,
                timeout=60.0,
            )
        return _huggingface_client
    elif provider == "ollama":
        if _ollama_client is None:
            ollama_host = os.getenv("OLLAMA_HOST", "ollama:11434")
            if "://" not in ollama_host:
                ollama_host = f"http://{ollama_host}"
            _ollama_client = AsyncOpenAI(
                base_url=f"{ollama_host.rstrip('/')}/v1",
                api_key="ollama",
            )
        return _ollama_client
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def _resolve_llm_client(provider: str = None) -> AsyncOpenAI:
    main_mod = _get_active_main()
    getter = getattr(main_mod, "get_llm_client", get_llm_client)
    return getter(provider) if provider is not None else getter()

def resolve_model_and_provider(fallback_model: str) -> tuple[str, str]:
    session_model = None
    if has_chainlit_context():
        session_model = cl.user_session.get("selected_model")

    if fallback_model and fallback_model != DEFAULT_MODEL_LLM:
        model_str = fallback_model
    else:
        model_str = session_model or fallback_model

    if ":" in model_str:
        provider, model_id = model_str.split(":", 1)
    else:
        provider = get_llm_provider()
        model_id = model_str

    if provider == "huggingface" and HF_PROVIDER and ":" not in model_id:
        model_id = f"{model_id}:{HF_PROVIDER}"

    return provider, model_id

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

# ─── Retry & Backoff Policy ──────────────────────────────────────────────────
def is_transient_llm_error(exc: Exception) -> bool:
    """Determine whether an exception from an LLM call is a transient rate-limit or queue issue."""
    if isinstance(exc, openai.RateLimitError):
        return True

    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True

    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 429:
        return True

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        if body.get("code") == "queue_exceeded" or body.get("type") == "too_many_requests_error":
            return True

    err_str = str(exc).lower()
    if re.search(r"\b429\b", err_str):
        return True

    transient_indicators = (
        "rate_limit",
        "rate-limit",
        "too_many_requests",
        "queue_exceeded",
        "queue-exceeded",
        "over capacity",
        "over_capacity",
    )
    return any(k in err_str for k in transient_indicators)

def compute_retry_delay(attempt: int, exc: Exception | None = None) -> float:
    """Calculate exponential backoff with jitter, respecting Retry-After header if present."""
    main_mod = _get_active_main()
    effective_base_delay = getattr(main_mod, "LLM_RETRY_BASE_DELAY", LLM_RETRY_BASE_DELAY)
    effective_max_delay = getattr(main_mod, "LLM_RETRY_MAX_DELAY", LLM_RETRY_MAX_DELAY)

    if exc is not None:
        response = getattr(exc, "response", None)
        if response is not None:
            headers = getattr(response, "headers", None)
            if headers:
                retry_after_str = headers.get("retry-after") or headers.get("Retry-After")
                if retry_after_str:
                    try:
                        retry_after = float(retry_after_str)
                    except (ValueError, TypeError):
                        try:
                            date_val = email.utils.parsedate_to_datetime(retry_after_str)
                            if date_val.tzinfo is None:
                                date_val = date_val.replace(tzinfo=datetime.timezone.utc)
                            now = datetime.datetime.now(datetime.timezone.utc)
                            retry_after = (date_val - now).total_seconds()
                        except Exception:
                            retry_after = None

                    if retry_after is not None and 0 < retry_after <= effective_max_delay:
                        return retry_after

    raw_delay = min(effective_base_delay * (2 ** attempt), effective_max_delay)
    jitter = random.uniform(0.75, 1.25)
    return min(raw_delay * jitter, effective_max_delay)

async def unified_chat_create(model: str, messages: list, system: str | list = None, max_tokens: int = 1024) -> str:
    provider, actual_model = resolve_model_and_provider(model)
    client = _resolve_llm_client()
    full_messages = _build_messages(messages, system)

    kwargs = {"model": actual_model, "max_tokens": max_tokens, "messages": full_messages, "timeout": 60.0}

    main_mod = _get_active_main()
    effective_max_retries = getattr(main_mod, "LLM_MAX_RETRIES", LLM_MAX_RETRIES)

    for attempt in range(effective_max_retries + 1):
        t0 = time.perf_counter()
        logger.info(f"[llm-call] Creating completion for actual_model='{actual_model}' on provider='{provider}' (attempt {attempt + 1}/{effective_max_retries + 1})...")
        try:
            resp = await client.chat.completions.create(**kwargs)
            elapsed = time.perf_counter() - t0
            logger.info(f"[llm-call] Completion creation attempt {attempt + 1} elapsed: {elapsed:.2f} seconds")
            content = resp.choices[0].message.content
            return content or ""
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.info(f"[llm-call] Completion creation attempt {attempt + 1} call duration: {elapsed:.2f} seconds")
            if attempt < effective_max_retries and is_transient_llm_error(exc):
                delay = compute_retry_delay(attempt, exc)
                logger.warning(
                    f"[llm-retry] Transient error in unified_chat_create ({exc}). "
                    f"Retrying in {delay:.2f}s (attempt {attempt + 1}/{effective_max_retries})..."
                )
                await asyncio.sleep(delay)
            else:
                raise

async def unified_chat_stream(model: str, messages: list, system: str | list = None, max_tokens: int = 2048) -> AsyncIterator[str]:
    provider, actual_model = resolve_model_and_provider(model)
    client = _resolve_llm_client()
    full_messages = _build_messages(messages, system)

    kwargs = {"model": actual_model, "max_tokens": max_tokens, "messages": full_messages, "stream": True, "timeout": 300.0}

    main_mod = _get_active_main()
    effective_max_retries = getattr(main_mod, "LLM_MAX_RETRIES", LLM_MAX_RETRIES)

    stream = None
    for attempt in range(effective_max_retries + 1):
        t0 = time.perf_counter()
        logger.info(f"[llm-call] Opening stream for actual_model='{actual_model}' on provider='{provider}' (attempt {attempt + 1}/{effective_max_retries + 1})...")
        try:
            stream = await client.chat.completions.create(**kwargs)
            elapsed = time.perf_counter() - t0
            logger.info(f"[llm-call] Stream connection attempt {attempt + 1} elapsed: {elapsed:.2f} seconds")
            break
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.info(f"[llm-call] Stream connection attempt {attempt + 1} call duration: {elapsed:.2f} seconds")
            if attempt < effective_max_retries and is_transient_llm_error(exc):
                delay = compute_retry_delay(attempt, exc)
                logger.warning(
                    f"[llm-retry] Transient error in unified_chat_stream ({exc}). "
                    f"Retrying in {delay:.2f}s (attempt {attempt + 1}/{effective_max_retries})..."
                )
                await asyncio.sleep(delay)
            else:
                raise

    # Stateful buffer for filtering <think> blocks (handles split-token tags)
    in_think_block = False
    buffer = ""
    start_tag = "<think>"
    end_tag = "</think>"

    first_chunk = True
    async for chunk in stream:
        if first_chunk:
            elapsed = time.perf_counter() - t0
            logger.info(f"[llm-call] First stream chunk received in {elapsed:.2f} seconds")
            first_chunk = False

        if chunk.choices:
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None) or ""
            if content:
                buffer += content

            while buffer:
                if not in_think_block:
                    start_idx = buffer.find(start_tag)
                    if start_idx != -1:
                        if start_idx > 0:
                            yield buffer[:start_idx]
                        in_think_block = True
                        buffer = buffer[start_idx + len(start_tag):]
                    else:
                        partial_idx = buffer.find("<")
                        if partial_idx != -1 and len(buffer[partial_idx:]) < len(start_tag):
                            if partial_idx > 0:
                                yield buffer[:partial_idx]
                            buffer = buffer[partial_idx:]
                            break
                        else:
                            yield buffer
                            buffer = ""
                else:
                    end_idx = buffer.find(end_tag)
                    if end_idx != -1:
                        in_think_block = False
                        buffer = buffer[end_idx + len(end_tag):]
                    else:
                        partial_end_idx = buffer.find("<")
                        if partial_end_idx != -1 and len(buffer[partial_end_idx:]) < len(end_tag):
                            buffer = buffer[partial_end_idx:]
                            break
                        else:
                            buffer = ""
                            break

    if buffer and not in_think_block:
        yield buffer

async def verify_response(assistant_response: str, context: str) -> str:
    main_mod = _get_active_main()
    effective_verify_enabled = getattr(main_mod, "VERIFY_ENABLED", VERIFY_ENABLED)
    effective_verify_model = getattr(main_mod, "VERIFY_MODEL", VERIFY_MODEL)

    if not effective_verify_enabled:
        return ""
    try:
        raw_verification = await unified_chat_create(
            model=effective_verify_model,
            max_tokens=512,
            system=VERIFY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"RESPONSE:\n{assistant_response}\n\nCONTEXT:\n{context}"}],
        )
        if not raw_verification:
            return "ALL_CLAIMS_VERIFIED"

        # Filter out false-alarm DISPUTED lines regarding explicit static form URLs or downloads
        lines = [line.strip() for line in raw_verification.split("\n") if line.strip()]
        filtered_lines = []
        for line in lines:
            line_lower = line.lower()
            if "disputed:" in line_lower and any(kw in line_lower for kw in ("/public/docs/forms/", "form download link")):
                continue
            filtered_lines.append(line)

        def _normalize_line(line: str) -> str:
            cleaned = line.strip()
            if cleaned.startswith(("- ", "* ")):
                cleaned = cleaned[2:].strip()
            return cleaned

        normalized_lines = [_normalize_line(line) for line in filtered_lines]
        if not normalized_lines or all(
            verification_line.startswith("VERIFIED:") or verification_line == "ALL_CLAIMS_VERIFIED"
            for verification_line in normalized_lines
        ):
            return "ALL_CLAIMS_VERIFIED"
        return "\n".join(filtered_lines)
    except Exception as exc:
        if is_transient_llm_error(exc):
            return "⚠️ Verification unavailable due to high traffic."
        return f"⚠️ Verification unavailable: {exc}"

async def trigger_verification_task(
    accumulated: str,
    context: str,
    message_obj: cl.Message,
    background_tasks: set[asyncio.Task],
) -> None:
    """Launch async background verification task if verification is enabled."""
    is_error_response = any(
        err in accumulated for err in (HIGH_TRAFFIC_MESSAGE, GENERIC_ERROR_MESSAGE, "⚠️ API error:")
    )
    main_mod = _get_active_main()
    effective_verify_enabled = getattr(main_mod, "VERIFY_ENABLED", VERIFY_ENABLED)
    if effective_verify_enabled and accumulated and not is_error_response:
        verify_fn = getattr(main_mod, "verify_response", verify_response)
        async def verify_and_update():
            try:
                report = await verify_fn(accumulated, context)
                if report and "ALL_CLAIMS_VERIFIED" not in report:
                    message_obj.content += f"\n\n---\n**Verification Note:**\n{report}"
                    await message_obj.update()
            except Exception as e:
                logger.error(f"[chat] Background verification task failed: {e}", exc_info=True)

        task = asyncio.create_task(verify_and_update())
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

# ─── Test Doctrine & Registry ───────────────────────────────────────────────
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
        if not directory.exists():
            return
        with self._lock:
            self.tests = []
            for f in directory.glob("*.md"):
                if f.name == "index.md":
                    continue
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

# ─── RAG Context & Helpers ──────────────────────────────────────────────────
@contextlib.asynccontextmanager
async def status_step(name: str, remove_on_exit: bool = False):
    """Safe context manager to show Chainlit steps only when a UI context exists."""
    if has_chainlit_context():
        async with cl.Step(name=name) as step:
            steps_list = cl.user_session.get("steps_to_remove")
            if isinstance(steps_list, list):
                steps_list.append(step)
            try:
                yield step
            finally:
                if remove_on_exit:
                    await step.remove()
    else:
        class DummyStep:
            def __init__(self):
                self.output = ""
            async def update(self):
                pass
            async def remove(self):
                pass
        yield DummyStep()

async def clear_active_status_steps() -> None:
    """Wipe any registered intermediate UI steps to keep chat history clean."""
    if not has_chainlit_context():
        return
    steps_to_remove = cl.user_session.get("steps_to_remove") or []
    for s in steps_to_remove:
        try:
            await s.remove()
        except Exception as e:
            logger.error(f"[chat] Failed to remove step: {e}")
    cl.user_session.set("steps_to_remove", [])

def _format_history(history: list[dict]) -> str:
    """Format conversation history list into a standardized string for LLM prompts."""
    history_text = ""
    for turn in history[-5:]:
        role = (turn["role"] if isinstance(turn, dict) else turn.role).capitalize()
        content = turn["content"] if isinstance(turn, dict) else turn.content
        if isinstance(content, list):
            content = "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in content])
        history_text += f"{role}: {content}\n"
    return history_text

async def condense_query(message: str, history: list[dict]) -> str:
    """Turn the conversation history and new message into a standalone search query."""
    if not history:
        return message

    history_text = _format_history(history)
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

def _get_active_index():
    main_mod = _get_active_main()
    if main_mod and hasattr(main_mod, "_index"):
        return main_mod._index
    return _index

def _get_active_chunks():
    main_mod = _get_active_main()
    if main_mod and hasattr(main_mod, "_chunks"):
        return main_mod._chunks
    return _chunks

def _get_active_search_batch():
    main_mod = _get_active_main()
    if main_mod and hasattr(main_mod, "search_index_batch"):
        return main_mod.search_index_batch
    return search_index_batch

def _get_active_test_registry():
    main_mod = _get_active_main()
    if main_mod and hasattr(main_mod, "_test_registry"):
        return main_mod._test_registry
    return _test_registry

async def get_rag_context(message: str, history: list[dict]) -> tuple[list[str], str, list[dict]]:
    if history and (not IS_DEV or os.getenv("AGNAV_FORCE_CONDENSE") == "true"):
        async with status_step("context synthesis...") as step:
            condensed = await condense_query(message, history)
            step.output = f'Condensed query: "{condensed}"'
    else:
        condensed = message

    queries = [condensed]

    async with status_step("retrieval...") as step:
        top_k_count = 3 if IS_DEV else 5
        curr_index = _get_active_index()
        curr_chunks = _get_active_chunks()
        search_fn = _get_active_search_batch()
        all_res = await asyncio.to_thread(search_fn, curr_index, curr_chunks, queries, [top_k_count] * len(queries))
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

        sources_found = set(c.get("source", "Unknown") for c in unique_snippets)
        step.output = f"Retrieved {len(unique_snippets)} matching excerpts from {len(sources_found)} reference documents."

    return queries, "\n\n".join(context_parts), unique_snippets

def format_rag_error_message(exc: Exception) -> str:
    """Map exceptions to user-facing error messages, hiding internal error details."""
    if is_transient_llm_error(exc):
        return HIGH_TRAFFIC_MESSAGE
    return GENERIC_ERROR_MESSAGE

async def rag_stream(message: str, history: list[dict]) -> AsyncIterator[tuple[str, str]]:
    """Yields (chunk, context) for tests and legacy callers."""
    curr_index = _get_active_index()
    if curr_index is None:
        yield "⚠️ Knowledge base not loaded.", ""
        return
    try:
        main_mod = _get_active_main()
        ctx_fn = getattr(main_mod, "get_rag_context", get_rag_context)
        queries, context, snippets = await ctx_fn(message, history)
        system = get_system_prompt().format(manifest="", verify_message="") + f"\n\nContext:\n{context}"

        capped = []
        for h in history[-2:]:
            c = h["content"] if isinstance(h["content"], str) else str(h["content"])
            capped.append({"role": h["role"], "content": c[:300] + "..." if len(c) > 300 else c})
        messages = capped + [{"role": "user", "content": message}]

        has_yielded_context = False
        stream_fn = getattr(main_mod, "unified_chat_stream", unified_chat_stream)
        async for chunk in stream_fn(
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
        logger.error(f"[rag] Stream error: {exc}", exc_info=True)
        yield f"⚠️ API error: {exc}", ""

async def rag_review_stream(
    message: str,
    history: list[dict],
    persona_mode: str = "Lookup",
    context: str | None = None,
    queries: list[str] | None = None
) -> AsyncIterator[str]:
    try:
        main_mod = _get_active_main()
        if not context or not queries:
            ctx_fn = getattr(main_mod, "get_rag_context", get_rag_context)
            q_new, c_new, s_new = await ctx_fn(message, history)
            context = context or c_new
            queries = queries or q_new

        base_persona = get_persona_prompt(persona_mode)
        audit_rules = ""
        if persona_mode in ("Grieve", "Manage"):
            registry = _get_active_test_registry()
            matched_tests = registry.find_matches(message + " " + queries[0])
            for test in matched_tests:
                audit_rules += f"\n\n--- MANDATORY LOGIC CHECK: {test.name.upper()} ---\n"
                audit_rules += f"This case involves potential issues related to {test.name}. You MUST follow the EXPLAIN/QUESTION/APPLY/CITE pattern (Explain the principle, ask the relevant Question, Apply it to the facts, and Cite the source).\n"
                audit_rules += f"Criteria:\n{test.content}\n"

        master_rules = get_system_prompt().format(manifest="", verify_message="")
        system = f"{master_rules}\n\n{base_persona}\n\n{audit_rules}\n\n--- KNOWLEDGE BASE CONTEXT ---\n{context}"

        capped = []
        for h in history[-2:]:
            c = h["content"] if isinstance(h["content"], str) else str(h["content"])
            capped.append({"role": h["role"], "content": c[:300] + "..." if len(c) > 300 else c})
        messages = capped + [{"role": "user", "content": message}]

        stream_fn = getattr(main_mod, "unified_chat_stream", unified_chat_stream)
        async for text in stream_fn(
            model=REVIEWER_MODEL,
            max_tokens=REVIEWER_MAX_TOKENS,
            system=system,
            messages=messages
        ):
            yield text
    except Exception as exc:
        logger.error(f"[rag] Pipeline error: {exc}", exc_info=True)
        yield format_rag_error_message(exc)

# ─── Documents & Index Lifecycle ─────────────────────────────────────────────
def _get_download_source_files() -> list[Path]:
    """Scan DATA_DIR for PDF and MD files. Excludes test_fixtures/."""
    if not DATA_DIR.exists():
        return []
    fixtures_dir = DATA_DIR / "test_fixtures"
    files = [
        p for p in DATA_DIR.rglob("*")
        if p.suffix.lower() in (".pdf", ".md")
        and not p.is_relative_to(fixtures_dir)
        and not p.name.endswith(".integrity.md")
    ]
    return sorted(list(set(files)), key=lambda p: str(p))

def resolve_pdf_path(md_path: Path) -> Path:
    """Resolve the matching PDF file path for a given Markdown source path."""
    if md_path.suffix.lower() == ".pdf":
        return md_path

    main_mod = _get_active_main()
    effective_docs_dir = getattr(main_mod, "PUBLIC_DOCS_DIR", PUBLIC_DOCS_DIR)

    pdf_same_dir = md_path.with_suffix(".pdf")
    if pdf_same_dir.exists():
        return pdf_same_dir

    exact_pdf = effective_docs_dir / f"{md_path.stem}.pdf"
    if exact_pdf.exists():
        return exact_pdf

    for found_pdf in effective_docs_dir.rglob(f"{md_path.stem}.pdf"):
        if found_pdf.is_file():
            return found_pdf

    if "_-_" in md_path.stem:
        base_stem = md_path.stem.split("_-_")[0]
        prefix_pdf = effective_docs_dir / f"{base_stem}.pdf"
        if prefix_pdf.exists():
            return prefix_pdf

        for found_pdf in effective_docs_dir.rglob(f"{base_stem}.pdf"):
            if found_pdf.is_file():
                return found_pdf

    return md_path

def build_reference_links(snippets: list[dict], source_path_map: dict[str, Path] | None = None) -> list[str]:
    """Build clean reference document links for retrieved sources."""
    main_mod = _get_active_main()
    active_path_map = source_path_map or getattr(main_mod, "_source_path_map", _source_path_map)
    effective_docs_dir = getattr(main_mod, "PUBLIC_DOCS_DIR", PUBLIC_DOCS_DIR)

    ref_links = []
    seen_sources = set()
    for s in snippets:
        source_name = s.get("source", "")
        if source_name and source_name not in seen_sources and source_name != "Unknown":
            md_path = active_path_map.get(source_name)
            if md_path:
                download_path = resolve_pdf_path(md_path)
                if download_path.exists():
                    try:
                        rel_path = download_path.relative_to(effective_docs_dir)
                        rel_url = f"/public/docs/{rel_path}"
                    except ValueError:
                        rel_url = f"/public/docs/{download_path.name}"
                    clean_title = source_name.replace("_", " ")
                    ref_links.append(f"- [{clean_title}]({rel_url})")
            seen_sources.add(source_name)
    return ref_links

def startup(force_rebuild: bool = False):
    global _index, _chunks, INTEGRITY_WARNING, _source_path_map

    provider = get_llm_provider()
    logger.info(f"[startup] AgNav {AGNAV_VERSION} starting...")
    logger.info(f"[startup] Provider: {provider}")
    logger.info(f"[startup] Default Model: {DEFAULT_MODEL_LLM}")
    logger.info(f"[startup] Build Integrity: {AGNAV_VERSION}")

    main_mod = _get_active_main()
    active_test_registry = getattr(main_mod, "_test_registry", _test_registry)
    active_test_registry.load(TESTS_DIR)

    import indexing
    indexing.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        test_file = indexing.CACHE_DIR / "permissions_test"
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        logger.warning(f"[startup] {CACHE_DIR} is not writable: {e}. Indexing may fail.")

    fetch_cache_fn = getattr(main_mod, "_fetch_pdf_cache_if_missing", _fetch_pdf_cache_if_missing)
    load_index_fn = getattr(main_mod, "load_precomputed_index", load_precomputed_index)
    build_index_fn = getattr(main_mod, "build_index_from_sources", build_index_from_sources)

    fetch_cache_fn()
    _index, _chunks = load_index_fn()
    if _index is None or force_rebuild:
        _index, _chunks = build_index_fn(force=True)
    if _index is not None:
        get_files_fn = getattr(main_mod, "_get_rag_source_files", _get_rag_source_files)
        get_name_fn = getattr(main_mod, "_get_source_name", _get_source_name)
        all_files = get_files_fn()
        _source_path_map = {get_name_fn(p.stem): p for p in all_files}
        report_fn = getattr(main_mod, "get_integrity_report", get_integrity_report)
        report = report_fn()
        failed_files = report.get("failed_files", []) if isinstance(report, dict) else []
        INTEGRITY_WARNING = (
            f"Integrity check failed for: {', '.join(failed_files)}"
            if failed_files
            else None
        )

    # Sync state into main module if present
    if main_mod:
        main_mod._index = _index
        main_mod._chunks = _chunks
        main_mod._source_path_map = _source_path_map
        main_mod.INTEGRITY_WARNING = INTEGRITY_WARNING

    get_download_files_fn = getattr(main_mod, "_get_download_source_files", _get_download_source_files)
    doc_list = get_download_files_fn()
    logger.info(f"[startup] {len(doc_list)} reference documents found.")

async def _ensure_startup() -> None:
    global _startup_done
    if _startup_done:
        return
    async with _startup_lock:
        if _startup_done:
            return
        await asyncio.to_thread(startup)
        _startup_done = True
