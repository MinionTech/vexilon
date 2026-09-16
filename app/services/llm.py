import os
import re
import sys
import time
import random
import email.utils
import asyncio
import datetime
import logging
from collections.abc import AsyncIterator

import openai
from openai import AsyncOpenAI
import chainlit as cl

from core.config import (
    AGNAV_APP_NAME,
    AGNAV_VERSION,
    HF_PROVIDER,
    DEFAULT_HF_MODEL_ID,
    CURRENT_MODEL_ID,
    VERIFY_ENABLED,
    VERIFY_MODEL,
    LLM_MAX_RETRIES,
    LLM_RETRY_BASE_DELAY,
    LLM_RETRY_MAX_DELAY,
    get_llm_provider,
)

logger = logging.getLogger(__name__)

def has_chainlit_context() -> bool:
    try:
        from chainlit.context import get_context
        return get_context() is not None
    except Exception:
        return False

# ─── Prompts & Rules ────────────────────────────────────────────────────────
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

# ─── Client Management ───────────────────────────────────────────────────────
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

def resolve_model_and_provider(fallback_model: str) -> tuple[str, str]:
    session_model = None
    if has_chainlit_context():
        session_model = cl.user_session.get("selected_model")

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

                    if retry_after is not None and 0 < retry_after <= LLM_RETRY_MAX_DELAY:
                        return retry_after

    raw_delay = min(LLM_RETRY_BASE_DELAY * (2 ** attempt), LLM_RETRY_MAX_DELAY)
    jitter = random.uniform(0.75, 1.25)
    return min(raw_delay * jitter, LLM_RETRY_MAX_DELAY)

def _resolve_llm_client() -> AsyncOpenAI:
    main_mod = sys.modules.get("main")
    if main_mod and hasattr(main_mod, "get_llm_client"):
        return main_mod.get_llm_client()
    curr_mod = sys.modules.get(__name__)
    if curr_mod and hasattr(curr_mod, "get_llm_client") and curr_mod.get_llm_client != get_llm_client:
        return curr_mod.get_llm_client()
    return get_llm_client()

async def unified_chat_create(model: str, messages: list, system: str | list = None, max_tokens: int = 1024) -> str:
    provider, actual_model = resolve_model_and_provider(model)
    client = _resolve_llm_client()
    full_messages = _build_messages(messages, system)

    kwargs = {"model": actual_model, "max_tokens": max_tokens, "messages": full_messages, "timeout": 60.0}

    for attempt in range(LLM_MAX_RETRIES + 1):
        t0 = time.perf_counter()
        logger.info(f"[llm-call] Creating completion for actual_model='{actual_model}' on provider='{provider}' (attempt {attempt + 1}/{LLM_MAX_RETRIES + 1})...")
        try:
            resp = await client.chat.completions.create(**kwargs)
            elapsed = time.perf_counter() - t0
            logger.info(f"[llm-call] Completion creation attempt {attempt + 1} elapsed: {elapsed:.2f} seconds")
            content = resp.choices[0].message.content
            return content or ""
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.info(f"[llm-call] Completion creation attempt {attempt + 1} call duration: {elapsed:.2f} seconds")
            if attempt < LLM_MAX_RETRIES and is_transient_llm_error(exc):
                delay = compute_retry_delay(attempt, exc)
                logger.warning(
                    f"[llm-retry] Transient error in unified_chat_create ({exc}). "
                    f"Retrying in {delay:.2f}s (attempt {attempt + 1}/{LLM_MAX_RETRIES})..."
                )
                await asyncio.sleep(delay)
            else:
                raise

async def unified_chat_stream(model: str, messages: list, system: str | list = None, max_tokens: int = 2048) -> AsyncIterator[str]:
    provider, actual_model = resolve_model_and_provider(model)
    client = _resolve_llm_client()
    full_messages = _build_messages(messages, system)

    kwargs = {"model": actual_model, "max_tokens": max_tokens, "messages": full_messages, "stream": True, "timeout": 300.0}

    stream = None
    for attempt in range(LLM_MAX_RETRIES + 1):
        t0 = time.perf_counter()
        logger.info(f"[llm-call] Opening stream for actual_model='{actual_model}' on provider='{provider}' (attempt {attempt + 1}/{LLM_MAX_RETRIES + 1})...")
        try:
            stream = await client.chat.completions.create(**kwargs)
            elapsed = time.perf_counter() - t0
            logger.info(f"[llm-call] Stream connection attempt {attempt + 1} elapsed: {elapsed:.2f} seconds")
            break
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.info(f"[llm-call] Stream connection attempt {attempt + 1} call duration: {elapsed:.2f} seconds")
            if attempt < LLM_MAX_RETRIES and is_transient_llm_error(exc):
                delay = compute_retry_delay(attempt, exc)
                logger.warning(
                    f"[llm-retry] Transient error in unified_chat_stream ({exc}). "
                    f"Retrying in {delay:.2f}s (attempt {attempt + 1}/{LLM_MAX_RETRIES})..."
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
            reasoning = getattr(delta, "reasoning", None) or ""
            if content:
                buffer += content

            while buffer:
                if not in_think_block:
                    # Look for start tag
                    start_idx = buffer.find(start_tag)
                    if start_idx != -1:
                        # Yield everything before the tag
                        if start_idx > 0:
                            yield buffer[:start_idx]
                        in_think_block = True
                        buffer = buffer[start_idx + len(start_tag):]  # Skip start tag
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
                    # In a think block, look for end tag
                    end_idx = buffer.find(end_tag)
                    if end_idx != -1:
                        in_think_block = False
                        buffer = buffer[end_idx + len(end_tag):]  # Skip end tag
                    else:
                        partial_end_idx = buffer.find("<")
                        if partial_end_idx != -1 and len(buffer[partial_end_idx:]) < len(end_tag):
                            buffer = buffer[partial_end_idx:]
                            break
                        else:
                            buffer = ""
                            break

    # Final flush of the buffer if there's remaining content that isn't a think block
    if buffer and not in_think_block:
        yield buffer

async def verify_response(assistant_response: str, context: str) -> str:
    main_mod = sys.modules.get("main")
    is_verify_enabled = getattr(
        main_mod, "VERIFY_ENABLED", getattr(sys.modules[__name__], "VERIFY_ENABLED", VERIFY_ENABLED)
    )
    if not is_verify_enabled:
        return ""
    try:
        raw_verification = await unified_chat_create(
            model=VERIFY_MODEL,
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
    from core.config import HIGH_TRAFFIC_MESSAGE, GENERIC_ERROR_MESSAGE
    is_error_response = any(err in accumulated for err in (HIGH_TRAFFIC_MESSAGE, GENERIC_ERROR_MESSAGE, "⚠️ API error:"))
    main_mod = sys.modules.get("main")
    is_verify_enabled = getattr(
        main_mod, "VERIFY_ENABLED", getattr(sys.modules[__name__], "VERIFY_ENABLED", VERIFY_ENABLED)
    )
    if is_verify_enabled and accumulated and not is_error_response:
        verify_fn = getattr(main_mod, "verify_response", getattr(sys.modules[__name__], "verify_response", verify_response))
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
