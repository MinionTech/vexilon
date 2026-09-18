import os
import asyncio
import logging

from patches import apply_patches
apply_patches()

import chainlit as cl
from chainlit.config import config as _cl_config
from chainlit.server import app as cl_app

from brand import AGNAV_APP_NAME, AGNAV_APP_DESCRIPTION, get_brand as _get_brand
_cl_config.ui.name = AGNAV_APP_NAME
_cl_config.ui.description = AGNAV_APP_DESCRIPTION

# ─── Backward-Compatible Re-exports ──────────────────────────────────────────
from core.config import *  # noqa: F401, F403
from core.security import *  # noqa: F401, F403
from services.llm import *  # noqa: F401, F403
from services.persistence import *  # noqa: F401, F403
from middleware.cookies import *  # noqa: F401, F403
from indexing import (  # noqa: F401
    _get_source_name,
    _get_rag_source_files,
    build_index_from_sources,
    get_integrity_report,
    load_precomputed_index,
    search_index_batch,
    _fetch_pdf_cache_if_missing,
    DATA_DIR,
    CACHE_DIR,
    PDF_CACHE_DIR,
    get_embed_model,
    EMBED_DIM,
    chunk_text,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from core.config import (  # noqa: F401
    _SIMPLE_KEYWORDS,
    _JOKE_KEYWORDS,
    _ALL_SIMPLE_KEYWORDS,
    _get_default_model,
)
from core.security import (  # noqa: F401
    _rate_limiter,
    _parse_client_uuid,
    _client_id,
)
from services.llm import (  # noqa: F401
    _index,
    _chunks,
    INTEGRITY_WARNING,
    _test_registry,
    _startup_done,
    _startup_lock,
    _ensure_startup,
    _huggingface_client,
    _ollama_client,
    _build_messages,
    _format_history,
    _get_download_source_files,
)

logger = logging.getLogger(__name__)
_background_tasks: set[asyncio.Task] = set()

# Register middleware and API routes
register_routes_and_middleware(cl_app)

# ─── Auth ───────────────────────────────────────────────────────────────────
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

    await cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="Persona",
                label="Navigator Persona",
                values=["Lookup", "Grieve", "Manage"],
                initial_index=0,
            ),
        ]
    ).send()

    cl.user_session.set("history", [])
    cl.user_session.set("persona", "Lookup")
    cl.user_session.set("selected_model", get_default_model_setting())

    if INTEGRITY_WARNING:
        await cl.Message(content=INTEGRITY_WARNING, author="system").send()

@cl.on_window_message
async def on_window_message(data):
    if not isinstance(data, dict) or data.get("type") != "vexilon_client_id":
        return
    client_uuid = _parse_client_uuid(data.get("clientId"))
    if client_uuid:
        cl.user_session.set("client_uuid", client_uuid)

async def on_persona_action(action: cl.Action):
    persona = action.payload.get("value")
    if persona:
        cl.user_session.set("persona", persona)

@cl.action_callback("starter_query")
async def on_action(action: cl.Action):
    query = action.payload.get("value")
    if not query:
        return
    await cl.Message(content=f"**Query:** {query}", author="System").send()
    await on_message(cl.Message(content=query))
    await action.remove()

@cl.action_callback("save_conversation")
async def on_save_conversation(action: cl.Action):
    await trigger_session_save(_client_id())

@cl.action_callback("load_conversation")
async def on_load_conversation(action: cl.Action):
    allowed, rate_msg = _rate_limiter.is_allowed(_client_id())
    if not allowed:
        await cl.Message(content=rate_msg, author="System").send()
        return

    task = asyncio.create_task(ask_for_session_file(_client_id()))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

@cl.on_message
async def on_message(message: cl.Message) -> None:
    cl.user_session.set("steps_to_remove", [])

    if (message.content or "").strip() == VEXILON_SAVE_SENTINEL:
        await trigger_session_save(_client_id())
        return

    await _ensure_startup()

    if message.elements:
        for element in message.elements:
            if (
                element.mime in ["text/markdown", "text/plain", "application/json", "application/octet-stream"]
                or element.name.lower().endswith((".md", ".json"))
            ):
                try:
                    with open(element.path, "r", encoding="utf-8") as f:
                        file_content = f.read()
                    await trigger_session_load(file_content, _client_id())
                    await cl.Message(content="✓ Session restored successfully.", author="System").send()
                except Exception as e:
                    logger.error(f"[load] Failed to read uploaded session file: {e}")
                    await cl.Message(content="Failed to read the uploaded session file.", author="System").send()
                return

    msg_str = (message.content or "").strip()
    if not msg_str:
        return

    prev_msg = cl.user_session.get("last_assistant_message")
    if prev_msg:
        try:
            prev_msg.actions = []
            await prev_msg.update()
        except Exception:
            pass

    allowed, rate_msg = _rate_limiter.is_allowed(_client_id())
    if not allowed:
        await cl.Message(content=rate_msg or "⚠️ Rate limit exceeded. Please wait before sending another message.").send()
        return

    sanitized, flagged = sanitize_input(msg_str)
    if flagged:
        await cl.Message(content="⚠️ Your request contained invalid input or exceeded maximum length.").send()
        return

    persona = cl.user_session.get("persona") or cl.user_session.get("chat_profile") or DEFAULT_PERSONA
    history: list[dict] = cl.user_session.get("history") or []

    out = cl.Message(content="")
    await out.send()

    accumulated = ""
    word_count = len(sanitized.split())
    char_count = len(sanitized)
    logger.info(f"[chat] Starting stream for {persona} mode (Words: {word_count}, Chars: {char_count})")
    try:
        queries, context, snippets = await get_rag_context(sanitized, history)
        ref_links = build_reference_links(snippets)
        out.elements = []

        first_token_received = False
        async for chunk in rag_review_stream(sanitized, history, persona, context=context, queries=queries):
            if not chunk:
                continue
            if not first_token_received:
                first_token_received = True
                await clear_active_status_steps()

            accumulated += chunk
            await out.stream_token(chunk)

        if ref_links:
            ref_section = "\n\n### 📄 Reference Documents\n" + "\n".join(ref_links)
            accumulated += ref_section
            await out.stream_token(ref_section)
    except Exception as exc:
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

    await trigger_verification_task(accumulated, context if "context" in locals() else "", out, _background_tasks)

    history.append({"role": "user", "content": sanitized})
    history.append({"role": "assistant", "content": accumulated})
    cl.user_session.set("history", history)
    cl.user_session.set("last_assistant_message", out)

    await clear_active_status_steps()
    logger.info(f"[chat] Stream completed. Total length: {len(accumulated)}")
