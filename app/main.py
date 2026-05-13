import os
import sys
import logging
import asyncio
import chainlit as cl
from pathlib import Path

# ── Environment & Path Setup ──────────────────────────────────────────────
# Ensure we prioritize the local indexing module
sys.path.insert(0, str(Path(__file__).parent))

# Ensure writable files directory for Chainlit
# In containerized environments, /app may be read-only, but ./.pdf_cache is usually a writable volume.
if not os.environ.get("CHAINLIT_FILES_DIR"):
    os.environ["CHAINLIT_FILES_DIR"] = "./.pdf_cache/.files"

# ── Python 3.14 Compatibility Patches ─────────────────────────────────────────
import sniffio
import anyio.to_thread
import anyio._backends._asyncio as _anyio_asyncio_backend

# 1. Patch sniffio
original_current_async_library = sniffio.current_async_library
def patched_current_async_library():
    try:
        return original_current_async_library()
    except Exception:
        return "asyncio"
sniffio.current_async_library = patched_current_async_library

# 2. Patch anyio.to_thread.run_sync (Fixes Starlette static file serving)
_orig_run_sync = anyio.to_thread.run_sync
async def _patched_run_sync(func, *args, abandon_on_cancel=False, limiter=None):
    current = asyncio.current_task()
    if current is None or current not in _anyio_asyncio_backend._task_states:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func, *args)
    return await _orig_run_sync(func, *args, abandon_on_cancel=abandon_on_cancel, limiter=limiter)
anyio.to_thread.run_sync = _patched_run_sync

# 3. Patch anyio CancelScope
from anyio._backends._asyncio import CancelScope
original_init = CancelScope.__init__
def patched_init(self, *args, **kwargs):
    try:
        original_init(self, *args, **kwargs)
    except Exception:
        self._deadline = float('inf')
        self._shield = False
        self._parent_scope = None
        self._cancel_called = False
CancelScope.__init__ = patched_init

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("agnav")

# ── Imports ──────────────────────────────────────────────────────────────────
try:
    from indexing import IndexManager, get_multi_perspective_context, rag_review_stream, sanitize_input
except ImportError as e:
    logger.error(f"Failed to import core modules: {e}")
    sys.exit(1)

# ── App Metadata ─────────────────────────────────────────────────────────────
AGNAV_VERSION = os.getenv("AGNAV_VERSION", "dev")
AGNAV_REPO_URL = os.getenv("AGNAV_REPO_URL", "https://github.com/MinionTech/vexilon")

# ── Global State ─────────────────────────────────────────────────────────────
_index_manager = None
_startup_lock = asyncio.Lock()
INTEGRITY_WARNING = None
_source_path_map = {}

async def _ensure_startup():
    global _index_manager, INTEGRITY_WARNING, _source_path_map
    async with _startup_lock:
        if _index_manager is not None:
            return
        
        logger.info("[startup] Initializing Index Manager...")
        _index_manager = IndexManager()
        report = await _index_manager.ensure_index()
        
        # Map source names to paths for download
        _source_path_map = {p.name: p for p in _index_manager.pdf_dir.glob("*.pdf")}
        
        if report.get("failed_files"):
            INTEGRITY_WARNING = f"⚠️ Index Incomplete: {len(report['failed_files'])} documents failed."

# ── Chainlit UI ──────────────────────────────────────────────────────────────
@cl.set_chat_profiles
async def set_chat_profile():
    return [
        cl.ChatProfile(
            name="Lookup",
            markdown_description="Focus on finding facts and article excerpts.",
            icon="public/persona_lookup.png"
        ),
        cl.ChatProfile(
            name="Grieve",
            markdown_description="Focus on grievance processing and strategy.",
            icon="public/persona_grieve.png"
        ),
        cl.ChatProfile(
            name="Manage",
            markdown_description="Focus on employer relations and management strategy.",
            icon="public/persona_manage.png"
        )
    ]

@cl.on_chat_start
async def start():
    await _ensure_startup()
    
    # ── Welcome Header ────────────────────────────────────────────────────
    welcome_msg = """# BCGEU Navigator
Welcome! I am your forensic labor law assistant. 

**Quick Tips:**
- Click the **Knowledge Base** tab (top-left) to access reference documents.
- Use the **Persona Selector** in the header to switch perspectives.
- Use the **Forensic Sidebar** (right) to access forensic references.
"""
    
    # ── Starter Questions ─────────────────────────────────────────────────
    starter_actions = [
        cl.Action(name="starter_query", payload={"value": "What are the just cause requirements for discipline?"}, label="Just Cause"),
        cl.Action(name="starter_query", payload={"value": "What rights do stewards have in investigation meetings?"}, label="Steward Rights"),
        cl.Action(name="starter_query", payload={"value": "What is the nexus test for establishing a link in off-duty conduct cases?"}, label="Nexus Test"),
    ]

    await cl.Message(
        content=welcome_msg, 
        author="System", 
        actions=starter_actions
    ).send()

    if INTEGRITY_WARNING:
        await cl.Message(content=INTEGRITY_WARNING, author="System").send()

# Callback for persona change is now handled by cl.set_chat_profiles native logic.

@cl.action_callback("starter_query")
async def on_starter_query(action):
    query = action.payload.get("value")
    await cl.Message(content=query, author="User").send()
    await on_message(cl.Message(content=query))

@cl.on_message
async def on_message(message: cl.Message):
    persona = cl.user_session.get("chat_profile") or "Lookup"
    history = cl.user_session.get("history") or []
    
    # 1. Sanitize
    sanitized, flagged = sanitize_input(message.content)
    if flagged:
        await cl.Message(content="⚠️ Input flagged for security review.").send()
        return

    # 2. RAG & Context
    msg = cl.Message(content="")
    await msg.send()
    
    accumulated = ""
    async for chunk in rag_review_stream(sanitized, history, persona):
        accumulated += chunk
        await msg.stream_token(chunk)
    
    await msg.update()
    
    # 3. Update History
    history.append({"role": "user", "content": sanitized})
    history.append({"role": "assistant", "content": accumulated})
    cl.user_session.set("history", history[-10:]) # Keep last 10 turns

if __name__ == "__main__":
    # This is only for local testing without the CLI
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)
