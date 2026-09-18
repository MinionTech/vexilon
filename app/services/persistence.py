import json
import re
import sys
import datetime
import logging
import chainlit as cl

from core.config import MAX_INPUT_LENGTH
from core.security import _rate_limiter

logger = logging.getLogger(__name__)

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

    for i, msg in enumerate(history, 1):
        role_label = "👤 You" if msg["role"] == "user" else "🤖 Assistant"
        md += f"## Turn {i}: {role_label}\n\n{msg['content']}\n\n"

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

    json_matches = re.findall(r'```json\s*\n(.*?)\n\s*```', content, re.DOTALL)
    if json_matches:
        json_str = json_matches[-1]
    else:
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

    persona = str(persona)[:50]
    saved_at = str(saved_at)[:50]

    if len(messages) > 100:
        messages = messages[:100]
        warnings.append("Conversation exceeded the limit of 100 turns. Truncated excess historical messages.")

    main_mod = sys.modules.get("main")
    effective_max_length = getattr(main_mod, "MAX_INPUT_LENGTH", MAX_INPUT_LENGTH)

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

        if len(orig_content) > effective_max_length:
            content_str = orig_content[:effective_max_length]
            truncated_count += 1
        else:
            content_str = orig_content

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
        warnings.append(f"Safely truncated {truncated_count} messages exceeding the length limit of {effective_max_length} characters.")

    if script_stripped:
        warnings.append("Security sanitization: Removed potential script injections from conversation history.")

    return sanitized_messages, persona, saved_at, warnings


async def trigger_session_save(client_id: str = "default") -> None:
    """Save conversation history to downloadable markdown file (PIPA-compliant)."""
    allowed, rate_msg = _rate_limiter.is_allowed(client_id)
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
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.md"

        msg = cl.Message(
            content="Conversation saved as markdown. Click below to download.",
            author="System",
            elements=[cl.File(name=filename, content=markdown_content, display="inline", mime="text/markdown")]
        )
        await msg.send()

        logger.info(f"[save] Conversation saved by {client_id} ({len(history)} messages)")
    except Exception as e:
        logger.error(f"[save] Failed to save conversation: {e}")
        await cl.Message(content=f"Error saving conversation: {e}", author="System").send()


async def trigger_session_load(file_content: str, client_id: str = "default") -> None:
    """Load conversation from uploaded markdown or JSON file."""
    allowed, rate_msg = _rate_limiter.is_allowed(client_id)
    if not allowed:
        await cl.Message(content=rate_msg, author="System").send()
        return

    try:
        messages, saved_persona, saved_at, warnings = deserialize_conversation(file_content)

        current_history: list[dict] = cl.user_session.get("history") or []
        current_history.extend(messages)
        cl.user_session.set("history", current_history)

        if warnings:
            warnings_text = "\n".join(f"- {w}" for w in warnings)
            await cl.Message(
                content=f"⚠️ **Upload Notice**\n\n{warnings_text}",
                author="System",
            ).send()

        msg = cl.Message(
            content=f"**Restored Conversation** (Persona: {saved_persona}, Saved: {saved_at})\n\nLoaded {len(messages)} messages. These are read-only.",
            author="System",
        )
        msg.metadata = {"restored": True}
        await msg.send()

        for loaded_msg in messages:
            display_role = "👤 You" if loaded_msg["role"] == "user" else "🤖 Assistant"
            msg_content = loaded_msg["content"]
            msg_obj = cl.Message(content=msg_content, author=display_role)
            msg_obj.metadata = {"restored": True, "readonly": True}
            await msg_obj.send()

        logger.info(f"[load] Conversation loaded by {client_id} ({len(messages)} messages)")
    except json.JSONDecodeError as e:
        logger.error(f"[load] Invalid JSON in file: {e}")
        await cl.Message(content="Invalid conversation file format. Expected JSON.", author="System").send()
    except ValueError as e:
        logger.error(f"[load] Conversation validation error: {e}")
        await cl.Message(content=f"Conversation file is invalid: {e}", author="System").send()
    except Exception as e:
        logger.error(f"[load] Failed to load conversation: {e}")
        await cl.Message(content=f"Error loading conversation: {e}", author="System").send()


async def ask_for_session_file(client_id: str = "default") -> None:
    """Background coroutine: prompts for a session file using AskFileMessage."""
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
            await trigger_session_load(file_content, client_id=client_id)
    except Exception as e:
        logger.error(f"[load] AskFileMessage background task failed: {e}")
        await cl.Message(content="Failed to load session file.", author="System").send()
