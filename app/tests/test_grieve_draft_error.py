"""Tests for Grievance Builder draft failure handling in on_message."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import main as app
from services.grievance_builder import (
    WIZARD_SESSION_KEY,
    WIZARD_STEPS,
    process_wizard_message,
    wizard_state_after_failed_draft,
)


def _sample_facts() -> dict[str, str]:
    return {
        "issue_summary": "Denied vacation",
        "incident_date": "2026-03-01",
        "articles": "Article 27",
        "management_response": "skip",
        "witnesses_evidence": "skip",
        "remedy": "Make whole",
    }


@pytest.mark.asyncio
async def test_on_message_preserves_wizard_facts_when_draft_stream_fails(monkeypatch):
    """Failed draft generation must not clear collected wizard facts."""
    monkeypatch.setattr(app, "_ensure_startup", AsyncMock())
    monkeypatch.setattr(app, "_rate_limiter", MagicMock(is_allowed=MagicMock(return_value=(True, ""))))
    monkeypatch.setattr(app, "sanitize_input", lambda s: (s, False))
    monkeypatch.setattr(app, "get_rag_context", AsyncMock(return_value=(["query"], "ctx", [])))
    monkeypatch.setattr(app, "trigger_verification_task", AsyncMock())
    monkeypatch.setattr(app, "clear_active_status_steps", AsyncMock())
    monkeypatch.setattr(app, "has_chainlit_context", lambda: False)

    facts = _sample_facts()

    async def mock_error_stream(*args, **kwargs):
        yield app.HIGH_TRAFFIC_MESSAGE

    monkeypatch.setattr(app, "rag_review_stream", mock_error_stream)

    session_store = {
        "persona": "Grieve",
        "history": [],
        WIZARD_SESSION_KEY: {
            "active": True,
            "step": len(WIZARD_STEPS),
            "facts": facts,
            "pending_draft": True,
        },
    }
    monkeypatch.setattr("chainlit.user_session.set", lambda k, v: session_store.__setitem__(k, v))
    monkeypatch.setattr("chainlit.user_session.get", lambda k, default=None: session_store.get(k, default))

    mock_msg_instance = MagicMock()
    mock_msg_instance.send = AsyncMock()
    mock_msg_instance.stream_token = AsyncMock()
    mock_msg_instance.update = AsyncMock()
    mock_msg_instance.actions = []
    mock_msg_instance.content = ""
    monkeypatch.setattr("chainlit.Message", MagicMock(return_value=mock_msg_instance))

    msg = MagicMock()
    msg.content = "retry draft"
    msg.elements = []
    await app.on_message(msg)
    await asyncio.sleep(0.01)

    saved = session_store[WIZARD_SESSION_KEY]
    assert saved["facts"] == facts
    assert saved.get("pending_draft") is True
    assert saved.get("active") is True


@pytest.mark.asyncio
async def test_on_message_clears_wizard_facts_after_successful_draft(monkeypatch):
    monkeypatch.setattr(app, "_ensure_startup", AsyncMock())
    monkeypatch.setattr(app, "_rate_limiter", MagicMock(is_allowed=MagicMock(return_value=(True, ""))))
    monkeypatch.setattr(app, "sanitize_input", lambda s: (s, False))
    monkeypatch.setattr(app, "get_rag_context", AsyncMock(return_value=(["query"], "ctx", [])))
    monkeypatch.setattr(app, "trigger_verification_task", AsyncMock())
    monkeypatch.setattr(app, "clear_active_status_steps", AsyncMock())
    monkeypatch.setattr(app, "has_chainlit_context", lambda: False)

    facts = _sample_facts()

    async def mock_ok_stream(*args, **kwargs):
        yield (
            "### 📝 Draft Statement of Grievance\nExample\n\n"
            "### 📋 Partial Form Fields (Steward Review Required)\n"
            "- **Form A — Grievor Case:** draft\n"
            "- **Form B — Notify Designates:** notes\n"
            "- **Form C — Steward Fact Sheet:** witnesses\n"
        )

    monkeypatch.setattr(app, "rag_review_stream", mock_ok_stream)

    session_store = {
        "persona": "Grieve",
        "history": [],
        WIZARD_SESSION_KEY: {
            "active": True,
            "step": len(WIZARD_STEPS),
            "facts": facts,
            "pending_draft": True,
        },
    }
    monkeypatch.setattr("chainlit.user_session.set", lambda k, v: session_store.__setitem__(k, v))
    monkeypatch.setattr("chainlit.user_session.get", lambda k, default=None: session_store.get(k, default))

    mock_msg_instance = MagicMock()
    mock_msg_instance.send = AsyncMock()
    mock_msg_instance.stream_token = AsyncMock()
    mock_msg_instance.update = AsyncMock()
    mock_msg_instance.actions = []
    mock_msg_instance.content = ""
    monkeypatch.setattr("chainlit.Message", MagicMock(return_value=mock_msg_instance))

    msg = MagicMock()
    msg.content = "retry draft"
    msg.elements = []
    await app.on_message(msg)
    await asyncio.sleep(0.01)

    saved = session_store[WIZARD_SESSION_KEY]
    assert saved.get("facts") == {}
    assert saved.get("pending_draft") is False


@pytest.mark.asyncio
async def test_on_message_preserves_wizard_when_draft_is_only_reference_links(monkeypatch):
    """Reference links alone must not count as a successful draft."""
    monkeypatch.setattr(app, "_ensure_startup", AsyncMock())
    monkeypatch.setattr(app, "_rate_limiter", MagicMock(is_allowed=MagicMock(return_value=(True, ""))))
    monkeypatch.setattr(app, "sanitize_input", lambda s: (s, False))
    monkeypatch.setattr(app, "get_rag_context", AsyncMock(return_value=(["query"], "ctx", [{"source": "Doc"}])))
    monkeypatch.setattr(app, "build_reference_links", lambda snippets: ["- [Doc](/doc)"])
    monkeypatch.setattr(app, "trigger_verification_task", AsyncMock())
    monkeypatch.setattr(app, "clear_active_status_steps", AsyncMock())
    monkeypatch.setattr(app, "has_chainlit_context", lambda: False)

    facts = _sample_facts()

    async def mock_empty_stream(*args, **kwargs):
        if False:
            yield ""

    monkeypatch.setattr(app, "rag_review_stream", mock_empty_stream)

    session_store = {
        "persona": "Grieve",
        "history": [],
        WIZARD_SESSION_KEY: {
            "active": True,
            "step": len(WIZARD_STEPS),
            "facts": facts,
            "pending_draft": True,
        },
    }
    monkeypatch.setattr("chainlit.user_session.set", lambda k, v: session_store.__setitem__(k, v))
    monkeypatch.setattr("chainlit.user_session.get", lambda k, default=None: session_store.get(k, default))

    mock_msg_instance = MagicMock()
    mock_msg_instance.send = AsyncMock()
    mock_msg_instance.stream_token = AsyncMock()
    mock_msg_instance.update = AsyncMock()
    mock_msg_instance.actions = []
    mock_msg_instance.content = ""
    monkeypatch.setattr("chainlit.Message", MagicMock(return_value=mock_msg_instance))

    msg = MagicMock()
    msg.content = "retry draft"
    msg.elements = []
    await app.on_message(msg)
    await asyncio.sleep(0.01)

    saved = session_store[WIZARD_SESSION_KEY]
    assert saved["facts"] == facts
    assert saved.get("pending_draft") is True


@pytest.mark.asyncio
async def test_pending_draft_does_not_hijack_freeform_question(monkeypatch):
    """While pending draft, normal questions use freeform Grieve, not auto-retry."""
    monkeypatch.setattr(app, "_ensure_startup", AsyncMock())
    monkeypatch.setattr(app, "_rate_limiter", MagicMock(is_allowed=MagicMock(return_value=(True, ""))))
    monkeypatch.setattr(app, "sanitize_input", lambda s: (s, False))
    monkeypatch.setattr(app, "get_rag_context", AsyncMock(return_value=(["query"], "ctx", [])))
    monkeypatch.setattr(app, "trigger_verification_task", AsyncMock())
    monkeypatch.setattr(app, "clear_active_status_steps", AsyncMock())
    monkeypatch.setattr(app, "has_chainlit_context", lambda: False)

    facts = _sample_facts()
    seen_messages: list[str] = []

    async def mock_stream(message, history, persona, **kwargs):
        seen_messages.append(message)
        assert kwargs.get("grieve_draft") is False
        yield "Article 8 governs the grievance procedure."

    monkeypatch.setattr(app, "rag_review_stream", mock_stream)

    session_store = {
        "persona": "Grieve",
        "history": [],
        WIZARD_SESSION_KEY: {
            "active": True,
            "step": len(WIZARD_STEPS),
            "facts": facts,
            "pending_draft": True,
        },
    }
    monkeypatch.setattr("chainlit.user_session.set", lambda k, v: session_store.__setitem__(k, v))
    monkeypatch.setattr("chainlit.user_session.get", lambda k, default=None: session_store.get(k, default))

    mock_msg_instance = MagicMock()
    mock_msg_instance.send = AsyncMock()
    mock_msg_instance.stream_token = AsyncMock()
    mock_msg_instance.update = AsyncMock()
    mock_msg_instance.actions = []
    mock_msg_instance.content = ""
    monkeypatch.setattr("chainlit.Message", MagicMock(return_value=mock_msg_instance))

    msg = MagicMock()
    msg.content = "What is Article 8?"
    msg.elements = []
    await app.on_message(msg)
    await asyncio.sleep(0.01)

    assert seen_messages == ["What is Article 8?"]
    assert session_store[WIZARD_SESSION_KEY]["facts"] == facts


@pytest.mark.asyncio
async def test_draft_fields_variant_counts_as_successful(monkeypatch):
    monkeypatch.setattr(app, "_ensure_startup", AsyncMock())
    monkeypatch.setattr(app, "_rate_limiter", MagicMock(is_allowed=MagicMock(return_value=(True, ""))))
    monkeypatch.setattr(app, "sanitize_input", lambda s: (s, False))
    monkeypatch.setattr(app, "get_rag_context", AsyncMock(return_value=(["query"], "ctx", [])))
    monkeypatch.setattr(app, "trigger_verification_task", AsyncMock())
    monkeypatch.setattr(app, "clear_active_status_steps", AsyncMock())
    monkeypatch.setattr(app, "has_chainlit_context", lambda: False)

    facts = _sample_facts()

    async def mock_stream(*args, **kwargs):
        yield (
            "### 📋 Partial Form Fields (Steward Review Required)\n"
            "- **Form A — Grievor Case (draft fields):** incident\n"
            "- **Form B — Notify Designates (draft notes):** step 2\n"
            "- **Form C — Steward Fact Sheet (draft fields):** witnesses\n"
        )

    monkeypatch.setattr(app, "rag_review_stream", mock_stream)

    session_store = {
        "persona": "Grieve",
        "history": [],
        WIZARD_SESSION_KEY: {
            "active": True,
            "step": len(WIZARD_STEPS),
            "facts": facts,
            "pending_draft": True,
        },
    }
    monkeypatch.setattr("chainlit.user_session.set", lambda k, v: session_store.__setitem__(k, v))
    monkeypatch.setattr("chainlit.user_session.get", lambda k, default=None: session_store.get(k, default))

    mock_msg_instance = MagicMock()
    mock_msg_instance.send = AsyncMock()
    mock_msg_instance.stream_token = AsyncMock()
    mock_msg_instance.update = AsyncMock()
    mock_msg_instance.actions = []
    mock_msg_instance.content = ""
    monkeypatch.setattr("chainlit.Message", MagicMock(return_value=mock_msg_instance))

    msg = MagicMock()
    msg.content = "retry draft"
    msg.elements = []
    await app.on_message(msg)
    await asyncio.sleep(0.01)

    assert session_store[WIZARD_SESSION_KEY].get("facts") == {}


def test_wizard_state_after_failed_draft():
    facts = _sample_facts()
    state = wizard_state_after_failed_draft(facts)
    assert state["facts"] == facts
    assert state["pending_draft"] is True
    assert state["active"] is True


def test_exit_wizard_clears_pending_draft():
    state = wizard_state_after_failed_draft(_sample_facts())
    state, turn = process_wizard_message("exit wizard", state)
    assert turn.handled
    assert state.get("pending_draft") is False
    assert state.get("active") is False
