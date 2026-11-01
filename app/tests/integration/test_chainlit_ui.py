"""
tests/integration/test_chainlit_ui.py — Integration: Chainlit handler wiring

Verifies that importing main registers the Chainlit lifecycle hooks
(on_chat_start, on_message) and exposes the expected configuration
(personas, examples). Does not boot the Chainlit server.
"""

from unittest.mock import AsyncMock, MagicMock

import chainlit as cl
import main as app
import pytest
import starters


def test_main_registers_required_handlers(monkeypatch, mock_llm_client):
    """Importing main must wire on_message + on_chat_start without error."""
    monkeypatch.setattr(app, "get_llm_client", lambda: mock_llm_client)
    monkeypatch.setattr(app, "_index", "not-none")

    # Module-level handlers must be callables exported from main.
    assert callable(getattr(app, "on_message", None)), \
        "main.on_message must be defined and decorated with @cl.on_message"
    assert callable(getattr(app, "on_chat_start", None)), \
        "main.on_chat_start must be defined and decorated with @cl.on_chat_start"
    assert callable(getattr(app, "chat_profiles", None)), \
        "main.chat_profiles must be defined and decorated with @cl.set_chat_profiles"


def test_personas_and_examples_exposed():
    """The persona list and example questions must remain available."""
    assert "Lookup" in app.PERSONAS
    assert "Grieve" in app.PERSONAS
    assert "Manage" in app.PERSONAS
    assert app.DEFAULT_PERSONA == "Lookup"
    assert len(app.EXAMPLES) >= 1
    assert all(isinstance(q, str) and q for q in app.EXAMPLES)


@pytest.mark.asyncio
async def test_chat_profiles_returns_chainlit_profiles_for_each_persona():
    """chat_profiles must yield one cl.ChatProfile per persona, with starters."""
    profiles = await app.chat_profiles(None)

    assert len(profiles) == len(app.PERSONAS)
    assert [p.name for p in profiles] == app.PERSONAS
    for p in profiles:
        assert isinstance(p, cl.ChatProfile)
        assert p.starters is not None and len(p.starters) >= 1
    assert sum(1 for p in profiles if p.default) == 1


@pytest.mark.asyncio
async def test_chat_profiles_all_starters_preserved():
    """Each persona profile shares the same mode how-to and Lookup starter chips."""
    profiles = await app.chat_profiles(None)
    assert len(profiles) == 3
    expected_labels = [
        "Discipline Just Cause",
        "Steward Rights",
        "Grievance Builder",
        "Management Guide",
    ]
    for p in profiles:
        assert p.starters is not None
        assert len(p.starters) == len(expected_labels), (
            f"Profile '{p.name}' had {len(p.starters)} starters; expected {len(expected_labels)}"
        )
        assert [s.label for s in p.starters] == expected_labels
        for starter in p.starters:
            assert isinstance(starter.label, str) and starter.label.strip()
            assert isinstance(starter.message, str) and starter.message.strip()


@pytest.mark.asyncio
async def test_on_message_informational_starter_skips_rag(monkeypatch, mock_llm_client):
    """Mode how-to chips return a fixed guide without invoking RAG."""
    monkeypatch.setattr(app, "get_llm_client", lambda: mock_llm_client)
    monkeypatch.setattr(app, "_index", "not-none")
    monkeypatch.setattr(app, "_ensure_startup", AsyncMock())
    monkeypatch.setattr(app, "_rate_limiter", type("RL", (), {"is_allowed": lambda self, _: (True, None)})())

    store: dict = {"persona": "Lookup", "history": []}
    user_session = MagicMock()
    user_session.get = lambda key, default=None: store.get(key, default)
    user_session.set = lambda key, value: store.__setitem__(key, value)
    monkeypatch.setattr(app.cl, "user_session", user_session)

    sent_messages: list = []

    class _Msg:
        def __init__(self, content="", author=None):
            self.content = content
            self.author = author
            self.actions = []
            self.elements = []

        async def send(self):
            sent_messages.append(self)
            return self

        async def update(self):
            return self

    monkeypatch.setattr(app.cl, "Message", _Msg)

    rag_called = {"value": False}

    async def _fail_rag(*_args, **_kwargs):
        rag_called["value"] = True
        yield "should not run"

    monkeypatch.setattr(app, "rag_review_stream", _fail_rag)
    monkeypatch.setattr(app, "get_rag_context", AsyncMock(return_value=([], "", [])))

    async def _noop_apply(message):
        spec = starters.resolve_starter(message)
        if spec and spec.target_persona != store.get("persona"):
            store["persona"] = spec.target_persona
        return spec

    monkeypatch.setattr(app, "apply_starter_persona", _noop_apply)

    await app.on_message(app.cl.Message(content="Grievance Builder"))

    assert rag_called["value"] is False
    assert len(sent_messages) == 1
    assert "Grieve" in sent_messages[0].content
    assert store["persona"] == "Grieve"
    assert len(store["history"]) == 2

