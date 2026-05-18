"""
tests/integration/test_chainlit_ui.py — Integration: Chainlit handler wiring

Verifies that importing main registers the Chainlit lifecycle hooks
(on_chat_start, on_message) and exposes the expected configuration
(personas, examples). Does not boot the Chainlit server.
"""

import chainlit as cl
import main as app


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


import pytest


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
