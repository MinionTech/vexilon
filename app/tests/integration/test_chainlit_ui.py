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


@pytest.mark.asyncio
async def test_chat_profiles_all_starters_preserved():
<<<<<<< HEAD
    """Grieve keeps the Builder starter; Lookup/Manage keep four general starters."""
=======
    """Each persona profile shares the same mode how-to and Lookup starter chips."""
>>>>>>> f4762ec (fix(ui): starter CTAs as mode how-tos with persona switch)
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
<<<<<<< HEAD
        expected = 5 if p.name == "Grieve" else 4
        assert len(p.starters) == expected, f"Profile '{p.name}' had {len(p.starters)} starters; expected {expected}"
=======
        assert len(p.starters) == len(expected_labels), (
            f"Profile '{p.name}' had {len(p.starters)} starters; expected {len(expected_labels)}"
        )
        assert [s.label for s in p.starters] == expected_labels
>>>>>>> f4762ec (fix(ui): starter CTAs as mode how-tos with persona switch)
        for starter in p.starters:
            assert isinstance(starter.label, str) and starter.label.strip()
            assert isinstance(starter.message, str) and starter.message.strip()

