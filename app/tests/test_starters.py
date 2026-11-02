"""Unit tests for welcome-screen starter chips."""

from unittest.mock import AsyncMock, MagicMock

import chainlit as cl
import pytest

import starters
from core.config import EXAMPLES, PERSONAS


def test_starter_specs_shape():
    assert 3 <= len(starters.STARTER_SPECS) <= 5
    labels = [s.label for s in starters.STARTER_SPECS]
    assert labels == [
        "Discipline Just Cause",
        "Steward Rights",
        "Grievance Builder",
        "Management Guide",
    ]


def test_grievance_builder_is_not_legacy_example_question():
    grievance = next(s for s in starters.STARTER_SPECS if s.label == "Grievance Builder")
    assert grievance.message != EXAMPLES[4]
    assert grievance.target_persona == "Grieve"
    assert grievance.informational is True
    assert grievance.assistant_reply and "Grieve" in grievance.assistant_reply


def test_grievance_builder_starter_starts_wizard():
    grievance = next(s for s in starters.STARTER_SPECS if s.label == "Grievance Builder")
    assert grievance.starts_wizard is True


def test_management_guide_targets_manage():
    guide = next(s for s in starters.STARTER_SPECS if s.label == "Management Guide")
    assert guide.target_persona == "Manage"
    assert guide.informational is True
    assert guide.assistant_reply
    assert "paperwork" in guide.assistant_reply.lower() or "forms" in guide.assistant_reply.lower()


def test_lookup_starters_use_high_traffic_examples():
    just_cause = starters.STARTER_SPECS[0]
    steward = starters.STARTER_SPECS[1]
    assert just_cause.message == EXAMPLES[0]
    assert steward.message == EXAMPLES[1]
    assert just_cause.target_persona == "Lookup"
    assert steward.target_persona == "Lookup"
    assert just_cause.informational is False


def test_chainlit_starters_match_specs():
    cls = starters.chainlit_starters()
    assert len(cls) == len(starters.STARTER_SPECS)
    for starter, spec in zip(cls, starters.STARTER_SPECS, strict=True):
        assert isinstance(starter, cl.Starter)
        assert starter.label == spec.label
        assert starter.message == spec.message


@pytest.mark.asyncio
async def test_apply_starter_persona_switches_grieve(monkeypatch):
    store: dict = {"persona": "Lookup"}
    user_session = MagicMock()
    user_session.get = lambda key, default=None: store.get(key, default)
    user_session.set = lambda key, value: store.__setitem__(key, value)
    monkeypatch.setattr(cl, "user_session", user_session)

    settings = MagicMock()
    settings.refresh = AsyncMock()
    monkeypatch.setattr(starters, "build_persona_settings", lambda idx: settings)

    spec = await starters.apply_starter_persona("Grievance Builder")

    assert spec is not None
    assert spec.label == "Grievance Builder"
    assert store["persona"] == "Grieve"
    settings.refresh.assert_awaited_once()


@pytest.mark.asyncio
async def test_apply_starter_persona_skips_refresh_when_already_in_mode(monkeypatch):
    store: dict = {"persona": "Grieve"}
    user_session = MagicMock()
    user_session.get = lambda key, default=None: store.get(key, default)
    user_session.set = lambda key, value: store.__setitem__(key, value)
    monkeypatch.setattr(cl, "user_session", user_session)

    settings = MagicMock()
    settings.refresh = AsyncMock()
    monkeypatch.setattr(starters, "build_persona_settings", lambda idx: settings)

    spec = await starters.apply_starter_persona("Grievance Builder")

    assert spec is not None
    assert store["persona"] == "Grieve"
    settings.refresh.assert_not_awaited()


@pytest.mark.asyncio
async def test_apply_starter_persona_unknown_message_is_noop(monkeypatch):
    store: dict = {"persona": "Lookup"}
    user_session = MagicMock()
    user_session.get = lambda key, default=None: store.get(key, default)
    user_session.set = lambda key, value: store.__setitem__(key, value)
    monkeypatch.setattr(cl, "user_session", user_session)

    settings = MagicMock()
    settings.refresh = AsyncMock()
    monkeypatch.setattr(starters, "build_persona_settings", lambda idx: settings)

    assert await starters.apply_starter_persona("random question") is None
    assert store["persona"] == "Lookup"
    settings.refresh.assert_not_awaited()


def test_build_persona_settings_uses_personas_list():
    settings = starters.build_persona_settings(PERSONAS.index("Manage"))
    assert len(settings.inputs) == 1
    widget = settings.inputs[0]
    assert widget.id == "Persona"
    assert widget.values == PERSONAS
    assert widget.initial_index == PERSONAS.index("Manage")
