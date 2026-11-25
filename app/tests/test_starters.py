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
        "Benefits",
        "Grievance Builder",
    ]


def test_grievance_builder_is_not_legacy_example_question():
    grievance = next(s for s in starters.STARTER_SPECS if s.label == "Grievance Builder")
    assert grievance.message != EXAMPLES[4]
    assert grievance.target_persona == "Lookup"
    assert grievance.informational is True
    assert grievance.assistant_reply and "Grievance Builder" in grievance.assistant_reply


def test_grievance_builder_starter_starts_wizard():
    grievance = next(s for s in starters.STARTER_SPECS if s.label == "Grievance Builder")
    assert grievance.starts_wizard is True


def test_lookup_starters_use_high_traffic_examples():
    just_cause = starters.STARTER_SPECS[0]
    steward = starters.STARTER_SPECS[1]
    assert just_cause.message == EXAMPLES[0]
    assert steward.message == EXAMPLES[1]
    assert just_cause.target_persona == "Lookup"
    assert steward.target_persona == "Lookup"
    assert just_cause.informational is False


def test_benefits_starter_targets_lookup():
    benefits = next(s for s in starters.STARTER_SPECS if s.label == "Benefits")
    assert benefits.message == "What are my benefits for critical illness?"
    assert benefits.target_persona == "Lookup"
    assert benefits.informational is False
    assert benefits.starts_wizard is False


def test_chainlit_starters_match_specs():
    cls = starters.chainlit_starters()
    assert len(cls) == len(starters.STARTER_SPECS)
    for starter, spec in zip(cls, starters.STARTER_SPECS, strict=True):
        assert isinstance(starter, cl.Starter)
        assert starter.label == spec.label
        assert starter.message == spec.message


@pytest.mark.asyncio
async def test_apply_starter_persona_returns_grievance_builder_spec():
    spec = await starters.apply_starter_persona("Grievance Builder")
    assert spec is not None
    assert spec.label == "Grievance Builder"
    assert spec.target_persona == "Lookup"


@pytest.mark.asyncio
async def test_apply_starter_persona_unknown_message_is_noop():
    assert await starters.apply_starter_persona("random question") is None
