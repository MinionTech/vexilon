"""Unit tests for welcome-screen starter chips."""

import chainlit as cl

import starters
from core.config import EXAMPLES


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
    assert "Grieve" in grievance.message


def test_management_guide_targets_manage():
    guide = next(s for s in starters.STARTER_SPECS if s.label == "Management Guide")
    assert guide.target_persona == "Manage"
    assert "paperwork" in guide.message.lower() or "forms" in guide.message.lower()


def test_lookup_starters_use_high_traffic_examples():
    just_cause = starters.STARTER_SPECS[0]
    steward = starters.STARTER_SPECS[1]
    assert just_cause.message == EXAMPLES[0]
    assert steward.message == EXAMPLES[1]
    assert just_cause.target_persona == "Lookup"
    assert steward.target_persona == "Lookup"


def test_chainlit_starters_match_specs():
    cls = starters.chainlit_starters()
    assert len(cls) == len(starters.STARTER_SPECS)
    for starter, spec in zip(cls, starters.STARTER_SPECS, strict=True):
        assert isinstance(starter, cl.Starter)
        assert starter.label == spec.label
        assert starter.message == spec.message
