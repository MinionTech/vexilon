"""Unit tests for Grievance Builder wizard (Grieve persona)."""

import main as app
from services.grievance_builder import (
    WIZARD_STEPS,
    build_draft_query,
    default_wizard_state,
    format_facts_block,
    is_builder_kickoff,
    process_wizard_message,
    should_activate_wizard,
    wizard_intro_message,
)


def test_is_builder_kickoff():
    assert is_builder_kickoff("I need to file a grievance for a member")
    assert not is_builder_kickoff("What is Article 14?")


def test_wizard_intro_lists_step_one():
    intro = wizard_intro_message()
    assert "Grievance Builder" in intro
    assert "steward review" in intro.lower()
    assert "Step 1 of" in intro


def test_process_wizard_advances_through_steps():
    state = default_wizard_state()
    for i, answer in enumerate(
        [
            "Manager denied vacation without cause",
            "2026-03-01",
            "Article 27",
            "skip",
            "skip",
            "Make the member whole",
        ]
    ):
        state, turn = process_wizard_message(answer, state)
        assert turn.handled
        if i < len(WIZARD_STEPS) - 1:
            assert not turn.generate_draft
        else:
            assert turn.generate_draft
            assert "Partial Form Fields" in turn.draft_query
            assert "Make the member whole" in turn.draft_query


def test_exit_wizard_deactivates():
    state = default_wizard_state()
    state, turn = process_wizard_message("exit wizard", state)
    assert turn.handled
    assert state["active"] is False


def test_format_facts_block_skips_skip_values():
    facts = {"issue_summary": "Overtime denied", "articles": "skip", "remedy": "Pay owed hours"}
    block = format_facts_block(facts)
    assert "Overtime denied" in block
    assert "skip" not in block.lower() or "articles" not in block.lower()


def test_should_activate_wizard_when_active():
    state = default_wizard_state()
    assert should_activate_wizard("Grieve", "anything", state)


def test_grieve_persona_loads_case_builder_prompt():
    prompt = app.get_persona_prompt("Grieve")
    assert "Grievance Builder" in prompt
    assert "Partial Form Fields" in prompt
    assert "Human review required" in prompt or "human review" in prompt.lower()


def test_build_draft_query_includes_all_step_titles():
    facts = {step["id"]: f"value-{step['id']}" for step in WIZARD_STEPS}
    query = build_draft_query(facts)
    assert "Form A" in query
    assert "Form C" in query
