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
    assert is_builder_kickoff("start grievance builder")
    assert is_builder_kickoff(
        "I need to file a grievance for a member—start the Grievance Builder wizard and outline the steps."
    )
    assert not is_builder_kickoff("What is Article 14 for filing a grievance?")


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


def test_kickoff_does_not_store_issue_summary():
    state = default_wizard_state()
    state, turn = process_wizard_message("start grievance builder", state)
    assert turn.handled
    assert state["step"] == 0
    assert "issue_summary" not in state.get("facts", {})


def test_required_step_rejects_skip():
    state = default_wizard_state()
    state, _ = process_wizard_message("Manager denied vacation", state)
    state, turn = process_wizard_message("skip", state)
    assert turn.handled
    assert "required" in turn.assistant_message.lower()


def test_format_facts_block_skips_skip_values():
    facts = {"issue_summary": "Overtime denied", "articles": "skip", "remedy": "Pay owed hours"}
    block = format_facts_block(facts)
    assert "Overtime denied" in block
    assert "**Contract articles:**" not in block


def test_should_activate_wizard_when_active():
    state = default_wizard_state()
    assert should_activate_wizard("Grieve", "anything", state)


def test_grieve_persona_loads_case_builder_prompt():
    draft_prompt = app.get_persona_prompt("Grieve", grieve_draft=True)
    assert "Grievance Builder" in draft_prompt
    assert "Partial Form Fields" in draft_prompt
    freeform = app.get_persona_prompt("Grieve")
    assert "Forensic Auditor" in freeform


def test_build_draft_query_includes_all_step_titles():
    facts = {step["id"]: f"value-{step['id']}" for step in WIZARD_STEPS}
    query = build_draft_query(facts)
    assert "Form A" in query
    assert "Form C" in query
