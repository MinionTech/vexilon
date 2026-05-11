import pytest
import main as app


@pytest.mark.asyncio
async def test_manager_mode_in_selector(monkeypatch, mock_llm_client):
    """Verifies that 'Manage' is available as a chat profile."""
    monkeypatch.setattr(app, "get_llm_client", lambda: mock_llm_client)
    monkeypatch.setattr(app, "_index", "not-none")

    profiles = await app.chat_profiles(None)
    names = [p.name for p in profiles]
    assert "Manage" in names, f"'Manage' not found in profiles: {names}"


def test_manager_persona_prompt(monkeypatch):
    """
    Verifies that get_persona_prompt returns the correctly formatted manager prompt.
    """
    prompt = app.get_persona_prompt("Manage")
    assert "Senior Strategic Management Consultant" in prompt
    assert "INADVERTENT BENEFIT WARNING" in prompt
    assert "Operational Framework" in prompt
    assert "> \"verbatim text\"" in prompt # Verbatim quote rule
