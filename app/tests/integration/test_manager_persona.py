import pytest
import main as app
import gradio as gr

def test_manager_mode_in_selector(monkeypatch, mock_llm_client):
    """
    Verifies that 'Manage' is available in the persona selector.
    """
    monkeypatch.setattr(app, "get_llm_client", lambda: mock_llm_client)
    monkeypatch.setattr(app, "_index", "not-none")
    
    import pathlib
    app_path = pathlib.Path(__file__).parent.parent.parent / "main.py"
    content = app_path.read_text()
    assert '"Manage"' in content, "'Manage' persona not found in Chainlit UI configuration"

def test_manager_persona_prompt(monkeypatch):
    """
    Verifies that get_persona_prompt returns the correctly formatted manager prompt.
    """
    prompt = app.get_persona_prompt("Manage")
    assert "Senior Strategic Management Consultant" in prompt
    assert "INADVERTENT BENEFIT WARNING" in prompt
    assert "Operational Framework" in prompt
    assert "> \"verbatim text\"" in prompt # Verbatim quote rule
