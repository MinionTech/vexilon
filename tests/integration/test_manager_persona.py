import pytest
import app
import gradio as gr

def test_manager_mode_in_selector(monkeypatch, mock_llm_client):
    """
    Verifies that 'Manage' is available in the persona selector.
    """
    monkeypatch.setattr(app, "get_async_openai_client", lambda: mock_llm_client)
    monkeypatch.setattr(app, "_index", "not-none")
    
    demo = app.build_ui()
    
    # Find the persona selector
    def find_persona_selector(parent):
        for child in getattr(parent, "children", []):
            if isinstance(child, gr.Dropdown) and child.elem_id == "persona_selector":
                return child
            found = find_persona_selector(child)
            if found:
                return found
        return None

    selector = find_persona_selector(demo)
            
    assert selector is not None, "Persona selector (Dropdown) not found in UI"
    # Gradio Radio choices can be a list of tuples (label, value)
    choice_values = [c[1] if isinstance(c, tuple) else c for c in selector.choices]
    assert "Manage" in choice_values, f"'Manage' not found in choices: {selector.choices}"

def test_manager_persona_prompt(monkeypatch):
    """
    Verifies that get_persona_prompt returns the correctly formatted manager prompt.
    """
    prompt = app.get_persona_prompt("Manage")
    assert "Senior Strategic Management Consultant" in prompt
    assert "INADVERTENT BENEFIT WARNING" in prompt
    assert "Operational Framework" in prompt
    assert "> \"verbatim text\"" in prompt # Verbatim quote rule
