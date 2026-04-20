import pytest
import app
import gradio as gr

def test_manager_mode_in_selector(monkeypatch, mock_anthropic):
    """
    Verifies that 'Manage' is available in the persona selector.
    """
    monkeypatch.setattr(app, "get_anthropic", lambda: mock_anthropic)
    monkeypatch.setattr(app, "_index", "not-none")
    
    demo = app.build_ui()
    
    # Find the persona selector
    def find_persona_selector(parent):
        for child in getattr(parent, "children", []):
            if isinstance(child, gr.Radio) and child.elem_id == "persona_selector":
                return child
            found = find_persona_selector(child)
            if found:
                return found
        return None

    radio = find_persona_selector(demo)
            
    assert radio is not None, "Persona selector (Radio) not found in UI"
    # Gradio Radio choices can be a list of tuples (label, value)
    choice_values = [c[1] if isinstance(c, tuple) else c for c in radio.choices]
    assert "Manage" in choice_values, f"'Manage' not found in choices: {radio.choices}"

def test_manager_persona_prompt(monkeypatch):
    """
    Verifies that get_persona_prompt returns the correctly formatted manager prompt.
    """
    prompt = app.get_persona_prompt("Manage")
    assert "Senior Strategic Management Consultant" in prompt
    assert "INADVERTENT BENEFIT WARNING" in prompt
    assert "Operational Framework" in prompt
    assert "> \"...\"" in prompt # Verbatim quote rule
