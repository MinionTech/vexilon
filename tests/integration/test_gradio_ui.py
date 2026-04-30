"""
tests/integration/test_gradio_ui.py — Integration: Gradio UI build check

Verifies that the Gradio interface can be constructed and handlers connected.
Does NOT rely on internal gradio.testing utilities which may vary across versions.
"""

import pytest
import app
import gradio as gr

def test_ui_builds_correctly(monkeypatch, mock_llm_client):
    """
    Verifies that build_ui() runs without error and returns a gr.Blocks instance.
    """
    # Setup mocks
    monkeypatch.setattr(app, "get_llm_client", lambda: mock_llm_client)
    monkeypatch.setattr(app, "_index", "not-none")
    
    # This catches syntax errors in handlers or missing component references
    demo = app.build_ui()
    
    assert isinstance(demo, gr.Blocks)
    assert len(demo.children) > 0
    
    # Verify expected components are present
    def find_component(parent, comp_type):
        for child in getattr(parent, "children", []):
            if isinstance(child, comp_type):
                return child
            found = find_component(child, comp_type)
            if found:
                return found
        return None

    chatbot = find_component(demo, gr.Chatbot)
    assert chatbot is not None
    
    textbox = find_component(demo, gr.Textbox)
    assert textbox is not None
