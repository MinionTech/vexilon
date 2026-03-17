"""
tests/integration/test_gradio_ui.py — Integration: Gradio UI build check

Verifies that the Gradio interface can be constructed and handlers connected.
Does NOT rely on internal gradio.testing utilities which may vary across versions.
"""

import pytest
import app
import gradio as gr

def test_ui_builds_correctly(monkeypatch, mock_anthropic):
    """
    Verifies that build_ui() runs without error and returns a gr.Blocks instance.
    """
    # Setup mocks
    monkeypatch.setattr(app, "get_anthropic", lambda: mock_anthropic)
    monkeypatch.setattr(app, "_index", "not-none")
    
    # This catches syntax errors in handlers or missing component references
    demo = app.build_ui()
    
    assert isinstance(demo, gr.Blocks)
    assert len(demo.children) > 0
    
    # Verify expected components are present
    chatbot = next((c for c in demo.children if isinstance(c, gr.Chatbot)), None)
    assert chatbot is not None
    
    textbox = None
    for child in demo.children:
        if isinstance(child, gr.Textbox):
            textbox = child
        elif isinstance(child, gr.Row):
            # Textbox is inside a Row in app.py
            for sub in child.children:
                if isinstance(sub, gr.Textbox):
                    textbox = sub
    
    assert textbox is not None
