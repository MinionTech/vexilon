"""
tests/integration/test_gradio_ui.py — Integration: Gradio UI build check

Verifies that the Gradio interface can be constructed and handlers connected.
Does NOT rely on internal gradio.testing utilities which may vary across versions.
"""

import pytest
import app as main_app
from src.vexilon import config, loader, utils
import gradio as gr

def test_ui_builds_correctly(monkeypatch, mock_anthropic):
    """
    Verifies that build_ui() runs without error and returns a gr.Blocks instance.
    """
    # Setup mocks
    monkeypatch.setattr(main_app, "get_anthropic", lambda: mock_anthropic)
    monkeypatch.setattr(main_app, "_index", "not-none")
    
    # This catches syntax errors in handlers or missing component references
    demo = main_app.build_ui()
    
    assert isinstance(demo, gr.Blocks)
    assert len(demo.children) > 0
    
    # Verify expected components are present
    chatbot = next((c for c in demo.children if isinstance(c, gr.Chatbot)), None)
    assert chatbot is not None
    
    def find_tb(block):
        if isinstance(block, gr.Textbox): return block
        for child in getattr(block, "children", []):
            tb = find_tb(child)
            if tb: return tb
        return None
    
    textbox = find_tb(demo)
    assert textbox is not None
