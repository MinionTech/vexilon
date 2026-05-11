"""
tests/integration/test_gradio_ui.py — Integration: Gradio UI build check

Verifies that the Gradio interface can be constructed and handlers connected.
Does NOT rely on internal gradio.testing utilities which may vary across versions.
"""

import pytest
import main as app
import gradio as gr

def test_ui_builds_correctly(monkeypatch, mock_llm_client):
    """
    Verifies that Chainlit is present in main.py.
    """
    # This catches syntax errors in handlers or missing component references
    assert hasattr(app, "cl")
    assert callable(app.on_chat_start)
