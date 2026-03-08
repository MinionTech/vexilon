"""
tests/test_rag_stream.py — Unit tests for rag_stream()

All external API calls (Anthropic, OpenAI/FAISS search) are mocked.
Tests verify guard-clause behaviour and correct prompt construction.
"""

# ── Model name sanity check ───────────────────────────────────────────────────
# This is cheap and catches the exact class of bug that broke production on 2026-03-08.
# Add every known-bad name here as they are discovered.
_KNOWN_BAD_MODEL_NAMES = {
    "claude-3-5-haiku-20241022",  # AI-hallucinated; never existed in the Anthropic API
}

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

import app


def test_claude_model_default_is_not_known_bad():
    """
    The module-level CLAUDE_MODEL constant must not be a known-broken model name.
    This catches AI-hallucinated model IDs before they ever reach the API.
    """
    assert app.CLAUDE_MODEL not in _KNOWN_BAD_MODEL_NAMES, (
        f"CLAUDE_MODEL='{app.CLAUDE_MODEL}' is a known-bad model name. "
        f"Update the default in app.py."
    )


def test_compose_yml_model_default_is_not_known_bad():
    """
    The CLAUDE_MODEL fallback in compose.yml must not be a known-broken model name.
    This is the value that actually reaches the container if no env var is set.
    """
    import re
    from pathlib import Path

    compose_text = (Path(__file__).parent.parent / "compose.yml").read_text()
    # Match:  CLAUDE_MODEL: ${CLAUDE_MODEL:-some-model-name}
    match = re.search(r"CLAUDE_MODEL:\s*\$\{CLAUDE_MODEL:-([^}]+)\}", compose_text)
    assert match, "Could not find CLAUDE_MODEL default in compose.yml"
    default_in_compose = match.group(1).strip()
    assert default_in_compose not in _KNOWN_BAD_MODEL_NAMES, (
        f"compose.yml CLAUDE_MODEL default='{default_in_compose}' is a known-bad model name. "
        f"Update compose.yml."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fake_chunks() -> list[dict]:
    return [
        {"text": "Article 1 says something important.", "page": 5, "chunk_index": 0},
        {"text": "Article 2 says something else.", "page": 6, "chunk_index": 0},
    ]


def _stream_yielding(tokens: list[str]):
    """Return a context-manager mock whose .text_stream yields *tokens*."""

    @contextmanager
    def _ctx(*args, **kwargs):
        mock_stream = MagicMock()
        mock_stream.text_stream = iter(tokens)
        yield mock_stream

    return _ctx


# ── Guard clauses ─────────────────────────────────────────────────────────────

def test_rag_stream_startup_error_yields_error_message(monkeypatch):
    """When _startup_error is set, rag_stream must yield an error string and stop."""
    monkeypatch.setattr(app, "_startup_error", "Something blew up")
    monkeypatch.setattr(app, "_index", None)

    output = list(app.rag_stream("What are my rights?", []))
    assert len(output) == 1
    assert "failed to start" in output[0].lower() or "⚠️" in output[0]


def test_rag_stream_no_index_yields_not_ready(monkeypatch):
    """When _index is None (but no startup error), yield the 'not ready' message."""
    monkeypatch.setattr(app, "_startup_error", None)
    monkeypatch.setattr(app, "_index", None)

    output = list(app.rag_stream("What are my rights?", []))
    assert len(output) == 1
    assert "not ready" in output[0].lower()


# ── Happy path ────────────────────────────────────────────────────────────────

def test_rag_stream_yields_tokens_from_claude(monkeypatch):
    """Happy path: tokens yielded by the Anthropic stream reach the caller."""
    fake_index = MagicMock()
    monkeypatch.setattr(app, "_startup_error", None)
    monkeypatch.setattr(app, "_index", fake_index)
    monkeypatch.setattr(app, "_chunks", _fake_chunks())
    monkeypatch.setattr(app, "search_index", lambda *a, **kw: _fake_chunks())

    mock_client = MagicMock()
    mock_client.messages.stream = _stream_yielding(["Hello", " there", "!"])
    monkeypatch.setattr(app, "get_anthropic", lambda: mock_client)

    output = list(app.rag_stream("Any question", []))
    assert output == ["Hello", " there", "!"]


def test_rag_stream_includes_page_context_in_system_prompt(monkeypatch):
    """The system prompt sent to Claude must reference the retrieved page numbers."""
    fake_index = MagicMock()
    monkeypatch.setattr(app, "_startup_error", None)
    monkeypatch.setattr(app, "_index", fake_index)
    monkeypatch.setattr(app, "_chunks", _fake_chunks())
    monkeypatch.setattr(app, "search_index", lambda *a, **kw: _fake_chunks())

    captured = {}

    @contextmanager
    def _capture_stream(**kwargs):
        captured.update(kwargs)
        mock_stream = MagicMock()
        mock_stream.text_stream = iter(["ok"])
        yield mock_stream

    mock_client = MagicMock()
    mock_client.messages.stream = _capture_stream
    monkeypatch.setattr(app, "get_anthropic", lambda: mock_client)

    list(app.rag_stream("What about overtime?", []))

    system = captured.get("system", "")
    assert "[Page 5]" in system
    assert "[Page 6]" in system
    assert "Article 1 says something important." in system


def test_rag_stream_appends_user_message_last(monkeypatch):
    """The last message in the messages list sent to Claude must be the user's query."""
    fake_index = MagicMock()
    monkeypatch.setattr(app, "_startup_error", None)
    monkeypatch.setattr(app, "_index", fake_index)
    monkeypatch.setattr(app, "_chunks", _fake_chunks())
    monkeypatch.setattr(app, "search_index", lambda *a, **kw: _fake_chunks())

    captured = {}

    @contextmanager
    def _capture_stream(**kwargs):
        captured.update(kwargs)
        mock_stream = MagicMock()
        mock_stream.text_stream = iter(["ok"])
        yield mock_stream

    mock_client = MagicMock()
    mock_client.messages.stream = _capture_stream
    monkeypatch.setattr(app, "get_anthropic", lambda: mock_client)

    history = [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"},
    ]
    list(app.rag_stream("New question", history))

    messages = captured.get("messages", [])
    assert messages[-1] == {"role": "user", "content": "New question"}
    assert len(messages) == 3  # 2 history + 1 new


def test_rag_stream_api_error_yields_error_message(monkeypatch):
    """An Anthropic APIError during streaming should yield an error string, not raise."""
    import anthropic as _anthropic

    fake_index = MagicMock()
    monkeypatch.setattr(app, "_startup_error", None)
    monkeypatch.setattr(app, "_index", fake_index)
    monkeypatch.setattr(app, "_chunks", _fake_chunks())
    monkeypatch.setattr(app, "search_index", lambda *a, **kw: _fake_chunks())

    @contextmanager
    def _raising_stream(**kwargs):
        raise _anthropic.APIStatusError(
            message="model: bad-model-name",
            response=MagicMock(status_code=404),
            body={"type": "error", "error": {"type": "not_found_error", "message": "model: bad-model-name"}},
        )
        yield  # pragma: no cover — unreachable

    mock_client = MagicMock()
    mock_client.messages.stream = _raising_stream
    monkeypatch.setattr(app, "get_anthropic", lambda: mock_client)

    output = list(app.rag_stream("Any question", []))
    assert len(output) == 1
    assert "⚠️" in output[0]
    assert "API error" in output[0]
