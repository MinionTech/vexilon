"""
tests/test_rag_stream.py — Unit tests for rag_stream()

All external API calls (Anthropic, OpenAI/FAISS search) are mocked.
Tests verify guard-clause behaviour and correct prompt construction.
"""

import re
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import anthropic
import pytest

import app


def test_compose_yml_does_not_hardcode_model_name():
    """
    compose.yml must not hardcode a CLAUDE_MODEL default.
    Defaults belong in app.py; putting them in compose.yml creates a second source
    of truth that can silently override the app default and cause production outages.
    """
    compose_text = (Path(__file__).parent.parent / "compose.yml").read_text()
    match = re.search(r"CLAUDE_MODEL", compose_text)
    assert not match, (
        "compose.yml must not set CLAUDE_MODEL. "
        "Remove it and let app.py own the default — having two places to update "
        "is exactly what caused the production outage."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fake_chunks() -> list[dict]:
    return [
        {"text": "Article 1 says something important.", "page": 5, "chunk_index": 0},
        {"text": "Article 2 says something else.", "page": 6, "chunk_index": 0},
    ]


def _stream_yielding(tokens: list[str]):
    """Return an async context-manager mock whose .text_stream yields *tokens*."""

    @asynccontextmanager
    async def _ctx(*args, **kwargs):
        mock_stream = MagicMock()
        async def _async_gen():
            for t in tokens:
                yield t
        mock_stream.text_stream = _async_gen()
        yield mock_stream

    return _ctx


# ── Guard clauses ─────────────────────────────────────────────────────────────

async def test_rag_stream_no_index_yields_not_ready(monkeypatch):
    """When _index is None, yield the 'not ready' message."""
    monkeypatch.setattr(app, "_index", None)

    output = []
    async for chunk in app.rag_stream("What are my rights?", []):
        output.append(chunk)
    assert len(output) == 1
    assert "not ready" in output[0].lower()


# ── Happy path ────────────────────────────────────────────────────────────────

async def test_rag_stream_yields_tokens_from_claude(monkeypatch):
    """Happy path: tokens yielded by the Anthropic stream reach the caller."""
    fake_index = MagicMock()
    monkeypatch.setattr(app, "_index", fake_index)
    monkeypatch.setattr(app, "_chunks", _fake_chunks())
    
    def mock_search(*a, **kw):
        return _fake_chunks()
    monkeypatch.setattr(app, "search_index", mock_search)

    mock_client = MagicMock()
    mock_client.messages.stream = _stream_yielding(["Hello", " there", "!"])
    monkeypatch.setattr(app, "get_anthropic", lambda: mock_client)

    output = []
    async for chunk in app.rag_stream("Any question", []):
        output.append(chunk)
    assert output == ["Hello", " there", "!"]


async def test_rag_stream_includes_page_context_in_system_prompt(monkeypatch):
    """The system prompt sent to Claude must reference the retrieved page numbers."""
    fake_index = MagicMock()
    monkeypatch.setattr(app, "_index", fake_index)
    monkeypatch.setattr(app, "_chunks", _fake_chunks())
    
    def mock_search(*a, **kw):
        return _fake_chunks()
    monkeypatch.setattr(app, "search_index", mock_search)

    captured = {}

    @asynccontextmanager
    async def _capture_stream(**kwargs):
        captured.update(kwargs)
        mock_stream = MagicMock()
        async def _async_gen():
            yield "ok"
        mock_stream.text_stream = _async_gen()
        yield mock_stream

    mock_client = MagicMock()
    mock_client.messages.stream = _capture_stream
    monkeypatch.setattr(app, "get_anthropic", lambda: mock_client)

    async for _ in app.rag_stream("What about overtime?", []):
        pass

    system_input = captured.get("system", "")
    if isinstance(system_input, list):
        # Handle block array format used for caching
        system_text = "".join(b["text"] for b in system_input if b["type"] == "text")
        # Also verify caching is enabled
        assert any(b.get("cache_control") == {"type": "ephemeral"} for b in system_input)
    else:
        system_text = system_input

    assert "[Page 5]" in system_text
    assert "[Page 6]" in system_text
    assert "Article 1 says something important." in system_text


async def test_rag_stream_appends_user_message_last(monkeypatch):
    """The last message in the messages list sent to Claude must be the user's query."""
    fake_index = MagicMock()
    monkeypatch.setattr(app, "_index", fake_index)
    monkeypatch.setattr(app, "_chunks", _fake_chunks())
    
    def mock_search(*a, **kw):
        return _fake_chunks()
    monkeypatch.setattr(app, "search_index", mock_search)

    captured = {}

    @asynccontextmanager
    async def _capture_stream(**kwargs):
        captured.update(kwargs)
        mock_stream = MagicMock()
        async def _async_gen():
            yield "ok"
        mock_stream.text_stream = _async_gen()
        yield mock_stream

    mock_client = MagicMock()
    mock_client.messages.stream = _capture_stream
    monkeypatch.setattr(app, "get_anthropic", lambda: mock_client)

    history = [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"},
    ]
    async for _ in app.rag_stream("New question", history):
        pass

    messages = captured.get("messages", [])
    assert messages[-1] == {"role": "user", "content": "New question"}
    assert len(messages) == 3  # 2 history + 1 new


async def test_rag_stream_api_error_yields_error_message(monkeypatch):
    """An Anthropic APIError during streaming should yield an error string, not raise."""
    fake_index = MagicMock()
    monkeypatch.setattr(app, "_index", fake_index)
    monkeypatch.setattr(app, "_chunks", _fake_chunks())
    
    def mock_search(*a, **kw):
        return _fake_chunks()
    monkeypatch.setattr(app, "search_index", mock_search)

    @asynccontextmanager
    async def _raising_stream(**kwargs):
        raise anthropic.APIStatusError(
            message="model: bad-model-name",
            response=MagicMock(status_code=404),
            body={"type": "error", "error": {"type": "not_found_error", "message": "model: bad-model-name"}},
        )
        yield  # pragma: no cover — unreachable

    mock_client = MagicMock()
    mock_client.messages.stream = _raising_stream
    monkeypatch.setattr(app, "get_anthropic", lambda: mock_client)

    output = []
    async for chunk in app.rag_stream("Any question", []):
        output.append(chunk)
    assert len(output) == 1
    assert "⚠️" in output[0]
    assert "API error" in output[0]
