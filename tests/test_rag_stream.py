"""
tests/test_rag_stream.py — Unit tests for rag_review_stream() logic and prompt construction.

Mocks Anthropic's AsyncStream and search_index() results.
Checks that chunk metadata and Article headers are properly formatted in system prompts.
"""

from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import anthropic
import pytest

import app as main_app
from src.vexilon import config, loader, vector, utils


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fake_chunks():
    """Return a mock chunk list with Article headers."""
    return [
        {"text": "Article 1 says something important.", "page": 5, "chunk_index": 0},
        {"text": "Article 2 says something else.", "page": 6, "chunk_index": 1},
    ]


@asynccontextmanager
async def _stream_yielding(tokens: list[str], input_tokens: int = 10, output_tokens: int = 5):
    """
    Simulate an Anthropic AsyncStream context manager.
    Yields 'tokens' one by one via text_stream.
    """
    mock_stream = MagicMock()

    async def _async_gen():
        for t in tokens:
            yield t

    mock_stream.text_stream = _async_gen()

    # Stub the usage/final response attributes
    fake_usage = MagicMock(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    fake_message = MagicMock()
    fake_message.usage = fake_usage

    async def _get_final():
        return fake_message

    mock_stream.get_final_message = _get_final

    yield mock_stream


# ── Logic Tests ─────────────────────────────────────────────────────────────

def test_compose_yml_does_not_hardcode_model_name():
    """Verify that we're using a modern model, not hardcoded ancient ones."""
    # This is a bit of a meta-test to catch if someone hardcodes 'claude-2.1' etc.
    assert "claude-3-5" in config.CLAUDE_MODEL or "claude-haiku" in config.CLAUDE_MODEL


@pytest.mark.asyncio
async def test_rag_review_stream_no_index_yields_not_ready(monkeypatch):
    """If the index is None at startup, the stream should yield a ⚠️ error message."""
    monkeypatch.setattr(main_app, "_index", None)
    monkeypatch.setattr(main_app, "_chunks", [])

    results = []
    async for chunk, ctx in main_app.rag_review_stream("Any question", []):
        results.append(chunk)

    assert any("not ready" in r.lower() for r in results)


@pytest.mark.asyncio
async def test_rag_review_stream_yields_tokens_from_claude(monkeypatch):
    """Happy path: tokens yielded by the Anthropic stream reach the caller."""
    fake_index = MagicMock()
    monkeypatch.setattr(main_app, "_index", fake_index)
    monkeypatch.setattr(main_app, "_chunks", _fake_chunks())
    monkeypatch.setattr(config, "VERIFY_ENABLED", False)

    def mock_search(*a, **kw):
        return _fake_chunks()

    monkeypatch.setattr(vector, "search_index", mock_search)

    mock_client = MagicMock()
    
    @asynccontextmanager
    async def mock_stream_func(*args, **kwargs):
        async with _stream_yielding(["Hello", " there", "!"]) as s:
            yield s
            
    mock_client.messages.stream = mock_stream_func
    monkeypatch.setattr(main_app, "get_anthropic", lambda: mock_client)

    output = []
    async for chunk, ctx in main_app.rag_review_stream("Any question", []):
        if chunk:  # Skip context-only yields
            output.append(chunk)

    assert output == ["Hello", " there", "!"]


@pytest.mark.asyncio
async def test_rag_review_stream_includes_page_context_in_system_prompt(monkeypatch):
    """Verify that the search context (Source/Page) is actually sent to Claude."""
    fake_index = MagicMock()
    monkeypatch.setattr(main_app, "_index", fake_index)
    monkeypatch.setattr(main_app, "_chunks", _fake_chunks())

    def mock_search(*a, **kw):
        return _fake_chunks()

    monkeypatch.setattr(vector, "search_index", mock_search)

    mock_client = MagicMock()
    # We'll use a wrapper to capture the kwargs passed to messages.stream
    captured_kwargs = {}

    @asynccontextmanager
    async def _capture_stream(**kwargs):
        nonlocal captured_kwargs
        captured_kwargs = kwargs
        async with _stream_yielding(["OK"]) as s:
            yield s

    mock_client.messages.stream = _capture_stream
    monkeypatch.setattr(main_app, "get_anthropic", lambda: mock_client)

    async for chunk, ctx in main_app.rag_review_stream("Question", []):
        pass

    # Check system prompt contents - correctly handling list-of-dicts system param
    system_text = "".join([s.get("text", "") for s in captured_kwargs.get("system", [])])
    assert "[Source: Unknown, Page: 5]" in system_text
    assert "Article 1 says" in system_text


@pytest.mark.asyncio
async def test_rag_review_stream_appends_user_message_last(monkeypatch):
    """Ensure history is respected and the current message is the final user role."""
    fake_index = MagicMock()
    monkeypatch.setattr(main_app, "_index", fake_index)
    monkeypatch.setattr(main_app, "_chunks", _fake_chunks())
    monkeypatch.setattr(vector, "search_index", lambda *a, **kw: _fake_chunks())

    # Mock condense_query to return a simple string
    async def _mock_condense(m, h): return m
    monkeypatch.setattr(main_app, "condense_query", _mock_condense)

    mock_client = MagicMock()
    captured_messages = []

    @asynccontextmanager
    async def _capture_msg(**kwargs):
        nonlocal captured_messages
        captured_messages = kwargs.get("messages", [])
        async with _stream_yielding(["OK"]) as s:
            yield s

    mock_client.messages.stream = _capture_msg
    monkeypatch.setattr(main_app, "get_anthropic", lambda: mock_client)

    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    async for chunk, ctx in main_app.rag_review_stream("New Q", history):
        pass

    assert len(captured_messages) == 3
    assert captured_messages[-1]["role"] == "user"
    assert captured_messages[-1]["content"] == "New Q"


@pytest.mark.asyncio
async def test_rag_review_stream_api_error_yields_error_message(monkeypatch):
    """An Anthropic APIError during streaming should yield an error string, not raise."""
    fake_index = MagicMock()
    monkeypatch.setattr(main_app, "_index", fake_index)
    monkeypatch.setattr(main_app, "_chunks", _fake_chunks())

    def mock_search(*a, **kw):
        return _fake_chunks()

    monkeypatch.setattr(vector, "search_index", mock_search)

    @asynccontextmanager
    async def _raising_stream(**kwargs):
        raise anthropic.APIStatusError(
            message="model: bad-model-name",
            response=MagicMock(status_code=404),
            body={
                "type": "error",
                "error": {
                    "type": "not_found_error",
                    "message": "model: bad-model-name",
                },
            },
        )
        yield  # pragma: no cover — unreachable

    mock_client = MagicMock()
    mock_client.messages.stream = _raising_stream
    monkeypatch.setattr(main_app, "get_anthropic", lambda: mock_client)

    output = []
    async for chunk, ctx in main_app.rag_review_stream("Any question", []):
        output.append(chunk)

    assert any("api error" in str(chunk).lower() for chunk in output)
