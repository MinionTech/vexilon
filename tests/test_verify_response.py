"""
tests/test_verify_response.py — Unit tests for verification bot

Tests verify the verify_response() function behavior.
"""

from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import anthropic
import pytest

import app


def _fake_verify_response(
    model: str = "claude-haiku-4-7-20260416",
    max_tokens: int = 512,
    system: list = None,
    messages: list = None,
):
    """Create a fake async messages.create that returns a mock response."""
    fake_message = MagicMock()
    fake_message.text = "ALL_CLAIMS_VERIFIED"

    fake_response = MagicMock()
    fake_response.content = [fake_message]

    async def _create(*args, **kwargs):
        return fake_response

    return _create


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client."""
    mock_client = MagicMock()
    mock_client.messages.create = _fake_verify_response()
    return mock_client


async def test_verify_response_disabled_when_flag_off(monkeypatch):
    """When VERIFY_ENABLED is False, verify_response returns empty string."""
    monkeypatch.setattr(app, "VERIFY_ENABLED", False)

    result = await app.verify_response("Some response", "Some context")
    assert result == ""


async def test_verify_response_calls_anthropic(monkeypatch):
    """verify_response should call Anthropic API with the response and context."""
    mock_client = MagicMock()

    fake_message = MagicMock()
    fake_message.text = "ALL_CLAIMS_VERIFIED"
    fake_response = MagicMock()
    fake_response.content = [fake_message]
    mock_client.messages.create = MagicMock(return_value=fake_response)

    monkeypatch.setattr(app, "get_anthropic", lambda: mock_client)

    result = await app.verify_response("The response", "The context")

    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert "The response" in call_kwargs["messages"][0]["content"]
    assert "The context" in call_kwargs["messages"][0]["content"]


async def test_verify_response_returns_verification_text(
    monkeypatch, mock_anthropic_client
):
    """verify_response returns the text from the verification model response."""
    monkeypatch.setattr(app, "get_anthropic", lambda: mock_anthropic_client)

    result = await app.verify_response("Response", "Context")

    assert result == "ALL_CLAIMS_VERIFIED"


async def test_verify_response_handles_api_error(monkeypatch):
    """verify_response should handle API errors gracefully."""
    mock_client = MagicMock()

    async def _raising_create(*args, **kwargs):
        raise anthropic.APIStatusError(
            message="API error",
            response=MagicMock(status_code=500),
            body={"type": "error"},
        )

    mock_client.messages.create = _raising_create
    monkeypatch.setattr(app, "get_anthropic", lambda: mock_client)

    result = await app.verify_response("Response", "Context")

    assert "Verification unavailable" in result


async def test_rag_stream_yields_context(monkeypatch):
    """rag_stream should yield context alongside text chunks."""
    fake_chunks = [
        {"text": "Article 1 content.", "page": 5, "chunk_index": 0},
    ]

    fake_index = MagicMock()
    monkeypatch.setattr(app, "_index", fake_index)
    monkeypatch.setattr(app, "_chunks", fake_chunks)

    def mock_search_batch(*a, **kw):
        return [fake_chunks]

    monkeypatch.setattr(app, "search_index_batch", mock_search_batch)

    @asynccontextmanager
    async def _stream_ctx(*args, **kwargs):
        mock_stream = MagicMock()

        async def _async_gen():
            yield "Hello"

        mock_stream.text_stream = _async_gen()

        fake_usage = MagicMock(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        fake_message = MagicMock()
        fake_message.usage = fake_usage

        async def _get_final():
            return fake_message

        mock_stream.get_final_message = _get_final
        yield mock_stream

    mock_client = MagicMock()
    mock_client.messages.stream = _stream_ctx
    monkeypatch.setattr(app, "get_anthropic", lambda: mock_client)

    # Mock generate_perspective_queries to avoid hitting the API in this test
    from unittest.mock import AsyncMock
    monkeypatch.setattr(app, "generate_perspective_queries", AsyncMock(return_value=["Question"]))

    yielded_contexts = []
    async for chunk, ctx in app.rag_stream("Question", []):
        if ctx:
            yielded_contexts.append(ctx)

    assert len(yielded_contexts) == 1
    assert "Article 1 content" in yielded_contexts[0]
    assert "Page: 5" in yielded_contexts[0]
