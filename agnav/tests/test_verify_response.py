"""
tests/test_verify_response.py — Unit tests for verification bot

Tests verify the verify_response() function behavior.
"""

from unittest.mock import MagicMock, patch, AsyncMock
import openai
import pytest

import app

@pytest.fixture
def mock_llm_client():
    """Create a mock OpenAI-compatible client."""
    mock_client = MagicMock()
    
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="ALL_CLAIMS_VERIFIED"))]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
    
    return mock_client

async def test_verify_response_disabled_when_flag_off(monkeypatch):
    """When VERIFY_ENABLED is False, verify_response returns empty string."""
    monkeypatch.setattr(app, "VERIFY_ENABLED", False)

    result = await app.verify_response("Some response", "Some context")
    assert result == ""

async def test_verify_response_calls_llm_client(monkeypatch):
    """verify_response should call LLM client API with the response and context."""
    mock_client = MagicMock()

    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="ALL_CLAIMS_VERIFIED"))]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

    monkeypatch.setattr(app, "get_llm_client", lambda: mock_client)

    result = await app.verify_response("The response", "The context")

    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert "The response" in call_kwargs["messages"][1]["content"]
    assert "The context" in call_kwargs["messages"][1]["content"]

async def test_verify_response_returns_verification_text(
    monkeypatch, mock_llm_client
):
    """verify_response returns the text from the verification model response."""
    monkeypatch.setattr(app, "get_llm_client", lambda: mock_llm_client)

    result = await app.verify_response("Response", "Context")

    assert result == "ALL_CLAIMS_VERIFIED"

async def test_verify_response_handles_api_error(monkeypatch):
    """verify_response should handle API errors gracefully."""
    mock_client = MagicMock()

    async def _raising_create(*args, **kwargs):
        raise openai.APIStatusError(
            message="API error",
            response=MagicMock(status_code=500),
            body={"type": "error"},
        )

    mock_client.chat.completions.create = _raising_create
    monkeypatch.setattr(app, "get_llm_client", lambda: mock_client)

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

    async def _mock_openai_stream(*args, **kwargs):
        if kwargs.get("stream"):
            async def _gen():
                chunk = MagicMock()
                chunk.choices = [MagicMock(delta=MagicMock(content="Hello"))]
                yield chunk
            return _gen()
        return MagicMock()

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=_mock_openai_stream)
    monkeypatch.setattr(app, "get_llm_client", lambda: mock_client)

    # Mock generate_perspective_queries to avoid hitting the API in this test
    monkeypatch.setattr(app, "generate_perspective_queries", AsyncMock(return_value=["Question"]))

    yielded_contexts = []
    async for chunk, ctx in app.rag_stream("Question", []):
        if ctx:
            yielded_contexts.append(ctx)

    assert len(yielded_contexts) == 1
    assert "Article 1 content" in yielded_contexts[0]
    assert "Page: 5" in yielded_contexts[0]
