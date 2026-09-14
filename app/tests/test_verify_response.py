"""
tests/test_verify_response.py — Unit tests for verification bot

Tests verify the verify_response() function behavior.
"""

import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import openai
import pytest

import main as app

@pytest.fixture(autouse=True)
def enable_verify(monkeypatch):
    monkeypatch.setattr(app, "VERIFY_ENABLED", True)

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

async def test_verify_response_filters_static_form_disputed_lines(monkeypatch):
    """verify_response should filter out false-alarm DISPUTED lines for explicit static form URLs."""
    mock_client = MagicMock()

    disputed_text = (
        "- DISPUTED: To file a grievance, official forms are available at /public/docs/forms/Grievance_-_0_-_Instructions.pdf — "
        "the provided source text makes no mention of official forms or download links."
    )
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content=disputed_text))]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

    monkeypatch.setattr(app, "get_llm_client", lambda: mock_client)

    result = await app.verify_response("Response with form links", "Context without form text")

    assert result == "ALL_CLAIMS_VERIFIED"

async def test_verify_response_preserves_mixed_disputed_contract_claims(monkeypatch):
    """verify_response must preserve genuine contract dispute lines even if form URL disputes are present."""
    mock_client = MagicMock()

    mixed_text = (
        "- DISPUTED: Form download link /public/docs/forms/Grievance_-_A_-_Grievor_Case.pdf is not in contract text\n"
        "- DISPUTED: Employee is entitled to 60 days of bereavement leave — Article 20.1 specifies 3 days."
    )
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content=mixed_text))]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

    monkeypatch.setattr(app, "get_llm_client", lambda: mock_client)

    result = await app.verify_response("Mixed response", "Mixed context")

    assert "60 days of bereavement leave" in result
    assert "/public/docs/forms/" not in result

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

    yielded_contexts = []
    async for chunk, ctx in app.rag_stream("Question", []):
        if ctx:
            yielded_contexts.append(ctx)

    assert len(yielded_contexts) == 1
    assert "Article 1 content" in yielded_contexts[0]
    assert "Page: 5" in yielded_contexts[0]

async def test_unified_chat_create_retries_transient_429(monkeypatch):
    """unified_chat_create should retry on transient 429 errors and succeed on subsequent attempt."""
    monkeypatch.setattr(app, "LLM_RETRY_BASE_DELAY", 0.0001)

    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="Retried success"))]

    calls = 0

    async def _flakey_create(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise openai.RateLimitError(
                message="queue_exceeded",
                response=MagicMock(status_code=429),
                body={"type": "too_many_requests_error", "code": "queue_exceeded"},
            )
        return mock_completion

    mock_client.chat.completions.create = AsyncMock(side_effect=_flakey_create)
    monkeypatch.setattr(app, "get_llm_client", lambda: mock_client)

    result = await app.unified_chat_create("test-model", [{"role": "user", "content": "hi"}])
    assert result == "Retried success"
    assert calls == 2

async def test_unified_chat_create_exhausts_retries_on_persistent_429(monkeypatch):
    """unified_chat_create should raise RateLimitError after exhausting all retries."""
    monkeypatch.setattr(app, "LLM_RETRY_BASE_DELAY", 0.0001)

    mock_client = MagicMock()

    async def _failing_create(*args, **kwargs):
        raise openai.RateLimitError(
            message="queue_exceeded",
            response=MagicMock(status_code=429),
            body={"type": "too_many_requests_error", "code": "queue_exceeded"},
        )

    mock_client.chat.completions.create = AsyncMock(side_effect=_failing_create)
    monkeypatch.setattr(app, "get_llm_client", lambda: mock_client)

    with pytest.raises(openai.RateLimitError):
        await app.unified_chat_create("test-model", [{"role": "user", "content": "hi"}])

    # Initial try + 3 retries = 4 attempts total
    assert mock_client.chat.completions.create.call_count == app.LLM_MAX_RETRIES + 1

async def test_verify_response_transient_429_sanitized_message(monkeypatch):
    """verify_response returns a clean high-traffic notice instead of raw dict on 429."""
    monkeypatch.setattr(app, "LLM_RETRY_BASE_DELAY", 0.0001)

    mock_client = MagicMock()

    async def _failing_create(*args, **kwargs):
        raise openai.RateLimitError(
            message="Error code: 429 - {'message': 'We are experiencing high traffic', 'code': 'queue_exceeded'}",
            response=MagicMock(status_code=429),
            body={"code": "queue_exceeded"},
        )

    mock_client.chat.completions.create = AsyncMock(side_effect=_failing_create)
    monkeypatch.setattr(app, "get_llm_client", lambda: mock_client)

    result = await app.verify_response("Response", "Context")
    assert result == "⚠️ Verification unavailable due to high traffic."
    assert "queue_exceeded" not in result
    assert "Error code: 429" not in result

def test_is_transient_llm_error():
    """Verify various 429 and queue_exceeded representations are recognized as transient."""
    rate_limit_err = openai.RateLimitError(
        message="rate limit",
        response=MagicMock(status_code=429),
        body=None,
    )
    assert app.is_transient_llm_error(rate_limit_err) is True

    status_err_429 = openai.APIStatusError(
        message="Too many requests",
        response=MagicMock(status_code=429),
        body={"type": "too_many_requests_error", "code": "queue_exceeded"},
    )
    assert app.is_transient_llm_error(status_err_429) is True

    runtime_queue_err = RuntimeError("Upstream returned 429 queue_exceeded")
    assert app.is_transient_llm_error(runtime_queue_err) is True

    unrelated_err = ValueError("Invalid argument")
    assert app.is_transient_llm_error(unrelated_err) is False

@pytest.mark.asyncio
async def test_on_message_skips_verification_on_error_message(monkeypatch):
    """on_message must not schedule verification when the assistant response is a high-traffic or error message."""
    monkeypatch.setattr(app, "_ensure_startup", AsyncMock())
    monkeypatch.setattr(app, "_rate_limiter", MagicMock(is_allowed=MagicMock(return_value=(True, ""))))
    monkeypatch.setattr(app, "sanitize_input", lambda s: (s, False))
    monkeypatch.setattr(app, "get_rag_context", AsyncMock(return_value=(["query"], "ctx", [])))

    async def mock_error_stream(*args, **kwargs):
        yield app.HIGH_TRAFFIC_MESSAGE

    monkeypatch.setattr(app, "rag_review_stream", mock_error_stream)
    session_store = {}
    monkeypatch.setattr("chainlit.user_session.set", lambda k, v: session_store.__setitem__(k, v))
    monkeypatch.setattr("chainlit.user_session.get", lambda k, default=None: session_store.get(k, default))
    monkeypatch.setattr(app, "clear_active_status_steps", AsyncMock())
    monkeypatch.setattr(app, "has_chainlit_context", lambda: False)

    mock_msg_instance = MagicMock()
    mock_msg_instance.send = AsyncMock()
    mock_msg_instance.stream_token = AsyncMock()
    mock_msg_instance.update = AsyncMock()
    mock_msg_instance.actions = []
    mock_msg_instance.content = ""
    monkeypatch.setattr("chainlit.Message", MagicMock(return_value=mock_msg_instance))

    mock_verify = AsyncMock(return_value="ALL_CLAIMS_VERIFIED")
    monkeypatch.setattr(app, "verify_response", mock_verify)

    msg = MagicMock()
    msg.content = "What is the policy?"
    msg.elements = []
    await app.on_message(msg)

    # Allow any background tasks to run if they were scheduled
    await asyncio.sleep(0.01)

    mock_verify.assert_not_called()

@pytest.mark.asyncio
async def test_on_message_runs_verification_on_normal_response(monkeypatch):
    """on_message should schedule verification when assistant response is valid substantive text."""
    monkeypatch.setattr(app, "_ensure_startup", AsyncMock())
    monkeypatch.setattr(app, "_rate_limiter", MagicMock(is_allowed=MagicMock(return_value=(True, ""))))
    monkeypatch.setattr(app, "sanitize_input", lambda s: (s, False))
    monkeypatch.setattr(app, "get_rag_context", AsyncMock(return_value=(["query"], "ctx", [])))

    async def mock_valid_stream(*args, **kwargs):
        yield "According to Article 1.2, stewards have rights."

    monkeypatch.setattr(app, "rag_review_stream", mock_valid_stream)
    session_store = {}
    monkeypatch.setattr("chainlit.user_session.set", lambda k, v: session_store.__setitem__(k, v))
    monkeypatch.setattr("chainlit.user_session.get", lambda k, default=None: session_store.get(k, default))
    monkeypatch.setattr(app, "clear_active_status_steps", AsyncMock())
    monkeypatch.setattr(app, "has_chainlit_context", lambda: False)

    mock_msg_instance = MagicMock()
    mock_msg_instance.send = AsyncMock()
    mock_msg_instance.stream_token = AsyncMock()
    mock_msg_instance.update = AsyncMock()
    mock_msg_instance.actions = []
    mock_msg_instance.content = ""
    monkeypatch.setattr("chainlit.Message", MagicMock(return_value=mock_msg_instance))

    mock_verify = AsyncMock(return_value="ALL_CLAIMS_VERIFIED")
    monkeypatch.setattr(app, "verify_response", mock_verify)

    msg = MagicMock()
    msg.content = "What is the policy?"
    msg.elements = []
    await app.on_message(msg)

    # Allow scheduled verification background task to run
    await asyncio.sleep(0.01)

    mock_verify.assert_called_once()
