"""
tests/test_rag_stream.py — Unit tests for rag_stream()

All external API calls (LLM, OpenAI/FAISS search) are mocked.
Tests verify guard-clause behaviour and correct prompt construction.
"""

import re
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import openai
import pytest

import main as app

def test_compose_yml_does_not_hardcode_model_name():
    """compose.yml must not hardcode a model name default."""
    compose_path = Path(__file__).parent.parent.parent / "app" / "compose.yml"
    if not compose_path.exists():
        pytest.skip("compose.yml not found (skipping meta-test in containerized environment)")
    compose_text = compose_path.read_text()
    match = re.search(r"DEFAULT_MODEL_LLM", compose_text)
    assert not match

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fake_chunks() -> list[dict]:
    return [
        {"text": "Article 1 says something important.", "page": 5, "chunk_index": 0},
        {"text": "Article 2 says something else.", "page": 6, "chunk_index": 0},
    ]

# ── Guard clauses ─────────────────────────────────────────────────────────────

async def test_rag_stream_no_index_yields_not_ready(monkeypatch):
    """When _index is None, yield the 'not ready' message."""
    monkeypatch.setattr(app, "_index", None)

    output = []
    async for chunk, ctx in app.rag_stream("What are my rights?", []):
        output.append(chunk)
    assert len(output) == 1
    assert "knowledge base not loaded" in output[0].lower()

# ── Happy path ────────────────────────────────────────────────────────────────

async def test_rag_stream_yields_tokens_from_llm(monkeypatch):
    """Happy path: tokens yielded by the LLM stream reach the caller."""
    fake_index = MagicMock()
    monkeypatch.setattr(app, "_index", fake_index)
    monkeypatch.setattr(app, "_chunks", _fake_chunks())

    def mock_search_batch(index, chunks, queries, top_ks):
        return [_fake_chunks() for _ in queries]

    monkeypatch.setattr(app, "search_index_batch", mock_search_batch)

    async def _mock_openai_stream(*args, **kwargs):
        if kwargs.get("stream"):
            async def _gen():
                for t in ["Hello", " there", "!"]:
                    chunk = MagicMock()
                    chunk.choices = [MagicMock(delta=MagicMock(content=t))]
                    yield chunk
            return _gen()
        return MagicMock()

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=_mock_openai_stream)
    monkeypatch.setattr(app, "get_llm_client", lambda: mock_client)

    output = []
    async for chunk, ctx in app.rag_stream("Any question", []):
        if chunk:  # Skip context-only yields
            output.append(chunk)
    assert output == ["Hello", " there", "!"]

async def test_rag_stream_includes_page_context_in_system_prompt(monkeypatch):
    """The system prompt sent to LLM must include agreement excerpts."""
    fake_index = MagicMock()
    monkeypatch.setattr(app, "_index", fake_index)
    monkeypatch.setattr(app, "_chunks", _fake_chunks())

    def mock_search_batch(index, chunks, queries, top_ks):
        return [_fake_chunks() for _ in queries]

    monkeypatch.setattr(app, "search_index_batch", mock_search_batch)

    captured = {}

    async def _capture_stream(**kwargs):
        captured.update(kwargs)
        if kwargs.get("stream"):
            async def _gen():
                chunk = MagicMock()
                chunk.choices = [MagicMock(delta=MagicMock(content="ok"))]
                yield chunk
            return _gen()
        return MagicMock()

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=_capture_stream)
    monkeypatch.setattr(app, "get_llm_client", lambda: mock_client)

    async for chunk, ctx in app.rag_stream("What about overtime?", []):
        pass

    messages = captured.get("messages", [])
    system_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
    
    assert "<<< SOURCE: Unknown | Page: 5 >>>" in system_prompt
    assert "<<< SOURCE: Unknown | Page: 6 >>>" in system_prompt
    assert "Article 1 says something important." in system_prompt

async def test_rag_stream_api_error_yields_error_message(monkeypatch):
    """An API error during streaming should yield an error string, not raise."""
    fake_index = MagicMock()
    monkeypatch.setattr(app, "_index", fake_index)
    monkeypatch.setattr(app, "_chunks", _fake_chunks())

    def mock_search_batch(index, chunks, queries, top_ks):
        return [_fake_chunks() for _ in queries]

    monkeypatch.setattr(app, "search_index_batch", mock_search_batch)

    async def _raising_stream(**kwargs):
        raise openai.APIStatusError(
            message="model: bad-model-name",
            response=MagicMock(status_code=404),
            body={"type": "error"},
        )

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=_raising_stream)
    monkeypatch.setattr(app, "get_llm_client", lambda: mock_client)

    output = []
    async for chunk, ctx in app.rag_stream("Any question", []):
        output.append(chunk)
    assert len(output) == 1
    assert "⚠️" in output[0]

async def test_rag_review_stream_rate_limit_error(monkeypatch):
    """When an upstream RateLimitError occurs, yield user-friendly 429 message."""
    async def _mock_context(*args, **kwargs):
        return ["query"], "context", []

    monkeypatch.setattr(app, "get_rag_context", _mock_context)

    async def _raising_stream(**kwargs):
        raise openai.RateLimitError(
            message="queue_exceeded",
            response=MagicMock(status_code=429),
            body={"type": "too_many_requests_error"},
        )
        yield  # Make it an async generator

    monkeypatch.setattr(app, "unified_chat_stream", _raising_stream)

    output = []
    async for chunk in app.rag_review_stream("Any question", []):
        output.append(chunk)
    assert len(output) == 1
    assert "⏳" in output[0]
    assert "experiencing high traffic" in output[0]

async def test_rag_review_stream_generic_error_sanitized(monkeypatch):
    """Generic errors in rag_review_stream must yield sanitized error messages."""
    async def _mock_context(*args, **kwargs):
        return ["query"], "context", []

    monkeypatch.setattr(app, "get_rag_context", _mock_context)

    async def _raising_stream(**kwargs):
        raise RuntimeError("Internal secret details")
        yield  # Make it an async generator

    monkeypatch.setattr(app, "unified_chat_stream", _raising_stream)

    output = []
    async for chunk in app.rag_review_stream("Any question", []):
        output.append(chunk)
    assert len(output) == 1
    assert "⚠️" in output[0]
    assert "Internal secret details" not in output[0]

async def test_rag_review_stream_queue_exceeded_fallback(monkeypatch):
    """RuntimeErrors containing 'queue_exceeded' must yield high-traffic message, distinct from generic error sanitization."""
    async def _mock_context(*args, **kwargs):
        return ["query"], "context", []

    monkeypatch.setattr(app, "get_rag_context", _mock_context)

    async def _raising_stream(**kwargs):
        raise RuntimeError("queue_exceeded")
        yield  # Make it an async generator

    monkeypatch.setattr(app, "unified_chat_stream", _raising_stream)

    output = []
    async for chunk in app.rag_review_stream("Any question", []):
        output.append(chunk)
    assert len(output) == 1
    assert "⏳" in output[0]
    assert "experiencing high traffic" in output[0]

async def test_rag_review_stream_get_rag_context_failure(monkeypatch):
    """Pre-stream failures in get_rag_context should be caught and formatted cleanly."""
    async def _failing_context(*args, **kwargs):
        raise RuntimeError("rate_limit exceeded during retrieval")

    monkeypatch.setattr(app, "get_rag_context", _failing_context)

    output = []
    async for chunk in app.rag_review_stream("Any question", []):
        output.append(chunk)
    assert len(output) == 1
    assert "⏳" in output[0]
    assert "experiencing high traffic" in output[0]

async def test_unified_chat_stream_retries_transient_429(monkeypatch):
    """unified_chat_stream should retry stream connection on transient 429 and stream successfully."""
    monkeypatch.setattr(app, "LLM_RETRY_BASE_DELAY", 0.0001)

    mock_client = MagicMock()
    calls = 0

    async def _mock_stream():
        chunk = MagicMock()
        chunk.choices = [MagicMock(delta=MagicMock(content="Chunk after retry"))]
        yield chunk

    async def _flakey_create(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise openai.RateLimitError(
                message="queue_exceeded",
                response=MagicMock(status_code=429),
                body={"code": "queue_exceeded"},
            )
        return _mock_stream()

    mock_client.chat.completions.create = AsyncMock(side_effect=_flakey_create)
    monkeypatch.setattr(app, "get_llm_client", lambda: mock_client)

    chunks = []
    async for chunk in app.unified_chat_stream("test-model", [{"role": "user", "content": "hi"}]):
        chunks.append(chunk)

    assert chunks == ["Chunk after retry"]
    assert calls == 2

def test_compute_retry_delay():
    """Verify compute_retry_delay honors Retry-After header and bounds."""
    # Test Retry-After header with seconds
    mock_resp = MagicMock()
    mock_resp.headers = {"retry-after": "3.5"}
    mock_exc = MagicMock(response=mock_resp)

    delay = app.compute_retry_delay(attempt=0, exc=mock_exc)
    assert delay == 3.5

    # Test Retry-After header with HTTP-date in the near future
    import datetime
    future_date = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=4)
    http_date_str = future_date.strftime("%a, %d %b %Y %H:%M:%S GMT")
    mock_resp_date = MagicMock()
    mock_resp_date.headers = {"retry-after": http_date_str}
    mock_exc_date = MagicMock(response=mock_resp_date)

    delay_date = app.compute_retry_delay(attempt=0, exc=mock_exc_date)
    assert 0 < delay_date <= app.LLM_RETRY_MAX_DELAY

    # Test default exponential delay without Retry-After
    d0 = app.compute_retry_delay(attempt=0, exc=None)
    assert 0.0 < d0 <= app.LLM_RETRY_MAX_DELAY
