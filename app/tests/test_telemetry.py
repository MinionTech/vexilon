"""
tests/test_telemetry.py — Unit tests for privacy-first operational telemetry logger (#340).
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from telemetry import (
    FORBIDDEN_KEYS,
    assert_safe_payload,
    build_telemetry_payload,
    log_telemetry,
)


def _recursively_check_keys(obj: object) -> None:
    """Ensure no forbidden keys exist at any nesting level."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert str(k).lower() not in FORBIDDEN_KEYS, f"Forbidden key '{k}' found in payload!"
            _recursively_check_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            _recursively_check_keys(item)


# ── Strict No-Op When TELEMETRY_URL is Unset ───────────────────────────────────

@pytest.mark.asyncio
async def test_telemetry_unset_url_is_noop(monkeypatch):
    """Verify zero network calls are attempted when TELEMETRY_URL is unset or empty."""
    monkeypatch.delenv("TELEMETRY_URL", raising=False)

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        task = log_telemetry(latency_ms=120.5)
        assert task is None
        mock_post.assert_not_called()

    # Also test empty string
    monkeypatch.setenv("TELEMETRY_URL", "   ")
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        task = log_telemetry(latency_ms=120.5)
        assert task is None
        mock_post.assert_not_called()


# ── Payload Schema & Strict Privacy Hygiene ───────────────────────────────────

@pytest.mark.asyncio
async def test_telemetry_payload_schema_and_privacy(monkeypatch):
    """
    Verify HTTP POST payload contains only operational metadata.
    Forbidden keys (content, message, prompt, response, query) must not exist.
    """
    telemetry_url = "https://telemetry.internal/v1/metrics"
    monkeypatch.setenv("TELEMETRY_URL", telemetry_url)

    mock_resp = httpx.Response(status_code=200, request=httpx.Request("POST", telemetry_url))

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp) as mock_post:
        raw_tokens = {
            "prompt_tokens": 128,
            "completion_tokens": 64,
            "total_tokens": 192,
            "unwanted_extra": "should_be_stripped",
        }
        raw_retrieval = {
            "chunk_count": 5,
            "top_score": 0.8842,
            "mean_score": 0.7231,
            "unwanted_query": "should_be_stripped",
        }

        task = log_telemetry(
            latency_ms=350.45,
            tokens=raw_tokens,
            retrieval_metadata=raw_retrieval,
            model="qwen3:8b",
            provider="ollama",
        )
        assert task is not None
        await task

        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == telemetry_url

        payload = kwargs["json"]

        # Check timestamp format
        assert "timestamp" in payload
        datetime.fromisoformat(payload["timestamp"])

        # Check latency
        assert payload["latency_ms"] == 350.45

        # Check tokens payload: only operational counts present
        assert payload["tokens"] == {
            "prompt_tokens": 128,
            "completion_tokens": 64,
            "total_tokens": 192,
        }

        # Check retrieval metadata: only operational numbers present
        assert payload["retrieval_metadata"] == {
            "chunk_count": 5,
            "top_score": 0.8842,
            "mean_score": 0.7231,
        }

        # Check model and provider identifiers
        assert payload["model"] == "qwen3:8b"
        assert payload["provider"] == "ollama"

        # Assert strict absence of forbidden keys throughout the entire payload structure
        _recursively_check_keys(payload)


@pytest.mark.asyncio
async def test_telemetry_payload_handles_none_values():
    """Verify build_telemetry_payload cleanly handles None tokens and retrieval metadata."""
    payload = build_telemetry_payload(
        latency_ms=99.9,
        tokens=None,
        retrieval_metadata=None,
        model=None,
        provider=None,
    )
    assert payload["latency_ms"] == 99.9
    assert payload["tokens"] is None
    assert payload["retrieval_metadata"] is None
    assert payload["model"] is None
    assert payload["provider"] is None
    _recursively_check_keys(payload)


# ── Non-Blocking Fault Tolerance ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_telemetry_handles_http_500(monkeypatch):
    """Verify non-blocking behavior when endpoint returns HTTP 500."""
    telemetry_url = "https://telemetry.internal/v1/metrics"
    monkeypatch.setenv("TELEMETRY_URL", telemetry_url)

    mock_resp = httpx.Response(status_code=500, request=httpx.Request("POST", telemetry_url))

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        task = log_telemetry(latency_ms=100.0)
        assert task is not None
        # Must not raise any exception
        await task


@pytest.mark.asyncio
async def test_telemetry_handles_timeout(monkeypatch):
    """Verify non-blocking behavior when endpoint times out."""
    telemetry_url = "https://telemetry.internal/v1/metrics"
    monkeypatch.setenv("TELEMETRY_URL", telemetry_url)

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=httpx.TimeoutException("Timeout")):
        task = log_telemetry(latency_ms=100.0)
        assert task is not None
        # Must not raise any exception
        await task


@pytest.mark.asyncio
async def test_telemetry_handles_connection_error(monkeypatch):
    """Verify non-blocking behavior when network connection fails."""
    telemetry_url = "https://telemetry.internal/v1/metrics"
    monkeypatch.setenv("TELEMETRY_URL", telemetry_url)

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=httpx.ConnectError("Connection refused")):
        task = log_telemetry(latency_ms=100.0)
        assert task is not None
        # Must not raise any exception
        await task


# ── Defensive Payload Key Validator ───────────────────────────────────────────

def test_assert_safe_payload_raises_on_forbidden_keys():
    """Verify assert_safe_payload raises ValueError if any forbidden key is present."""
    for key in FORBIDDEN_KEYS:
        with pytest.raises(ValueError, match=f"Forbidden key '{key}' detected"):
            assert_safe_payload({key: "some_value"})

        with pytest.raises(ValueError, match=f"Forbidden key '{key}' detected"):
            assert_safe_payload({"nested": {key: "nested_value"}})


# ── Integration with app/main.py ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_main_on_message_telemetry_integration(monkeypatch):
    """Verify that completing on_message in main.py triggers log_telemetry with metadata."""
    import main as app

    mock_telemetry = patch("main.log_telemetry").start()
    try:
        monkeypatch.setattr(app, "_ensure_startup", AsyncMock())
        monkeypatch.setattr(app, "_rate_limiter", MagicMock(is_allowed=MagicMock(return_value=(True, ""))))
        monkeypatch.setattr(app, "sanitize_input", lambda s: (s, False))
        monkeypatch.setattr(app, "get_rag_context", AsyncMock(return_value=(["query"], "ctx", [{"text": "t", "score": 0.85}])))

        async def mock_stream(*args, **kwargs):
            yield "Hello"

        monkeypatch.setattr(app, "rag_review_stream", mock_stream)
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
        monkeypatch.setattr("chainlit.Message", MagicMock(return_value=mock_msg_instance))

        msg = MagicMock()
        msg.content = "What is the policy?"
        msg.elements = []
        await app.on_message(msg)

        mock_telemetry.assert_called_once()
        _, kwargs = mock_telemetry.call_args
        assert kwargs["latency_ms"] >= 0
        assert kwargs["retrieval_metadata"] == {
            "chunk_count": 1,
            "top_score": 0.85,
            "mean_score": 0.85,
        }
    finally:
        patch.stopall()

