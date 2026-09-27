"""POST /api/golden-question returns a Lookup answer and does not call a live model."""

import json
import logging
import secrets

from fastapi import FastAPI, HTTPException, Request

from core.config import GENERIC_ERROR_MESSAGE, HIGH_TRAFFIC_MESSAGE
from middleware.cookies import post_golden_question, register_routes_and_middleware
import services.llm as llm

TOKEN = secrets.token_hex(16)


def _request(
    body: bytes,
    content_type: str = "application/json",
    *,
    token: str | None = TOKEN,
    forwarded_for: str | None = None,
) -> tuple[Request, dict[str, bool]]:
    headers = [(b"content-type", content_type.encode())]
    if token is not None:
        headers.append((b"x-golden-question-token", token.encode()))
    if forwarded_for is not None:
        headers.append((b"x-forwarded-for", forwarded_for.encode()))
    seen = {"read": False}
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/golden-question",
        "headers": headers,
        "client": ("127.0.0.1", 1234),
        "query_string": b"",
    }

    async def receive():
        seen["read"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive), seen


def _allow(monkeypatch, token: str = TOKEN) -> None:
    monkeypatch.setenv("GOLDEN_QUESTION_TOKEN", token)


def _patch_pipeline(monkeypatch, stream):
    async def fake_ensure():
        return None

    async def fake_context(message, history):
        assert history == []
        return ["q"], "context", [{"source": "BCGEU 20th Main Agreement", "text": "clause"}]

    def fake_links(*args, **kwargs):
        raise AssertionError("reference links were added to the scored answer")

    def live_client(*args, **kwargs):
        raise AssertionError("live model client was called")

    monkeypatch.setattr(llm, "_ensure_startup", fake_ensure)
    monkeypatch.setattr(llm, "get_rag_context", fake_context)
    monkeypatch.setattr(llm, "rag_review_stream", stream)
    monkeypatch.setattr(llm, "build_reference_links", fake_links)
    monkeypatch.setattr(llm, "get_llm_client", live_client)


def test_golden_route_registered_once():
    app = FastAPI()
    register_routes_and_middleware(app)
    register_routes_and_middleware(app)
    paths = [getattr(route, "path", None) for route in app.routes]
    assert paths.count("/api/golden-question") == 1
    golden = next(route for route in app.routes if getattr(route, "path", None) == "/api/golden-question")
    assert golden.methods == {"POST"}


async def test_golden_question_returns_lookup_answer(monkeypatch):
    async def fake_stream(message, history, persona, context=None, queries=None):
        assert message == "Who bears the burden?"
        assert history == []
        assert persona == "Lookup"
        assert context == "context"
        yield "[BCGEU 20th Main Agreement - 10.1 Burden of Proof]"

    _allow(monkeypatch)
    _patch_pipeline(monkeypatch, fake_stream)
    body = json.dumps({"question": "Who bears the burden?"}).encode()
    request, seen = _request(body)
    result = await post_golden_question(request)
    assert seen["read"] is True
    assert result["answer"] == "[BCGEU 20th Main Agreement - 10.1 Burden of Proof]"


async def test_golden_question_model_error_is_503(monkeypatch):
    markers = (
        HIGH_TRAFFIC_MESSAGE,
        GENERIC_ERROR_MESSAGE,
        "⚠️ API error: boom",
    )
    _allow(monkeypatch)
    body = json.dumps({"question": "Who bears the burden?"}).encode()
    for marker in markers:
        async def fake_stream(message, history, persona, context=None, queries=None, marker=marker):
            yield "[BCGEU 20th Main Agreement - 10.1 Burden of Proof] "
            yield marker

        _patch_pipeline(monkeypatch, fake_stream)
        request, _seen = _request(body)
        try:
            await post_golden_question(request)
        except HTTPException as exc:
            assert exc.status_code == 503
            assert exc.detail == "answer failed"
        else:
            raise AssertionError(f"marker after answer text was returned: {marker}")


async def test_golden_question_rejects_empty_and_rate_limit(monkeypatch):
    from core.security import _rate_limiter

    async def fake_stream(message, history, persona, context=None, queries=None):
        raise AssertionError("model was called")
        yield ""

    _allow(monkeypatch)
    _patch_pipeline(monkeypatch, fake_stream)
    blank, blank_seen = _request(json.dumps({"question": "  "}).encode())
    try:
        await post_golden_question(blank)
    except HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("blank question was accepted")
    assert blank_seen["read"] is True

    invalid, _invalid_seen = _request(b"not-json")
    try:
        await post_golden_question(invalid)
    except HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("invalid JSON was accepted")

    captured: dict[str, str] = {}

    def deny(user_id="default"):
        captured["user_id"] = user_id
        return False, "limited"

    monkeypatch.setattr(_rate_limiter, "is_allowed", deny)
    limited, limited_seen = _request(
        json.dumps({"question": "Who bears the burden?"}).encode(),
        forwarded_for="203.0.113.9, 10.0.0.1",
    )
    try:
        await post_golden_question(limited)
    except HTTPException as exc:
        assert exc.status_code == 429
    else:
        raise AssertionError("rate limit did not stop the question")
    assert limited_seen["read"] is False
    assert captured["user_id"] == "golden:127.0.0.1"
    assert "203.0.113.9" not in captured["user_id"]
    assert "x-forwarded-for" not in captured["user_id"].lower()


async def test_golden_question_rejects_anonymous_before_the_body(monkeypatch, caplog):
    _allow(monkeypatch)

    async def fake_stream(message, history, persona, context=None, queries=None):
        raise AssertionError("model was called")
        yield ""

    _patch_pipeline(monkeypatch, fake_stream)
    body = json.dumps({"question": "Who bears the burden?"}).encode()
    missing, missing_seen = _request(body, token=None)
    try:
        await post_golden_question(missing)
    except HTTPException as exc:
        assert exc.status_code == 401
        assert exc.detail == "unauthorized"
        assert TOKEN not in exc.detail
    else:
        raise AssertionError("missing token was accepted")
    assert missing_seen["read"] is False

    wrong, wrong_seen = _request(body, token=secrets.token_hex(8))
    try:
        await post_golden_question(wrong)
    except HTTPException as exc:
        assert exc.status_code == 401
    else:
        raise AssertionError("wrong token was accepted")
    assert wrong_seen["read"] is False

    for value in (None, " ", "\n", "\r"):
        if value is None:
            monkeypatch.delenv("GOLDEN_QUESTION_TOKEN", raising=False)
        else:
            monkeypatch.setenv("GOLDEN_QUESTION_TOKEN", value)
        unconfigured, unconfigured_seen = _request(body)
        caplog.clear()
        with caplog.at_level(logging.ERROR):
            try:
                await post_golden_question(unconfigured)
            except HTTPException as exc:
                assert exc.status_code == 503
                assert exc.detail == "unavailable"
                assert "GOLDEN_QUESTION_TOKEN" not in exc.detail
            else:
                raise AssertionError("rejected token env was accepted")
        assert unconfigured_seen["read"] is False
        assert "GOLDEN_QUESTION_TOKEN is unset or invalid" in caplog.text
        assert TOKEN not in caplog.text


async def test_golden_question_logs_the_exception(monkeypatch, caplog):
    _allow(monkeypatch)

    async def fake_stream(message, history, persona, context=None, queries=None):
        raise RuntimeError("retrieval exploded")
        yield ""

    _patch_pipeline(monkeypatch, fake_stream)
    request, _seen = _request(json.dumps({"question": "Who bears the burden?"}).encode())
    with caplog.at_level(logging.ERROR):
        try:
            await post_golden_question(request)
        except HTTPException as exc:
            assert exc.status_code == 503
            assert exc.detail == "answer failed"
            assert "retrieval exploded" not in exc.detail
        else:
            raise AssertionError("model exception was returned")
    assert "retrieval exploded" in caplog.text
    assert "RuntimeError" in caplog.text
