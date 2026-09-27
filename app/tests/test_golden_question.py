"""POST /api/golden-question returns a Lookup answer and does not call a live model."""

import json

from fastapi import FastAPI, HTTPException, Request

from core.config import HIGH_TRAFFIC_MESSAGE
from middleware.cookies import post_golden_question, register_routes_and_middleware
import services.llm as llm


def _request(body: bytes, content_type: str = "application/json") -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/golden-question",
        "headers": [(b"content-type", content_type.encode())],
        "client": ("127.0.0.1", 1234),
        "query_string": b"",
    }

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


def _patch_pipeline(monkeypatch, stream):
    async def fake_ensure():
        return None

    async def fake_context(message, history):
        assert history == []
        return ["q"], "context", [{"source": "BCGEU 20th Main Agreement", "text": "clause"}]

    def fake_links(snippets):
        assert snippets[0]["source"] == "BCGEU 20th Main Agreement"
        return ["- [BCGEU 20th Main Agreement](/public/docs/BCGEU_20th_Main_Agreement.pdf)"]

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

    _patch_pipeline(monkeypatch, fake_stream)
    body = json.dumps({"question": "Who bears the burden?"}).encode()
    result = await post_golden_question(_request(body))
    answer = result["answer"]
    assert "[BCGEU 20th Main Agreement - 10.1 Burden of Proof]" in answer
    assert "/public/docs/BCGEU_20th_Main_Agreement.pdf" in answer


async def test_golden_question_model_error_is_503(monkeypatch):
    async def fake_stream(message, history, persona, context=None, queries=None):
        yield HIGH_TRAFFIC_MESSAGE

    _patch_pipeline(monkeypatch, fake_stream)
    body = json.dumps({"question": "Who bears the burden?"}).encode()
    try:
        await post_golden_question(_request(body))
    except HTTPException as exc:
        assert exc.status_code == 503
    else:
        raise AssertionError("model error was returned as an answer")


async def test_golden_question_rejects_empty_and_rate_limit(monkeypatch):
    from core.security import _rate_limiter

    async def fake_stream(message, history, persona, context=None, queries=None):
        raise AssertionError("model was called")
        yield ""

    _patch_pipeline(monkeypatch, fake_stream)
    try:
        await post_golden_question(_request(json.dumps({"question": "  "}).encode()))
    except HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("blank question was accepted")

    try:
        await post_golden_question(_request(b"not-json"))
    except HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("invalid JSON was accepted")

    monkeypatch.setattr(_rate_limiter, "is_allowed", lambda user_id="default": (False, "limited"))
    try:
        await post_golden_question(_request(json.dumps({"question": "Who bears the burden?"}).encode()))
    except HTTPException as exc:
        assert exc.status_code == 429
    else:
        raise AssertionError("rate limit did not stop the question")
