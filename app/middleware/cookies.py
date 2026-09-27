import logging
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.routing import APIRoute
from fastapi import HTTPException, Request

from core.config import AGNAV_VERSION, GENERIC_ERROR_MESSAGE, HIGH_TRAFFIC_MESSAGE
from brand import get_brand as _get_brand

logger = logging.getLogger(__name__)

class PartitionedCookieMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        cookie_headers = response.headers.getlist("set-cookie")
        if cookie_headers:
            del response.headers["set-cookie"]
            for header in cookie_headers:
                # Append Partitioned for SameSite=None cookies to comply with CHIPS
                if "samesite=none" in header.lower() and "partitioned" not in header.lower():
                    header += "; Partitioned"
                response.headers.append("set-cookie", header)
        return response

def get_version():
    return {
        "version": AGNAV_VERSION
    }

def get_brand_config():
    return _get_brand()

async def get_health():
    from services.llm import get_llm_client
    try:
        client = get_llm_client()
        # Fast lightweight check to ensure credentials and network are valid
        await client.models.list(timeout=10.0)
        return {"status": "ok", "llm": "connected"}
    except Exception as e:
        logger.error(f"[health] LLM connection failed: {e}")
        raise HTTPException(status_code=503, detail="LLM connection failed")

brand_route = APIRoute(
    "/api/brand",
    endpoint=get_brand_config,
    methods=["GET"],
    include_in_schema=False
)

version_route = APIRoute(
    "/api/version",
    endpoint=get_version,
    methods=["GET"],
    include_in_schema=False
)

health_route = APIRoute(
    "/api/health",
    endpoint=get_health,
    methods=["GET"],
    include_in_schema=False
)

def _answer_failed(text: str) -> bool:
    if not text or not text.strip():
        return True
    # Same markers as trigger_verification_task: a sentinel after real tokens still fails.
    return any(
        marker in text
        for marker in (HIGH_TRAFFIC_MESSAGE, GENERIC_ERROR_MESSAGE, "⚠️ API error:")
    )

async def post_golden_question(request: Request):
    """Answer one Lookup question. The promotion gate checks the citation."""
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="question is required") from None
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="question is required")
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise HTTPException(status_code=400, detail="question is required")

    from core.security import _rate_limiter, sanitize_input
    from services.llm import (
        _ensure_startup,
        get_rag_context,
        rag_review_stream,
    )

    client_host = request.client.host if request.client else "golden-question"
    allowed, rate_msg = _rate_limiter.is_allowed(f"golden:{client_host}")
    if not allowed:
        raise HTTPException(status_code=429, detail=rate_msg or "Rate limit exceeded.")

    sanitized, flagged = sanitize_input(question.strip())
    if flagged or not sanitized.strip():
        raise HTTPException(status_code=400, detail="invalid input")

    try:
        await _ensure_startup()
        queries, context, _snippets = await get_rag_context(sanitized, [])
        accumulated = ""
        async for chunk in rag_review_stream(
            sanitized,
            [],
            "Lookup",
            context=context,
            queries=queries,
        ):
            if chunk:
                accumulated += chunk
        if _answer_failed(accumulated):
            raise HTTPException(status_code=503, detail="answer failed")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[golden] answer failed: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="answer failed") from None

    return {"answer": accumulated}

golden_route = APIRoute(
    "/api/golden-question",
    endpoint=post_golden_question,
    methods=["POST"],
    include_in_schema=False,
)

def register_routes_and_middleware(app):
    """Register partitioned cookie middleware and custom FastAPI routes on the app."""
    if not any(
        getattr(m, "cls", None) is PartitionedCookieMiddleware
        for m in getattr(app, "user_middleware", [])
    ):
        app.add_middleware(PartitionedCookieMiddleware)

    existing_paths = {getattr(r, "path", None) for r in getattr(app.router, "routes", [])}
    if "/api/brand" not in existing_paths:
        app.router.routes.insert(0, brand_route)
    if "/api/version" not in existing_paths:
        app.router.routes.insert(1, version_route)
    if "/api/health" not in existing_paths:
        app.router.routes.insert(2, health_route)
    if "/api/golden-question" not in existing_paths:
        app.router.routes.insert(3, golden_route)
