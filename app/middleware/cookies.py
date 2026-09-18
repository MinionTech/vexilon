import logging
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.routing import APIRoute
from fastapi import HTTPException

from core.config import AGNAV_VERSION
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

def register_routes_and_middleware(app):
    """Register partitioned cookie middleware and custom FastAPI routes on the app."""
    if getattr(app, "middleware_stack", None) is None:
        app.add_middleware(PartitionedCookieMiddleware)

    existing_paths = {getattr(r, "path", None) for r in getattr(app.router, "routes", [])}
    if "/api/brand" not in existing_paths:
        app.router.routes.insert(0, brand_route)
    if "/api/version" not in existing_paths:
        app.router.routes.insert(1, version_route)
    if "/api/health" not in existing_paths:
        app.router.routes.insert(2, health_route)
