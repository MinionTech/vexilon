from starlette.middleware.base import BaseHTTPMiddleware

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
