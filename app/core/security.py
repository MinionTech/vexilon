import re
import time
import uuid
import logging
from threading import Lock

from core.config import (
    MAX_INPUT_LENGTH,
    LOG_SUSPICIOUS_INPUTS,
    RATE_LIMIT_PER_MINUTE,
    RATE_LIMIT_PER_HOUR,
    _get_active_main,
)

logger = logging.getLogger(__name__)

PROMPT_INJECTION_PATTERNS = [
    re.compile(r, re.IGNORECASE)
    for r in [
        r"ignore\s+.*instructions",
        r"forget\s+.*instructions",
        r"disregard\s+.*rules",
        r"you\s+are\s+now\s+.+\s+instead",
        r"new\s+(system\s+|)prompt:",
        r"#\#\#\s*(system\s+|)instructions",
        r"\[\[SYSTEM\]\]",
        r"override\s+.*instructions",
        r"disable\s+.*safety",
        r"\bjailbreak\b",
        r"developer\s+mode",
        r"sudo\s+mode",
        r"roleplay\s+as",
        r"pretend\s+(you\s+are|to\s+be)",
        r"forget\s+everything\s+above",
        r"discard\s+.*instructions",
    ]
]

def sanitize_input(user_input: str) -> tuple[str, bool]:
    """Check for prompt injection patterns and length limits."""
    if not user_input:
        return user_input, False

    main_mod = _get_active_main()
    effective_log_suspicious = getattr(main_mod, "LOG_SUSPICIOUS_INPUTS", LOG_SUSPICIOUS_INPUTS)
    effective_max_length = getattr(main_mod, "MAX_INPUT_LENGTH", MAX_INPUT_LENGTH)

    injection_found = False
    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(user_input):
            injection_found = True
            if effective_log_suspicious:
                logger.warning(f"[security] Prompt injection detected: {pattern.pattern[:100]}...")
            break

    too_long = len(user_input) > effective_max_length
    if too_long and effective_log_suspicious:
        logger.warning(f"[security] Input too long: {len(user_input)}")

    return user_input[:effective_max_length], injection_found or too_long


class RateLimiter:
    def __init__(self, max_per_minute: int = 10, max_per_hour: int = 100):
        self.minute_limit = max_per_minute
        self.hour_limit = max_per_hour
        self.requests: dict[str, list[float]] = {}
        self._lock = Lock()

    def _clean_old_requests(self, key: str) -> None:
        now = time.time()
        hour_ago = now - 3600
        if key in self.requests:
            self.requests[key] = [t for t in self.requests[key] if t > hour_ago]
            if not self.requests[key]:
                del self.requests[key]

    def is_allowed(self, user_id: str = "default") -> tuple[bool, str]:
        with self._lock:
            self._clean_old_requests(user_id)
            now = time.time()
            minute_ago = now - 60
            requests = self.requests.get(user_id, [])
            recent = [t for t in requests if t > minute_ago]
            if len(recent) >= self.minute_limit:
                return False, f"Rate limit exceeded: {self.minute_limit} per minute."
            if len(requests) >= self.hour_limit:
                return False, f"Rate limit exceeded: {self.hour_limit} per hour."
            self.requests.setdefault(user_id, []).append(now)
            return True, ""


_rate_limiter = RateLimiter(RATE_LIMIT_PER_MINUTE, RATE_LIMIT_PER_HOUR)


def _parse_client_uuid(candidate) -> str | None:
    """Validate a client-supplied UUID string, or None if invalid.

    Only accepts well-formed UUID v4 strings so an untrusted window_message
    payload can't inject arbitrary/oversized content into session state or
    logs (same defensive posture as sanitize_input()).
    """
    if not isinstance(candidate, str):
        return None
    try:
        parsed = uuid.UUID(candidate)
    except (ValueError, AttributeError, TypeError):
        return None
    if parsed.version != 4:
        return None
    return str(parsed)


def _client_id() -> str:
    """Pseudonymous client identifier for rate limiting and log correlation.

    Prefers the persistent, client-generated UUID captured by
    on_window_message (survives page reloads); falls back to Chainlit's
    ephemeral session id for the brief window before the client posts it.
    """
    try:
        import chainlit as cl
        persistent = cl.user_session.get("client_uuid")
        if persistent:
            return persistent
        sid = getattr(cl.user_session, "id", None) or cl.user_session.get("id")
        return str(sid) if sid else "default"
    except Exception:
        return "default"
