"""Plausible analytics (#796): anonymous usage counts, nothing stored in the app.

Events carry only an event name and enumerated property values. Content, file
names, and user input are never sent. When ``PLAUSIBLE_DOMAIN`` is unset,
nothing is sent. Sending runs on a small dedicated thread pool with a short timeout, excess
events are dropped, and every error is swallowed, so analytics can never break or slow a user action.
"""
import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import chainlit as cl
import requests

logger = logging.getLogger(__name__)

PLAUSIBLE_DOMAIN = os.getenv("PLAUSIBLE_DOMAIN", "").strip()
PLAUSIBLE_API_HOST = "https://plausible.io"
TIMEOUT_SECONDS = 2
# Own pool, so a slow Plausible never holds the default executor that chat retrieval uses.
MAX_PENDING = 32
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="analytics")

# The only events and property values that may ever be sent.
ALLOWED_EVENTS: dict[str, dict[str, frozenset[str]]] = {
    "Import": {"result": frozenset({"success", "failure"})},
    "Export": {"result": frozenset({"success", "failure"})},
    "Persona Selected": {"persona": frozenset({"Lookup", "Grieve", "Manage"})},
}

_background_tasks: set[asyncio.Future] = set()


def build_payload(name: str, props: dict[str, str], url: str) -> dict:
    """Build an Events API body; raise ValueError for anything not allow-listed."""
    allowed = ALLOWED_EVENTS.get(name)
    if allowed is None:
        raise ValueError(f"Unknown analytics event: {name!r}")
    if set(props) != set(allowed):
        raise ValueError(f"Invalid properties for {name!r}: {sorted(props)}")
    for key, value in props.items():
        if value not in allowed[key]:
            raise ValueError(f"Invalid value for {name!r}.{key}")
    return {"name": name, "domain": PLAUSIBLE_DOMAIN, "url": url, "props": dict(props)}


def _visitor_headers() -> tuple[dict[str, str], str]:
    """Forward the visitor's User-Agent and IP so Plausible can hash them.

    These are passed straight through to Plausible and are never logged or stored.
    """
    environ = cl.context.session.environ or {}
    # The last X-Forwarded-For entry is the one the proxy in front of the app
    # appended. Earlier entries come from the client and can be forged.
    forwarded = [part.strip() for part in environ.get("HTTP_X_FORWARDED_FOR", "").split(",") if part.strip()]
    ip = forwarded[-1] if forwarded else environ.get("REMOTE_ADDR", "")
    headers = {
        "User-Agent": environ.get("HTTP_USER_AGENT", ""),
        "X-Forwarded-For": ip,
        "Content-Type": "application/json",
    }
    # Same hostname as the browser's pageviews, so events join the visit.
    origin = environ.get("HTTP_ORIGIN") or f"https://{PLAUSIBLE_DOMAIN}"
    url = f"{origin}/"
    return headers, url


def _post(payload: dict, headers: dict[str, str]) -> None:
    try:
        requests.post(
            f"{PLAUSIBLE_API_HOST}/api/event", json=payload, headers=headers, timeout=TIMEOUT_SECONDS
        )
    except Exception as e:
        logger.warning(f"[analytics] Plausible event not sent: {type(e).__name__}")


def track(name: str, **props: str) -> None:
    """Send one event in the background. Never raises, never blocks."""
    if not PLAUSIBLE_DOMAIN:
        return
    try:
        headers, url = _visitor_headers()
        payload = build_payload(name, props, url)
        loop = asyncio.get_running_loop()
        if len(_background_tasks) >= MAX_PENDING:
            logger.debug("[analytics] Event dropped: too many pending")
            return
        task = loop.run_in_executor(_EXECUTOR, _post, payload, headers)
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
    except Exception as e:
        logger.warning(f"[analytics] Event skipped: {type(e).__name__}")
