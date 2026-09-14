"""
app/telemetry.py — Privacy-first operational telemetry logger.

Dispatches non-blocking async HTTP POST requests with operational metadata
(latency, tokens, retrieval scores) when TELEMETRY_URL is configured.
Ensures zero user queries, prompt text, assistant responses, or PII are logged.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Strong references for fire-and-forget tasks to prevent premature GC
_active_telemetry_tasks: set[asyncio.Task] = set()

# Forbidden keys that must NEVER appear anywhere in operational telemetry
FORBIDDEN_KEYS: set[str] = {"content", "message", "prompt", "response", "query"}


def assert_safe_payload(obj: Any) -> None:
    """Recursively verify that no forbidden keys exist in the payload structure."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                raise ValueError(f"Forbidden key '{key}' detected in telemetry payload.")
            assert_safe_payload(value)
    elif isinstance(obj, list):
        for item in obj:
            assert_safe_payload(item)


def build_telemetry_payload(
    latency_ms: float,
    tokens: dict[str, Any] | None = None,
    retrieval_metadata: dict[str, Any] | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    """
    Construct an operational telemetry payload strictly excluding any user queries,
    prompts, responses, or PII.
    """
    token_payload: dict[str, Any] | None = None
    if isinstance(tokens, dict):
        token_payload = {
            "prompt_tokens": tokens.get("prompt_tokens"),
            "completion_tokens": tokens.get("completion_tokens"),
            "total_tokens": tokens.get("total_tokens"),
        }

    retrieval_payload: dict[str, Any] | None = None
    if isinstance(retrieval_metadata, dict):
        retrieval_payload = {
            "chunk_count": int(retrieval_metadata.get("chunk_count", 0)),
            "top_score": (
                round(float(retrieval_metadata["top_score"]), 4)
                if retrieval_metadata.get("top_score") is not None
                else None
            ),
            "mean_score": (
                round(float(retrieval_metadata["mean_score"]), 4)
                if retrieval_metadata.get("mean_score") is not None
                else None
            ),
        }

    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_ms": round(float(latency_ms), 2),
        "tokens": token_payload,
        "retrieval_metadata": retrieval_payload,
        "model": model,
        "provider": provider,
    }

    # Defensive guarantee: strictly verify no forbidden keys slipped in
    assert_safe_payload(payload)

    return payload


async def _dispatch_http(url: str, payload: dict[str, Any], timeout: float = 2.0) -> None:
    """Send async HTTP POST to telemetry endpoint, swallowing errors to protect UX."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code >= 400:
                logger.debug(f"[telemetry] Telemetry endpoint returned status {resp.status_code}")
    except Exception as exc:
        logger.debug(f"[telemetry] Telemetry dispatch failed: {exc}")


def log_telemetry(
    latency_ms: float,
    tokens: dict[str, Any] | None = None,
    retrieval_metadata: dict[str, Any] | None = None,
    model: str | None = None,
    provider: str | None = None,
    url: str | None = None,
) -> asyncio.Task | None:
    """
    Dispatch operational telemetry if TELEMETRY_URL is configured.

    Returns the spawned asyncio.Task if scheduled, or None if disabled (strict no-op).
    """
    target_url = url if url is not None else os.getenv("TELEMETRY_URL", "").strip()
    if not target_url:
        return None

    try:
        payload = build_telemetry_payload(
            latency_ms=latency_ms,
            tokens=tokens,
            retrieval_metadata=retrieval_metadata,
            model=model,
            provider=provider,
        )
    except Exception as exc:
        logger.debug(f"[telemetry] Failed to build telemetry payload: {exc}")
        return None

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.debug("[telemetry] No running event loop to schedule telemetry dispatch")
        return None

    task = loop.create_task(_dispatch_http(target_url, payload))
    _active_telemetry_tasks.add(task)
    task.add_done_callback(_active_telemetry_tasks.discard)
    return task


# Aliases for operational clarity
send_telemetry = log_telemetry
record_telemetry = log_telemetry
