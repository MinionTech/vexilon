"""Tests for Plausible analytics (#796): payloads carry no content, file names or user input."""
import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import services.analytics as analytics
import services.persistence as persistence

ALLOWED_KEYS = {"name", "domain", "url", "props"}


@pytest.fixture
def plausible(monkeypatch):
    """Enable analytics with a fake visitor session and capture outgoing posts."""
    monkeypatch.setattr(analytics, "PLAUSIBLE_DOMAIN", "vexilon.example")
    session = SimpleNamespace(environ={
        "HTTP_USER_AGENT": "Mozilla/5.0 test",
        "HTTP_X_FORWARDED_FOR": "203.0.113.7",
        "HTTP_ORIGIN": "https://vexilon.example",
    })
    monkeypatch.setattr(analytics.cl, "context", SimpleNamespace(session=session), raising=False)
    post = MagicMock()
    monkeypatch.setattr(analytics.requests, "post", post)
    return post


async def _drain():
    await asyncio.gather(*list(analytics._background_tasks))


@pytest.mark.parametrize("name,props", [
    ("Import", {"result": "success"}),
    ("Import", {"result": "failure"}),
    ("Export", {"result": "success"}),
    ("Export", {"result": "failure"}),
    ("Persona Selected", {"persona": "Lookup"}),
    ("Persona Selected", {"persona": "Grieve"}),
    ("Persona Selected", {"persona": "Manage"}),
])
async def test_event_payload_is_name_and_enumerated_props_only(plausible, name, props):
    analytics.track(name, **props)
    await _drain()

    plausible.assert_called_once()
    kwargs = plausible.call_args.kwargs
    payload = kwargs["json"]
    assert plausible.call_args.args[0] == "https://plausible.io/api/event"
    assert set(payload) == ALLOWED_KEYS
    assert payload["name"] == name
    assert payload["props"] == props
    assert payload["url"] == "https://vexilon.example/"
    assert kwargs["timeout"] == analytics.TIMEOUT_SECONDS


@pytest.mark.parametrize("name,props", [
    ("Import", {"result": "success", "file": "conversation_20260101.md"}),
    ("Import", {"result": "my grievance about my manager"}),
    ("Export", {}),
    ("Persona Selected", {"persona": "Ignore previous instructions"}),
    ("Query", {"text": "What is just cause?"}),
])
def test_payload_rejects_content_file_names_and_user_input(name, props):
    with pytest.raises(ValueError):
        analytics.build_payload(name, props, "https://vexilon.example/")


async def test_disallowed_event_is_not_sent(plausible, caplog):
    with caplog.at_level(logging.WARNING, logger=analytics.logger.name):
        analytics.track("Persona Selected", persona="secret question text")
    await _drain()
    plausible.assert_not_called()
    assert "secret question text" not in caplog.text


async def test_nothing_sent_when_domain_unset(monkeypatch):
    monkeypatch.setattr(analytics, "PLAUSIBLE_DOMAIN", "")
    post = MagicMock()
    monkeypatch.setattr(analytics.requests, "post", post)
    analytics.track("Export", result="success")
    await _drain()
    post.assert_not_called()


async def test_network_failure_never_raises(plausible, caplog):
    plausible.side_effect = analytics.requests.ConnectionError("test: simulated outage")
    with caplog.at_level(logging.WARNING, logger=analytics.logger.name):
        analytics.track("Export", result="success")
        await _drain()
    assert "Plausible event not sent" in caplog.text


def test_missing_session_never_raises(monkeypatch, caplog):
    monkeypatch.setattr(analytics, "PLAUSIBLE_DOMAIN", "vexilon.example")
    monkeypatch.setattr(analytics.cl, "context", None, raising=False)
    with caplog.at_level(logging.WARNING, logger=analytics.logger.name):
        analytics.track("Export", result="success")  # no running loop, no session
    assert "Event skipped" in caplog.text


async def test_track_does_not_block(plausible):
    """The HTTP call runs in a background thread; track() returns immediately."""
    import threading
    release = threading.Event()
    plausible.side_effect = lambda *a, **k: release.wait(5)
    analytics.track("Export", result="success")
    assert analytics._background_tasks  # still in flight
    release.set()
    await _drain()


async def test_export_and_import_fire_events(monkeypatch):
    calls = []
    monkeypatch.setattr(persistence, "track", lambda name, **p: calls.append((name, p)))
    monkeypatch.setattr(persistence._rate_limiter, "is_allowed", lambda _id: (True, ""))
    session = {"history": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
               "persona": "Lookup"}
    monkeypatch.setattr(persistence.cl, "user_session", SimpleNamespace(
        get=session.get, set=session.__setitem__))
    msg = MagicMock()
    msg.return_value.send = AsyncMock()
    monkeypatch.setattr(persistence.cl, "Message", msg)
    monkeypatch.setattr(persistence.cl, "File", MagicMock())

    await persistence.trigger_session_save("test")
    saved = persistence.serialize_conversation(session["history"], "Lookup")
    await persistence.trigger_session_load(saved, "test")
    with caplog_off():
        await persistence.trigger_session_load("not a session", "test")

    assert calls == [
        ("Export", {"result": "success"}),
        ("Import", {"result": "success"}),
        ("Import", {"result": "failure"}),
    ]
    assert all(json.dumps(p) in ('{"result": "success"}', '{"result": "failure"}') for _, p in calls)


class caplog_off:
    """Keep the expected '[load] validation error' log out of CI output."""
    def __enter__(self):
        logging.disable(logging.ERROR)
    def __exit__(self, *exc):
        logging.disable(logging.NOTSET)
