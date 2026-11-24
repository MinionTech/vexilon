"""Unit tests for optional GoatCounter config, starter sync, and /api/goatcounter."""

import json
import re
from pathlib import Path

from fastapi import FastAPI

from core.config import get_goatcounter_code
from middleware.cookies import get_goatcounter_config, register_routes_and_middleware
from starters import STARTER_SPECS


def test_code_unset(monkeypatch):
    monkeypatch.delenv("GOATCOUNTER_CODE", raising=False)
    assert get_goatcounter_code() == ""
    assert get_goatcounter_config() == {"goatcounter_code": None}


def test_code_empty(monkeypatch):
    monkeypatch.setenv("GOATCOUNTER_CODE", "   ")
    assert get_goatcounter_code() == ""
    assert get_goatcounter_config() == {"goatcounter_code": None}


def test_code_valid(monkeypatch):
    monkeypatch.setenv("GOATCOUNTER_CODE", "vexilon-ci")
    assert get_goatcounter_code() == "vexilon-ci"
    assert get_goatcounter_config() == {"goatcounter_code": "vexilon-ci"}


def test_code_is_normalized(monkeypatch):
    monkeypatch.setenv("GOATCOUNTER_CODE", "  Vexilon-CI  ")
    assert get_goatcounter_code() == "vexilon-ci"


def test_code_rejects_invalid(monkeypatch):
    for value in (
        "https://evil.example/count",
        "foo.bar",
        "foo_bar",
        "-abc",
        "abc-",
        "a" * 64,
        "../x",
    ):
        monkeypatch.setenv("GOATCOUNTER_CODE", value)
        assert get_goatcounter_code() == "", value
        assert get_goatcounter_config() == {"goatcounter_code": None}


def test_route_registered():
    app = FastAPI()
    register_routes_and_middleware(app)
    route = next(
        r for r in app.routes if getattr(r, "path", None) == "/api/goatcounter"
    )
    assert "GET" in route.methods
    assert route.endpoint is get_goatcounter_config


def test_route_endpoint_unset(monkeypatch):
    monkeypatch.delenv("GOATCOUNTER_CODE", raising=False)
    assert get_goatcounter_config() == {"goatcounter_code": None}


def test_route_endpoint_set(monkeypatch):
    monkeypatch.setenv("GOATCOUNTER_CODE", "vexilon-ci")
    assert get_goatcounter_config() == {"goatcounter_code": "vexilon-ci"}


def test_starter_persona_map_matches_starters_py():
    index_js = (Path(__file__).resolve().parents[1] / "public" / "index.js").read_text(
        encoding="utf-8"
    )
    match = re.search(r"const STARTER_PERSONAS\s*=\s*(\{[^}]+\})", index_js)
    assert match is not None
    # Convert trailing comma before closing brace to parse cleanly as JSON.
    raw_obj = re.sub(r",\s*\}", "}", match.group(1))
    js_map = json.loads(raw_obj)
    expected = {
        f"starter-{spec.label.strip().lower().replace(' ', '-')}": spec.target_persona.lower()
        for spec in STARTER_SPECS
    }
    assert js_map == expected
