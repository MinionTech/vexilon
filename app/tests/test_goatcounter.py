"""Unit tests for optional GoatCounter config and the /api/goatcounter route."""

from fastapi import FastAPI

from core.config import get_goatcounter_code
from middleware.cookies import get_goatcounter_config, register_routes_and_middleware


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
