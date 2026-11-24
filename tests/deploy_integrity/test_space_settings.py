"""Tests for applying common Space settings (.github/space.toml) + CLI overrides."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / ".github" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import apply_space_settings as settings  # noqa: E402


class FakeApi:
    def __init__(self, *, visibility="public", hardware="cpu-basic", sleep=172800, variables=None):
        self.visibility, self.hardware, self.sleep = visibility, hardware, sleep
        self.variables = variables if variables is not None else {"GOATCOUNTER_CODE": ""}
        self.calls: list[tuple[str, dict]] = []

    def space_info(self, repo_id):
        return SimpleNamespace(private=(self.visibility == "private"), visibility=self.visibility)

    def get_space_runtime(self, repo_id):
        return SimpleNamespace(
            hardware=SimpleNamespace(value=self.hardware),
            requested_hardware=SimpleNamespace(value=self.hardware),
            sleep_time=None,
            raw={"gcTimeout": self.sleep},
        )

    def get_space_variables(self, repo_id):
        return {k: SimpleNamespace(value=v) for k, v in self.variables.items()}

    def update_repo_settings(self, **kw):
        self.calls.append(("update_repo_settings", kw))

    def request_space_hardware(self, **kw):
        self.calls.append(("request_space_hardware", kw))

    def set_space_sleep_time(self, **kw):
        self.calls.append(("set_space_sleep_time", kw))

    def add_space_variable(self, **kw):
        self.calls.append(("add_space_variable", kw))

    def add_space_secret(self, **kw):
        self.calls.append(("add_space_secret", kw))


_CFG = {"visibility": "public", "hardware": "cpu-basic", "sleep_seconds": 172800, "required_vars": ["GOATCOUNTER_CODE"]}


def test_noop_when_live_matches_and_preserves_undeclared_vars():
    api = FakeApi(variables={"GOATCOUNTER_CODE": "", "UNRELATED": "keep"})
    applied = settings.apply_space_settings(
        repo="bcgeu/navigator-test", config=_CFG, secret_names=[], dry_run=False,
        environ={"HF_TOKEN": "tok", "SPACE_VARS_JSON": '{"GOATCOUNTER_CODE": ""}'}, api=api,
    )
    assert applied == [] and api.calls == []


def test_applies_only_changed_settings_and_variables():
    api = FakeApi(visibility="private", variables={})
    applied = settings.apply_space_settings(
        repo="bcgeu/navigator", config=_CFG, secret_names=[], dry_run=False,
        environ={"HF_TOKEN": "tok", "SPACE_VARS_JSON": '{"GOATCOUNTER_CODE": "site"}'}, api=api,
    )
    assert applied == ["visibility=public", "var:GOATCOUNTER_CODE"]
    assert api.calls == [
        ("update_repo_settings", {"repo_id": "bcgeu/navigator", "visibility": "public", "repo_type": "space"}),
        ("add_space_variable", {"repo_id": "bcgeu/navigator", "key": "GOATCOUNTER_CODE", "value": "site"}),
    ]


def test_common_config_and_cli_override_in_dry_run(capsys):
    code = settings.main(
        ["--repo", "bcgeu/navigator-test", "--hardware", "cpu-upgrade", "--sleep-seconds", "3600", "--dry-run"],
        environ={"SPACE_VARS_JSON": '{"GOATCOUNTER_CODE": "rehearsal"}'},
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "hardware=cpu-upgrade" in out and "sleep_seconds=3600" in out


def test_rejects_custom_sleep_on_cpu_basic(capsys):
    code = settings.main(
        ["--repo", "bcgeu/navigator-test", "--sleep-seconds", "3600", "--dry-run"],
        environ={"SPACE_VARS_JSON": '{"GOATCOUNTER_CODE": ""}'},
    )
    assert code == 1 and "cpu-basic" in capsys.readouterr().err


def test_missing_vs_empty_required_var(capsys):
    assert settings.main(["--repo", "bcgeu/navigator-test", "--dry-run"], environ={"SPACE_VARS_JSON": "{}"}) == 0
    assert settings.main(["--repo", "bcgeu/navigator", "--require-var", "GOATCOUNTER_CODE", "--dry-run"], environ={"SPACE_VARS_JSON": "{}"}) == 1
    assert "GOATCOUNTER_CODE" in capsys.readouterr().err
    assert settings.main(["--repo", "bcgeu/navigator", "--require-var", "GOATCOUNTER_CODE", "--dry-run"], environ={"SPACE_VARS_JSON": '{"GOATCOUNTER_CODE": ""}'}) == 0


def test_rejects_hf_token_in_vars_and_secrets(capsys):
    for args, env in (
        (["--repo", "bcgeu/navigator-test", "--dry-run"], {"SPACE_VARS_JSON": '{"GOATCOUNTER_CODE":"","HF_TOKEN":"x"}'}),
        (["--repo", "bcgeu/navigator-test", "--secret", "HF_TOKEN", "--dry-run"], {"SPACE_VARS_JSON": '{"GOATCOUNTER_CODE":""}', "HF_TOKEN": "x"}),
    ):
        assert settings.main(args, environ=env) == 1
        assert "HF_TOKEN" in capsys.readouterr().err


def test_merge_workflow_wires_vars_json_for_both_spaces():
    merge = (REPO_ROOT / ".github" / "workflows" / "merge.yml").read_text()
    assert "--repo bcgeu/navigator-test" in merge
    assert "--repo bcgeu/navigator --require-var GOATCOUNTER_CODE" in merge
    assert "SPACE_VARS_JSON: ${{ toJSON(vars) }}" in merge
