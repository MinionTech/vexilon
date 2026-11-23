"""Tests for applying Hugging Face Space settings from GitHub Environments."""

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
        self.visibility = visibility
        self.hardware = hardware
        self.sleep = sleep
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

    def update_repo_settings(self, **kwargs):
        self.calls.append(("update_repo_settings", kwargs))

    def request_space_hardware(self, **kwargs):
        self.calls.append(("request_space_hardware", kwargs))

    def set_space_sleep_time(self, **kwargs):
        self.calls.append(("set_space_sleep_time", kwargs))

    def add_space_variable(self, **kwargs):
        self.calls.append(("add_space_variable", kwargs))

    def add_space_secret(self, **kwargs):
        self.calls.append(("add_space_secret", kwargs))


def test_noop_when_live_space_matches_and_keeps_undeclared_variables():
    api = FakeApi(variables={"GOATCOUNTER_CODE": "", "UNRELATED": "keep-me"})
    applied = settings.apply_space_settings(
        repo="bcgeu/navigator-test",
        visibility="public",
        hardware="cpu-basic",
        sleep_seconds=172800,
        required_vars=["GOATCOUNTER_CODE"],
        secret_names=[],
        dry_run=False,
        environ={"HF_TOKEN": "tok", "SPACE_VARS_JSON": '{"GOATCOUNTER_CODE": ""}'},
        api=api,
    )
    assert applied == []
    assert api.calls == []


def test_applies_only_changed_settings_and_variables():
    api = FakeApi(visibility="private", variables={})
    applied = settings.apply_space_settings(
        repo="bcgeu/navigator",
        visibility="public",
        hardware="cpu-basic",
        sleep_seconds=172800,
        required_vars=["GOATCOUNTER_CODE"],
        secret_names=[],
        dry_run=False,
        environ={"HF_TOKEN": "tok", "SPACE_VARS_JSON": '{"GOATCOUNTER_CODE": "site"}'},
        api=api,
    )
    assert applied == ["visibility=public", "variable:GOATCOUNTER_CODE"]
    assert api.calls == [
        ("update_repo_settings", {"repo_id": "bcgeu/navigator", "visibility": "public", "repo_type": "space"}),
        ("add_space_variable", {"repo_id": "bcgeu/navigator", "key": "GOATCOUNTER_CODE", "value": "site"}),
    ]


def test_rejects_custom_sleep_on_cpu_basic():
    with pytest.raises(settings.ApplyError, match="cpu-basic"):
        settings.apply_space_settings(
            repo="bcgeu/navigator-test",
            visibility="public",
            hardware="cpu-basic",
            sleep_seconds=3600,
            required_vars=[],
            secret_names=[],
            dry_run=True,
            environ={},
        )


def test_sets_sleep_on_upgraded_hardware():
    api = FakeApi()
    settings.apply_space_settings(
        repo="bcgeu/navigator-test",
        visibility="public",
        hardware="cpu-upgrade",
        sleep_seconds=3600,
        required_vars=[],
        secret_names=[],
        dry_run=False,
        environ={"HF_TOKEN": "tok", "SPACE_VARS_JSON": '{"GOATCOUNTER_CODE": ""}'},
        api=api,
    )
    assert ("request_space_hardware", {"repo_id": "bcgeu/navigator-test", "hardware": "cpu-upgrade"}) in api.calls
    assert ("set_space_sleep_time", {"repo_id": "bcgeu/navigator-test", "sleep_time": 3600}) in api.calls


def test_missing_required_variable_fails_even_in_dry_run(capsys):
    code = settings.main(
        ["--repo", "bcgeu/navigator", "--require-var", "GOATCOUNTER_CODE", "--dry-run"],
        environ={"SPACE_VARS_JSON": "{}"},
    )
    assert code == 1
    assert "GOATCOUNTER_CODE" in capsys.readouterr().err


def test_empty_required_variable_is_allowed_in_dry_run():
    code = settings.main(
        ["--repo", "bcgeu/navigator-test", "--require-var", "GOATCOUNTER_CODE", "--dry-run"],
        environ={"SPACE_VARS_JSON": '{"GOATCOUNTER_CODE": ""}'},
    )
    assert code == 0


def test_live_apply_requires_hf_token(capsys):
    code = settings.main(
        ["--repo", "bcgeu/navigator-test", "--require-var", "GOATCOUNTER_CODE"],
        environ={"SPACE_VARS_JSON": '{"GOATCOUNTER_CODE": ""}'},
    )
    assert code == 1
    assert "HF_TOKEN" in capsys.readouterr().err


def test_refuses_to_sync_hf_token_as_space_secret(capsys):
    code = settings.main(
        ["--repo", "bcgeu/navigator-test", "--secret", "HF_TOKEN", "--dry-run"],
        environ={"HF_TOKEN": "tok"},
    )
    assert code == 1
    assert "HF_TOKEN" in capsys.readouterr().err


def test_merge_workflow_wires_vars_json_for_both_spaces():
    merge = (REPO_ROOT / ".github" / "workflows" / "merge.yml").read_text()
    assert "--repo bcgeu/navigator-test" in merge
    assert "--repo bcgeu/navigator" in merge
    assert "SPACE_VARS_JSON: ${{ toJSON(vars) }}" in merge
    assert "pip install --user" not in merge
