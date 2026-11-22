"""Tests for applying Hugging Face Space settings from .github/spaces.toml."""

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


CONFIG = REPO_ROOT / ".github" / "spaces.toml"


def test_config_declares_live_space_settings():
    """Desired state matches the live Spaces (public, cpu-basic, 48h sleep)."""
    config = settings.load_config(CONFIG)
    for name, repo in (
        ("test", "bcgeu/navigator-test"),
        ("prod", "bcgeu/navigator"),
    ):
        space = settings.select_space(config, name)
        assert space["repo"] == repo
        assert space["visibility"] == "public"
        assert space["hardware"] == "cpu-basic"
        assert space["sleep_seconds"] == 172800
        assert space["variables"] == ["GOATCOUNTER_CODE"]
        assert space["secrets"] == []


def test_env_is_required(capsys):
    with pytest.raises(SystemExit):
        settings.parse_args([])
    with pytest.raises(SystemExit):
        settings.parse_args(["--dry-run"])


def test_unknown_environment_is_rejected():
    config = settings.load_config(CONFIG)
    with pytest.raises(settings.ApplyError, match="must be one of"):
        settings.select_space(config, "staging")


def test_missing_process_env_variable_fails():
    with pytest.raises(settings.ApplyError, match="GOATCOUNTER_CODE"):
        settings.require_env_values(["GOATCOUNTER_CODE"], {}, "variable")


def test_empty_variable_value_is_allowed():
    values = settings.require_env_values(
        ["GOATCOUNTER_CODE"], {"GOATCOUNTER_CODE": ""}, "variable"
    )
    assert values == {"GOATCOUNTER_CODE": ""}


def test_github_missing_variable_fails():
    def getter(url: str, token: str) -> int:
        assert token == "ghs_test"
        assert url.endswith("/environments/test/variables/GOATCOUNTER_CODE")
        return 404

    with pytest.raises(settings.ApplyError, match="variable GOATCOUNTER_CODE"):
        settings.assert_github_environment_items(
            "MinionTech/vexilon",
            "test",
            ["GOATCOUNTER_CODE"],
            [],
            "ghs_test",
            getter=getter,
        )


def test_github_defined_variable_passes():
    settings.assert_github_environment_items(
        "MinionTech/vexilon",
        "prod",
        ["GOATCOUNTER_CODE"],
        [],
        "ghs_test",
        getter=lambda url, token: 200,
    )


def test_github_api_error_fails():
    with pytest.raises(settings.ApplyError, match="HTTP 500"):
        settings.assert_github_environment_items(
            "MinionTech/vexilon",
            "test",
            ["GOATCOUNTER_CODE"],
            [],
            "ghs_test",
            getter=lambda url, token: 500,
        )


def _space():
    return {
        "repo": "bcgeu/navigator-test",
        "visibility": "public",
        "hardware": "cpu-basic",
        "sleep_seconds": 172800,
        "variables": ["GOATCOUNTER_CODE"],
        "secrets": [],
    }


def test_plan_is_empty_when_current_matches():
    current = {
        "visibility": "public",
        "hardware": "cpu-basic",
        "requested_hardware": "cpu-basic",
        "sleep_seconds": 172800,
        "variables": {"GOATCOUNTER_CODE": "", "UNRELATED": "keep-me"},
    }
    actions = settings.plan_changes(_space(), current, {"GOATCOUNTER_CODE": ""}, [])
    assert actions == []


def test_plan_does_not_delete_undeclared_variables():
    current = {
        "visibility": "public",
        "hardware": "cpu-basic",
        "requested_hardware": "cpu-basic",
        "sleep_seconds": 172800,
        "variables": {"GOATCOUNTER_CODE": "abc", "UNRELATED": "keep-me"},
    }
    actions = settings.plan_changes(_space(), current, {"GOATCOUNTER_CODE": "abc"}, [])
    assert actions == []
    assert not any(action["op"] == "delete" for action in actions)


def test_plan_upserts_changed_variable_and_visibility():
    current = {
        "visibility": "private",
        "hardware": "cpu-basic",
        "requested_hardware": "cpu-basic",
        "sleep_seconds": 172800,
        "variables": {},
    }
    actions = settings.plan_changes(_space(), current, {"GOATCOUNTER_CODE": "site"}, [])
    ops = [(action["op"], action.get("name") or action.get("value")) for action in actions]
    assert ("visibility", "public") in ops
    assert ("variable", "GOATCOUNTER_CODE") in ops


def test_plan_rejects_custom_sleep_on_cpu_basic():
    space = _space()
    space["sleep_seconds"] = 3600
    current = {
        "visibility": "public",
        "hardware": "cpu-basic",
        "requested_hardware": "cpu-basic",
        "sleep_seconds": 172800,
        "variables": {"GOATCOUNTER_CODE": ""},
    }
    with pytest.raises(settings.ApplyError, match="cpu-basic"):
        settings.plan_changes(space, current, {"GOATCOUNTER_CODE": ""}, [])


def test_plan_sets_sleep_on_upgraded_hardware():
    space = _space()
    space["hardware"] = "cpu-upgrade"
    space["sleep_seconds"] = 3600
    current = {
        "visibility": "public",
        "hardware": "cpu-basic",
        "requested_hardware": "cpu-basic",
        "sleep_seconds": 172800,
        "variables": {"GOATCOUNTER_CODE": ""},
    }
    actions = settings.plan_changes(space, current, {"GOATCOUNTER_CODE": ""}, [])
    assert {"op": "hardware", "value": "cpu-upgrade", "repo": space["repo"]} in actions
    assert {"op": "sleep", "value": 3600, "repo": space["repo"]} in actions


def test_dry_run_exits_zero_and_does_not_need_tokens():
    code = settings.main(
        ["--env", "test", "--dry-run", "--config", str(CONFIG)],
        environ={"GOATCOUNTER_CODE": "rehearsal"},
    )
    assert code == 0


def test_dry_run_fails_when_variable_missing(capsys):
    code = settings.main(
        ["--env", "prod", "--dry-run", "--config", str(CONFIG)],
        environ={},
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "GOATCOUNTER_CODE" in err


def test_live_apply_without_hf_token_fails(capsys):
    code = settings.main(
        ["--env", "test", "--config", str(CONFIG)],
        environ={"GOATCOUNTER_CODE": ""},
    )
    assert code == 1
    assert "HF_TOKEN" in capsys.readouterr().err


def test_apply_actions_call_expected_api():
    calls: list[tuple] = []

    class Api:
        def update_repo_settings(self, **kwargs):
            calls.append(("update_repo_settings", kwargs))

        def request_space_hardware(self, **kwargs):
            calls.append(("request_space_hardware", kwargs))

        def set_space_sleep_time(self, **kwargs):
            calls.append(("set_space_sleep_time", kwargs))

        def add_space_variable(self, **kwargs):
            calls.append(("add_space_variable", kwargs))

        def add_space_secret(self, **kwargs):
            calls.append(("add_space_secret", kwargs))

        def delete_space_variable(self, **kwargs):
            calls.append(("delete_space_variable", kwargs))

    space = _space()
    actions = [
        {"op": "visibility", "value": "public", "repo": space["repo"]},
        {"op": "variable", "name": "GOATCOUNTER_CODE", "repo": space["repo"]},
    ]
    settings.apply_actions(
        Api(), space, actions, {"GOATCOUNTER_CODE": "site"}, {}
    )
    assert calls == [
        (
            "update_repo_settings",
            {"repo_id": space["repo"], "visibility": "public", "repo_type": "space"},
        ),
        (
            "add_space_variable",
            {"repo_id": space["repo"], "key": "GOATCOUNTER_CODE", "value": "site"},
        ),
    ]
    assert not any(name.startswith("delete_") for name, _ in calls)


def test_current_state_reads_runtime_and_variables():
    class Api:
        def space_info(self, repo_id):
            return SimpleNamespace(private=False, visibility="public")

        def get_space_runtime(self, repo_id):
            return SimpleNamespace(
                hardware=SimpleNamespace(value="cpu-basic"),
                requested_hardware=SimpleNamespace(value="cpu-basic"),
                sleep_time=None,
                raw={"gcTimeout": 172800},
            )

        def get_space_variables(self, repo_id):
            return {"GOATCOUNTER_CODE": SimpleNamespace(value="")}

    state = settings.current_state(Api(), "bcgeu/navigator-test")
    assert state["visibility"] == "public"
    assert state["hardware"] == "cpu-basic"
    assert state["sleep_seconds"] == 172800
    assert state["variables"] == {"GOATCOUNTER_CODE": ""}


def test_merge_workflow_applies_settings_for_both_environments():
    merge = (REPO_ROOT / ".github" / "workflows" / "merge.yml").read_text()
    assert "apply_space_settings.py --env test" in merge
    assert "apply_space_settings.py --env prod" in merge
    assert "requirements-space-settings.txt" in merge
    requirements = (REPO_ROOT / ".github" / "scripts" / "requirements-space-settings.txt").read_text()
    assert "huggingface_hub==2.4.0" in requirements
    # The deploy token stays an Environment secret; it is not synced to the Space.
    assert "vars.GOATCOUNTER_CODE" in merge


def test_managed_secrets_do_not_include_hf_token():
    config = settings.load_config(CONFIG)
    for name in ("test", "prod"):
        space = settings.select_space(config, name)
        assert "HF_TOKEN" not in space["secrets"]
        assert "HF_TOKEN" not in space["variables"]
