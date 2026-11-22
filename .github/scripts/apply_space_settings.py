#!/usr/bin/env python3
"""Apply desired Hugging Face Space settings declared in .github/spaces.toml.

Variable and secret values are read from the process environment (passed in
from the GitHub Environment). The script upserts declared items only; it
does not delete undeclared Space variables or secrets.

Usage:
    python apply_space_settings.py --env test|prod [--dry-run] [--config PATH]
"""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / ".github" / "spaces.toml"
CPU_BASIC_DEFAULT_SLEEP_SECONDS = 172800
VALID_ENVIRONMENTS = ("test", "prod")


class ApplyError(Exception):
    """A required value is missing or an apply step failed."""


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ApplyError(f"Config file not found: {path}")
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    spaces = data.get("spaces")
    if not isinstance(spaces, dict) or not spaces:
        raise ApplyError(f"{path} has no [spaces.*] tables")
    return data


def select_space(config: dict[str, Any], environment: str) -> dict[str, Any]:
    if environment not in VALID_ENVIRONMENTS:
        raise ApplyError(
            f"Environment must be one of {VALID_ENVIRONMENTS}, got {environment!r}"
        )
    space = (config.get("spaces") or {}).get(environment)
    if not isinstance(space, dict):
        raise ApplyError(f"No [spaces.{environment}] table in the config")
    for key in ("repo", "visibility", "hardware", "sleep_seconds"):
        if key not in space:
            raise ApplyError(f"[spaces.{environment}] is missing {key}")
    if not isinstance(space["sleep_seconds"], int):
        raise ApplyError(f"[spaces.{environment}].sleep_seconds must be an integer")
    space = dict(space)
    space["variables"] = list(space.get("variables") or [])
    space["secrets"] = list(space.get("secrets") or [])
    return space


def require_env_values(names: list[str], environ: dict[str, str], kind: str) -> dict[str, str]:
    """Return values for each name. Empty strings are allowed; missing names fail."""
    values: dict[str, str] = {}
    missing = [name for name in names if name not in environ]
    if missing:
        raise ApplyError(
            f"Required Space {kind} missing from the process environment: {', '.join(missing)}"
        )
    for name in names:
        values[name] = environ[name]
    return values


def _hardware_name(value: Any) -> str | None:
    if value is None:
        return None
    return value.value if hasattr(value, "value") else str(value)


def current_state(api: Any, repo: str) -> dict[str, Any]:
    info = api.space_info(repo_id=repo)
    runtime = api.get_space_runtime(repo_id=repo)
    visibility = getattr(info, "visibility", None)
    if visibility is None:
        visibility = "private" if getattr(info, "private", False) else "public"
    current_hw = _hardware_name(getattr(runtime, "hardware", None))
    requested_hw = _hardware_name(getattr(runtime, "requested_hardware", None))
    sleep_time = getattr(runtime, "sleep_time", None)
    if sleep_time is None and hasattr(runtime, "raw") and isinstance(runtime.raw, dict):
        sleep_time = runtime.raw.get("gcTimeout")
    variables = api.get_space_variables(repo_id=repo)
    current_vars = {
        key: (var if isinstance(var, str) else getattr(var, "value", ""))
        for key, var in variables.items()
    }
    return {
        "visibility": visibility,
        "hardware": current_hw or requested_hw,
        "requested_hardware": requested_hw,
        "sleep_seconds": sleep_time,
        "variables": current_vars,
    }


def plan_changes(
    space: dict[str, Any],
    current: dict[str, Any] | None,
    variable_values: dict[str, str],
    secret_names: list[str],
) -> list[dict[str, Any]]:
    """Build the list of writes needed to reach the desired state.

    current=None means the live Space was not read (dry-run); every declared
    setting is reported as a planned write.
    """
    actions: list[dict[str, Any]] = []
    hardware = space["hardware"]
    sleep_seconds = space["sleep_seconds"]
    upgraded = hardware != "cpu-basic"

    if current is None or current.get("visibility") != space["visibility"]:
        actions.append(
            {"op": "visibility", "value": space["visibility"], "repo": space["repo"]}
        )

    current_hw = None if current is None else (current.get("requested_hardware") or current.get("hardware"))
    if current is None or current_hw != hardware:
        actions.append({"op": "hardware", "value": hardware, "repo": space["repo"]})

    current_sleep = None if current is None else current.get("sleep_seconds")
    sleep_matches = current is not None and current_sleep == sleep_seconds
    if not upgraded and sleep_seconds != CPU_BASIC_DEFAULT_SLEEP_SECONDS:
        raise ApplyError(
            f"Cannot set sleep_seconds={sleep_seconds} on cpu-basic "
            f"({space['repo']}); free hardware is fixed at "
            f"{CPU_BASIC_DEFAULT_SLEEP_SECONDS} seconds"
        )
    if not sleep_matches and upgraded:
        actions.append(
            {"op": "sleep", "value": sleep_seconds, "repo": space["repo"]}
        )

    current_vars = {} if current is None else current.get("variables") or {}
    for name, value in variable_values.items():
        if current is None or current_vars.get(name) != value:
            actions.append(
                {"op": "variable", "name": name, "repo": space["repo"]}
            )

    # Secret values cannot be read back; if any are declared, they are always upserted.
    for name in secret_names:
        actions.append({"op": "secret", "name": name, "repo": space["repo"]})

    return actions


def apply_actions(
    api: Any,
    space: dict[str, Any],
    actions: list[dict[str, Any]],
    variable_values: dict[str, str],
    secret_values: dict[str, str],
) -> None:
    repo = space["repo"]
    for action in actions:
        op = action["op"]
        if op == "visibility":
            api.update_repo_settings(
                repo_id=repo, visibility=action["value"], repo_type="space"
            )
        elif op == "hardware":
            api.request_space_hardware(repo_id=repo, hardware=action["value"])
        elif op == "sleep":
            api.set_space_sleep_time(repo_id=repo, sleep_time=action["value"])
        elif op == "variable":
            name = action["name"]
            api.add_space_variable(repo_id=repo, key=name, value=variable_values[name])
        elif op == "secret":
            name = action["name"]
            api.add_space_secret(repo_id=repo, key=name, value=secret_values[name])
        else:
            raise ApplyError(f"Unknown action {op!r}")


def _format_plan(actions: list[dict[str, Any]]) -> str:
    if not actions:
        return "No Space setting changes; already at the desired state."
    lines = ["Planned Space setting changes:"]
    for action in actions:
        op = action["op"]
        if op in {"variable", "secret"}:
            lines.append(f"  - upsert {op} {action['name']} on {action['repo']}")
        else:
            lines.append(f"  - set {op}={action['value']} on {action['repo']}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        required=True,
        choices=VALID_ENVIRONMENTS,
        help="GitHub Environment / spaces.* table to apply (test or prod). Required.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan only; do not call the HF API")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, environ: dict[str, str] | None = None) -> int:
    args = parse_args(argv)
    env = environ if environ is not None else dict(os.environ)
    try:
        config = load_config(args.config)
        space = select_space(config, args.env)
        variable_values = require_env_values(space["variables"], env, "variable")
        secret_values = require_env_values(space["secrets"], env, "secret")

        if args.dry_run:
            actions = plan_changes(space, None, variable_values, list(secret_values))
            print(_format_plan(actions))
            print("Dry-run: no Hugging Face API writes.")
            return 0

        hf_token = env.get("HF_TOKEN")
        if not hf_token:
            raise ApplyError("HF_TOKEN is required unless --dry-run is set")

        from huggingface_hub import HfApi

        api = HfApi(token=hf_token)
        current = current_state(api, space["repo"])
        actions = plan_changes(space, current, variable_values, list(secret_values))
        print(_format_plan(actions))
        apply_actions(api, space, actions, variable_values, secret_values)
        print("Space settings applied.")
        return 0
    except ApplyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
