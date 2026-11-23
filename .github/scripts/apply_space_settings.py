#!/usr/bin/env python3
"""Apply Hugging Face Space settings and GitHub Environment variables/secrets.

Variables are passed via SPACE_VARS_JSON (e.g. from GitHub Actions `toJSON(vars)`).
Secrets are named with --secret and read from the process environment.
Undeclared Space variables and secrets are left untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from typing import Any

CPU_BASIC_DEFAULT_SLEEP_SECONDS = 172800


class ApplyError(Exception):
    """A required input is missing or invalid."""


def _parse_vars_json(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ApplyError(f"SPACE_VARS_JSON is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ApplyError("SPACE_VARS_JSON must be a JSON object")
    return {str(k): "" if v is None else str(v) for k, v in parsed.items()}


def _collect_inputs(
    environ: Mapping[str, str],
    required_vars: list[str],
    secret_names: list[str],
) -> tuple[dict[str, str], dict[str, str]]:
    variables = _parse_vars_json(environ.get("SPACE_VARS_JSON"))
    missing_vars = [name for name in required_vars if name not in variables]
    if missing_vars:
        raise ApplyError(
            "Required Space variable(s) missing from GitHub Environment: "
            + ", ".join(missing_vars)
        )

    if "HF_TOKEN" in secret_names:
        raise ApplyError("HF_TOKEN is the deploy credential and cannot be synced to the Space")

    missing_secrets = [name for name in secret_names if name not in environ]
    if missing_secrets:
        raise ApplyError(
            "Required Space secret(s) missing from process environment: "
            + ", ".join(missing_secrets)
        )
    secrets = {name: environ[name] for name in secret_names}
    return variables, secrets


def _hardware_name(value: Any) -> str | None:
    if value is None:
        return None
    return value.value if hasattr(value, "value") else str(value)


def _current_state(api: Any, repo: str) -> dict[str, Any]:
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
    raw_vars = api.get_space_variables(repo_id=repo)
    current_vars = {
        key: (var if isinstance(var, str) else getattr(var, "value", ""))
        for key, var in raw_vars.items()
    }
    return {
        "visibility": visibility,
        "hardware": requested_hw or current_hw,
        "sleep_seconds": sleep_time,
        "variables": current_vars,
    }


def apply_space_settings(
    *,
    repo: str,
    visibility: str,
    hardware: str,
    sleep_seconds: int,
    required_vars: list[str],
    secret_names: list[str],
    dry_run: bool,
    environ: Mapping[str, str],
    api: Any = None,
) -> list[str]:
    if hardware == "cpu-basic" and sleep_seconds != CPU_BASIC_DEFAULT_SLEEP_SECONDS:
        raise ApplyError(
            f"Cannot set sleep_seconds={sleep_seconds} on cpu-basic ({repo}); "
            f"free hardware is fixed at {CPU_BASIC_DEFAULT_SLEEP_SECONDS} seconds"
        )

    variables, secrets = _collect_inputs(environ, required_vars, secret_names)
    upgraded = hardware != "cpu-basic"

    if dry_run:
        summary = [
            f"visibility={visibility}",
            f"hardware={hardware}",
            *(["sleep_seconds=" + str(sleep_seconds)] if upgraded else []),
            *(f"variable:{k}" for k in sorted(variables)),
            *(f"secret:{k}" for k in sorted(secrets)),
        ]
        print(f"Dry-run for {repo}: {', '.join(summary)}")
        return summary

    if api is None:
        hf_token = environ.get("HF_TOKEN")
        if not hf_token:
            raise ApplyError("HF_TOKEN is required unless --dry-run is set")
        from huggingface_hub import HfApi

        api = HfApi(token=hf_token)

    current = _current_state(api, repo)
    applied: list[str] = []

    if current["visibility"] != visibility:
        api.update_repo_settings(repo_id=repo, visibility=visibility, repo_type="space")
        applied.append(f"visibility={visibility}")

    if current["hardware"] != hardware:
        api.request_space_hardware(repo_id=repo, hardware=hardware)
        applied.append(f"hardware={hardware}")

    if upgraded and current["sleep_seconds"] != sleep_seconds:
        api.set_space_sleep_time(repo_id=repo, sleep_time=sleep_seconds)
        applied.append(f"sleep_seconds={sleep_seconds}")

    for name, value in sorted(variables.items()):
        if current["variables"].get(name) != value:
            api.add_space_variable(repo_id=repo, key=name, value=value)
            applied.append(f"variable:{name}")

    # Space secrets cannot be read back from HF, so declared secrets are always upserted.
    for name, value in sorted(secrets.items()):
        api.add_space_secret(repo_id=repo, key=name, value=value)
        applied.append(f"secret:{name}")

    if applied:
        print(f"Applied to {repo}: {', '.join(applied)}")
    else:
        print(f"No Space setting changes for {repo}; already at desired state.")
    return applied


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Target HF Space repo (e.g. bcgeu/navigator-test)")
    parser.add_argument("--visibility", choices=("public", "private"), default="public")
    parser.add_argument("--hardware", default="cpu-basic")
    parser.add_argument("--sleep-seconds", type=int, default=CPU_BASIC_DEFAULT_SLEEP_SECONDS)
    parser.add_argument(
        "--require-var",
        action="append",
        default=[],
        help="Variable name that must exist in SPACE_VARS_JSON (repeatable)",
    )
    parser.add_argument(
        "--secret",
        action="append",
        default=[],
        help="Secret env var name to upsert onto the Space (repeatable)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without calling HF API")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, environ: Mapping[str, str] | None = None) -> int:
    args = parse_args(argv)
    env = environ if environ is not None else os.environ
    try:
        apply_space_settings(
            repo=args.repo,
            visibility=args.visibility,
            hardware=args.hardware,
            sleep_seconds=args.sleep_seconds,
            required_vars=args.require_var,
            secret_names=args.secret,
            dry_run=args.dry_run,
            environ=env,
        )
        return 0
    except ApplyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
