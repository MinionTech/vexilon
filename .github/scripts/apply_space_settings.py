#!/usr/bin/env python3
"""Apply common Hugging Face Space settings (.github/space.toml) + CLI overrides."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "space.toml"
CPU_BASIC_DEFAULT_SLEEP_SECONDS = 172800


class ApplyError(Exception):
    """A required input or Space setting is invalid."""


def _load_inputs(environ: Mapping[str, str], required_vars: list[str], secret_names: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    raw = environ.get("SPACE_VARS_JSON") or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ApplyError(f"SPACE_VARS_JSON is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ApplyError("SPACE_VARS_JSON must be a JSON object")

    variables = {str(k): "" if v is None else str(v) for k, v in parsed.items()}
    if "HF_TOKEN" in variables or "HF_TOKEN" in secret_names:
        raise ApplyError("HF_TOKEN is the deploy credential and cannot be synced to the Space")

    missing_vars = [name for name in required_vars if name not in variables]
    if missing_vars:
        raise ApplyError("Missing GitHub Environment variable(s): " + ", ".join(missing_vars))

    missing_secrets = [name for name in secret_names if name not in environ]
    if missing_secrets:
        raise ApplyError("Missing Space secret(s) in environment: " + ", ".join(missing_secrets))

    return variables, {name: environ[name] for name in secret_names}


def _hw_name(value: Any) -> str | None:
    return value.value if hasattr(value, "value") else (None if value is None else str(value))


def apply_space_settings(
    *,
    repo: str,
    config: dict[str, Any],
    secret_names: list[str],
    dry_run: bool,
    environ: Mapping[str, str],
    api: Any = None,
) -> list[str]:
    visibility = config["visibility"]
    hardware = config["hardware"]
    sleep_seconds = int(config["sleep_seconds"])
    required_vars = list(config.get("required_vars") or [])
    upgraded = hardware != "cpu-basic"

    if not upgraded and sleep_seconds != CPU_BASIC_DEFAULT_SLEEP_SECONDS:
        raise ApplyError(f"Cannot set sleep_seconds={sleep_seconds} on cpu-basic ({repo}); fixed at {CPU_BASIC_DEFAULT_SLEEP_SECONDS}")

    variables, secrets = _load_inputs(environ, required_vars, secret_names)
    if dry_run:
        items = [f"visibility={visibility}", f"hardware={hardware}", *(f"var:{k}" for k in sorted(variables)), *(f"secret:{k}" for k in sorted(secrets))]
        if upgraded:
            items.insert(2, f"sleep_seconds={sleep_seconds}")
        print(f"Dry-run for {repo}: {', '.join(items)}")
        return items

    if api is None:
        token = environ.get("HF_TOKEN")
        if not token:
            raise ApplyError("HF_TOKEN is required unless --dry-run is set")
        from huggingface_hub import HfApi

        api = HfApi(token=token)

    info = api.space_info(repo_id=repo)
    runtime = api.get_space_runtime(repo_id=repo)
    cur_vis = getattr(info, "visibility", None) or ("private" if getattr(info, "private", False) else "public")
    cur_hw = _hw_name(getattr(runtime, "requested_hardware", None)) or _hw_name(getattr(runtime, "hardware", None))
    cur_sleep = getattr(runtime, "sleep_time", None)
    if cur_sleep is None and isinstance(getattr(runtime, "raw", None), dict):
        cur_sleep = runtime.raw.get("gcTimeout")
    cur_vars = {k: (v if isinstance(v, str) else getattr(v, "value", "")) for k, v in api.get_space_variables(repo_id=repo).items()}

    applied: list[str] = []
    if cur_vis != visibility:
        api.update_repo_settings(repo_id=repo, visibility=visibility, repo_type="space")
        applied.append(f"visibility={visibility}")
    if cur_hw != hardware:
        api.request_space_hardware(repo_id=repo, hardware=hardware)
        applied.append(f"hardware={hardware}")
    if upgraded and cur_sleep != sleep_seconds:
        api.set_space_sleep_time(repo_id=repo, sleep_time=sleep_seconds)
        applied.append(f"sleep_seconds={sleep_seconds}")
    for name, value in sorted(variables.items()):
        if cur_vars.get(name) != value:
            api.add_space_variable(repo_id=repo, key=name, value=value)
            applied.append(f"var:{name}")
    for name, value in sorted(secrets.items()):
        api.add_space_secret(repo_id=repo, key=name, value=value)
        applied.append(f"secret:{name}")

    print(f"Applied to {repo}: {', '.join(applied)}" if applied else f"No Space setting changes for {repo}.")
    return applied


def main(argv: list[str] | None = None, environ: Mapping[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--visibility", choices=("public", "private"))
    parser.add_argument("--hardware")
    parser.add_argument("--sleep-seconds", type=int)
    parser.add_argument("--require-var", action="append", default=[])
    parser.add_argument("--secret", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not args.config.is_file():
            raise ApplyError(f"Config file not found: {args.config}")
        with args.config.open("rb") as fh:
            config = tomllib.load(fh)
        for key, val in (("visibility", args.visibility), ("hardware", args.hardware), ("sleep_seconds", args.sleep_seconds)):
            if val is not None:
                config[key] = val
        if args.require_var:
            config["required_vars"] = list(dict.fromkeys([*config.get("required_vars", []), *args.require_var]))
        apply_space_settings(
            repo=args.repo,
            config=config,
            secret_names=args.secret,
            dry_run=args.dry_run,
            environ=environ if environ is not None else os.environ,
        )
        return 0
    except ApplyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
