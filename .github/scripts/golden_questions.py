#!/usr/bin/env python3
"""Ask the committed golden questions on a Hugging Face Space.

A citation miss exits 1. Callers on the TEST-to-PROD path treat that as a
failed gate and do not promote. This file does not call a model itself.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Promotion-path budget. The workflow step timeout stays above this.
# worst_case_seconds=765
QUESTION_ATTEMPTS_DEFAULT = "2"
CURL_MAX_TIME_DEFAULT = "120"
RETRY_DELAY_DEFAULT = "15"
CITATION_WINDOW = 240
RETRY_STATUSES = {408, 429, 500, 502, 503, 504}
SPACE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
QUESTIONS_PATH = Path(__file__).resolve().parents[1] / "golden-questions.json"


def _collapse(text: str) -> str:
    return " ".join(text.casefold().split())


def citation_found(
    answer: str,
    document: str,
    locator: str,
    window: int = CITATION_WINDOW,
) -> bool:
    """True when document and locator occur together as one citation."""
    if not isinstance(answer, str) or not document or not locator:
        return False
    haystack = _collapse(answer)
    doc = _collapse(document)
    loc = _collapse(locator)
    if not doc or not loc:
        return False
    start = 0
    while True:
        idx = haystack.find(loc, start)
        if idx < 0:
            return False
        end = idx + len(loc)
        before = haystack[idx - 1] if idx else ""
        after = haystack[end] if end < len(haystack) else ""
        bounded = (not before or not before.isalnum()) and (not after or not after.isalnum())
        if bounded:
            left = max(0, idx - window)
            right = min(len(haystack), end + window)
            if doc in haystack[left:right]:
                return True
        start = idx + 1


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("questions") if isinstance(raw, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} has no questions")
    seen: set[str] = set()
    questions: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{path} has a question that is not an object")
        item = {
            "id": row.get("id"),
            "question": row.get("question"),
            "document": row.get("document"),
            "locator": row.get("locator"),
        }
        if any(not isinstance(value, str) or not value.strip() for value in item.values()):
            raise ValueError(f"{path} has a question missing id, question, document, or locator")
        if item["id"] in seen:
            raise ValueError(f"duplicate golden question id {item['id']}")
        seen.add(item["id"])
        questions.append({key: value.strip() for key, value in item.items()})
    return questions


def space_answer_url(space_id: str) -> str:
    if not SPACE_ID_RE.fullmatch(space_id):
        raise ValueError(f"invalid space id {space_id!r}")
    host = space_id.lower().replace("/", "-")
    return f"https://{host}.hf.space/api/golden-question"


def _budget() -> tuple[int, int, int]:
    attempts = int(os.environ.get("GOLDEN_QUESTION_ATTEMPTS", QUESTION_ATTEMPTS_DEFAULT))
    max_time = int(os.environ.get("GOLDEN_CURL_MAX_TIME", CURL_MAX_TIME_DEFAULT))
    delay = int(os.environ.get("GOLDEN_RETRY_DELAY", RETRY_DELAY_DEFAULT))
    if attempts < 1 or max_time < 1 or delay < 0:
        raise ValueError("golden question budget must be positive")
    return attempts, max_time, delay


def post_with_curl(url: str, question: str, max_time: int) -> tuple[int, int, str]:
    """POST one question via curl. Returns (curl_exit, http_status, body)."""
    with tempfile.TemporaryDirectory() as tmp:
        payload_path = Path(tmp) / "payload.json"
        body_path = Path(tmp) / "body.json"
        payload_path.write_text(json.dumps({"question": question}), encoding="utf-8")
        result = subprocess.run(
            [
                "curl",
                "-sS",
                "--max-time",
                str(max_time),
                "-H",
                "Content-Type: application/json",
                "-H",
                "Accept: application/json",
                "-o",
                str(body_path),
                "-w",
                "%{http_code}",
                "-X",
                "POST",
                "--data-binary",
                f"@{payload_path}",
                url,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        body = body_path.read_text(encoding="utf-8") if body_path.exists() else ""
    if result.returncode != 0:
        detail = (result.stderr or "").strip()
        if detail:
            print(f"[golden] curl exit {result.returncode}: {detail}", file=sys.stderr)
        return result.returncode, 0, body
    code_text = result.stdout.strip()
    if not code_text.isdigit():
        print(f"[golden] curl returned no HTTP status: {code_text!r}", file=sys.stderr)
        return 1, 0, body
    return 0, int(code_text), body


def _answer_from_body(body: str) -> str | None:
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    answer = parsed.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return None
    return answer


def _show(answer: str) -> str:
    shown = " ".join(answer.split())
    if len(shown) > 400:
        return shown[:400] + "..."
    return shown


def run(space_id: str, post=None) -> int:
    questions = load_questions()
    url = space_answer_url(space_id)
    attempts, max_time, delay = _budget()
    send = post or post_with_curl
    for item in questions:
        print(f"[golden] asking {item['id']}", flush=True)
        answer: str | None = None
        for attempt in range(1, attempts + 1):
            curl_exit, status, body = send(url, item["question"], max_time)
            if curl_exit == 0 and status == 200:
                answer = _answer_from_body(body)
                if answer is None:
                    print(f"[golden] {item['id']} response was not an answer", file=sys.stderr)
                    return 1
                break
            retryable = curl_exit != 0 or status in RETRY_STATUSES
            if retryable and attempt < attempts:
                print(
                    f"[golden] {item['id']} attempt {attempt} failed "
                    f"(curl {curl_exit}, HTTP {status}); retrying",
                    flush=True,
                )
                if delay:
                    time.sleep(delay)
                continue
            print(
                f"[golden] {item['id']} failed (curl {curl_exit}, HTTP {status})",
                file=sys.stderr,
            )
            return 1
        if answer is None or not citation_found(answer, item["document"], item["locator"]):
            print(
                f"[golden] {item['id']} missed citation "
                f"{item['document']} / {item['locator']}",
                file=sys.stderr,
            )
            if answer is not None:
                print(f"[golden] answer: {_show(answer)}", file=sys.stderr)
            return 1
        print(f"[golden] passed {item['id']}", flush=True)
    print(f"[golden] {len(questions)} questions cited the expected clauses", flush=True)
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: golden_questions.py <space_id>", file=sys.stderr)
        return 1
    try:
        return run(argv[1])
    except (OSError, ValueError) as exc:
        print(f"[golden] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
