"""Golden-question gate: committed citations, and no promotion on a miss."""

import importlib.util
import json
import re
import secrets
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
AGREEMENT = REPO_ROOT / "app" / "data" / "01_primary" / "BCGEU_20th_Main_Agreement.md"
QUESTIONS = REPO_ROOT / ".github" / "golden-questions.json"
MERGE = REPO_ROOT / ".github" / "workflows" / "merge.yml"
PR = REPO_ROOT / ".github" / "workflows" / "pr.yml"
GATE = REPO_ROOT / ".github" / "scripts" / "golden_questions.py"


def _gate():
    spec = importlib.util.spec_from_file_location("golden_questions", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _headings(markdown: str) -> set[str]:
    found = set()
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            found.add(stripped.lstrip("#").strip())
    return found


def test_questions_cite_20th_main_agreement_headings():
    payload = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    rows = payload["questions"]
    assert 1 <= len(rows) <= 5
    headings = _headings(AGREEMENT.read_text(encoding="utf-8"))
    ids = set()
    for row in rows:
        assert row["document"] == "BCGEU 20th Main Agreement"
        assert row["locator"] in headings
        assert row["locator"].casefold() not in row["question"].casefold()
        assert row["document"].casefold() not in row["question"].casefold()
        assert "19th" not in row["question"].casefold()
        assert row["id"] not in ids
        ids.add(row["id"])


def test_merge_blocks_prod_and_opens_a_github_issue():
    merge = MERGE.read_text(encoding="utf-8")
    pr = PR.read_text(encoding="utf-8")
    header, jobs = merge.split("jobs:", 1)
    deploy_test = jobs.split("  deploy-test:", 1)[1].split("  deploy-prod:", 1)[0]
    assert header.count("group: huggingface-test-space") == 1
    assert "cancel-in-progress: false" in header
    assert "group: huggingface-test-space" not in jobs
    assert "timeout-minutes: 40" in deploy_test
    assert "timeout-minutes: 15" in deploy_test
    assert "steps.golden.outcome == 'failure'" in deploy_test
    assert "run: ./.github/scripts/golden_questions.sh bcgeu/navigator-test" in deploy_test
    assert "secrets.GOLDEN_QUESTION_TOKEN" in deploy_test
    wrapper = (GATE.parent / "golden_questions.sh").read_text(encoding="utf-8")
    gate_text = GATE.read_text(encoding="utf-8")
    assert re.search(r"GOLDEN_QUESTION_TOKEN\s*=", gate_text + wrapper + merge) is None
    assert "Golden Question Failure: Agreement Navigator (AgNav)" in deploy_test
    assert "bcgov/actions/workflow-notifier@" in deploy_test
    assert "secrets.GITHUB_TOKEN" in deploy_test
    assert "  golden-questions:" not in merge
    assert "needs: [deploy-test]" in merge.split("  deploy-prod:", 1)[1]
    assert "needs: [deploy-test, golden-questions]" not in merge
    assert "huggingface-prod-space" in merge
    assert merge.count("cancel-in-progress: false") == 2
    lowered = merge.lower()
    assert "slack" not in lowered
    assert "smtp" not in lowered
    assert "mailto:" not in lowered
    assert "golden_questions.sh" not in pr
    assert "tests/golden_questions/offline.sh" in pr


def test_citation_window_pass_and_miss():
    gate = _gate()
    cited = "[BCGEU 20th Main Agreement - 10.1 Burden of Proof]"
    assert gate.citation_found(cited, "BCGEU 20th Main Agreement", "10.1 Burden of Proof")
    assert not gate.citation_found(
        "The Employer bears the burden.",
        "BCGEU 20th Main Agreement",
        "10.1 Burden of Proof",
    )
    far = "BCGEU 20th Main Agreement " + ("x" * (gate.CITATION_WINDOW + 5)) + " 10.1 Burden of Proof"
    assert not gate.citation_found(far, "BCGEU 20th Main Agreement", "10.1 Burden of Proof")
    assert not gate.citation_found(
        "[Other Agreement - 10.1 Burden of Proof]",
        "BCGEU 20th Main Agreement",
        "10.1 Burden of Proof",
    )


def test_gate_pass_fail_and_retry_with_fake_responder(monkeypatch):
    gate = _gate()
    monkeypatch.setenv("GOLDEN_RETRY_DELAY", "0")
    questions = gate.load_questions()

    def good_post(url, question, max_time):
        item = next(row for row in questions if row["question"] == question)
        body = json.dumps({"answer": f"[{item['document']} - {item['locator']}]"})
        return 0, 200, body

    assert gate.run("bcgeu/navigator-test", post=good_post) == 0

    calls = {"n": 0}

    def miss_post(url, question, max_time):
        calls["n"] += 1
        return 0, 200, json.dumps({"answer": "The Employer bears the burden."})

    assert gate.run("bcgeu/navigator-test", post=miss_post) == 1
    assert calls["n"] == 1

    state = {"n": 0}

    def retry_post(url, question, max_time):
        state["n"] += 1
        if state["n"] == 1:
            return 0, 503, ""
        return good_post(url, question, max_time)

    assert gate.run("bcgeu/navigator-test", post=retry_post) == 0
    assert state["n"] == len(questions) + 1

    down = {"n": 0}

    def down_post(url, question, max_time):
        down["n"] += 1
        return 0, 503, ""

    assert gate.run("bcgeu/navigator-test", post=down_post) == 1
    assert down["n"] == 2


def test_curl_sends_the_token_header_and_refuses_when_unset(monkeypatch):
    gate = _gate()
    rejected = (None, " ", " x", "x ", " x ", "\n", "\r", "bad\r\n")
    for value in rejected:
        if value is None:
            monkeypatch.delenv("GOLDEN_QUESTION_TOKEN", raising=False)
        else:
            monkeypatch.setenv("GOLDEN_QUESTION_TOKEN", value)
        try:
            gate.post_with_curl("https://example.test/api/golden-question", "q", 1)
        except ValueError as exc:
            assert str(exc) == "GOLDEN_QUESTION_TOKEN is unset or invalid"
        else:
            raise AssertionError("curl ran with a rejected token")

    token = secrets.token_hex(16)
    monkeypatch.setenv("GOLDEN_QUESTION_TOKEN", token)
    captured: dict[str, object] = {}

    def fake_run(args, **kwargs):
        captured["args"] = [str(arg) for arg in args]
        header_text = ""
        header_mode = None
        for index, arg in enumerate(args):
            if arg not in ("-H", "--header"):
                continue
            value = str(args[index + 1])
            if not value.startswith("@"):
                continue
            path = Path(value[1:])
            text = path.read_text(encoding="utf-8")
            if "X-Golden-Question-Token:" in text:
                header_text = text
                header_mode = path.stat().st_mode & 0o777
        captured["header"] = header_text
        captured["mode"] = header_mode
        body_path = args[args.index("-o") + 1]
        Path(body_path).write_text('{"answer":"x"}', encoding="utf-8")

        class Result:
            returncode = 0
            stdout = "200"
            stderr = ""

        return Result()

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    exit_code, status, body = gate.post_with_curl(
        "https://example.test/api/golden-question", "q", 5
    )
    assert (exit_code, status, body) == (0, 200, '{"answer":"x"}')
    assert captured["header"] == f"X-Golden-Question-Token: {token}\n"
    assert captured["mode"] == 0o600
    assert all(token not in arg for arg in captured["args"])
    assert token not in body
