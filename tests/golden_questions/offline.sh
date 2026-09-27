#!/bin/bash
# Offline golden-question cases. Run inside a container with no network:
#   podman run --rm --network=none -v "$PWD:/src:ro" -w /tmp \
#     docker.io/library/python:3.14-slim bash /src/tests/golden_questions/offline.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GATE="$ROOT/.github/scripts/golden_questions.py"
WRAPPER="$ROOT/.github/scripts/golden_questions.sh"
QUESTIONS="$ROOT/.github/golden-questions.json"
MERGE="$ROOT/.github/workflows/merge.yml"
PR="$ROOT/.github/workflows/pr.yml"
AGREEMENT="$ROOT/app/data/01_primary/BCGEU_20th_Main_Agreement.md"

fail() {
    echo "FAIL: $*"
    exit 1
}

pass() {
    echo "PASS: $*"
}

python3 - "$GATE" "$WRAPPER" "$QUESTIONS" "$MERGE" "$PR" "$AGREEMENT" <<'PY' || fail "static checks failed"
import json, pathlib, re, sys

gate, wrapper, questions, merge, pr, agreement = [pathlib.Path(p).read_text() for p in sys.argv[1:]]

def assign(name):
    match = re.search(rf'^{name} = "(\d+)"', gate, re.M)
    if not match:
        raise SystemExit(f"missing {name}")
    return int(match.group(1))

attempts = assign("QUESTION_ATTEMPTS_DEFAULT")
max_time = assign("CURL_MAX_TIME_DEFAULT")
delay = assign("RETRY_DELAY_DEFAULT")
count = len(json.loads(questions)["questions"])
worst = count * (attempts * max_time + (attempts - 1) * delay)
documented = int(re.search(r"^# worst_case_seconds=(\d+)", gate, re.M).group(1))
if worst != documented:
    raise SystemExit(f"documented {documented} != computed {worst}")
if worst >= 900:
    raise SystemExit(f"worst case {worst}s is not under 900")

deploy_test = merge.split("  deploy-test:", 1)[1].split("  deploy-prod:", 1)[0]
if deploy_test.count("group: huggingface-test-space") != 1:
    raise SystemExit("TEST space lock is not a single group on deploy-test")
if "cancel-in-progress: false" not in deploy_test:
    raise SystemExit("TEST promotion cancels an in-progress run")
if deploy_test.index("group: huggingface-test-space") > deploy_test.index("golden_questions.sh bcgeu/navigator-test"):
    raise SystemExit("golden questions run outside the TEST space lock")
if "timeout-minutes: 15" not in deploy_test or "timeout-minutes: 40" not in deploy_test:
    raise SystemExit("question step timeout must stay 15 and the deploy-test job 40")
if "steps.golden.outcome == 'failure'" not in deploy_test:
    raise SystemExit("golden failure does not have its own notifier")
if "bcgov/actions/workflow-notifier@" not in deploy_test or "secrets.GITHUB_TOKEN" not in deploy_test:
    raise SystemExit("golden failure does not use workflow-notifier")
if "Golden Question Failure: Agreement Navigator (AgNav)" not in deploy_test:
    raise SystemExit("golden failure title changed")
if "  golden-questions:" in merge:
    raise SystemExit("golden questions are a separate job, so the TEST lock is released first")
prod = merge.split("  deploy-prod:", 1)[1]
if "needs: [deploy-test]" not in prod or "golden-questions" in prod.split("steps:", 1)[0]:
    raise SystemExit("prod does not wait on the deploy-test job that runs the gate")
if "cancel-in-progress: false" not in prod:
    raise SystemExit("PROD promotion cancels an in-progress run")
lowered = merge.lower()
for banned in ("slack", "smtp", "mailto:"):
    if banned in lowered:
        raise SystemExit(f"merge.yml contains {banned}")
if "golden_questions.sh" in pr:
    raise SystemExit("PR workflow calls the live golden-question script")

rows = json.loads(questions)["questions"]
if not rows or len(rows) > 5:
    raise SystemExit("question set is empty or no longer small")
headings = set()
for line in agreement.splitlines():
    stripped = line.strip()
    if stripped.startswith("#"):
        headings.add(stripped.lstrip("#").strip())
for row in rows:
    if row["document"] != "BCGEU 20th Main Agreement":
        raise SystemExit(f"{row['id']} does not cite the 20th main agreement")
    if row["locator"] not in headings:
        raise SystemExit(f"{row['id']} locator is not a heading in the 20th agreement")
    question = row["question"].casefold()
    if row["locator"].casefold() in question or row["document"].casefold() in question:
        raise SystemExit(f"{row['id']} question already contains its citation")
    if "19th" in question:
        raise SystemExit(f"{row['id']} mentions the 19th agreement")

combined = gate + wrapper
for banned in ("/pause", "/restart", "api/spaces", "slack", "smtp", "mailto:"):
    if banned in combined.lower():
        raise SystemExit(f"gate contains {banned}")
print(f"PASS: worst case {worst}s < 900; {count} questions cite 20th agreement headings")
PY

mkdir -p /tmp/bin
cat > /tmp/bin/curl <<'PY'
#!/usr/bin/env python3
import json, os, sys

args = sys.argv[1:]
url = None
max_time = None
output = None
data_path = None
i = 0
no_arg = {"-s", "-S", "-sS", "--silent", "--show-error", "--location"}
takes_arg = {
    "--max-time", "-m", "-w", "--write-out", "-o", "--output",
    "-H", "--header", "-X", "--request", "--data-binary", "--data",
}
while i < len(args):
    arg = args[i]
    if arg in no_arg:
        i += 1
        continue
    if arg in takes_arg:
        value = args[i + 1]
        if arg in ("--max-time", "-m"):
            max_time = value
        elif arg in ("-o", "--output"):
            output = value
        elif arg in ("--data-binary", "--data"):
            data_path = value[1:] if value.startswith("@") else None
        i += 2
        continue
    if arg.startswith("-"):
        sys.stderr.write(f"fake curl: unhandled flag {arg}\n")
        sys.exit(2)
    url = arg
    i += 1

log_path = os.environ["CURL_LOG"]
state_path = os.environ["CURL_STATE"]
scenario = os.environ.get("SCENARIO", "")

def load_state():
    if not os.path.exists(state_path):
        return {}
    with open(state_path) as handle:
        return json.load(handle)

def save_state(state):
    with open(state_path, "w") as handle:
        json.dump(state, handle)

def bump(key):
    state = load_state()
    state[key] = state.get(key, 0) + 1
    save_state(state)
    return state[key]

def log(exit_code, status):
    shown = "" if max_time is None else str(max_time)
    with open(log_path, "a") as handle:
        handle.write(f"{url}\tmax={shown}\tstatus={status}\texit={exit_code}\n")

if not data_path:
    sys.stderr.write("fake curl: missing body\n")
    sys.exit(2)
with open(data_path) as handle:
    question = json.load(handle)["question"]

rows = json.loads(open(os.environ["QUESTIONS_FILE"]).read())["questions"]
item = next(row for row in rows if row["question"] == question)
order = [row["question"] for row in rows]

def emit(body, status, exit_code):
    log(exit_code, status)
    if exit_code != 0:
        sys.exit(exit_code)
    if output is not None:
        with open(output, "w") as handle:
            handle.write(body)
    sys.stdout.write(str(status))
    sys.exit(0)

def good():
    return json.dumps({"answer": f"[{item['document']} - {item['locator']}]"})

count = bump(question)
if scenario == "curl-retry" and question == order[0] and count == 1:
    log(28, 0)
    sys.exit(28)
if scenario == "retry-then-pass" and count == 1:
    emit("", 503, 0)
if scenario == "transport-down":
    emit("", 503, 0)
if scenario == "missing-locator" and question == order[0]:
    emit(json.dumps({"answer": f"See {item['document']}."}), 200, 0)
if scenario == "document-far" and question == order[0]:
    gap = "x" * 300
    emit(json.dumps({"answer": f"{item['document']} {gap} {item['locator']}"}), 200, 0)
if scenario == "wrong-document" and question == order[0]:
    emit(json.dumps({"answer": f"[Other Agreement - {item['locator']}]"}), 200, 0)
if scenario == "bad-body" and question == order[0]:
    emit('{"nope": true}', 200, 0)
if scenario == "http-400" and question == order[0]:
    emit('{"detail":"invalid input"}', 400, 0)
emit(good(), 200, 0)
PY
chmod +x /tmp/bin/curl

run_case() {
    local name="$1"
    local scenario="$2"
    local expect_rc="$3"
    local expect_curls="$4"
    local work log out
    work="$(mktemp -d)"
    log="$work/curl.log"
    out="$work/out.txt"
    : >"$log"
    set +e
    SCENARIO="$scenario" \
        CURL_LOG="$log" \
        CURL_STATE="$work/state.json" \
        QUESTIONS_FILE="$QUESTIONS" \
        PATH="/tmp/bin:${PATH}" \
        GOLDEN_RETRY_DELAY=0 \
        bash "$WRAPPER" "bcgeu/navigator-test" >"$out" 2>&1
    local rc=$?
    set -e
    if [ "$rc" -ne "$expect_rc" ]; then
        echo "---- output ----"
        cat "$out"
        fail "$name: exit $rc, expected $expect_rc"
    fi
    local curls
    curls="$(grep -c $'\tmax=' "$log" || true)"
    if [ "$curls" -ne "$expect_curls" ]; then
        cat "$log"
        cat "$out"
        fail "$name: curl count $curls, expected $expect_curls"
    fi
    if grep -Ev $'\tmax=[0-9]+\tstatus=' "$log" >/dev/null; then
        cat "$log"
        fail "$name: a curl ran without --max-time"
    fi
    pass "$name"
}

question_count="$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["questions"]))' "$QUESTIONS")"

run_case "pass" pass 0 "$question_count"
run_case "missing locator" missing-locator 1 1
run_case "document outside citation window" document-far 1 1
run_case "wrong document" wrong-document 1 1
run_case "non-answer body" bad-body 1 1
run_case "HTTP 400" http-400 1 1
run_case "HTTP 503 then pass" retry-then-pass 0 "$((question_count * 2))"
run_case "HTTP 503 exhausted" transport-down 1 2
run_case "curl failure then pass" curl-retry 0 "$((question_count + 1))"

echo "All offline golden-question cases passed."
