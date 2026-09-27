#!/bin/bash
# Offline health-probe cases. Run inside a container with no network:
#   podman run --rm --network=none -v "$PWD:/src:ro" -w /tmp \
#     docker.io/library/python:3.12-slim bash /src/tests/probe_health/offline.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROBE="$ROOT/.github/scripts/probe_health.sh"
VERIFY="$ROOT/.github/scripts/verify_deployment.sh"
WORKFLOW="$ROOT/.github/workflows/health-probe.yml"
MERGE="$ROOT/.github/workflows/merge.yml"
DEPLOY="$ROOT/.github/scripts/deploy.sh"

fail() {
    echo "FAIL: $*"
    exit 1
}

pass() {
    echo "PASS: $*"
}

python3 - "$PROBE" "$VERIFY" "$WORKFLOW" "$MERGE" "$DEPLOY" <<'PY' || fail "budget check failed"
import pathlib, re, sys

probe, verify, workflow, merge, deploy = [pathlib.Path(p).read_text() for p in sys.argv[1:]]

def assign(text, name):
    match = re.search(rf"^{name}=(\S+)", text, re.M)
    if not match:
        raise SystemExit(f"missing {name}")
    return match.group(1)

def default_int(text, name):
    match = re.search(rf"^{name}=\"?\${{{name}:-(\d+)}}\"?", text, re.M)
    if not match:
        raise SystemExit(f"missing default for {name}")
    return int(match.group(1))

meta_attempts = int(assign(probe, "METADATA_ATTEMPTS"))
delays = [int(n) for n in re.search(r"^METADATA_RETRY_DELAYS=\(([^)]*)\)", probe, re.M).group(1).split()]
probe_max = default_int(probe, "PROBE_CURL_MAX_TIME")
health_attempts = int(assign(probe, "HEALTH_ATTEMPTS"))
health_interval = default_int(probe, "HEALTH_RETRY_INTERVAL")
smoke_budget = default_int(probe, "VERIFY_SMOKE_BUDGET")
api_max = default_int(verify, "VERIFY_API_MAX_TIME")
health_max = default_int(verify, "VERIFY_HEALTH_MAX_TIME")
if not re.search(r"^VERIFY_SMOKE_BUDGET=\${VERIFY_SMOKE_BUDGET:-}\s*$", verify, re.M):
    raise SystemExit("deploy smoke budget default is not empty")
if "MAX_RETRIES=12" not in verify:
    raise SystemExit("deploy smoke retries changed")
if "VERIFY_SMOKE_BUDGET" in merge or "VERIFY_SMOKE_BUDGET" in deploy:
    raise SystemExit("deploy path sets a smoke budget")
if not re.search(r'probe_health\.sh "\$SPACE_ID" 300', workflow):
    raise SystemExit("workflow deploy-wait argument is not 300")
if "timeout-minutes: 15" not in workflow or "timeout-minutes: 20" not in workflow:
    raise SystemExit("step timeout must stay 15 and job timeout 20")

documented = int(re.search(r"^# worst_case_seconds=(\d+)", probe, re.M).group(1))
worst = (
    meta_attempts * probe_max + sum(delays)
    + probe_max
    + 300 + api_max
    + smoke_budget
    + health_attempts * probe_max + (health_attempts - 1) * health_interval
)
print(
    f"worst_case metadata={meta_attempts * probe_max + sum(delays)} "
    f"wake={probe_max} deploy={300 + api_max} smoke={smoke_budget} "
    f"health={health_attempts * probe_max + (health_attempts - 1) * health_interval} "
    f"total={worst}"
)
if worst != documented:
    raise SystemExit(f"documented {documented} != computed {worst}")
if worst >= 900:
    raise SystemExit(f"worst case {worst}s is not under 900")
if health_max < 10:
    raise SystemExit("health max-time is below the app models.list timeout")
print(f"PASS: worst case {worst}s < 900; api max-time {api_max}s; health max-time {health_max}s")
PY

mkdir -p /tmp/bin
cat > /tmp/bin/curl <<'PY'
#!/usr/bin/env python3
import json, os, sys, time

args = sys.argv[1:]
url = None
max_time = None
write_out = None
output = None
fail_with_body = False
i = 0
no_arg = {"-s", "-S", "-sS", "-L", "--fail-with-body", "--location", "--silent", "--show-error"}
takes_arg = {"--max-time", "-m", "-w", "--write-out", "-o", "--output", "-H", "--header", "--retry", "--retry-delay", "--connect-timeout"}
while i < len(args):
    arg = args[i]
    if arg in no_arg:
        if arg == "--fail-with-body":
            fail_with_body = True
        i += 1
        continue
    if arg in takes_arg:
        value = args[i + 1]
        if arg in ("--max-time", "-m"):
            max_time = value
        elif arg in ("-w", "--write-out"):
            write_out = value
        elif arg in ("-o", "--output"):
            output = value
        i += 2
        continue
    if arg.startswith("-"):
        sys.stderr.write(f"fake curl: unhandled flag {arg}\n")
        sys.exit(2)
    url = arg
    i += 1

scenario = os.environ.get("SCENARIO", "")
log_path = os.environ["CURL_LOG"]
state_path = os.environ["CURL_STATE"]

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

def log(kind, exit_code):
    shown = "" if max_time is None else str(max_time)
    with open(log_path, "a") as handle:
        handle.write(f"{kind}\t{url}\tmax={shown}\texit={exit_code}\n")

def hang():
    if scenario != "hung":
        return
    time.sleep(30 if max_time is None else float(max_time))

def emit(kind, body, code, exit_code):
    log(kind, exit_code)
    if exit_code != 0:
        sys.exit(exit_code)
    if output is not None:
        with open(output, "w") as handle:
            handle.write(body)
    else:
        sys.stdout.write(body)
        if write_out is not None:
            sys.stdout.write(write_out.replace("%{http_code}", code))
    sys.exit(0)

SPECS = {
    "paid-sleeping": ("SLEEPING", {"current": "t4-small", "requested": "t4-small"}, 3600, "ok"),
    "paid-stopped": ("STOPPED", {"current": "t4-small", "requested": "t4-small"}, 3600, "ok"),
    "paid-running-sleep": ("RUNNING", {"current": "t4-small", "requested": "t4-small"}, 172800, "ok"),
    "pending-upgrade": ("RUNNING", {"current": "cpu-basic", "requested": "t4-small"}, 172800, "ok"),
    "unknown-sleeping": ("SLEEPING", {}, None, "ok"),
    "paid-running-nosleep": ("RUNNING", {"current": "t4-small", "requested": "t4-small"}, None, "ok"),
    "free-ok": ("RUNNING", {"current": "cpu-basic", "requested": "cpu-basic"}, 172800, "ok"),
    "free-503": ("RUNNING", {"current": "cpu-basic", "requested": "cpu-basic"}, 172800, "503"),
    "free-degraded": ("RUNNING", {"current": "cpu-basic", "requested": "cpu-basic"}, 172800, "degraded"),
    "free-nonjson": ("RUNNING", {"current": "cpu-basic", "requested": "cpu-basic"}, 172800, "nonjson"),
    "free-runtime-error": ("RUNTIME_ERROR", {"current": "cpu-basic", "requested": "cpu-basic"}, None, "ok"),
    "free-sleeping": ("SLEEPING", {"current": "cpu-basic", "requested": "cpu-basic"}, 172800, "ok"),
    "free-retry-ok": ("RUNNING", {"current": "cpu-basic", "requested": "cpu-basic"}, 172800, "timeout-then-ok"),
    "free-three-fail": ("RUNNING", {"current": "cpu-basic", "requested": "cpu-basic"}, 172800, "fail-three"),
    "smoke-empty": ("RUNNING", {"current": "cpu-basic", "requested": "cpu-basic"}, 172800, "ok"),
    "smoke-empty-immediate": ("RUNNING", {"current": "cpu-basic", "requested": "cpu-basic"}, 172800, "ok"),
    "smoke-exit-7": ("RUNNING", {"current": "cpu-basic", "requested": "cpu-basic"}, 172800, "ok"),
}

def spaces_body(stage, hardware, gc):
    runtime = {"stage": stage, "hardware": hardware}
    if gc is not None:
        runtime["gcTimeout"] = gc
    return json.dumps({"runtime": runtime})

def strict_response(mode):
    count = bump("strict")
    if mode == "timeout-then-ok":
        if count == 1:
            log("strict", 28)
            sys.exit(28)
        return '{"status":"ok"}', "200", 0
    if mode == "fail-three":
        if count == 2:
            log("strict", 28)
            sys.exit(28)
        return '{"status":"down"}', "503", 0
    if mode == "503":
        return '{"status":"down"}', "503", 0
    if mode == "degraded":
        return '{"status":"degraded"}', "200", 0
    if mode == "nonjson":
        return "not-json", "200", 0
    return '{"status":"ok"}', "200", 0

if scenario == "hung":
    hang()
    if "huggingface.co/api/spaces/" in url:
        count = bump("spaces")
        if count <= 3:
            log("spaces", 28)
            sys.exit(28)
        if count == 4:
            emit("spaces", spaces_body("SLEEPING", {"current": "cpu-basic", "requested": "cpu-basic"}, 172800), "200", 0)
        emit("spaces", spaces_body("RUNNING", {"current": "cpu-basic", "requested": "cpu-basic"}, 172800), "200", 0)
    if url.endswith("/api/health") and output is not None:
        emit("wake", "", "200", 0)
    log("health", 28)
    sys.exit(28)

stage, hardware, gc, mode = SPECS[scenario]
if "huggingface.co/api/spaces/" in url:
    emit("spaces", spaces_body(stage, hardware, gc), "200", 0)
if url.endswith("/api/health") and output is not None:
    emit("wake", "", "200", 0)
if url.endswith("/api/health") and fail_with_body and scenario == "smoke-empty":
    if max_time is not None:
        time.sleep(float(max_time))
    emit("smoke", "", "200", 0)
if url.endswith("/api/health") and fail_with_body and scenario == "smoke-empty-immediate":
    emit("smoke", "", "200", 0)
if url.endswith("/api/health") and fail_with_body and scenario == "smoke-exit-7":
    log("smoke", 7)
    sys.exit(7)
if url.endswith("/api/health") and fail_with_body:
    emit("smoke", '{"status":"ok"}', "200", 0)
if url.endswith("/api/health") and write_out is not None:
    body, code, exit_code = strict_response(mode)
    if fail_with_body and int(code) >= 400:
        log("strict", 22)
        sys.exit(22)
    emit("strict", body, code, exit_code)

sys.stderr.write(f"fake curl: unhandled url {url}\n")
sys.exit(2)
PY
chmod +x /tmp/bin/curl

run_case() {
    local name="$1"
    local scenario="$2"
    local timeout="$3"
    local expect_rc="$4"
    local expect_health="$5"
    local poll_interval="${6:-0}"
    local work
    work="$(mktemp -d)"
    local log="$work/curl.log"
    local out="$work/out.txt"
    : >"$log"
    set +e
    SCENARIO="$scenario" \
        CURL_LOG="$log" \
        CURL_STATE="$work/state.json" \
        PATH="/tmp/bin:${PATH}" \
        VERIFY_POLL_INTERVAL="$poll_interval" \
        HEALTH_RETRY_INTERVAL=0 \
        bash "$PROBE" "bcgeu/navigator-test" "$timeout" >"$out" 2>&1
    local rc=$?
    set -e
    if [ "$rc" -ne "$expect_rc" ]; then
        echo "---- output ----"
        cat "$out"
        fail "$name: exit $rc, expected $expect_rc"
    fi
    local health_count
    health_count="$(grep -c '/api/health' "$log" || true)"
    if [ "$expect_health" = "yes" ]; then
        local strict_count
        strict_count="$(grep -c $'^strict\t' "$log" || true)"
        if [ "$strict_count" -eq 0 ]; then
            cat "$log"
            fail "$name: expected a strict /api/health check"
        fi
    fi
    if [ "$expect_health" = "no" ] && [ "$health_count" -ne 0 ]; then
        cat "$log"
        fail "$name: /api/health was called"
    fi
    if [ -s "$log" ] && grep -Ev $'\tmax=[0-9]+(\.[0-9]+)?\texit=' "$log" >/dev/null; then
        cat "$log"
        fail "$name: a curl ran without --max-time"
    fi
    CASE_LOG="$log"
    pass "$name"
}

set -euo pipefail

run_case "paid, SLEEPING" paid-sleeping 5 0 no
run_case "paid, STOPPED" paid-stopped 5 0 no
run_case "paid, RUNNING, sleep time set" paid-running-sleep 5 0 no
run_case "current cpu-basic, requested paid, RUNNING, sleep time set" pending-upgrade 5 0 no
run_case "unknown hardware, SLEEPING" unknown-sleeping 5 0 no
run_case "paid, RUNNING, never sleeps" paid-running-nosleep 5 0 yes
run_case "free, RUNNING, status ok" free-ok 5 0 yes
run_case "free, RUNNING, HTTP 503" free-503 5 1 yes
run_case "free, RUNNING, status degraded" free-degraded 5 1 yes
run_case "free, RUNNING, non-JSON body" free-nonjson 5 1 yes
run_case "free, RUNTIME_ERROR" free-runtime-error 5 1 no
# Wake-up only. expect_health=yes requires a strict line; this case checks that below.
run_case "free, SLEEPING, never reaches RUNNING" free-sleeping 2 1 wake 1
wake_count="$(grep -c $'^wake\t' "$CASE_LOG" || true)"
strict_count="$(grep -c $'^strict\t' "$CASE_LOG" || true)"
if [ "$wake_count" -ne 1 ] || [ "$strict_count" -ne 0 ]; then
    cat "$CASE_LOG"
    fail "free sleeping: wake=$wake_count strict=$strict_count"
fi
pass "free sleeping called only the wake-up /api/health"

run_case "free, RUNNING, first health attempt times out" free-retry-ok 5 0 yes
strict_count="$(grep -c $'^strict\t' "$CASE_LOG" || true)"
if [ "$strict_count" -ne 2 ]; then
    cat "$CASE_LOG"
    fail "retry-ok strict attempts=$strict_count, expected 2"
fi
pass "retry-ok stopped after the second strict check"

run_case "free, RUNNING, three health attempts fail" free-three-fail 5 1 yes
strict_count="$(grep -c $'^strict\t' "$CASE_LOG" || true)"
if [ "$strict_count" -ne 3 ]; then
    cat "$CASE_LOG"
    fail "three-fail strict attempts=$strict_count, expected 3"
fi
pass "three-fail used 3 strict checks"

hung_work="$(mktemp -d)"
hung_log="$hung_work/curl.log"
hung_out="$hung_work/out.txt"
: >"$hung_log"
hung_start="$(date +%s)"
set +e
SCENARIO=hung \
    CURL_LOG="$hung_log" \
    CURL_STATE="$hung_work/state.json" \
    PATH="/tmp/bin:${PATH}" \
    PROBE_CURL_MAX_TIME=1 \
    VERIFY_API_MAX_TIME=1 \
    VERIFY_HEALTH_MAX_TIME=1 \
    VERIFY_SMOKE_BUDGET=3 \
    VERIFY_POLL_INTERVAL=0 \
    HEALTH_RETRY_INTERVAL=0 \
    bash "$PROBE" "bcgeu/navigator-test" 2 >"$hung_out" 2>&1
hung_rc=$?
set -e
hung_elapsed="$(( $(date +%s) - hung_start ))"
# Same sum as the production budget, with the scaled ceilings above:
# metadata 4*1+(1+2+4)=11, wake 1, deploy 2+1=3, smoke 3, health 3*1+2*0=3.
hung_ceiling=21
if [ "$hung_rc" -eq 0 ]; then
    cat "$hung_out"
    fail "hung request returned success"
fi
if [ "$hung_elapsed" -gt $((hung_ceiling + 15)) ]; then
    echo "---- hung output ----"
    cat "$hung_out"
    cat "$hung_log"
    fail "hung request took ${hung_elapsed}s, ceiling is ${hung_ceiling}s"
fi
if [ "$hung_elapsed" -lt 12 ]; then
    cat "$hung_log"
    fail "hung request finished in ${hung_elapsed}s, so a stall was not actually waited out"
fi
if grep -Ev $'\tmax=[0-9]+(\.[0-9]+)?\texit=' "$hung_log" >/dev/null; then
    cat "$hung_log"
    fail "hung path issued a curl without --max-time"
fi
pass "hung request finished in ${hung_elapsed}s (ceiling ${hung_ceiling}s, slack 15s)"

run_verify_budget() {
    local name="$1"
    local scenario="$2"
    local budget="$3"
    local expect_report="$4"
    local retry_interval="${5:-}"
    local expect_smokes="${6:-}"
    local work out
    work="$(mktemp -d)"
    out="$work/out.txt"
    local log="$work/curl.log"
    local -a env_args=(
        "SCENARIO=$scenario"
        "CURL_LOG=$log"
        "CURL_STATE=$work/state.json"
        "PATH=/tmp/bin:${PATH}"
        "VERIFY_SMOKE_BUDGET=$budget"
        "VERIFY_POLL_INTERVAL=0"
    )
    if [ -n "$retry_interval" ]; then
        env_args+=("VERIFY_SMOKE_RETRY_INTERVAL=$retry_interval")
    fi
    set +e
    env "${env_args[@]}" bash "$VERIFY" "bcgeu/navigator-test" 5 >"$out" 2>&1
    local rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then
        cat "$out"
        fail "$name: verify succeeded"
    fi
    if ! grep -F "curl returned: ${expect_report}." "$out" >/dev/null; then
        cat "$out"
        fail "$name: missing curl returned: ${expect_report}"
    fi
    if [ "$expect_report" != "28" ] && grep -F "curl returned: 28." "$out" >/dev/null; then
        cat "$out"
        fail "$name: invented curl exit 28"
    fi
    if [ "$expect_report" != "0" ] && grep -F "curl returned: 0." "$out" >/dev/null; then
        cat "$out"
        fail "$name: reported curl exit 0 for a failed smoke check"
    fi
    if [ -n "$expect_smokes" ]; then
        local smoke_count
        smoke_count="$(grep -c $'^smoke\t' "$log" || true)"
        if [ "$smoke_count" -ne "$expect_smokes" ]; then
            cat "$log"
            fail "$name: smoke attempts=$smoke_count, expected $expect_smokes"
        fi
    fi
    pass "$name"
}

run_verify_budget "smoke budget exhausted before a curl" smoke-exit-7 0 budget-exhausted
run_verify_budget "smoke budget keeps the last real curl exit" smoke-exit-7 1 7
run_verify_budget "empty 200 at the smoke budget does not report curl exit 0" smoke-empty 1 budget-exhausted
run_verify_budget "smoke retries exhausted on an empty 200" smoke-empty-immediate "" empty-200 0 12

echo "All offline probe cases passed."
