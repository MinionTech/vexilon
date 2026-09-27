#!/bin/bash
# .github/scripts/probe_health.sh <space_id> [timeout_seconds]
# Read-only health probe for a Hugging Face Space: never deploys, restarts, or pushes.
# Usage: ./.github/scripts/probe_health.sh bcgeu/navigator-test 300

set -euo pipefail

SPACE_ID=${1:-}
TIMEOUT_SECONDS=${2:-300}
# The only Spaces hardware tier with no hourly price.
FREE_HARDWARE="cpu-basic"

if [ -z "$SPACE_ID" ]; then
    echo "Error: SPACE_ID argument missing."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPACE_URL="https://$(echo "$SPACE_ID" | tr '[:upper:]' '[:lower:]' | tr '/' '-').hf.space"

echo "[probe] Reading runtime for Hugging Face Space: $SPACE_ID"
SPACE_JSON=$(curl -sS --fail-with-body --max-time 30 --retry 3 --retry-all-errors "https://huggingface.co/api/spaces/$SPACE_ID")
# Paid unless every reported hardware value (current and requested) is free; a pending upgrade counts as paid.
RUNTIME=$(echo "$SPACE_JSON" | python3 -c "
import sys, json
r = json.load(sys.stdin).get('runtime') or {}
h = r.get('hardware') or {}
current, requested = h.get('current') or 'none', h.get('requested') or 'none'
paid = {current, requested} - {'none'} != {sys.argv[1]}
print(str(r.get('stage') or 'UNKNOWN').upper(), current + '/' + requested, str(paid).lower(), r.get('gcTimeout') or 'none')
" "$FREE_HARDWARE")
read -r STAGE HARDWARE PAID SLEEP_TIMEOUT <<< "$RUNTIME"
echo "[probe] stage=$STAGE hardware(current/requested)=$HARDWARE paid=$PAID sleep_timeout_seconds=$SLEEP_TIMEOUT"

# HF reports an idle Space as SLEEPING (free hardware) or STOPPED (paid hardware with a sleep time).
ASLEEP=false
case "$STAGE" in
    SLEEPING|STOPPED) ASLEEP=true ;;
esac

# A request to a paid Space wakes it (if asleep) or resets its sleep timer (if awake), which bills hardware time.
if [ "$PAID" == "true" ]; then
    if [ "$ASLEEP" == "true" ]; then
        echo "✅ $SPACE_ID is asleep on paid hardware '$HARDWARE'. Treating as healthy without calling /api/health."
        exit 0
    fi
    if [ "$STAGE" == "RUNNING" ] && [ "$SLEEP_TIMEOUT" != "none" ]; then
        echo "✅ $SPACE_ID is running on paid hardware '$HARDWARE' with a ${SLEEP_TIMEOUT}s sleep time. Skipping /api/health so the probe does not keep it awake."
        exit 0
    fi
fi

if [ "$ASLEEP" == "true" ]; then
    echo "[probe] $SPACE_ID is asleep on free hardware '$HARDWARE'. Sending a wake-up request..."
    WAKE_EXIT=0
    curl -s -o /dev/null --max-time 30 "$SPACE_URL/api/health" || WAKE_EXIT=$?
    echo "[probe] Wake-up request sent (curl exit code: $WAKE_EXIT). Waiting for the Space to start..."
fi

bash "$SCRIPT_DIR/verify_deployment.sh" "$SPACE_ID" "$TIMEOUT_SECONDS"

# Each attempt is a fresh request with a fresh body, so one network blip does not raise an issue.
HEALTH_ATTEMPTS=3
HEALTH_RETRY_INTERVAL=30

for i in $(seq 1 "$HEALTH_ATTEMPTS"); do
    echo "[probe] Checking $SPACE_URL/api/health for HTTP 200 and \"status\": \"ok\" (Attempt $i/$HEALTH_ATTEMPTS)..."
    CURL_EXIT=0
    RESPONSE=$(curl -sS --max-time 30 -w $'\n%{http_code}' "$SPACE_URL/api/health") || CURL_EXIT=$?
    if [ $CURL_EXIT -ne 0 ]; then
        echo "[probe] Attempt $i failed: /api/health request failed (curl exit code: $CURL_EXIT)."
    else
        HTTP_STATUS="${RESPONSE##*$'\n'}"
        BODY="${RESPONSE%$'\n'*}"
        HEALTH_STATUS=$(echo "$BODY" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('status', '') if isinstance(d, dict) else '')" 2>/dev/null || echo "")
        if [ "$HTTP_STATUS" == "200" ] && [ "$HEALTH_STATUS" == "ok" ]; then
            echo "✅ $SPACE_ID is healthy. HTTP $HTTP_STATUS, body: $BODY"
            exit 0
        fi
        echo "[probe] Attempt $i failed: HTTP $HTTP_STATUS, body: $BODY"
    fi
    if [ "$i" -lt "$HEALTH_ATTEMPTS" ]; then
        echo "[probe] Retrying in $HEALTH_RETRY_INTERVAL seconds..."
        sleep "$HEALTH_RETRY_INTERVAL"
    fi
done

echo "❌ Error: $SPACE_ID is unhealthy. /api/health failed $HEALTH_ATTEMPTS attempts."
exit 1
