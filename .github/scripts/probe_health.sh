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
RUNTIME=$(echo "$SPACE_JSON" | python3 -c "
import sys, json
r = json.load(sys.stdin).get('runtime') or {}
h = r.get('hardware') or {}
print(str(r.get('stage') or 'UNKNOWN').upper(), h.get('current') or h.get('requested') or 'unknown', r.get('gcTimeout') or 'none')
")
read -r STAGE HARDWARE SLEEP_TIMEOUT <<< "$RUNTIME"
echo "[probe] stage=$STAGE hardware=$HARDWARE sleep_timeout_seconds=$SLEEP_TIMEOUT"

# A request to a paid Space wakes it (if asleep) or resets its sleep timer (if awake), which bills hardware time.
if [ "$HARDWARE" != "$FREE_HARDWARE" ]; then
    if [ "$STAGE" == "SLEEPING" ]; then
        echo "✅ $SPACE_ID is asleep on paid hardware '$HARDWARE'. Treating as healthy without calling /api/health."
        exit 0
    fi
    if [ "$STAGE" == "RUNNING" ] && [ "$SLEEP_TIMEOUT" != "none" ]; then
        echo "✅ $SPACE_ID is running on paid hardware '$HARDWARE' with a ${SLEEP_TIMEOUT}s sleep time. Skipping /api/health so the probe does not keep it awake."
        exit 0
    fi
fi

if [ "$STAGE" == "SLEEPING" ]; then
    echo "[probe] $SPACE_ID is asleep on free hardware '$HARDWARE'. Sending a wake-up request..."
    WAKE_EXIT=0
    curl -s -o /dev/null --max-time 30 "$SPACE_URL/api/health" || WAKE_EXIT=$?
    echo "[probe] Wake-up request sent (curl exit code: $WAKE_EXIT). Waiting for the Space to start..."
fi

bash "$SCRIPT_DIR/verify_deployment.sh" "$SPACE_ID" "$TIMEOUT_SECONDS"

echo "[probe] Checking $SPACE_URL/api/health for HTTP 200 and \"status\": \"ok\"..."
CURL_EXIT=0
RESPONSE=$(curl -sS --max-time 30 -w $'\n%{http_code}' "$SPACE_URL/api/health") || CURL_EXIT=$?
if [ $CURL_EXIT -ne 0 ]; then
    echo "❌ Error: /api/health request failed (curl exit code: $CURL_EXIT)."
    exit 1
fi

HTTP_STATUS="${RESPONSE##*$'\n'}"
BODY="${RESPONSE%$'\n'*}"
HEALTH_STATUS=$(echo "$BODY" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('status', '') if isinstance(d, dict) else '')" 2>/dev/null || echo "")

if [ "$HTTP_STATUS" != "200" ] || [ "$HEALTH_STATUS" != "ok" ]; then
    echo "❌ Error: $SPACE_ID is unhealthy. HTTP $HTTP_STATUS, body: $BODY"
    exit 1
fi

echo "✅ $SPACE_ID is healthy. HTTP $HTTP_STATUS, body: $BODY"
