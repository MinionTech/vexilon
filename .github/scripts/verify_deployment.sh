#!/bin/bash
# .github/scripts/verify_deployment.sh <space_id> [timeout_seconds]
# Polls Hugging Face API until Space status is 'running'.
# Usage: ./.github/scripts/verify_deployment.sh DerekRoberts/vexilon 600

set -eo pipefail

SPACE_ID=$1
TIMEOUT_SECONDS=${2:-600} # Default 10 minutes to maintain headroom under CI timeout
# One Spaces status call. A healthy body is a small JSON document and finishes
# inside 60s. A single timed-out poll is retried until TIMEOUT_SECONDS, so one
# slow call does not fail a deploy.
VERIFY_API_MAX_TIME=${VERIFY_API_MAX_TIME:-60}
# One /api/health call. The app aborts models.list after 10s, so a healthy
# body returns inside 30s. Attempts still retry while the server is down.
VERIFY_HEALTH_MAX_TIME=${VERIFY_HEALTH_MAX_TIME:-30}
# probe_health.sh sets this. Deploy leaves it empty and keeps all 12 attempts.
VERIFY_SMOKE_BUDGET=${VERIFY_SMOKE_BUDGET:-}
VERIFY_POLL_INTERVAL=${VERIFY_POLL_INTERVAL:-30}
INTERVAL=$VERIFY_POLL_INTERVAL

if [ -z "$SPACE_ID" ]; then
    echo "Error: SPACE_ID argument missing."
    exit 1
fi

echo "[verify] Monitoring Hugging Face Space: $SPACE_ID"
echo "[verify] Timeout: $TIMEOUT_SECONDS seconds"

START_TIME=$(date +%s)
END_TIME=$((START_TIME + TIMEOUT_SECONDS))

while [ "$(date +%s)" -lt "$END_TIME" ]; do
  # Use Bash array for safer argument handling
  CURL_ARGS=( -s -L --max-time "$VERIFY_API_MAX_TIME" )
  if [ -n "${HF_TOKEN:-}" ]; then
      CURL_ARGS+=( -H "Authorization: Bearer $HF_TOKEN" )
  fi

  # Temporarily disable -e to handle network errors during polling
  set +e
  # Capture both body and HTTP status code
  HTTP_RESPONSE=$(curl "${CURL_ARGS[@]}" -w "%{http_code}" "https://huggingface.co/api/spaces/$SPACE_ID")
  CURL_EXIT=$?
  set -e

  HTTP_STATUS="${HTTP_RESPONSE: -3}"
  STATUS_JSON="${HTTP_RESPONSE:0:${#HTTP_RESPONSE}-3}"

  if [ $CURL_EXIT -ne 0 ]; then
      echo "[verify] curl command failed (exit code $CURL_EXIT). Retrying in $INTERVAL seconds..."
      sleep "$INTERVAL"
      continue
  fi

  if [ "$HTTP_STATUS" != "200" ]; then
      echo "[verify] API returned HTTP $HTTP_STATUS. Body: $STATUS_JSON. Retrying..."
      sleep "$INTERVAL"
      continue
  fi

  # Check if we got a valid response (not empty)
  if [ -z "$STATUS_JSON" ]; then
      echo "[verify] Received empty response from API. Retrying..."
      sleep "$INTERVAL"
      continue
  fi

  # Robust status extraction
  CURRENT_STATUS=$(echo "$STATUS_JSON" | python3 -c "import sys, json; data=json.load(sys.stdin); print(str(data.get('runtime', {}).get('stage', 'unknown')).lower())" 2>/dev/null || echo "unknown")
  
  echo "[verify] Current status: $CURRENT_STATUS ($(($(date +%s) - START_TIME))s)"
  
  if [ "$CURRENT_STATUS" == "running" ]; then
    echo "✅ Success: Space $SPACE_ID is running!"
    break
  fi
  
  # Fail immediately on terminal error states
  case "$CURRENT_STATUS" in
      *crashed*|*error*|*failed*|*deleted*)
          echo "❌ Error: Space $SPACE_ID state is '$CURRENT_STATUS'."
          echo "Check logs at: https://huggingface.co/spaces/$SPACE_ID"
          exit 1
          ;;
  esac
  
  sleep "$INTERVAL"
done

# Check if we exited the loop because of success or timeout
if [ "$CURRENT_STATUS" != "running" ]; then
  echo "❌ Error: Timeout waiting for Space $SPACE_ID to become ready after $TIMEOUT_SECONDS seconds."
  exit 1
fi

# --- Functional Smoke Test ---
echo "[verify] 🔍 Running functional smoke test..."
SPACE_URL="https://$(echo "$SPACE_ID" | tr '[:upper:]' '[:lower:]' | tr '/' '-').hf.space"

# Since the server may take a few seconds to fully initialize even after the Space
# status reports 'running', we run the probe with a brief retry loop.
MAX_RETRIES=12
RETRY_INTERVAL=10
CURL_EXIT=0
HEALTH_JSON=""
SMOKE_DEADLINE=""
SMOKE_STOP=""
if [ -n "$VERIFY_SMOKE_BUDGET" ]; then
  SMOKE_DEADLINE=$(( $(date +%s) + VERIFY_SMOKE_BUDGET ))
fi

for i in $(seq 1 $MAX_RETRIES); do
  this_max=$VERIFY_HEALTH_MAX_TIME
  if [ -n "$SMOKE_DEADLINE" ]; then
    remaining=$((SMOKE_DEADLINE - $(date +%s)))
    if [ "$remaining" -le 0 ]; then
      echo "[verify] Smoke-test budget of ${VERIFY_SMOKE_BUDGET}s exhausted."
      SMOKE_STOP=budget-exhausted
      break
    fi
    if [ "$remaining" -lt "$this_max" ]; then
      this_max=$remaining
    fi
  fi

  echo "[verify] Querying /api/health (Attempt $i/$MAX_RETRIES)..."
  CURL_EXIT=0
  HEALTH_JSON=$(curl -s --fail-with-body --max-time "$this_max" "$SPACE_URL/api/health") || CURL_EXIT=$?
  
  if [ $CURL_EXIT -eq 0 ] && [ -n "$HEALTH_JSON" ]; then
    break
  fi

  sleep_for=$RETRY_INTERVAL
  if [ -n "$SMOKE_DEADLINE" ]; then
    remaining=$((SMOKE_DEADLINE - $(date +%s)))
    if [ "$remaining" -le 0 ]; then
      echo "[verify] Smoke-test budget of ${VERIFY_SMOKE_BUDGET}s exhausted."
      SMOKE_STOP=budget-exhausted
      break
    fi
    if [ "$remaining" -lt "$sleep_for" ]; then
      sleep_for=$remaining
    fi
  fi
  
  echo "[verify] Attempt $i failed (exit code: $CURL_EXIT). Retrying in $sleep_for seconds..."
  sleep "$sleep_for"
done

if [ $CURL_EXIT -ne 0 ] || [ -z "$HEALTH_JSON" ]; then
  # A budget stop before any curl, or after an empty 200, has no failed curl exit to report.
  smoke_report=$CURL_EXIT
  if [ -n "$SMOKE_STOP" ] && [ "$CURL_EXIT" -eq 0 ]; then
    smoke_report=$SMOKE_STOP
  fi
  echo "❌ Error: Functional smoke test failed. Could not query /api/health after $MAX_RETRIES attempts. curl returned: $smoke_report. Response: $HEALTH_JSON"
  exit 1
fi

echo "[verify] Health check returned: $HEALTH_JSON"
echo "✅ Success: Functional smoke test passed! AgNav is fully operational at $SPACE_URL"
exit 0
