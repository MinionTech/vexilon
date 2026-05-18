#!/bin/bash
# .github/scripts/verify_deployment.sh <space_id> [timeout_seconds]
# Polls Hugging Face API until Space status is 'running'.
# Usage: ./.github/scripts/verify_deployment.sh DerekRoberts/vexilon 600

set -eo pipefail

SPACE_ID=$1
TIMEOUT_SECONDS=${2:-900} # Default 15 minutes because building can be slow
INTERVAL=30

if [ -z "$SPACE_ID" ]; then
    echo "Error: SPACE_ID argument missing."
    exit 1
fi

echo "[verify] Monitoring Hugging Face Space: $SPACE_ID"
echo "[verify] Timeout: $TIMEOUT_SECONDS seconds"

START_TIME=$(date +%s)
END_TIME=$((START_TIME + TIMEOUT_SECONDS))

while [ $(date +%s) -lt $END_TIME ]; do
  # Use Bash array for safer argument handling
  CURL_ARGS=( -s -L )
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
      sleep $INTERVAL
      continue
  fi

  if [ "$HTTP_STATUS" != "200" ]; then
      echo "[verify] API returned HTTP $HTTP_STATUS. Body: $STATUS_JSON. Retrying..."
      sleep $INTERVAL
      continue
  fi

  # Check if we got a valid response (not empty)
  if [ -z "$STATUS_JSON" ]; then
      echo "[verify] Received empty response from API. Retrying..."
      sleep $INTERVAL
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
  
  sleep $INTERVAL
done

# Check if we exited the loop because of success or timeout
if [ "$CURRENT_STATUS" != "running" ]; then
  echo "❌ Error: Timeout waiting for Space $SPACE_ID to become ready after $TIMEOUT_SECONDS seconds."
  exit 1
fi

# --- Functional Smoke Test ---
echo "[verify] 🔍 Running functional smoke test..."
SPACE_URL="https://$(echo "$SPACE_ID" | tr '[:upper:]' '[:lower:]' | tr '/' '-').hf.space"

# 1. Query the custom /api/version route to verify the FastAPI backend is running and responding
VERSION_JSON=$(curl -s -f "$SPACE_URL/api/version")
CURL_EXIT=$?

if [ $CURL_EXIT -ne 0 ] || [ -z "$VERSION_JSON" ]; then
  echo "❌ Error: Functional smoke test failed. Could not query /api/version (exit code: $CURL_EXIT). Response: $VERSION_JSON"
  exit 1
fi

# Extract version from response JSON
APP_VERSION=$(echo "$VERSION_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin).get('version', ''))" 2>/dev/null || echo "")

if [ -z "$APP_VERSION" ]; then
  echo "❌ Error: Functional smoke test failed. API returned invalid JSON or version is missing: $VERSION_JSON"
  exit 1
fi

echo "[verify] App version detected: $APP_VERSION"
echo "✅ Success: Functional smoke test passed! AgNav is fully operational at $SPACE_URL"
exit 0

