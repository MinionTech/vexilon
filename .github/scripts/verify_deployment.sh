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
  CURL_ARGS=( -s )
  if [ -n "${HF_TOKEN:-}" ]; then
      CURL_ARGS+=( -H "Authorization: Bearer $HF_TOKEN" )
  fi

  # Temporarily disable -e to handle network errors during polling
  set +e
  STATUS_JSON=$(curl "${CURL_ARGS[@]}" "https://huggingface.co/api/spaces/$SPACE_ID")
  CURL_EXIT=$?
  set -e

  if [ $CURL_EXIT -ne 0 ]; then
      echo "[verify] curl command failed (exit code $CURL_EXIT). This might be a network glitch. Retrying in $INTERVAL seconds..."
      sleep $INTERVAL
      continue
  fi

  # Check if we got a valid response (not empty or error)
  if [ -z "$STATUS_JSON" ] || [[ "$STATUS_JSON" == *"\"error\":\""* ]]; then
      echo "[verify] API returned error or empty response. Body: $STATUS_JSON. Retrying in $INTERVAL seconds..."
      sleep $INTERVAL
      continue
  fi

  # Robust status extraction using python (already available in GH Actions)
  CURRENT_STATUS=$(echo "$STATUS_JSON" | python3 -c "import sys, json; print(str(json.load(sys.stdin).get('runtime', {}).get('stage', 'unknown')).lower())")
  
  echo "[verify] Current status: $CURRENT_STATUS ($(($(date +%s) - START_TIME))s)"
  
  if [ "$CURRENT_STATUS" == "running" ]; then
    echo "✅ Success: Space $SPACE_ID is running!"
    exit 0
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

echo "❌ Error: Timeout waiting for Space $SPACE_ID to become ready after $TIMEOUT_SECONDS seconds."
exit 1
