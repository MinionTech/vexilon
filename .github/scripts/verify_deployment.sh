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
  # We use -H "Authorization: Bearer $HF_TOKEN" if it's set
  AUTH_HEADER=""
  if [ -n "${HF_TOKEN:-}" ]; then
      AUTH_HEADER="-H \"Authorization: Bearer $HF_TOKEN\""
  fi

  STATUS_JSON=$(curl -s $AUTH_HEADER "https://huggingface.co/api/spaces/$SPACE_ID")
  
  # Check if we got a valid response (not empty or error)
  if [ -z "$STATUS_JSON" ] || [[ "$STATUS_JSON" == *"\"error\":\""* ]]; then
      echo "[verify] API error or empty response. Retrying in $INTERVAL seconds..."
      sleep $INTERVAL
      continue
  fi

  CURRENT_STATUS=$(echo "$STATUS_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin).get('runtime', {}).get('stage', 'unknown'))")
  
  echo "[verify] Current status: $CURRENT_STATUS ($(($(date +%s) - START_TIME))s)"
  
  if [ "$CURRENT_STATUS" == "running" ]; then
    echo "✅ Success: Space $SPACE_ID is running!"
    
    # Optional: Smoke test the URL
    # SUBDOMAIN=$(echo "$SPACE_ID" | sed 's/\//-/g' | tr '[:upper:]' '[:lower:]')
    # URL="https://${SUBDOMAIN}.hf.space"
    # echo "[verify] Performing smoke test on $URL..."
    # if curl -s --fail "$URL" > /dev/null; then
    #     echo "✅ Smoke test passed!"
    # else
    #     echo "⚠️ Smoke test failed (URL not reachable yet), but HF says it's running."
    # fi
    
    exit 0
  fi
  
  if [[ "$CURRENT_STATUS" == *"crashed"* ]]; then
    echo "❌ Error: Space $SPACE_ID has CRASHED (status: $CURRENT_STATUS)"
    echo "Check logs at: https://huggingface.co/spaces/$SPACE_ID"
    exit 1
  fi

  if [[ "$CURRENT_STATUS" == "no_container_error" ]]; then
    echo "❌ Error: Space $SPACE_ID reports no_container_error."
    exit 1
  fi
  
  sleep $INTERVAL
done

echo "❌ Error: Timeout waiting for Space $SPACE_ID to become ready."
exit 1
