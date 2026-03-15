#!/usr/bin/env bash
set -euo pipefail

# Deployment script for Hugging Face Spaces
# Usage: ./scripts/deploy.sh <space_name>

if [ -z "${1:-}" ]; then
    echo "Error: Space name must be provided."
    echo "Usage: $0 <space_name>"
    exit 1
fi

SPACE_NAME=$1

if [ -z "${HF_TOKEN:-}" ]; then
    echo "Error: HF_TOKEN environment variable must be set."
    exit 1
fi

# Ensure working directory is clean before proceeding locally
if [ -z "${GITHUB_ACTIONS:-}" ] && ! git diff --quiet; then
    echo "Error: Working directory must be clean before deploying locally. Please commit or stash your changes."
    exit 1
fi

if [ -n "${GITHUB_ACTIONS:-}" ]; then
    git config user.email "github-actions@github.com"
    git config user.name "GitHub Actions"
fi

# Make sure there is no previous branch
git branch -D hf-snapshot 2>/dev/null || true

# Make sure we are at the root of the repo
cd "$(dirname "$0")/.."

# Store original ref for cleanup, and set up a trap to ensure cleanup happens on exit.
ORIGINAL_REF=$(git symbolic-ref -q --short HEAD || git rev-parse HEAD)
function cleanup() {
  # Only checkout if we are on the snapshot branch
  if [[ "$(git branch --show-current)" == "hf-snapshot" ]]; then
    git checkout "$ORIGINAL_REF" 2>/dev/null || true
  fi
  git branch -D hf-snapshot 2>/dev/null || true
  git config --local --unset credential.https://huggingface.co.helper 2>/dev/null || true
  git remote remove hf 2>/dev/null || true
}
trap cleanup EXIT

# Remove the remote if it already exists
git remote remove hf 2>/dev/null || true
git remote add hf "https://huggingface.co/spaces/${SPACE_NAME}"
git config --local credential.https://huggingface.co.helper '!f() { echo "username=api"; echo "password=${HF_TOKEN}"; }; f'

COMMIT_MSG=$(git log -1 --format='%s')

# Resolve the Provenance Chain (The Auditor Logic)
CURRENT_SHA=$(git rev-parse HEAD)
if [ -n "${GITHUB_ACTIONS:-}" ]; then
    echo "Auditing Provenance for commit: $CURRENT_SHA"
    # 1. Resolve Parent PR from System of Record
    PR_DATA=$(gh api "repos/${GITHUB_REPOSITORY}/commits/${CURRENT_SHA}/pulls" --jq '.[0]')
    PR_NUM=$(echo "$PR_DATA" | jq -r '.number')
    
    if [ "$PR_NUM" != "null" ]; then
        # 2. Resolve Head SHA (The unforgeable fingerprint)
        IMAGE_SHA=$(echo "$PR_DATA" | jq -r '.head.sha')
        echo "Protocol Verified: PR #$PR_NUM resolved to Head SHA $IMAGE_SHA"
    else
        echo "Warning: No PR link found for this commit. Falling back to current SHA."
        IMAGE_SHA=$CURRENT_SHA
    fi
else
    # Local fallback
    IMAGE_SHA=$CURRENT_SHA
fi

REPO_NAME=$(echo "ghcr.io/${GITHUB_REPOSITORY:-derekroberts/vexilon}" | tr '[:upper:]' '[:lower:]')

# Create an orphaned branch for the snapshot
git branch -D hf-snapshot 2>/dev/null || true
git checkout --orphan hf-snapshot

# Implement the "Stub" logic: Replace the complex Dockerfile with a 1-line pointer
echo "FROM ${REPO_NAME}:sha-${IMAGE_SHA}" > Dockerfile

# Remove unneeded files from index to keep the snapshot minimal
git rm -rf --ignore-unmatch pdf_cache/ .github/ .pytest_cache/ tests/ 2>/dev/null || true

# Commit the stub snapshot
git add Dockerfile
git commit -m "deploy: $COMMIT_MSG (image: sha-${IMAGE_SHA})"

# Force push to Hugging Face
git push hf hf-snapshot:main --force --no-verify

echo "Deployment to ${SPACE_NAME} complete!"
