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

# Resolve the Provenance Pointer (The Auditor)
CURRENT_SHA=$(git rev-parse HEAD)
REPO_NAME=$(echo "ghcr.io/${GITHUB_REPOSITORY:-derekroberts/vexilon}" | tr '[:upper:]' '[:lower:]')
IMAGE_TAG="main-sha-${CURRENT_SHA}"

echo "Auditing Deployment for commit: ${CURRENT_SHA}"
echo "Pointer: ${REPO_NAME}:${IMAGE_TAG}"

# Create an orphaned branch for the snapshot
git branch -D hf-snapshot 2>/dev/null || true
git checkout --orphan hf-snapshot

# Implement the "Stub" logic: Replace the Dockerfile with a 1-line pointer
# This ensures Hugging Face ONLY pulls the binary vouched by our Fortress.
echo "FROM ${REPO_NAME}:${IMAGE_TAG}" > Dockerfile

# Remove unneeded files from index to keep the snapshot minimal
git rm -rf --ignore-unmatch pdf_cache/ .github/ .pytest_cache/ tests/ 2>/dev/null || true

# Commit the stub snapshot
git add Dockerfile
git commit -m "deploy: $COMMIT_MSG (vouched: $IMAGE_TAG)"

# Force push to Hugging Face
git push hf hf-snapshot:main --force --no-verify

echo "Deployment to ${SPACE_NAME} complete!"
