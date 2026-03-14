#!/usr/bin/env bash
set -euo pipefail

# Vexilon Deployment Script (Phase 1: Metadata-Aware Stub Deployer)
# Usage: ./scripts/deploy.sh <space_name> [image_url] [--pr <number>] [--sha <commit>]

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <space_name> [image_url] [--pr <num>] [--sha <commit>]"
    exit 1
fi

SPACE_NAME=$1
IMAGE_URL=${2:-""}
PR_NUM=""
SHA=""

# Parse optional flags
shift
[ $# -gt 0 ] && shift || true # shift past image_url if it was there
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pr) PR_NUM="$2"; shift 2 ;;
        --sha) SHA="$2"; shift 2 ;;
        *) shift ;;
    esac
done

if [ -z "${HF_TOKEN:-}" ]; then
    echo "Error: HF_TOKEN environment variable must be set."
    exit 1
fi

# Safety check: No local pushes allowed in the new "Build-Once" paradigm
if [ "${GITHUB_ACTIONS:-}" != "true" ]; then
    echo "Error: This script is restricted to GitHub Actions only."
    echo "To deploy locally, use Podman Compose: 'podman-compose up --build'"
    exit 1
fi

# Git configuration for actions
git config user.email "github-actions@github.com"
git config user.name "GitHub Actions"

# Make sure we are at the root of the repo
cd "$(dirname "$0")/.."

# Store original ref for cleanup
ORIGINAL_REF=$(git symbolic-ref -q --short HEAD || git rev-parse HEAD)
function cleanup() {
  if [[ "$(git branch --show-current)" == "hf-snapshot" ]]; then
    git checkout "$ORIGINAL_REF" 2>/dev/null || true
  fi
  git branch -D hf-snapshot 2>/dev/null || true
  git remote remove hf 2>/dev/null || true
}
trap cleanup EXIT

# Set up the HF remote
git remote add hf "https://huggingface.co/spaces/${SPACE_NAME}"
git config --local credential.https://huggingface.co.helper '!f() { echo "username=api"; echo "password=${HF_TOKEN}"; }; f'

# Create an orphaned branch for the deployment snapshot
git branch -D hf-snapshot 2>/dev/null || true
git checkout --orphan hf-snapshot

if [ -n "$IMAGE_URL" ]; then
    # --- STUB MODE (Promotion) ---
    echo "Deploying in STUB MODE for image: ${IMAGE_URL}"
    
    # Create the 1-line stub
    echo "FROM ${IMAGE_URL}" > Dockerfile
    
    # Identify files to keep (Dockerfile and README.md are mandatory)
    # We use a temp file to track files to preserve
    keep_list=("Dockerfile" "README.md")
    
    # Remove everything from the git index except the keep_list
    # We do this by clearing the whole index and adding back only what we want.
    git rm -rf . > /dev/null
    for file in "${keep_list[@]}"; do
        if [ -f "$file" ]; then
            git add "$file"
        fi
    done
    
    DESC="promote ${IMAGE_URL}"
    [ -n "$PR_NUM" ] && DESC="${DESC} (PR #${PR_NUM})"
    [ -n "$SHA" ] && DESC="${DESC} @ ${SHA}"
    COMMIT_MSG="deploy: ${DESC}"
else
    # --- LEGACY MODE (Source Snapshot) ---
    echo "Deploying in LEGACY MODE (Full Source Snapshot)"
    
    # Remove large binary or cache files that HF shouldn't see
    git rm -rf --ignore-unmatch pdf_cache/ 2>/dev/null || true
    
    ORIG_MSG=$(git log -1 --format='%s' "$ORIGINAL_REF")
    COMMIT_MSG="deploy: ${ORIG_MSG}"
fi

git commit -m "$COMMIT_MSG"

# Force push to Hugging Face
git push hf hf-snapshot:main --force --no-verify

echo "Deployment to ${SPACE_NAME} complete!"
