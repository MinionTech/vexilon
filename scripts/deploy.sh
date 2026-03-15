#!/usr/bin/env bash
set -euo pipefail

# Deployment script for Hugging Face Spaces (Consolidated Auditor V26)
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

if [ -n "${GITHUB_ACTIONS:-}" ]; then
    git config user.email "github-actions@github.com"
    git config user.name "GitHub Actions"
fi

# Make sure there is no previous branch
git branch -D hf-snapshot 2>/dev/null || true

# Make sure we are at the root of the repo
cd "$(dirname "$0")/.."

# Store original ref for cleanup
ORIGINAL_REF=$(git symbolic-ref -q --short HEAD || git rev-parse HEAD)
function cleanup() {
  if [[ "$(git branch --show-current)" == "hf-snapshot" ]]; then
    git checkout "$ORIGINAL_REF" 2>/dev/null || true
  fi
  git branch -D hf-snapshot 2>/dev/null || true
  git config --local --unset credential.https://huggingface.co.helper 2>/dev/null || true
  git remote remove hf 2>/dev/null || true
}
trap cleanup EXIT

# Resolve the Provenance Proof (The Cryptographic Auditor)
CURRENT_SHA=$(git rev-parse HEAD)
REPO_NAME=$(echo "ghcr.io/${GITHUB_REPOSITORY?Error: GITHUB_REPOSITORY not set}" | tr '[:upper:]' '[:lower:]')

echo "Auditing Cryptographic Provenance for commit: ${CURRENT_SHA}"

if [ -n "${GITHUB_ACTIONS:-}" ]; then
    # 1. Query the System of Record for the Vouched Digest
    echo "Querying GitHub Attestations..."
    IMAGE_DIGEST=$(gh api "repos/${GITHUB_REPOSITORY}/attestations/${CURRENT_SHA}" --jq '.attestations[0].bundle.content.subject[0].digest.sha256')
    
    if [ -z "$IMAGE_DIGEST" ] || [ "$IMAGE_DIGEST" == "null" ]; then
        echo "Error: No cryptographic attestation found for commit ${CURRENT_SHA}."
        echo "This commit has not been vouched by the Fortress."
        exit 1
    fi

    # 2. Verify the Proof
    echo "Verifying Platform Signature for digest: sha256:${IMAGE_DIGEST}..."
    gh attestation verify "oci://${REPO_NAME}@sha256:${IMAGE_DIGEST}" --owner "${GITHUB_REPOSITORY_OWNER}"
    
    IMAGE_POINTER="${REPO_NAME}@sha256:${IMAGE_DIGEST}"
    echo "Proof Verified: ${IMAGE_POINTER}"
else
    # Local fallback for development (unvouched)
    echo "Warning: Running in local mode. Skipping cryptographic verification."
    IMAGE_POINTER="${REPO_NAME}:latest"
fi

# Create an orphaned branch for the snapshot (The Clean Room)
git checkout --orphan hf-snapshot

# Implement the "Immutable Stub" logic: Replace everything with a 1-line Dockerfile pointer
# This ensures Hugging Face ONLY pulls the exact binary signed by our Fortress.
git rm -rf . >/dev/null 2>&1 || true
echo "FROM ${IMAGE_POINTER}" > Dockerfile

# Commit the stub snapshot
git add Dockerfile
git commit -m "deploy: bit-for-bit snapshot (vouched: $IMAGE_POINTER)"

# Configure HF remote and credentials
git remote remove hf 2>/dev/null || true
git remote add hf "https://huggingface.co/spaces/${SPACE_NAME}"
git config --local credential.https://huggingface.co.helper '!f() { echo "username=api"; echo "password=${HF_TOKEN}"; }; f'

# Force push the 1-line stub to Hugging Face
git push hf hf-snapshot:main --force --no-verify

echo "Deployment to ${SPACE_NAME} complete via cryptographic pointer!"
