#!/bin/bash
# Usage: ./.github/scripts/deploy.sh <space_name> [image_ref] [--dry-run]
# <space_name>: Full name of the Hugging Face Space (e.g. 'DerekRoberts/vexilon')
# [image_ref]: Tag or digest of the image to deploy (falls back to short SHA if omitted)
#
# Strict mode + Trace
set -euo pipefail

# Usage function
usage() {
    echo "Usage: $0 <space_name> [image_ref] [--dry-run]"
    echo "  <space_name>: Full name of the Hugging Face Space (e.g. 'DerekRoberts/vexilon')"
    echo "  [image_ref]: Tag or digest of the image to deploy"
    echo "  --dry-run: Show what would be done without performing it"
    exit 1
}

SPACE_NAME=""
IMAGE_REF=""
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            ;;
        *)
            if [ -z "$SPACE_NAME" ]; then
                SPACE_NAME="$1"
            elif [ -z "$IMAGE_REF" ]; then
                IMAGE_REF="$1"
            else
                echo "Error: Extra argument provided: $1"
                usage
            fi
            shift
            ;;
    esac
done

if [ -z "$SPACE_NAME" ]; then
    echo "Error: space_name (e.g. 'DerekRoberts/vexilon') must be provided."
    usage
fi

# Fallback to current short SHA if no image ref provided
if [ -z "$IMAGE_REF" ]; then
    IMAGE_REF=$(git rev-parse --short HEAD)
    echo "[info] No image reference provided. Falling back to current SHA: $IMAGE_REF"
fi

if [ -z "${HF_TOKEN:-}" ] && [ "$DRY_RUN" == "false" ]; then
    echo "Error: HF_TOKEN environment variable must be set."
    exit 1
fi

# Ensure working directory is clean before proceeding locally
if [ -z "${GITHUB_ACTIONS:-}" ] && [ "$DRY_RUN" == "false" ] && ! git diff --quiet; then
    echo "Error: Working directory must be clean before deploying locally."
    exit 1
fi

# Make sure we are at the root of the repo
cd "$(git rev-parse --show-toplevel)"
ORIGINAL_REF=$(git symbolic-ref -q --short HEAD || git rev-parse HEAD)

function cleanup() {
  echo "[cleanup] Returning to $ORIGINAL_REF..."
  git checkout -f "$ORIGINAL_REF" 2>/dev/null || true
  git branch -D hf-snapshot 2>/dev/null || true
  git remote remove hf 2>/dev/null || true
  # Nuke any potential leftover clutter from the orphan reset
  git clean -fd 2>/dev/null || true
}
trap cleanup EXIT

# Create an orphaned branch and clear it
git branch -D hf-snapshot 2>/dev/null || true
git checkout --orphan hf-snapshot
git reset # Clears the index

# Create the Stub Dockerfile
# Digests use @ syntax, tags use : syntax
[[ "$IMAGE_REF" == sha256:* ]] && separator='@' || separator=':'

# Use Bash-native expansion for the organization name (lowercase)
# The package name is hardcoded to 'agnav' to match the workflow configuration
_REPO_REF="${GITHUB_REPOSITORY:-miniontech/agnav}"
ORG_NAME="${_REPO_REF%/*}"
REPO_PATH="${ORG_NAME,,}/agnav"

cat <<EOF > Dockerfile
FROM ghcr.io/${REPO_PATH}${separator}$IMAGE_REF
EOF

if [ "$DRY_RUN" == "true" ]; then
    echo "--- DRY RUN MODE ---"
    echo "Target: $SPACE_NAME"
    echo "Image:  $IMAGE_REF"
    echo "Dockerfile content:"
    cat Dockerfile
    echo "--- DRY RUN COMPLETE ---"
    exit 0
fi

if [ -n "${GITHUB_ACTIONS:-}" ]; then
    git config user.email "github-actions@github.com"
    git config user.name "GitHub Actions"
fi

# Re-add only the essentials (including app.py as requested)
# Ensure SDK is set to docker in README.md (portable sed)
sed 's/^sdk: .*/sdk: docker/' README.md > README.md.tmp && mv README.md.tmp README.md
git add Dockerfile README.md app.py
git commit -m "promote: $IMAGE_REF from $ORIGINAL_REF"

# Auth and Push
# Remove existing remote to avoid collision/stale URLs
git remote remove hf 2>/dev/null || true
git remote add hf "https://huggingface.co/spaces/${SPACE_NAME}"
git config --local credential.https://huggingface.co.helper '!f() { echo "username=api"; echo "password=${HF_TOKEN}"; }; f'
git push hf hf-snapshot:main --force --no-verify

echo "Deployment to ${SPACE_NAME} complete!"
