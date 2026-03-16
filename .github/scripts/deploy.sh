# Usage: ./.github/scripts/deploy.sh <image_ref> [--prod] [--dry-run]
# <image_ref> can be a tag (e.g. 'sha-abc123') or a digest (e.g. 'sha256:abc123...')
# Default: Targets "DerekRoberts/landru" (TEST).
# Use --prod as second argument to target "DerekRoberts/vexilon".

# Strict mode + Trace
set -euo pipefail

IMAGE_REF="${1:-}"
MODE="${2:-}"
DRY_RUN=false

if [ -z "$IMAGE_REF" ]; then
    echo "Error: Image reference (e.g. 'sha-abc123' or 'sha256:abc123...') must be provided."
    exit 1
fi

SPACE_NAME="DerekRoberts/landru"
if [[ "$MODE" == "--prod" ]]; then
    echo "[safety] Production mode enabled."
    SPACE_NAME="DerekRoberts/vexilon"
fi

# Detect --dry-run in any position
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=true
done

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
cat <<EOF > Dockerfile
FROM ghcr.io/derekroberts/vexilon${separator}$IMAGE_REF
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
# We also need to fix the README.md metadata on-the-fly to use sdk: docker
sed -i 's/^sdk: gradio/sdk: docker/' README.md
git add Dockerfile README.md app.py
git commit -m "promote: $IMAGE_REF from $ORIGINAL_REF"

# Auth and Push
git remote add hf "https://huggingface.co/spaces/${SPACE_NAME}" 2>/dev/null || true
git config --local credential.https://huggingface.co.helper '!f() { echo "username=api"; echo "password=${HF_TOKEN}"; }; f'
git push hf hf-snapshot:main --force --no-verify

echo "Deployment to ${SPACE_NAME} complete!"
