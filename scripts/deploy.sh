# Usage: ./scripts/deploy.sh [image_tag] [--prod] [--dry-run]
# Default: Targets "DerekRoberts/landru" (TEST).
# Use --prod to target "DerekRoberts/vexilon".

SPACE_NAME="DerekRoberts/landru"
IMAGE_TAG=""
DRY_RUN=false
TMP_META=""

# Argument Parsing
for arg in "$@"; do
  case "$arg" in
    --prod)
      echo "[safety] Production mode enabled."
      SPACE_NAME="DerekRoberts/vexilon"
      # Prioritize PRODUCTION token if available
      HF_TOKEN=${HF_TOKEN_PRODUCTION:-$HF_TOKEN}
      ;;
    --dry-run)
      DRY_RUN=true
      ;;
    -*)
      echo "Unknown flag: $arg"
      exit 1
      ;;
    *)
      # Assume any non-flag argument is the image tag
      IMAGE_TAG="$arg"
      ;;
  esac
done

if [ -z "${HF_TOKEN:-}" ] && [ "$DRY_RUN" == "false" ]; then
    echo "Error: HF_TOKEN environment variable must be set."
    exit 1
fi

# Ensure working directory is clean before proceeding locally
if [ -z "${GITHUB_ACTIONS:-}" ] && [ "$DRY_RUN" == "false" ] && ! git diff --quiet; then
    echo "Error: Working directory must be clean before deploying locally. Please commit or stash your changes."
    exit 1
fi

if [ -n "${GITHUB_ACTIONS:-}" ]; then
    git config user.email "github-actions@github.com"
    git config user.name "GitHub Actions"
fi

# Make sure we are at the root of the repo
cd "$(dirname "$0")/.."

# Store original ref for cleanup, and set up a trap
ORIGINAL_REF=$(git symbolic-ref -q --short HEAD || git rev-parse HEAD)
function cleanup() {
  if [[ "$(git branch --show-current)" == "hf-snapshot" ]]; then
    git checkout "$ORIGINAL_REF" 2>/dev/null || true
  fi
  git branch -D hf-snapshot 2>/dev/null || true
  git config --local --unset credential.https://huggingface.co.helper 2>/dev/null || true
  git remote remove hf 2>/dev/null || true
  [ -n "$TMP_META" ] && rm -rf "$TMP_META"
}
trap cleanup EXIT

# Create an orphaned branch for the snapshot
git branch -D hf-snapshot 2>/dev/null || true
git checkout --orphan hf-snapshot

# --- SOURCE SCRUBBING / STUB GENERATION ---
if [ -n "$IMAGE_TAG" ]; then
    echo "[promote] Creating stub for image: $IMAGE_TAG"
    
    # Identify mandatory HF files (README.md for metadata)
    TMP_META=$(mktemp -d)
    [ -f README.md ] && cp README.md "$TMP_META/"
    
    # Nuke everything
    git rm -rf . > /dev/null 2>&1 || true
    
    # Restore metadata
    [ -f "$TMP_META/README.md" ] && cp "$TMP_META/README.md" . && git add README.md
    
    # Create the Stub Dockerfile
    cat <<EOF > Dockerfile
FROM ghcr.io/derekroberts/vexilon:$IMAGE_TAG
EOF
    git add Dockerfile
    COMMIT_MSG="promote: $IMAGE_TAG"
else
    echo "[deploy] Code-only deployment (no image tag provided)"
    # Fallback: existing behavior (just remove cache)
    git rm -rf --ignore-unmatch pdf_cache/ 2>/dev/null || true
    COMMIT_MSG="deploy: $(git log -1 --format='%s' "$ORIGINAL_REF" 2>/dev/null || echo 'initial snapshot')"
fi

if [ "$DRY_RUN" == "true" ]; then
    echo "--- DRY RUN MODE ---"
    echo "Target Space: $SPACE_NAME"
    echo "Commit Message: $COMMIT_MSG"
    echo "Generated Dockerfile content:"
    cat Dockerfile 2>/dev/null || echo "(No Dockerfile generated - full source deploy)"
    echo "Files in snapshot:"
    ls -A
    echo "--- DRY RUN COMPLETE ---"
    exit 0
fi

# Commit the snapshot
git commit -m "$COMMIT_MSG"

# Force push to Hugging Face
git remote add hf "https://huggingface.co/spaces/${SPACE_NAME}" 2>/dev/null || true
git config --local credential.https://huggingface.co.helper '!f() { echo "username=api"; echo "password=${HF_TOKEN}"; }; f'
git push hf hf-snapshot:main --force --no-verify

echo "Deployment to ${SPACE_NAME} complete!"
