---
description: CI/CD workflow architecture and constraints — read before modifying any workflow
---

# CI/CD Workflow Specification

This document describes the deployment pipeline architecture. **Read this entire document before modifying any workflow file.**

## Architecture Overview

```
PR opened → pr-open.yml (test + build image) → merge to main → deploy-test.yml (push to HF test Space)
                                                             → release published → deploy-prod.yml (push to HF prod Space)
```

## Critical Constraints

### GHCR Image URLs MUST be fully lowercase
GHCR requires lowercase repository paths. GitHub's `${{ github.repository }}` returns `MinionTech/vexilon` (mixed case). You MUST use `declare -l` to lowercase the entire image URL:

```bash
# ✅ CORRECT — lowercases entire string including repo path
declare -l IMAGE="ghcr.io/${{ github.repository }}:${TAG}"

# ❌ WRONG — only lowercases the tag, repo path stays mixed case
IMAGE="ghcr.io/${{ github.repository }}:${TAG,,}"
```

### Image tags use the PR head SHA
Images are tagged with `sha-<PR head commit SHA>` during CI. The deploy workflows look up the PR associated with each merge commit and resolve its head SHA to find the matching image.

### Deploy workflows need git history for walk-back
Both deploy workflows walk back through recent commits to find a valid image. `actions/checkout` defaults to `fetch-depth: 1` (shallow). The checkout step MUST specify `fetch-depth: 20` so the walk-back has commits to search.

### paths-ignore in pr-open.yml is intentional
Documentation-only PRs (`.md`, `LICENSE`, etc.) skip CI builds. This means some merge commits on main will NOT have a corresponding image in GHCR. The deploy workflows handle this via the walk-back fallback.

### HuggingFace Docker Space metadata (README.md)
The README.md YAML frontmatter controls the HF Space configuration:
- `sdk: docker` — REQUIRED. Tells HF this is a Docker Space, not Gradio.
- `app_port: 7860` — REQUIRED. Tells HF which port to healthcheck.
- `startup_duration_timeout: 10m` — REQUIRED. Model loading takes ~30-60s.
- `sdk_version` — DO NOT USE. This is a Gradio-only field.
- `app_file` — DO NOT USE. This is a Gradio-only field.

### hf_cache must remain root-owned
The `COPY --from=builder /app/hf_cache /app/hf_cache` line in the Containerfile must NOT include `--chown`. The cache directory must remain root-owned (read-only) for security.

### Thread pinning is runtime-only
`OMP_NUM_THREADS` and `MKL_NUM_THREADS` must ONLY be set inside functions (at runtime), never at module level. Global thread pinning causes build-time hangs on HuggingFace Spaces.

## File Reference

| File | Purpose |
|------|---------|
| `.github/workflows/pr-open.yml` | Tests + builds Docker image on PR |
| `.github/workflows/deploy-test.yml` | Deploys to test HF Space on push to main |
| `.github/workflows/deploy-prod.yml` | Deploys to prod HF Space on release |
| `.github/scripts/deploy.sh` | Pushes stub Dockerfile + README to HF Space. Usage: `<space_name> [image_ref] [--dry-run]` |
| `Containerfile` | Multi-stage Docker build |
| `tests/test_deploy_integrity.py` | Automated checks for the constraints above |
