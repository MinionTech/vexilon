"""
tests/test_deploy_integrity.py — Automated Deployment Readiness Checks
----------------------------------------------------------------------
Verifies that the codebase is ready for a 'Green' Hugging Face Space deploy.
Ensures metadata (README), Docker safety (Containerfile), and performance 
optimizations (app.py) are correctly synchronized.
"""
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def test_readme_metadata_sync():
    """Ensures README.md has the critical metadata for Docker HF Spaces."""
    readme_path = REPO_ROOT / "README.md"
    content = readme_path.read_text()
    
    # 1. Must be sdk: docker
    assert re.search(r"^sdk: docker", content, re.MULTILINE), \
        "README.md MUST have 'sdk: docker' for the current deployment strategy."
    
    # 2. Must have app_port: 7860 (prevents 'Still Building' status ghost)
    assert re.search(r"^app_port: 7860", content, re.MULTILINE), \
        "README.md MUST have 'app_port: 7860' so Hugging Face can perform healthchecks."


def test_app_py_build_safety():
    """Ensures app.py doesn't contain global thread-pinning that hangs Docker builds."""
    app_path = REPO_ROOT / "app.py"
    content = app_path.read_text()
    
    # Check for OMP_NUM_THREADS or MKL_NUM_THREADS at top level (not inside a function)
    # This specifically checks for assignments happening outside of any def/if/with blocks
    # which is what caused our previous build-time hangs.
    
    for line in content.splitlines():
        if ("OMP_NUM_THREADS" in line or "MKL_NUM_THREADS" in line) and "=" in line:
            # If the assignment is at the start of the line (no indentation), it's global.
            # This is a basic safety check against regressions like PR #238.
            assert line.startswith(" ") or line.startswith("\t") or "os.getenv" in line or "os.environ.get" in line, \
                f"Potentially dangerous global thread pinning detected: {line.strip()}. Move this inside a function or conditioned check."


def test_containerfile_healthcheck_sync():
    """Ensures the Docker HEALTHCHECK port matches the app port."""
    containerfile_path = REPO_ROOT / "Containerfile"
    if not containerfile_path.exists():
        return # Skip if no Containerfile
        
    content = containerfile_path.read_text()
    
    # Check for the HEALTHCHECK port
    # CMD python -c "... http://localhost:7860"
    assert "localhost:7860" in content or "0.0.0.0:7860" in content, \
        "Containerfile HEALTHCHECK port must match the app port (7860)."


def test_hf_cache_security_lock():
    """Ensures hf_cache ownership has not been loosened (must remain root for security)."""
    containerfile_path = REPO_ROOT / "Containerfile"
    content = containerfile_path.read_text()
    
    # Ensure there is NO --chown=1001:1001 on the hf_cache line
    # Broken version: COPY --from=builder --chown=1001:1001 /app/hf_cache /app/hf_cache
    # Safe version: COPY --from=builder /app/hf_cache /app/hf_cache
    assert "--chown=1001:1001 /app/hf_cache" not in content, \
        "Security Breach: hf_cache MUST NOT be owned by the app user. Revert the chown to root."
