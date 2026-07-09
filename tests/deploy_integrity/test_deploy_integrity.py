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

REPO_ROOT = Path(__file__).parent.parent.parent


def test_readme_metadata_sync():
    """Ensures metadata.yml has the critical metadata for Docker HF Spaces."""
    metadata_path = REPO_ROOT / "app" / "metadata.yml"
    content = metadata_path.read_text()
    
    # 1. Must be sdk: docker
    assert re.search(r"^sdk: docker", content, re.MULTILINE), \
        "README.md MUST have 'sdk: docker' for the current deployment strategy."
    
    # 2. Must have app_port: 7860 (prevents 'Still Building' status ghost)
    # Plus a warning comment to prevent drift
    assert re.search(r"^app_port: 7860.*drift", content, re.MULTILINE), \
        "README.md MUST have 'app_port: 7860' AND the sync-drift warning comment."
    
    # 3. Must NOT have Gradio-specific fields that confuse Docker mode
    assert not re.search(r"^sdk_version:", content, re.MULTILINE), \
        "README.md must NOT have 'sdk_version' — that's a Gradio SDK field, not Docker."
    assert not re.search(r"^app_file:", content, re.MULTILINE), \
        "README.md must NOT have 'app_file' — that's a Gradio SDK field, not Docker."
    
    # 4. Must have startup_duration_timeout to prevent HF from killing slow model loads
    assert re.search(r"^startup_duration_timeout: 10m", content, re.MULTILINE), \
        "README.md MUST have 'startup_duration_timeout: 10m' to prevent HF from killing the container during model loading."


def test_app_py_build_safety():
    """Ensures app.py doesn't contain global thread-pinning that hangs Docker builds."""
    app_path = REPO_ROOT / "app" / "main.py"
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
    containerfile_path = REPO_ROOT / "app" / "Containerfile"
    if not containerfile_path.exists():
        return # Skip if no Containerfile
        
    content = containerfile_path.read_text()
    
    # Check for the HEALTHCHECK port
    # CMD python -c "... http://localhost:7860"
    assert "localhost:7860" in content or "0.0.0.0:7860" in content, \
        "Containerfile HEALTHCHECK port must match the app port (7860)."


def test_hf_cache_security_lock():
    """Ensures hf_cache ownership has not been loosened (must remain root for security)."""
    containerfile_path = REPO_ROOT / "app" / "Containerfile"
    content = containerfile_path.read_text()
    
    # Ensure there is NO chown assigning hf_cache to the app user
    assert not re.search(r"chown\s+.*?\b(?:app|1000)\b.*?/hf_cache", content), \
        "Security Breach: hf_cache MUST NOT be owned by the app user. Revert the chown to root."


def test_compose_llm_provider_valid():
    """Ensures compose.yml uses only officially supported LLM providers."""
    compose_path = REPO_ROOT / "compose.yml"
    if not compose_path.exists():
        return
        
    content = compose_path.read_text()
    
    # Extract all lines setting AGNAV_LLM_PROVIDER at the start of a line
    # Supporting standard variables and interpolations with default values, e.g., ${AGNAV_LLM_PROVIDER:-ollama}
    providers = re.findall(r"^\s*AGNAV_LLM_PROVIDER:\s*(?:\${[A-Z0-9_]+:-)?([a-zA-Z0-9_-]+)}?", content, re.MULTILINE)
    
    supported_providers = {"ollama", "huggingface", "mock"}
    for p in providers:
        assert p.lower() in supported_providers, \
            f"Unsupported LLM provider '{p}' found in compose.yml. Supported: {supported_providers}"


def test_chainlit_markdown_links_exist():
    """Ensures all local static file links inside chainlit.md actually exist on disk."""
    chainlit_md_path = REPO_ROOT / "app" / "chainlit.md"
    if not chainlit_md_path.exists():
        return
        
    content = chainlit_md_path.read_text()
    
    # Match local links starting with /public/docs/
    # e.g., [BCGEU 19th Main Agreement](/public/docs/BCGEU_19th_Main_Agreement.pdf)
    local_links = re.findall(r"\]\((/public/docs/[^\)]+)\)", content)
    
    public_docs_root = REPO_ROOT / "app" / "public" / "docs"
    
    for link in local_links:
        # Strip the /public/docs/ prefix to get the relative path inside app/public/docs
        relative_path_str = link.replace("/public/docs/", "")
        
        # Resolve target physical file path
        target_file = public_docs_root / relative_path_str
        
        assert target_file.exists(), \
            f"Broken Link in chainlit.md: The asset '{link}' was linked, but '{target_file}' does not exist on disk."


def test_manifest_source_files_exist():
    """Ensures all source files listed in data/manifest.json actually exist on disk."""
    import json
    manifest_path = REPO_ROOT / "app" / "data" / "manifest.json"
    if not manifest_path.exists():
        return
        
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as e:
        assert False, f"manifest.json is not valid JSON: {e}"
        
    sources = manifest.get("sources", {})
    data_root = REPO_ROOT / "app" / "data"
    
    for relative_path_str in sources.keys():
        target_file = data_root / relative_path_str
        assert target_file.exists(), \
            f"Missing indexed resource: Source file '{relative_path_str}' is listed in manifest.json, but '{target_file}' does not exist on disk."


def test_default_model_is_gemma_4():
    """Ensures that the LLM configuration remains strictly aligned to Gemma 4 models."""
    app_path = REPO_ROOT / "app" / "main.py"
    content = app_path.read_text()
    
    # Assert that no default returns or fallbacks mention any Qwen models
    assert "Qwen" not in content, \
        "Code Quality regression: Swapping default fallback models to Qwen is prohibited. Keep flagship Gemma 4."
    
    # Verify the fallback model returned in _get_default_model is exactly google/gemma-4-31B-it
    assert re.search(r'return\s+["\']google/gemma-4-31B-it["\']', content), \
        "Code Quality regression: fallback model return value in main.py must be the flagship google/gemma-4-31B-it model."


def test_python_version_integrity():
    """Ensures Python runtime version does not drop below Python 3.14 in Containerfile and CI workflows."""
    # 1. Check Containerfile for base image version (must be >= 3.14)
    containerfile_path = REPO_ROOT / "app" / "Containerfile"
    if containerfile_path.exists():
        content = containerfile_path.read_text()
        match = re.search(r"FROM\s+python:([\d\.]+)", content)
        if match:
            version_str = match.group(1)
            major, minor = map(int, version_str.split(".")[:2])
            assert (major > 3) or (major == 3 and minor >= 14), \
                f"Container base image is running Python {version_str}. Downgrading below Python 3.14 is prohibited."

    # 2. Check CI workflows for python-version (must be >= 3.14)
    workflows_dir = REPO_ROOT / ".github" / "workflows"
    if workflows_dir.exists():
        for f in workflows_dir.glob("*.yml"):
            content = f.read_text()
            matches = re.findall(r"python-version:\s*['\"]?([\d\.]+)['\"]?", content)
            for ver in matches:
                major, minor = map(int, ver.split(".")[:2])
                assert (major > 3) or (major == 3 and minor >= 14), \
                    f"CI Workflow '{f.name}' specifies python-version '{ver}'. Downgrading below Python 3.14 is prohibited."
