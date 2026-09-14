#!/usr/bin/env python3
"""
Generate a manifest.json for the FAISS cache directory.
This tracks hashes of source files to validate cache freshness.

Issue #239: FAISS Cache Persistence & Binary Bloat
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, UTC


def hash_file(filepath: Path) -> str:
    """Generate SHA256 hash of file contents."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


_PKG_ROOT = Path(__file__).parent.parent

def _get_default_data_dir() -> Path:
    if "AGNAV_DATA_DIR" in os.environ:
        return Path(os.environ["AGNAV_DATA_DIR"])
    container_sources = Path("/data/sources")
    if container_sources.exists():
        return container_sources
    return _PKG_ROOT / "data"

def generate_manifest(
    data_dir: Path | None = None,
    output_path: Path | None = None
) -> dict:
    """Generate manifest of all source files in data directory."""
    if data_dir is None:
        data_dir = _get_default_data_dir()
    if output_path is None:
        output_path = data_dir / "manifest.json"

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "version": "1.0",
        "sources": {}
    }
    
    # Find all source files (PDF and MD)
    fixtures_dir = data_dir / "test_fixtures"
    source_files = []
    for pattern in ["*.md", "*.pdf"]:
        for file_path in data_dir.rglob(pattern):
            # Skip hidden, tests, cache, and integrity files
            if (not file_path.name.startswith(".") 
                and ".workspaces" not in file_path.parts
                and not file_path.is_relative_to(fixtures_dir)
                and not file_path.name.endswith(".integrity.md")
                and "cache" not in file_path.parts
                and ".pdf_cache" not in file_path.parts):
                source_files.append(file_path)
    
    # Sort for deterministic output
    source_files = sorted(source_files, key=lambda p: str(p))
    
    for file_path in source_files:
        relative_path = file_path.relative_to(data_dir)
        manifest["sources"][str(relative_path)] = {
            "hash": hash_file(file_path),
            "size_bytes": file_path.stat().st_size
        }
    
    # Write manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    return manifest


def validate_cache(
    data_dir: Path | None = None,
    manifest_path: Path | None = None
) -> bool:
    """Validate that cache matches current source files."""
    if data_dir is None:
        data_dir = _get_default_data_dir()
    if manifest_path is None:
        manifest_path = data_dir / "manifest.json"

    if not manifest_path.exists():
        print(f"ERROR: No manifest.json found at {manifest_path}. Run generate_cache_manifest.py first.")
        return False
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Collect current source files
    fixtures_dir = data_dir / "test_fixtures"
    current_files = {}
    for pattern in ["*.md", "*.pdf"]:
        for file_path in data_dir.rglob(pattern):
            if (not file_path.name.startswith(".") 
                and ".workspaces" not in file_path.parts
                and not file_path.is_relative_to(fixtures_dir)
                and not file_path.name.endswith(".integrity.md")
                and "cache" not in file_path.parts
                and ".pdf_cache" not in file_path.parts):
                relative_path = file_path.relative_to(data_dir)
                current_files[str(relative_path)] = hash_file(file_path)
    
    errors = []
    
    # Check for new/modified files
    for file_path, current_hash in current_files.items():
        if file_path not in manifest["sources"]:
            errors.append(f"NEW: {file_path} (not in manifest)")
        elif manifest["sources"][file_path]["hash"] != current_hash:
            errors.append(f"MODIFIED: {file_path} (hash mismatch)")
    
    # Check for removed files
    for file_path in manifest["sources"]:
        if file_path not in current_files:
            errors.append(f"REMOVED: {file_path} (in manifest but not in data/)")
    
    if errors:
        print("Cache validation FAILED:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print(f"Cache validation PASSED: {len(current_files)} source files match manifest")
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        success = validate_cache()
        sys.exit(0 if success else 1)
    else:
        manifest = generate_manifest()
        print(f"Generated manifest.json with {len(manifest['sources'])} source files")

