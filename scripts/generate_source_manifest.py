import hashlib
import json
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Adjust paths to be relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "data" / "labour_law"
MANIFEST_PATH = SOURCE_DIR / "manifest.json"

def get_source_files() -> list[Path]:
    files = []
    for pattern in ["*.md", "*.pdf"]:
        for p in SOURCE_DIR.rglob(pattern):
            if (not p.name.startswith(".") 
                and ".workspaces" not in p.parts
                and "tests" not in p.parts):
                files.append(p)
    return sorted(files, key=lambda p: str(p))

def main():
    logger.info(f"[manifest] Generating source manifest for {SOURCE_DIR}...")
    files = get_source_files()
    if not files:
        logger.warning("[manifest] No source files found.")
        sys.exit(0)

    manifest = {}
    for source_file in files:
        hasher = hashlib.sha256()
        with open(source_file, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        # Store relative path to handle different mount points
        rel_path = source_file.relative_to(SOURCE_DIR)
        manifest[str(rel_path)] = hasher.hexdigest()

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"[manifest] Saved manifest with {len(manifest)} files to {MANIFEST_PATH}")

if __name__ == "__main__":
    main()
