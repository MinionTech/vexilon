#!/usr/bin/env python3
import sys
import logging
from pathlib import Path

# Add project root to sys.path to import from agnav/
sys.path.append(str(Path(__file__).parent.parent))

from agnav.indexing import build_index_from_sources

if __name__ == "__main__":
    # Ensure logs from indexing.py are visible (Issue #196)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    force_rebuild = "--rebuild-index" in sys.argv
    print(f"[build_index] Starting standalone index build (force={force_rebuild})...")
    build_index_from_sources(force=force_rebuild)
    print("[build_index] Finished.")
