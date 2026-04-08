#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to sys.path to import from vexilon/
sys.path.append(str(Path(__file__).parent.parent))

try:
    from vexilon.indexing import build_index_from_sources
except ImportError:
    # Handle environment where vexilon/ is not yet in Python path correctly
    sys.path.append(str(Path(__file__).parent.parent / "src"))
    from indexing import build_index_from_sources

if __name__ == "__main__":
    force_rebuild = "--force" in sys.argv
    print(f"[build_index] Starting standalone index build (force={force_rebuild})...")
    build_index_from_sources(force=force_rebuild)
    print("[build_index] Finished.")
