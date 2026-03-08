"""
conftest.py — pytest root configuration

Adds the project root to sys.path so `import app` works from tests/
regardless of how pytest is invoked.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
