import logging
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

class TestDoctrine:
    """Represents a legal/arbitration test doctrine loaded from markdown fixtures."""
    def __init__(self, name: str, keywords: set[str], content: str, file_path: Path):
        self.name = name
        self.keywords = keywords
        self.content = content
        self.file_path = file_path


class TestRegistry:
    """Thread-safe registry of arbitration doctrines matched against steward queries."""
    def __init__(self):
        self.tests: list[TestDoctrine] = []
        self._lock = Lock()

    def load(self, directory: Path) -> None:
        if not directory.exists():
            return
        with self._lock:
            self.tests = []
            for f in directory.glob("*.md"):
                if f.name == "index.md":
                    continue
                try:
                    text = f.read_text(encoding="utf-8")
                    lines = text.split("\n")
                    keywords = set()
                    content_start = 0
                    for i, line in enumerate(lines):
                        if line.startswith("**Keywords:**"):
                            kw_line = line.replace("**Keywords:**", "").strip()
                            keywords = {k.strip().lower() for k in kw_line.split(",") if k.strip()}
                            content_start = i + 1
                            break
                    self.tests.append(TestDoctrine(
                        name=f.stem.replace("_", " ").title(),
                        keywords=keywords,
                        content="\n".join(lines[content_start:]).strip(),
                        file_path=f
                    ))
                except Exception as e:
                    logger.error(f"[registry] Failed to load {f.name}: {e}")

    def find_matches(self, query: str) -> list[TestDoctrine]:
        q_lower = query.lower()
        with self._lock:
            return [test for test in self.tests if any(k in q_lower for k in test.keywords)]


_test_registry = TestRegistry()
