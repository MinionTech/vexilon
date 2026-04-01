"""
app.py — BCGEU Steward Assistant
--------------------------------------------
Tech stack:
  - pymupdf             : PDF extraction for Markdown conversion
  - sentence-transformers: Local CPU embeddings (BAAI/bge-small-en-v1.5, no API key)
  - FAISS                : In-memory vector index (no server process)
  - Anthropic            : Claude (claude-haiku-4-5-20251001) for responses
  - Gradio 6             : Web UI at http://localhost:7860

Quick start:
  1. export ANTHROPIC_API_KEY=sk-ant-...
  2. uv sync
  3. uv run python app.py

Index pre-computation (run once after updating the PDF):
  python -c "from app import startup; startup(force_rebuild=True)"
  # Saves .pdf_cache/index.faiss + .pdf_cache/chunks.json for fast cold starts.
"""

# ─── Standard Library ────────────────────────────────────────────────────────
import sys
import threading

print("[boot] Python started, importing stdlib...", flush=True)
import json
import os
import re
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
import datetime
import tempfile


# Ensure the HuggingFace model cache is writable and persistent.
# Inside the container (WORKDIR /app), this resolves to /app/hf_cache.
# Locally, it resolves to ./hf_cache in the repo root.
if not os.getenv("HF_HOME"):
    os.environ["HF_HOME"] = str(Path("./hf_cache").absolute())

# ─── Third-party: Deferred Imports ───────────────────────────────────────────
# (numpy, anthropic, faiss, sentence_transformers, gradio)
# are imported inside functions to keep startup and test-loading fast.
print("[boot] All boilerplate complete.", flush=True)

# ─── Configuration ───────────────────────────────────────────────────────────
VEXILON_REPO_URL = os.getenv("VEXILON_REPO_URL", "https://github.com/DerekRoberts/vexilon")
_GITHUB_RAW_BASE = os.getenv("VEXILON_RAW_URL_BASE", "https://raw.githubusercontent.com/DerekRoberts/vexilon/main")

# Public GitHub raw URL base for labour_law PDFs.
# Used for folder/file links in the UI.
GITHUB_LABOUR_LAW_URL = os.getenv(
    "VEXILON_KNOWLEDGE_URL", f"{VEXILON_REPO_URL}/tree/main/data/labour_law"
)

# Raw URL base for downloading pre-computed index from GitHub.
# Used by _fetch_pdf_cache_if_missing() for HF Spaces bootstrap.
PDF_CACHE_DIR = Path("./.pdf_cache")
LABOUR_LAW_DIR = Path("./data/labour_law")
TESTS_DIR = LABOUR_LAW_DIR / "tests"
INDEX_PATH = PDF_CACHE_DIR / "index.faiss"
CHUNKS_PATH = PDF_CACHE_DIR / "chunks.json"
MANIFEST_PATH = PDF_CACHE_DIR / "manifest.json"
_CSS_PATH = Path(__file__).parent / "style.css"

# Models
DEFAULT_MODEL_LLM = os.getenv("VEXILON_DEFAULT_MODEL", "claude-haiku-4-5-20251001")
CLAUDE_MODEL = os.getenv("VEXILON_CLAUDE_MODEL", DEFAULT_MODEL_LLM)
REVIEWER_MODEL = os.getenv("VEXILON_REVIEWER_MODEL", DEFAULT_MODEL_LLM)
CONDENSE_MODEL = os.getenv("VEXILON_CONDENSE_MODEL", DEFAULT_MODEL_LLM)
VERIFY_MODEL = os.getenv("VEXILON_VERIFY_MODEL", DEFAULT_MODEL_LLM)


# Generation Limits (Tokens)
RAG_MAX_TOKENS = 4096
REVIEWER_MAX_TOKENS = 4096

# Brain: Local Embeddings (Search) + Cloud LLM (Claude)
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")  # 512-token window
MAX_EMBED_TOKENS = int(os.getenv("VEXILON_MAX_EMBED_TOKENS", 4096))  # Sane offset-mapping limit
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 450))  # Sized for BGE-small
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", 40))  # More context depth

# Memory / Context Condensation
CONDENSE_QUERY_HISTORY_TURNS = int(os.getenv("CONDENSE_QUERY_HISTORY_TURNS", 3))
CONDENSE_QUERY_CONTENT_MAX_LEN = int(os.getenv("CONDENSE_QUERY_CONTENT_MAX_LEN", 200))

import re
import logging

# Input Sanitization (for prompt injection prevention)
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", 10000))
LOG_SUSPICIOUS_INPUTS = os.getenv("LOG_SUSPICIOUS_INPUTS", "true").lower() == "true"

PROMPT_INJECTION_PATTERNS = [
    re.compile(r, re.IGNORECASE)
    for r in [
        r"ignore\s+.*instructions",
        r"forget\s+.*instructions",
        r"disregard\s+.*rules",
        r"you\s+are\s+now\s+.+\s+instead",
        r"new\s+(system\s+|)prompt:",
        r"#\#\#\s*(system\s+|)instructions",
        r"\[\[SYSTEM\]\]",
        r"override\s+.*instructions",
        r"disable\s+.*safety",
        r"\bjailbreak\b",
        r"developer\s+mode",
        r"sudo\s+mode",
        r"roleplay\s+as",
        r"pretend\s+(you\s+are|to\s+be)",
        r"forget\s+everything\s+above",
        r"discard\s+.*instructions",
    ]
]


def sanitize_input(user_input: str) -> tuple[str, bool]:
    """
    Check for prompt injection patterns and length limits.
    Returns (sanitized_input, was_flagged).
    """
    if not user_input:
        return user_input, False

    injection_found = False
    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(user_input):
            injection_found = True
            if LOG_SUSPICIOUS_INPUTS:
                logging.warning(
                    f"[security] Prompt injection pattern detected: {pattern.pattern[:100]}..."
                )
            break

    too_long = len(user_input) > MAX_INPUT_LENGTH
    if too_long and LOG_SUSPICIOUS_INPUTS:
        logging.warning(
            f"[security] Input length ({len(user_input)}) exceeds limit of {MAX_INPUT_LENGTH}. Input rejected."
        )

    flagged = injection_found or too_long
    sanitized = user_input[:MAX_INPUT_LENGTH]

    return sanitized, flagged


# Verification Bot (for reducing hallucinations)
VERIFY_ENABLED = os.getenv("VERIFY_ENABLED", "true").lower() == "true"

# Rate Limiting (for abuse prevention and cost control)
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "100"))

VERIFY_SYSTEM_PROMPT = """You are a verification assistant. Your job is to verify that the claims made in an AI response are supported by the provided source citations.

For each claim in the response:
1. Check if the quoted text actually supports the claim being made
2. Check if the citation (document name, article/section, page number) is accurate
3. Identify any hallucinations, misquotes, or unsupported claims

Respond in this format:
- VERIFIED: [claim summary] — the quote supports the claim
- DISPUTED: [claim summary] — the quote does NOT support the claim
- UNCERTAIN: [claim summary] — cannot verify due to unclear citation

If all claims are verified, respond with "ALL_CLAIMS_VERIFIED".
If there are disputed claims, list them with explanations."""


_SECONDS_IN_MINUTE = 60
_SECONDS_IN_HOUR = 3600


class RateLimiter:
    """Simple in-memory rate limiter for request throttling."""

    def __init__(self, max_per_minute: int = 10, max_per_hour: int = 100):
        self.minute_limit = max_per_minute
        self.hour_limit = max_per_hour
        self.requests: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _clean_old_requests(self, key: str) -> None:
        """Remove requests older than 1 hour and cleans up the user entry if empty."""
        now = time.time()
        hour_ago = now - _SECONDS_IN_HOUR
        if key in self.requests:
            self.requests[key] = [t for t in self.requests[key] if t > hour_ago]
            if not self.requests[key]:
                del self.requests[key]

    def is_allowed(self, user_id: str = "default") -> tuple[bool, str]:
        """Check if request is allowed. Returns (allowed, message)."""
        with self._lock:
            self._clean_old_requests(user_id)
            now = time.time()
            minute_ago = now - _SECONDS_IN_MINUTE

            requests = self.requests.get(user_id, [])
            recent_requests = [t for t in requests if t > minute_ago]

            if len(recent_requests) >= self.minute_limit:
                return (
                    False,
                    f"Rate limit exceeded: {self.minute_limit} requests per minute. Please wait before trying again.",
                )

            if len(requests) >= self.hour_limit:
                return (
                    False,
                    f"Rate limit exceeded: {self.hour_limit} requests per hour. Please try again later.",
                )

            self.requests.setdefault(user_id, []).append(now)
            return True, ""


_rate_limiter = RateLimiter(
    max_per_minute=RATE_LIMIT_PER_MINUTE, max_per_hour=RATE_LIMIT_PER_HOUR
)

# Two-Bot Self-Review Pipeline (Issue #104)
USE_REVIEWER = os.getenv("USE_REVIEWER", "false").lower() == "true"


def get_vexilon_info():
    """
    1. Get Version (Priority: Env Var -> Baked File -> Fallback)
    2. Get Python and OS metadata
    3. Print a beautiful startup banner
    """
    import platform

    # Priority 1: Env Var
    version = os.getenv("VEXILON_VERSION")
    source = "External/CI"

    if not version:
        # Priority 2: Baked-in Build File
        try:
            with open("/app/build_version.txt", "r") as f:
                version = f.read().strip()
                source = "Local Build"
        except FileNotFoundError:
            # Priority 3: Fallback (Never dev!)
            version = "unspecified-local"
            source = "fallback"

    py_ver = sys.version.split()[0]
    os_info = platform.system()

    # The Banner
    print("=" * 50)
    print(f" VEXILON VERSION : {version} ({source})")
    print(f" PYTHON VERSION  : {py_ver}")
    print(f" RUNTIME OS      : {os_info}")
    print("=" * 50, flush=True)

    return {"ver": version, "src": source, "py": py_ver, "os": os_info}


# Initialise version and logging at imports
_info = get_vexilon_info()
VEXILON_VERSION = _info["ver"]
VEXILON_USERNAME = os.getenv("VEXILON_USERNAME", "admin")
VEXILON_PASSWORD = os.getenv("VEXILON_PASSWORD")

# ─── Typing ──────────────────────────────────────────────────────────────────
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anthropic
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import faiss
    import gradio as gr

# ─── Clients ─────────────────────────────────────────────────────────────────
_embed_model: "SentenceTransformer | None" = None
_anthropic_client: "anthropic.AsyncAnthropic | None" = None


def get_embed_model() -> "SentenceTransformer":
    global _embed_model
    if _embed_model is None:
        # Stabilize CPU usage in shared-resource environments (HF Spaces/CI)
        # We only do this at RUNTIME. Doing this during BUILD causes infinite hangs.
        if os.getenv("HF_SPACE_ID") or os.getenv("EXTERNAL_CI"):
            for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(var, "1")

        print(f"[embed] Loading local embedding model '{EMBED_MODEL}'…")
        
        # Ensure we are truly offline to avoid lock-file permission errors 
        # in the root-owned (read-only) hf_cache.
        if os.getenv("TRANSFORMERS_OFFLINE") == "1":
             os.environ["HF_HUB_OFFLINE"] = "1"

        from sentence_transformers import SentenceTransformer

        # Use the requested device (cpu) to avoid CUDA detection overhead.
        _embed_model = SentenceTransformer(EMBED_MODEL, device="cpu")

        # Sane limit for offset mapping (4096 is plenty for any single page).
        _embed_model.max_seq_length = MAX_EMBED_TOKENS
        if hasattr(_embed_model, "tokenizer"):
            _embed_model.tokenizer.model_max_length = MAX_EMBED_TOKENS
        print("[embed] Embedding model ready.")
    return _embed_model


# Embedding dimension (derived from model to prevent FAISS mismatch)
# Default is 384 for BAAI/bge-small-en-v1.5
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))


def get_anthropic() -> "anthropic.AsyncAnthropic":
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic

        # Reads ANTHROPIC_API_KEY from environment automatically; raises AuthenticationError if missing
        _anthropic_client = anthropic.AsyncAnthropic()
    return _anthropic_client


def _get_rag_source_files() -> list[Path]:
    """
    Recursively scan LABOUR_LAW_DIR for Markdown files ONLY.
    PDFs and Forensic Integrity Audits are completely ignored for indexing.
    The tests/ subdirectory is excluded.
    """
    if not LABOUR_LAW_DIR.exists():
        return []
    tests_dir = LABOUR_LAW_DIR / "tests"
    mds = [
        p for p in LABOUR_LAW_DIR.rglob("*.md") 
        if not p.is_relative_to(tests_dir) and not p.name.endswith(".integrity.md")
    ]
    return sorted(mds, key=lambda p: str(p))

def _get_download_source_files() -> list[Path]:
    """
    Recursively scan LABOUR_LAW_DIR for PDF files ONLY (for human downloads).
    The tests/ subdirectory is excluded.
    """
    if not LABOUR_LAW_DIR.exists():
        return []
    tests_dir = LABOUR_LAW_DIR / "tests"
    pdfs = [p for p in LABOUR_LAW_DIR.rglob("*.pdf") if not p.is_relative_to(tests_dir)]
    return sorted(pdfs, key=lambda p: str(p))


def _get_source_name(stem: str) -> str:
    """
    Parse source_name from internal filename convention: [Index]_[Category]_[Title].
    Also handles [Index]_[Title] and fallback to title-cased filename.
    """
    parts = stem.split("_", 2)
    if len(parts) == 3:
        # Index_Category_Title
        return parts[2]
    elif len(parts) == 2:
        # Index_Title
        return parts[1]
    # Fallback / No underscores
    return stem.replace("_", " ").title()


def get_knowledge_manifest() -> str:
    """
    Dynamically scan the labour_law directory and build a formatted list for the system prompt.
    Files follow the naming convention: [Index]_[Category]_[Title].pdf
    Example: 1_Primary_BCGEU 19th Main Agreement.pdf
    """
    files = _get_rag_source_files()
    if not files:
        return "No documents available."

    lines = []
    for f in files:
        stem = f.stem
        source_name = _get_source_name(stem)
        parts = stem.split("_", 2)
        if len(parts) == 3:
            idx, cat, _ = parts
            lines.append(f"{idx}. {source_name} ({cat})")
        elif len(parts) == 2:
            idx, _ = parts
            lines.append(f"{idx}. {source_name} (Reference)")
        else:
            lines.append(f"- {source_name} (Uncategorized)")

    return "\n".join(lines)


def build_pdf_download_links() -> str:
    """
    Generate HTML for individual PDF and MD download links using Gradio's /gradio_api/file= endpoint.
    Returns HTML-formatted links for each source in the labour_law directory.
    """
    import html

    files = _get_download_source_files()
    if not files:
        return ""

    # Use relative path for cross-environment compatibility (local dev + Docker container)
    lines = ["<b>Download Documents:</b>", "<ul>"]
    for f in files:
        stem = f.stem
        source_name = _get_source_name(stem)
        parts = stem.split("_", 2)
        if len(parts) == 3:
            idx, cat, _ = parts
            display_name = f"{idx}. {source_name} ({cat})"
        elif len(parts) == 2:
            idx, _ = parts
            display_name = f"{idx}. {source_name} (Reference)"
        else:
            display_name = source_name

        # Use relative path for Gradio's /gradio_api/file= endpoint (works in both local and container)
        file_path = str(f.relative_to(Path(".")))
        lines.append(
            f'<li><a href="/gradio_api/file={file_path}" target="_blank">{html.escape(display_name)}</a></li>'
        )

    lines.append("</ul>")
    return "\n".join(lines)


DEVELOPER_MODE = os.getenv("DEVELOPER_MODE", "false").lower() == "true"

PROMPTS_DIR = Path(__file__).parent / "prompts"

def get_system_prompt(developer_mode: bool = False) -> str:
    """Load the default system prompt, optionally with developer extensions."""
    path = PROMPTS_DIR / ("developer.txt" if developer_mode else "steward.txt")
    if path.is_file():
        return path.read_text(encoding="utf-8")
    # Robust fallback including required formatting placeholders
    return (
        "You are Vexilon, a professional assistant for BCGEU union stewards.\n\n"
        "Knowledge Base:\n{manifest}\n\n"
        "{verify_message}"
    )

GLOBAL_MANDATORY_RULES = """--- MANDATORY OPERATIONAL RULES (OVERRIDING) ---
1. ANSWER FROM EXCERPTS ONLY: Base your answer strictly on the provided excerpts. If the specific text was not retrieved, suggest the user ask about that section directly. NEVER fabricate contract language.
2. CITATIONS: Every claim MUST be supported by a verbatim quote in a blockquote (> "...") followed by its citation (Document, Article, Page).
3. HIERARCHY: Lead with the Collective Agreement. Use Statutes only to reinforce the legal framework.
4. GRIEVANCE FILING (CRITICAL): If a steward asks for resolution steps or once the facts of a potential violation are gathered, you MUST proactively recommend filing a grievance. 
   - YOU MUST APPEND a final section titled '### 📁 Resolution & Next Steps'.
   - THIS SECTION MUST CONTAIN this link: [Download BCGEU Grievance Form](/gradio_api/file=data/labour_law/forms/BCGEU%20Grievance%20Form.pdf)
   - YOU MUST ALSO mention the 'BCGEU Grievance Form Guide.md' for instructions.
   - DISCLAIMER: You MUST include this verbatim: "Note: Viability of this grievance will be assessed by the staff representative and/or arbitrator, not by the steward."
5. NO MERIT ASSESSMENT: Do NOT judge the merit, viability, or likelihood of success of a grievance. Your role is to identify potential violations and facilitate the filing process.
6. TONE: Professional, authoritative, and forensic.
----------------------------------
"""


def get_persona_prompt(mode_name: str) -> str:
    """Helper to load system prompts for different operational modes."""
    paths = {
        "Direct": PROMPTS_DIR / "direct_staff_rep.txt",
        "Defend": PROMPTS_DIR / "case_builder.txt",
    }
    fallbacks = {
        "Direct": "You are a BCGEU Staff Rep providing DIRECT OPERATIONAL GUIDANCE.\n\nKnowledge Base:\n{manifest}\n\n{verify_message}",
        "Defend": "You are a BCGEU Staff Rep specializing in Grievance Drafting.\n\nKnowledge Base:\n{manifest}\n\n{verify_message}",
    }
    
    path = paths.get(mode_name)
    content = path.read_text(encoding="utf-8") if path and path.is_file() else fallbacks.get(mode_name, get_system_prompt(DEVELOPER_MODE))
        
    return f"{GLOBAL_MANDATORY_RULES}\n\n{content}"



VERIFY_STEWARD_MESSAGE = os.getenv(
    "STEWARD_VERIFY_MESSAGE", "Verify w/ Area Office: 604-291-9611"
)

# ─── Two-Bot Review Prompt (Bot B) ──────────────────────────────────────────────
REVIEWER_SYSTEM_PROMPT = """You are a Senior BCGEU Staff Representative reviewing a junior steward's output for accuracy and completeness.

Your role is to VERIFY the steward's response before it reaches the member. You must critically evaluate:
1. CITATIONS: Are all quoted articles/sections accurate and verbatim?
2. NEXUS: Does the analysis properly connect the facts to the relevant contract language?
3. PROCEDURES: Are the suggested steps correct and in proper order?
4. GAPS: Did the steward miss anything important?

Scoring criteria (1-10):
- 9-10: Approved - ready for member
- 7-8: Minor issues - recommend corrections but safe to use
- 5-6: Significant issues - needs revision before use
- 1-4: Escalate to Area Office - contains errors or dangerous advice

Response format:
SCORE: [1-10]
VERIFIED STEPS: [final safe instructions, corrections if needed]
ISSUES: [specific errors or gaps found, if any]
ESCALATE: [yes/no - only if score < 5]
"""

# Millhaven Factors for Off-Duty Conduct Audit (Issue #154) - DEPRECATED in favor of TestRegistry
# (kept as fallback constants until logic migration is complete)
MILLHAVEN_FACTORS_PATH = Path("./prompts/millhaven_audit_criteria.txt")
MILLHAVEN_FACTORS = ""
if MILLHAVEN_FACTORS_PATH.is_file():
    MILLHAVEN_FACTORS = MILLHAVEN_FACTORS_PATH.read_text(encoding="utf-8")

OFF_DUTY_KEYWORDS = {
    "off-duty", "personal conduct", "nexus", "outside of work", "facebook",
    "reddit", "social media", "arrest", "charged", "personal life",
    "instagram", "twitter", "tiktok", "personal blog", "off-site"
}

# ─── Test Registry (Issue #160) ────────────────────────────────────────────────
class TestDoctrine:
    """A labour law test or doctrine with trigger keywords and logic excerpts."""
    def __init__(self, name: str, keywords: set[str], content: str, file_path: Path):
        self.name = name
        self.keywords = keywords
        self.content = content
        self.file_path = file_path

class TestRegistry:
    """Registry for modular labour-law tests/doctrines."""
    def __init__(self):
        self.tests: list[TestDoctrine] = []
        self._lock = threading.Lock()

    def load(self, directory: Path) -> None:
        """Scan directory for .md files and parse them into the registry."""
        if not directory.exists():
            print(f"[registry] Warning: {directory} does not exist.")
            return

        with self._lock:
            self.tests = []
            for f in directory.glob("*.md"):
                try:
                    text = f.read_text(encoding="utf-8")
                    lines = text.split("\n")
                    
                    # Simple parser for "Keywords: k1, k2"
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
                    print(f"[registry] Failed to load {f.name}: {e}")
            print(f"[registry] Loaded {len(self.tests)} tests from {directory.name}")

    def find_matches(self, query: str) -> list[TestDoctrine]:
        """Find all tests whose keywords appear in the lowercased query."""
        q_lower = query.lower()
        with self._lock:
            return [test for test in self.tests if any(k in q_lower for k in test.keywords)]

_test_registry = TestRegistry()

# ─── Chunking ─────────────────────────────────────────────────────────────────


def chunk_text(
    full_text: str, token_data: list[tuple[int, int, int, str]], source_name: str
) -> list[dict]:
    """
    Split *full_text* into overlapping token-based chunks across the whole document.
    Uses 'token_data' [(char_start, char_end, page_num, header)] to preserve metadata.
    Returns list of dicts: {text, page, source, header, chunk_index}.
    """
    chunks = []
    if not token_data:
        return chunks

    # Safety guard: Ensure the loop always advances. If configuration is
    # invalid (size <= overlap), we force a minimum step of 1 token.
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)

    idx = 0
    start = 0
    while start < len(token_data):
        end = min(start + CHUNK_SIZE, len(token_data))

        # Get metadata from the first token of the chunk
        char_start, _, page_num, header = token_data[start]
        # End Char index is from the last token of the chunk
        _, char_end, _, _ = token_data[end - 1]

        # Contextual Breadcrumb: Prepend source + header
        prefix = f"[{source_name} - {header}] " if header else f"[{source_name}] "
        chunk_text_str = prefix + full_text[char_start:char_end]

        chunks.append(
            {
                "text": chunk_text_str,
                "page": page_num,
                "source": source_name,
                "header": header,
                "chunk_index": idx,
            }
        )
        idx += 1
        start += step
    return chunks


# ─── PDF Loader ───────────────────────────────────────────────────────────────


def _is_toc_or_index_page(page_text: str) -> bool:
    """
    Detect whether a page is a Table of Contents or alphabetical Index page.

    These navigational pages mention every article/clause by name but contain
    no substantive content. Indexing them causes TOC entries to dominate
    semantic search results, drowning out actual contract text.

    Returns True if the page appears to be TOC/index content.
    """
    import re

    lines = [l.strip() for l in page_text.split("\n") if l.strip()]
    if not lines:
        return False

    # Heuristic 1: 3+ dot-leader lines (..........) strongly indicates a TOC page
    dot_leader_count = sum(1 for l in lines if l.count(".") >= 8 and ".." in l)
    if dot_leader_count >= 3:
        return True

    # Heuristic 2: >40% of lines match index-style pattern "Some Text ... NN"
    # e.g. "Abandonment of Position, 10.10 ........ 23"
    index_line_re = re.compile(r".{10,}\.\s*\d{1,3}\s*$")
    index_count = sum(1 for l in lines if index_line_re.search(l))
    if len(lines) >= 5 and index_count / len(lines) > 0.4:
        return True

    return False


def _clean_page_text(page_text: str) -> str:
    """
    Remove noise artifacts injected by web-based PDF extraction.

    BC statute PDFs (ESA, Human Rights Code, Labour Relations Code) were
    exported from bclaws.gov.bc.ca and contain repeated URL lines and
    date-stamps that waste ~30 tokens per chunk and dilute embedding quality.
    """
    import re

    # Remove bclaws.gov.bc.ca URL lines
    page_text = re.sub(
        r"https?://www\.bclaws\.gov\.bc\.ca/\S*",
        "",
        page_text,
    )
    # Remove date/time stamps from web-to-PDF artifacts, e.g.:
    # "17/03/2026, 08:44 Employment Standards Act"
    page_text = re.sub(
        r"\d{2}/\d{2}/\d{4},?\s*\d{2}:\d{2}\s+[A-Z][^\n]*",
        "",
        page_text,
    )
    # Collapse runs of 3+ blank lines down to a single blank line
    page_text = re.sub(r"\n{3,}", "\n\n", page_text)
    return page_text.strip()


def load_md_chunks(md_path: Path) -> list[dict]:
    """
    Parse a Markdown file into tokens and chunks.
    Markdown is preferred for structured summaries as it preserves semantic hierarchies
    better than PDF extraction.
    """
    content = md_path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    source_name = _get_source_name(md_path.stem)
    print(f"[loader] Parsing Markdown '{source_name}'…")

    tokenizer = get_embed_model().tokenizer
    token_metadata = []
    
    # Split into header-delimited sections and filter TOC blocks
    current_header = ""
    lines = content.split("\n")
    
    # Group lines into sections by header, then filter TOC sections
    sections: list[tuple[str, list[str]]] = []  # (header, lines)
    current_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            if current_lines:
                sections.append((current_header, current_lines))
            current_header = stripped.lstrip("#").strip().upper()
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_header, current_lines))
    
    # Rebuild content without TOC sections and tokenize
    filtered_lines: list[str] = []
    for header, section_lines in sections:
        section_text = "\n".join(section_lines)
        if _is_toc_or_index_page(section_text):
            print(f"[loader]   Skipping TOC/index section: {header or '(untitled)'}")
            continue
        filtered_lines.extend(section_lines)
    
    filtered_content = "\n".join(filtered_lines)
    if not filtered_content.strip():
        return []
    
    # Tokenize filtered content
    current_header = ""
    char_offset = 0
    for line in filtered_lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            current_header = stripped.lstrip("#").strip().upper()
        
        # Page estimation removed per user request (Issue #177)
        # Markdown sources are treated as continuous text (Default to p. 1)
        page_num = 1
        
        encoding = tokenizer(
            line,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        for start_off, end_off in encoding.offset_mapping:
            token_metadata.append(
                (char_offset + start_off, char_offset + end_off, page_num, current_header)
            )
        
        char_offset += len(line) + 1  # +1 for newline

    return chunk_text(filtered_content, token_metadata, source_name)


# ─── FAISS Index ──────────────────────────────────────────────────────────────
def embed_texts(texts: list[str]) -> "np.ndarray":
    """Embed a list of texts using the local sentence-transformers model. Returns (N, EMBED_DIM) float32 array."""
    import numpy as np

    model = get_embed_model()
    # encode() handles batching internally; show_progress_bar=False keeps logs clean
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.astype(np.float32)


def build_index(chunks: list[dict]) -> "faiss.IndexFlatIP":
    """
    Embed all chunks and build a FAISS inner-product index.
    Vectors are L2-normalised so inner product == cosine similarity.
    """
    import faiss

    texts = [c["text"] for c in chunks]
    print(f"[index] Embedding {len(texts)} chunks locally (may take 30–90 s on CPU)…")
    t0 = time.time()
    vectors = embed_texts(texts)
    print(f"[index] Embeddings complete in {time.time() - t0:.1f}s")
    # L2-normalise for cosine similarity via inner product
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)
    print(f"[index] FAISS index built — {index.ntotal} vectors")
    return index


def search_index(
    index: "faiss.IndexFlatIP",
    chunks: list[dict],
    query: str,
    top_k: int = SIMILARITY_TOP_K,
) -> list[dict]:
    """Return the top-k most similar chunks for *query*."""
    import faiss

    query_vec = embed_texts([query])  # (1, EMBED_DIM)
    faiss.normalize_L2(query_vec)
    _scores, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


# ─── RAG App State (module-level, built at startup) ──────────────────────────
_chunks: list[dict] = []
_index: "faiss.IndexFlatIP | None" = None


def save_index(index: "faiss.IndexFlatIP", chunks: list[dict]) -> None:
    """Persist the FAISS index and chunk metadata to .pdf_cache/ for fast cold starts."""
    import faiss

    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    print(f"[index] Saved index → {INDEX_PATH} and chunks → {CHUNKS_PATH}")


def load_precomputed_index() -> (
    tuple["faiss.IndexFlatIP", list[dict]] | tuple[None, None]
):
    """
    Load a pre-computed FAISS index and chunks from disk if both exist.
    Returns (index, chunks) on success, (None, None) if files are missing.
    """
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        return None, None
    print(f"[startup] Loading pre-computed index from {INDEX_PATH}…")
    import faiss

    index = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    print(
        f"[startup] Pre-computed index loaded — {index.ntotal} vectors, {len(chunks)} chunks."
    )
    return index, chunks


def _fetch_pdf_cache_if_missing() -> None:
    """
    Download pre-computed index and chunks from GitHub raw if not present locally.
    This enables fast cold starts on HuggingFace Spaces where PDFs aren't bundled.
    """
    import urllib.request
    import urllib.error

    # Ensure cache directory exists
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Check if both files already exist
    if INDEX_PATH.exists() and CHUNKS_PATH.exists():
        return

    # Build URLs for the missing files
    base = _GITHUB_RAW_BASE
    urls = {}
    if not INDEX_PATH.exists():
        urls[INDEX_PATH] = f"{base}/.pdf_cache/index.faiss"
    if not CHUNKS_PATH.exists():
        urls[CHUNKS_PATH] = f"{base}/.pdf_cache/chunks.json"

    # Download missing files
    for dest_path, url in urls.items():
        print(f"[fetch] Downloading {dest_path.name} from {url}…")
        try:
            urllib.request.urlretrieve(url, dest_path)
            print(f"[fetch] Saved {dest_path}")
        except (urllib.error.URLError, OSError) as e:
            print(f"[fetch] Warning: could not fetch {dest_path.name}: {e}. Will build index from source.")


def build_index_from_sources(force: bool = False) -> None:
    """
    Parse all source files (PDF and MD) in LABOUR_LAW_DIR, embed them, 
    and write the pre-built index to .pdf_cache/index.faiss + .pdf_cache/chunks.json.

    Args:
        force (bool): If True, ignores the existing manifest and rebuilds the
                      index from scratch. Defaults to False.

    SMART REFRESH: This function generates a manifest.json containing SHA256 hashes
    of all source files. If the current files match the stored manifest (and the
    index exists), the expensive embedding process is skipped.
    """
    import hashlib

    global _chunks, _index

    all_files = _get_rag_source_files()
    if not all_files:
        print("[build] No source files found to index!")
        return

    # Calculate hashes for the current source set
    current_manifest = {}
    for source_file in all_files:
        hasher = hashlib.sha256()
        with open(source_file, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        current_manifest[source_file.name] = hasher.hexdigest()

    # Check against stored manifest
    if not force and MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r") as f:
                stored_manifest = json.load(f)
            if (
                stored_manifest == current_manifest
                and INDEX_PATH.exists()
                and CHUNKS_PATH.exists()
            ):
                print(
                    "[build] Smart Refresh: No changes detected in data/labour_law/. Skipping indexing."
                )
                return
        except Exception as e:
            print(f"[build] Failed to read manifest: {e}. Rebuilding anyway.")

    # Ensure cache directory exists before writing
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[build] Scanning Sources in {LABOUR_LAW_DIR}…")
    _chunks = []
    for f in all_files:
        if f.suffix.lower() == ".pdf":
            _chunks.extend(load_pdf_chunks(f))
        elif f.suffix.lower() == ".md":
            _chunks.extend(load_md_chunks(f))

    num_chunks = len(_chunks)
    print(f"[build] Total {num_chunks} chunks loaded from {len(all_files)} files.")
    _index = build_index(_chunks)
    save_index(_index, _chunks)

    # Save the new manifest
    with open(MANIFEST_PATH, "w") as f:
        json.dump(current_manifest, f, indent=2)
    print(f"[build] Index and manifest written to {PDF_CACHE_DIR}.")
    print("[build] Indexing complete.")


def startup(force_rebuild: bool = False) -> None:
    """
    Initialise the vector index and load document chunks.

    If force_rebuild=False (default), we try to load a pre-computed index
    from .pdf_cache/. If missing, we fall back to build_index_from_sources()
    which embeds documents from scratch.
    """
    global _chunks, _index
    get_anthropic()  # Ping early to catch missing ANTHROPIC_API_KEY

    # Try to fetch pre-computed index from GitHub if not present locally
    # (Needed for HuggingFace Spaces where PDFs aren't bundled)
    print(f"[startup] Starting Vexilon {VEXILON_VERSION}…")
    if DEVELOPER_MODE:
        print("[startup] DEVELOPER_MODE is ACTIVE. Proactive suggestions enabled.")
    _fetch_pdf_cache_if_missing()
    _test_registry.load(TESTS_DIR)

    if not force_rebuild:
        index, chunks = load_precomputed_index()
        if index is not None and chunks is not None:
            _index = index
            _chunks = chunks
            # Warm the embedding model so the first query isn't slow
            get_embed_model()
            print("[startup] Ready.")
            return

    # ── Slow path: delegate to the API-key-free build function ────────────
    build_index_from_sources(force=force_rebuild)
    print("[startup] Ready.")


# ─── RAG Query ────────────────────────────────────────────────────────────────
async def condense_query(message: str, history: list[dict]) -> str:
    """
    Use Claude to condense conversation history and the latest message into a
    standalone, search-friendly query.
    """
    if not history:
        return message

    client = get_anthropic()

    # Simple history formatting
    context_lines = []
    for turn in history[
        -CONDENSE_QUERY_HISTORY_TURNS:
    ]:  # Uses configured history context
        role = "User" if turn["role"] == "user" else "Assistant"
        # Truncate content for the condensation prompt.
        # In Gradio 6, content can be a string or a list of blocks.
        raw_content = turn["content"]
        if isinstance(raw_content, list):
            # Extract text from message parts
            text_parts = [
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in raw_content
            ]
            raw_content = "".join(text_parts)

        # Ensure it is a string before slicing/concatenating
        raw_content = str(raw_content)
        msg_len = CONDENSE_QUERY_CONTENT_MAX_LEN
        content = raw_content[:msg_len] + ("..." if len(raw_content) > msg_len else "")
        context_lines.append(f"{role}: {content}")

    context_str = "\n".join(context_lines)

    prompt = (
        "Given the following conversation history and a follow-up question, "
        "rephrase the question to be a standalone search query that captures "
        "the full intent. If the question is about the bot's own capabilities, "
        "make the query specifically about its documentation and manifest. "
        "Only provide the rephrased query, nothing else.\n\n"
        f"History:\n{context_str}\n\n"
        f"Follow-up question: {message}\n\n"
        "Standalone query:"
    )

    try:
        response = await client.messages.create(
            model=CONDENSE_MODEL,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        condensed = response.content[0].text.strip().strip('"')
        return condensed
    except Exception as exc:
        # We catch generic Exception here since anthropic is deferredly imported
        print(f"[rag] Query condensation failed: {exc}. Using raw message.")
        return message



# ─── Export / Import Functions ────────────────────────────────────────────────
MAX_IMPORT_SIZE_BYTES = 500 * 1024  # 500KB limit





async def rag_stream(
    message: str, history: list[dict]
) -> AsyncIterator[tuple[str, str]]:
    """
    Stream response tokens from Claude, yielding (text_chunk, context) tuples.
    The context is yielded once at the start with empty text, then text chunks follow with empty context.
    """
    if not _index or not _chunks:
        yield (
            "\n\n⚠️ Knowledge base not loaded. Please refresh or rebuild the index.",
            "",
        )
        return

    # Rewrite query for RAG if there is history
    query = await condense_query(message, history)
    relevant_chunks = search_index(_index, _chunks, query)

    # Build context block from retrieved chunks
    context_parts = []
    for chunk in relevant_chunks:
        context_parts.append(
            f"[Source: {chunk.get('source', 'Unknown')}, Page: {chunk['page']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # Build message list for Claude: prior history + new user message
    messages = []
    for turn in history:
        if turn["role"] in ("user", "assistant"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": message})

    client = get_anthropic()
    try:
        async with client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=RAG_MAX_TOKENS,
            # Two cache breakpoints:
            # 1. Static instructions — identical every request; cached once per session.
            # 2. Dynamic excerpts — changes per query; cached separately.
            # This avoids re-caching the instructions block whenever the excerpts change.
            system=[
                {
                    "type": "text",
                    "text": get_system_prompt(DEVELOPER_MODE).format(
                        manifest=get_knowledge_manifest(),
                        verify_message=VERIFY_STEWARD_MESSAGE,
                    ),
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": (
                        "--- AGREEMENT EXCERPTS ---\n\n"
                        + context
                        + "\n\n--- END EXCERPTS ---"
                    ),
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            messages=messages,
        ) as stream:
            # Yield context first, then stream text chunks
            yield ("", context)
            async for text_chunk in stream.text_stream:
                yield (text_chunk, "")
            # Log cache effectiveness so we can verify caching is working.
            final = await stream.get_final_message()
            usage = final.usage
            cache_created = usage.cache_creation_input_tokens or 0
            cache_read = usage.cache_read_input_tokens or 0
            print(
                f"[rag] Tokens — input: {usage.input_tokens}, "
                f"cache_create: {cache_created}, cache_read: {cache_read}, "
                f"output: {usage.output_tokens}"
            )
            if final.stop_reason == "max_tokens":
                yield ("\n\n⚠️ Response truncated. Please ask for the rest of the answer.", "")
    except Exception as exc:
        yield (f"\n\n⚠️ API error: {exc}", "")


async def get_rag_context(message: str, history: list[dict]) -> tuple[str, str]:
    """Get context (excerpts) for a query without generating a response."""
    query = await condense_query(message, history)
    relevant_chunks = search_index(_index, _chunks, query)

    context_parts = []
    for chunk in relevant_chunks:
        context_parts.append(
            f"[Source: {chunk.get('source', 'Unknown')}, Page: {chunk['page']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)
    return query, context


async def verify_response(assistant_response: str, context: str) -> str:
    """
    Use a second bot to verify that the claims in the response are supported
    by the provided context/citations. Returns verification result.
    """
    if not VERIFY_ENABLED:
        return ""

    client = get_anthropic()
    try:
        verification_messages = [
            {
                "role": "user",
                "content": f"""RESPONSE TO VERIFY:
{assistant_response}

SOURCE CITATIONS AND CONTEXT:
{context}""",
            }
        ]

        verify_resp = await client.messages.create(
            model=VERIFY_MODEL,
            max_tokens=512,
            system=[{"type": "text", "text": VERIFY_SYSTEM_PROMPT}],
            messages=verification_messages,
        )

        return verify_resp.content[0].text
    except Exception as exc:
        return f"⚠️ Verification unavailable: {exc}"


# ─── Verification Bot (for reducing hallucinations) ───────────────────────────


def get_ground_truth_for_review(response: str, all_chunks: list[dict]) -> str:
    """
    Extract cited articles from Bot A's response and fetch their full text from context.
    This prevents circular verification (Issue #183) by giving Bot B independent context.
    Improved robustness (dashes, plurals, header matching) based on PR feedback.
    """
    # 1. Match formal Bot A citations (handles various dashes and optional brackets)
    # e.g., '— [Doc Name], Article 10.4' or '- Doc Name, Section 5'
    formal_regex = re.compile(
        r"(?:[-–—]\s*)?(?:\[(?P<doc>[^\]]+)\]|(?P<doc_raw>[^,]+)),\s*(?:Article|Section|Clause|Appendix)\s*(?P<art>[\w\d.]+)", 
        re.IGNORECASE
    )
    # 2. Match informal mentions in text: 'Article 10.4'
    informal_regex = re.compile(
        r"(?:Article|Section|Clause|Appendix)\s+(?P<art>[\w\d.]+)", 
        re.IGNORECASE
    )
    
    cited_targets = []
    # Extract from formal citations first
    for m in formal_regex.finditer(response):
        doc_hint = (m.group("doc") or m.group("doc_raw")).strip().lower()
        art_root = m.group("art").split(".")[0].upper()
        cited_targets.append((doc_hint, art_root))
    
    # Supplement with informal mentions
    for m in informal_regex.finditer(response):
        art_root = m.group("art").split(".")[0].upper()
        if not any(t[1] == art_root for t in cited_targets):
            cited_targets.append((None, art_root))
            
    if not cited_targets:
        return ""
        
    truth_parts = []
    seen_chunk_ids = set()
    
    # 3. Pulll chunks matching these targets (using regex for robust header parsing)
    header_num_re = re.compile(r"(?:ARTICLE|SECTION|APPENDIX|CLAUSE)\s+(?P<num>[\w\d.]+)", re.IGNORECASE)
    
    for doc_hint, art_num in cited_targets:
        for chunk in all_chunks:
            cid = (chunk["source"], chunk["chunk_index"])
            if cid in seen_chunk_ids:
                continue
            
            header = chunk.get("header", "").upper()
            m_header = header_num_re.search(header)
            
            match_header = m_header and m_header.group("num") == art_num
            match_doc = (doc_hint is None) or (doc_hint in chunk["source"].lower())
            
            if match_header and match_doc:
                truth_parts.append(
                    f"[Source: {chunk['source']}, Page: {chunk['page']}]\n{chunk['text']}"
                )
                seen_chunk_ids.add(cid)
                
    # Limit to 15 chunks (roughly 6k-8k tokens) for performance and window safety
    return "\n\n---\n\n".join(truth_parts[:15])


# ─── Two-Bot Review Stream (Bot B) ─────────────────────────────────────────────
async def review_stream(
    raw_response: str, query: str, context: str
) -> AsyncIterator[str]:
    """
    Bot B: Senior BCGEU rep reviewing Bot A's (steward) output.
    Yields the review result with score and verified steps.
    """
    client = get_anthropic()
    # Independent re-retrieval for Bot B (Issue #183)
    # Extracts cited articles from Bot A's response and fetches ground truth context.
    ground_truth = get_ground_truth_for_review(raw_response, _chunks)
    if not ground_truth:
        # Fallback to Bot A's context if no citations were generated or re-retrieval failed
        ground_truth = context

    review_prompt = f"""Review the following steward's response for accuracy and completeness using the provided GROUND TRUTH context.

QUERY: {query}

STEWARD'S RESPONSE:
{raw_response}

GROUND TRUTH CONTEXT (FOR VERIFICATION):
{ground_truth[:4000]}

{REVIEWER_SYSTEM_PROMPT}
"""

    try:
        async with client.messages.stream(
            model=REVIEWER_MODEL,
            max_tokens=REVIEWER_MAX_TOKENS,
            messages=[{"role": "user", "content": review_prompt}],
        ) as stream:
            async for text_chunk in stream.text_stream:
                yield text_chunk
            final = await stream.get_final_message()
            review_text = final.content[0].text if final.content else ""

            # Parse score from response
            score = 5  # default
            import re

            score_match = re.search(r"SCORE:\s*(\d+)", review_text, re.IGNORECASE)
            if score_match:
                score = int(score_match.group(1))

            # Log the review
            print(f"[review] Score: {score}/10")
    except Exception as exc:
        yield f"\n\n⚠️ Review error: {exc}"


# ─── Combined RAG + Review Stream ─────────────────────────────────────────────
async def rag_review_stream(
    message: str,
    history: list[dict],
    use_reviewer: bool = False,
    persona_mode: str = "Explorer",
) -> AsyncIterator[str]:
    """
    Retrieve relevant chunks, build the prompt, stream from Bot A (RAG),
    and optionally pass through Bot B (reviewer) for verification.
    If direct_mode is True, swaps the system prompt for a more operational persona.
    If case_builder is True, swaps the system prompt for a formal drafting persona.
    """

    if _index is None:
        yield "⚠️ The index is not ready yet. Please wait a moment and try again."
        return

    # Rewrite query for RAG if there is history
    query = await condense_query(message, history)
    relevant_chunks = search_index(_index, _chunks, query)

    # Build context block from retrieved chunks
    context_parts = []
    for chunk in relevant_chunks:
        context_parts.append(
            f"[Source: {chunk.get('source', 'Unknown')}, Page: {chunk['page']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # Build message list for Claude: prior history + new user message
    messages = []
    for turn in history:
        if turn["role"] in ("user", "assistant"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": message})

    client = get_anthropic()

    try:
        # 1. Resolve System Prompt based on Persona
        if persona_mode == "Explore":
            base_prompt = get_system_prompt(DEVELOPER_MODE)
        elif persona_mode in ["Direct", "Defend"]:
            base_prompt = get_persona_prompt(persona_mode)
        else:
            # Catch-all fallback to prevent crashes from unexpected persona_mode values
            base_prompt = get_system_prompt(DEVELOPER_MODE)

        # Standardized formatting for all personas (Issue #216 feedback: use .replace for safety)
        formatted_prompt = base_prompt.replace("{manifest}", get_knowledge_manifest()).replace("{verify_message}", VERIFY_STEWARD_MESSAGE)



        # 2. Audit Logic (Issue #161 Refactor)
        if persona_mode != "Explore":
            matched_tests = _test_registry.find_matches(message + " " + query)
            
            # 1. New Registry Tests
            for test in matched_tests:
                formatted_prompt += f"\n\n--- MANDATORY LOGIC CHECK: {test.name.upper()} ---\n"
                formatted_prompt += f"This case involves potential {test.name}. You MUST follow these criteria and apply them to the facts:\n{test.content}\n"
                formatted_prompt += f"In your response, follow the strategic guidance and instructions in the {test.name} module, specifically identifying any criteria or factors that have not been met or proven."

            # 2. Legacy Millhaven Fallback (if registry doesn't catch it)
            if not matched_tests and MILLHAVEN_FACTORS:
                msg_lower = message.lower()
                query_lower = query.lower()
                is_off_duty = any(k in msg_lower or k in query_lower for k in OFF_DUTY_KEYWORDS)
                if is_off_duty:
                    formatted_prompt += f"\n\n--- MANDATORY LOGIC CHECK: MILLHAVEN AUDIT ---\n"
                    formatted_prompt += f"This case involves potential off-duty conduct. You MUST audit the facts against these 5 factors:\n{MILLHAVEN_FACTORS}\n"
                    formatted_prompt += "In your response, identify which factors management HAS NOT PROVEN."

        # Bot A: Get raw RAG response
        raw_response = ""
        async with client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=RAG_MAX_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": formatted_prompt,
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": (
                        "--- AGREEMENT EXCERPTS ---\n\n"
                        + context
                        + "\n\n--- END EXCERPTS ---"
                    ),
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            messages=messages,
        ) as stream:
            async for text_chunk in stream.text_stream:
                raw_response += text_chunk
                yield text_chunk
            final = await stream.get_final_message()
            usage = final.usage
            print(
                f"[rag] Tokens — input: {usage.input_tokens}, "
                f"output: {usage.output_tokens}"
            )
            if final.stop_reason == "max_tokens":
                yield "\n\n⚠️ Response truncated. Please ask for the rest of the answer."

        # Bot B: Review the response if enabled
        if use_reviewer:
            yield "\n\n---\n\n**🔍 Senior Rep Review:**\n"
            async for review_chunk in review_stream(raw_response, query, context):
                yield review_chunk

    except Exception as exc:
        yield f"\n\n⚠️ API error: {exc}"


# ─── Export & Import ──────────────────────────────────────────────────────────


def history_to_markdown(history: list[dict]) -> str:
    """Convert chat history to a Markdown string."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md = f"# Vexilon Conversation Export - {timestamp}\n\n"

    for turn in history:
        role = turn["role"].capitalize()
        # content can be a string or a list of blocks in Gradio 6
        content = turn["content"]
        if isinstance(content, list):
            # Extract text from message parts
            text_parts = [
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            ]
            content = "".join(text_parts)

        md += f"### {role}\n{content}\n\n"

    return md


def markdown_to_history(file_path: str) -> list[dict]:
    """Parse a Markdown conversation file back into a list of dicts."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    history = []
    current_role = None
    current_content = []

    for line in lines:
        new_role = None
        if line.startswith("### User"):
            new_role = "user"
        elif line.startswith("### Assistant"):
            new_role = "assistant"

        if new_role:
            if current_role:
                history.append(
                    {"role": current_role, "content": "\n".join(current_content).strip()}
                )
            current_role = new_role
            current_content = []
        elif current_role:
            current_content.append(line.rstrip("\n"))

    # Append the last turn
    if current_role:
        history.append(
            {"role": current_role, "content": "\n".join(current_content).strip()}
        )

    return history


# ─── Gradio UI ────────────────────────────────────────────────────────────────
EXAMPLE_QUESTIONS = [
    "What are the just cause requirements for discipline?",
    "What rights do stewards have in investigation meetings?",
    "What is the nexus test for establishing a link in off-duty conduct cases?",
    "Does my employer have a social media policy?",
]




ATTRIBUTION_HTML = f"""
<div style='text-align: center; color: #6b7280; font-size: 0.85rem; margin-top: 1rem;'>
    <a href='{VEXILON_REPO_URL}' target='_blank' style='color: #005691; text-decoration: none;'>View code on GitHub</a>
    <span style='margin-left: 0.5rem; opacity: 0.7;'>•</span>
    <a href='{VEXILON_REPO_URL}/blob/main/docs/PRIVACY.md' target='_blank' style='color: #008542; text-decoration: none;'>Privacy Policy (PIPA)</a>
    <span style='margin-left: 0.5rem; opacity: 0.7;'>•</span>
    <a href='{VEXILON_REPO_URL}/pkgs/container/vexilon' target='_blank' style='color: #005691; text-decoration: none;'>{VEXILON_VERSION}</a>
</div>
"""




def build_ui() -> "gr.Blocks":
    """Assemble and return the Gradio Blocks application."""
    import gradio as gr

    with gr.Blocks(
        title="Collective Agreement Explorer",
        js="""
        function() {
            document.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    const textarea = document.querySelector('#msg_input textarea');
                    if (textarea && document.activeElement === textarea) {
                        e.preventDefault();
                        const sendBtn = document.querySelector('#send_btn');
                        if (sendBtn) sendBtn.click();
                    }
                }
            });
        }
        """,
    ) as demo:
        # ── Header ────────────────────────────────────────────────────────────
        # ── Header ────────────────────────────────────────────────────────────
        gr.Markdown("## Unofficial Ephemeral BCGEU Steward Assistant")

        with gr.Accordion("Knowledge Base & Priority", open=False):
            gr.Markdown(
                "**The Collective Agreement and Primary Statutes** are our primary references. Anything else provides additional context."
            )
            # Use gr.HTML() to preserve clickable links (gr.Markdown sanitizes HTML)
            gr.HTML(build_pdf_download_links())
            gr.Markdown(
                f"[📁 Browse Knowledge Base on GitHub]({GITHUB_LABOUR_LAW_URL})"
            )

        with gr.Row(visible=True) as chip_row:
            chip_btns = [gr.Button(q, size="sm") for q in EXAMPLE_QUESTIONS]

        # ── Chat interface ────────────────────────────────────────────────────
        chatbot = gr.Chatbot(
            height=600,
            buttons=["copy"],
            render_markdown=True,
            show_label=False,
            elem_id="chatbot",
        )

        # ── Reviewer Toggle & Management ──────────────────────────────────────
        with gr.Row(variant="compact", elem_classes="compact-row"):
            persona_selector = gr.Radio(
                choices=["Explore", "Direct", "Defend"],
                value="Explore",
                show_label=False,
                container=False,
                scale=3,
                elem_id="persona_selector",
            )
            reviewer_toggle = gr.Checkbox(
                label="Reviewer",
                value=USE_REVIEWER,
                container=False,
                scale=1,
                elem_id="reviewer_toggle",
            )
            export_btn = gr.DownloadButton("⬇️ Save", variant="secondary", size="sm", scale=1, elem_classes="sm-btn")
            import_btn = gr.UploadButton("⬆️ Load", file_types=[".md"], variant="secondary", size="sm", scale=1, elem_classes="sm-btn")



        # ── Input row ─────────────────────────────────────────────────────────
        with gr.Row(elem_id="input_row"):
            msg_input = gr.Textbox(
                placeholder="Ask about the collective agreement…",
                label="",
                lines=2,
                max_lines=6,
                scale=5,
                show_label=False,
                container=False,
                elem_id="msg_input",
            )
            send_btn = gr.Button("Send ➤", scale=1, variant="primary", elem_id="send_btn")

        # ── Submit handlers ───────────────────────────────────────────────────
        async def submit(
            message: str,
            history: list[dict],
            use_reviewer: bool,
            persona_mode: str,
            **kwargs,
        ) -> AsyncIterator[tuple[list[dict], str, dict, dict]]:
            import gradio as gr
            
            # Onboarding visibility logic



            request = kwargs.get("request")
            hide = gr.update(visible=False)
            show = gr.update(visible=True)
            if not message.strip():
                yield history, "", show, gr.update()
                return

            user_id = request.client.host if request else "default"
            allowed, rate_msg = _rate_limiter.is_allowed(user_id)
            if not allowed:
                history = list(history) + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": rate_msg},
                ]
                yield history, "", show
                return

            message, was_flagged = sanitize_input(message)
            if was_flagged:
                yield (
                    history,
                    "Your input was flagged for security review. Please try a different question.",
                    show,
                )
                return
            prior_history = list(history)
            # Append user turn; seed an empty assistant bubble for streaming.
            # Hide onboarding components on first message.
            history = prior_history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": ""},
            ]
            yield history, "", hide
            # Stream tokens from RAG; accumulate into the assistant bubble
            accumulated = ""
            async for chunk in rag_review_stream(
                message, prior_history, use_reviewer, persona_mode
            ):
                accumulated += chunk
                history[-1]["content"] = accumulated
                yield history, gr.update(), hide



        submit_inputs = [msg_input, chatbot, reviewer_toggle, persona_selector]
        submit_outputs = [chatbot, msg_input, chip_row]

        send_btn.click(fn=submit, inputs=submit_inputs, outputs=submit_outputs)
        msg_input.submit(fn=submit, inputs=submit_inputs, outputs=submit_outputs)

        # ── Chip click handlers — populate input and auto-submit ──────────────
        for chip in chip_btns:
            chip.click(
                fn=lambda q: q,
                inputs=[chip],
                outputs=[msg_input],
            ).then(
                fn=submit,
                inputs=submit_inputs,
                outputs=submit_outputs,
            )

        # ── Export/Import Handlers ───────────────────────────────────────────
        def handle_export(history):
            if not history:
                return None
            md_str = history_to_markdown(history)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
            filename = f"vexilon_chat_{timestamp}.md"
            save_path = os.path.join(tempfile.gettempdir(), filename)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(md_str)
            threading.Timer(600, lambda: os.path.exists(save_path) and os.remove(save_path)).start()
            return save_path

        export_btn.click(fn=handle_export, inputs=[chatbot], outputs=[export_btn])

        def handle_import(file):
            if file is None:
                return gr.update()
            try:
                new_history = markdown_to_history(file.name)
                # Hide onboarding if history is restored
                return new_history, gr.update(visible=False)
            except Exception:
                logging.error("[ui] Import failed", exc_info=True)
                return gr.update(), gr.update()

        import_btn.upload(
            fn=handle_import, inputs=[import_btn], outputs=[chatbot, chip_row]
        )

        # ── Attribution Footer ────────────────────────────────────────────────
        gr.HTML(ATTRIBUTION_HTML)

    return demo


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Vexilon RAG Service")
    parser.add_argument(
        "--rebuild-index", action="store_true", help="Force rebuild of FAISS index from sources"
    )
    args = parser.parse_args()

    if args.rebuild_index:
        startup(force_rebuild=True)
        print("[startup] Index rebuild complete. Exiting (--rebuild-index specified).")
        sys.exit(0)

    # Standard startup sequence
    startup(force_rebuild=False)
    app = build_ui()
    # Enable authentication if a password is set in the environment.
    auth_creds = None
    if VEXILON_PASSWORD:
        auth_creds = (VEXILON_USERNAME, VEXILON_PASSWORD)
        print(f"[startup] Authentication enabled for user '{VEXILON_USERNAME}'")

    # Build allowed_paths: allow labour_law directory for PDF downloads
    # Use relative path for cross-environment compatibility (local dev + Docker container)
    # Note: This allows the entire directory rather than specific files for cross-environment
    # compatibility. The directory only contains PDF files per project structure, so this is acceptable.
    allowed_paths = [str(LABOUR_LAW_DIR), str(Path("docs"))]
    
    # ── Final Build Report ──────────────────────────────────────────────────
    print(f"[startup] Vexilon UI initialized. Ready to serve at port {os.getenv('PORT', 7860)}.")
    print(f"[startup] Version: {VEXILON_VERSION} | Threads: {os.getenv('OMP_NUM_THREADS', 'Auto')}")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
        allowed_paths=allowed_paths,
        css=_CSS_PATH.read_text() if _CSS_PATH.exists() else "",
        auth=auth_creds,
    )
