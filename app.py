# Vexilon RAG App: Cache Performance Check commit
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
  # Saves .pdf_cache/index.faiss + .pdf_cache/chunks.pkl for fast cold starts.
"""

# ─── Standard Library ────────────────────────────────────────────────────────
import sys
import threading
import logging

# Configure structured logging (Issue #196)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from vexilon.indexing import (
    _get_source_name,
    _get_rag_source_files,
    build_index_from_sources,
    get_integrity_report,
    load_precomputed_index,
    search_index,
    _fetch_pdf_cache_if_missing,
    embed_texts,
    build_index,
    chunk_text,
    get_embed_model,
    PDF_CACHE_DIR,
    LABOUR_LAW_DIR,
    INDEX_PATH,
    CHUNKS_PATH,
    MANIFEST_PATH,
    EMBED_MODEL,
    MAX_EMBED_TOKENS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBED_DIM,
    SIMILARITY_TOP_K,
)
TESTS_DIR = LABOUR_LAW_DIR / "tests"
_chunks: list[dict] = []
_index: "faiss.IndexFlatIP | None" = None

logger.info("[boot] Python started, importing stdlib...")
import json
import os
import re
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
import datetime
import tempfile
import textwrap


# Ensure the HuggingFace model cache is writable and persistent.
# Inside the container (WORKDIR /app), this resolves to /app/hf_cache.
# Locally, it resolves to ./hf_cache in the repo root.
if not os.getenv("HF_HOME"):
    os.environ["HF_HOME"] = str(Path("./hf_cache").absolute())

# ─── Third-party: Deferred Imports ───────────────────────────────────────────
# (numpy, anthropic, faiss, sentence_transformers, gradio)
# are imported inside functions to keep startup and test-loading fast.
logger.info("[boot] All boilerplate complete.")

# ─── Configuration ───────────────────────────────────────────────────────────
VEXILON_REPO_URL = os.getenv("VEXILON_REPO_URL", "https://github.com/DerekRoberts/vexilon")
_GITHUB_RAW_BASE = os.getenv("VEXILON_RAW_URL_BASE", "https://raw.githubusercontent.com/DerekRoberts/vexilon/main")

# Public GitHub raw URL base for labour_law PDFs.
# Used for folder/file links in the UI.
GITHUB_LABOUR_LAW_URL = os.getenv(
    "VEXILON_KNOWLEDGE_URL", f"{VEXILON_REPO_URL}/tree/main/data/labour_law"
)

# Raw URL base for downloading pre-computed index from GitHub.
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


# Memory / Context Condensation
CONDENSE_QUERY_HISTORY_TURNS = int(os.getenv("CONDENSE_QUERY_HISTORY_TURNS", 3))
CONDENSE_QUERY_CONTENT_MAX_LEN = int(os.getenv("CONDENSE_QUERY_CONTENT_MAX_LEN", 200))

import re


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

# Build Integrity Warning (populated at startup)
INTEGRITY_WARNING: str | None = None

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
    logger.info("=" * 50)
    logger.info(f" VEXILON VERSION : {version} ({source})")
    logger.info(f" PYTHON VERSION  : {py_ver}")
    logger.info(f" RUNTIME OS      : {os_info}")
    logger.info("=" * 50)

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
# get_embed_model, EMBED_DIM, _get_rag_source_files, _get_source_name,
# and others are imported from vexilon.indexing above.

_anthropic_client: "anthropic.AsyncAnthropic | None" = None

def get_anthropic() -> "anthropic.AsyncAnthropic":
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.AsyncAnthropic()
    return _anthropic_client

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

def get_mandatory_header() -> str:
    """Centralized helper for standard prompt headers (Date + Rules)."""
    today = datetime.datetime.now().strftime("%A, %B %d, %Y")
    return f"Current Date: {today}\n{GLOBAL_MANDATORY_RULES}\n\n"

def get_system_prompt(developer_mode: bool = False) -> str:
    """Load the default system prompt, optionally with developer extensions."""
    path = PROMPTS_DIR / ("developer.txt" if developer_mode else "steward.txt")
    if path.is_file():
        content = path.read_text(encoding="utf-8")
    else:
        # Robust fallback including required formatting placeholders
        content = (
            "You are Vexilon, a professional assistant for BCGEU union stewards.\n\n"
            "Knowledge Base:\n{manifest}\n\n"
            "{verify_message}"
        )
    # Prepend mandatory header (Date + Rules)
    return f"{get_mandatory_header()}{content}"

GLOBAL_MANDATORY_RULES = """--- MANDATORY OPERATIONAL RULES (OVERRIDING - v272-FIXED) ---
1. ANSWER FROM EXCERPTS ONLY: Base your answer strictly on the provided excerpts. If the specific text was not retrieved, suggest the user ask about that section directly. NEVER fabricate contract language.
2. CITATIONS: Every claim MUST be supported by a verbatim quote in a blockquote (> "...") followed by its citation (Document, Article, Page).
3. HIERARCHY: Lead with the Collective Agreement. Use Statutes only to reinforce the legal framework.
4. GRIEVANCE TIMELINES & CHECKPOINT (CRITICAL): Before generating ANY grievance form guidance, you MUST AUTOMATICALLY extract and display all relevant timeline provisions from the collective agreement as a separate, highlighted section labeled "### ⏰ CRITICAL DEADLINES."
   - You MUST cross-reference those timelines against the article citations used in the grievance content. If there's a mismatch or missing timeline, you MUST flag it clearly (e.g. 🔴 **MISSING TIMELINE / MISMATCH**).
   - You MUST require explicit confirmation — a checkpoint where you ask the user to approve the timeline display before proceeding to form completion. Provide exactly this prompt: "Do these timelines match your collective agreement?" STOP generating further guidance until the user confirms.
   - In any form completion sections, you MUST ONLY use timeline language that directly quotes or references the collective agreement sections shown in step 1. No paraphrasing allowed.
5. GRIEVANCE FILING: If a steward asks for resolution steps or once the facts are verified and timelines confirmed, you MUST proactively recommend filing a grievance. 
   - YOU MUST APPEND a final section titled '### 📁 Resolution & Next Steps'.
   - THIS SECTION MUST CONTAIN ONLY these 4 absolute links (DO NOT PARAPHRASE OR SUMMARIZE):
       - [Grievance - 0 - Instructions](https://github.com/DerekRoberts/vexilon/raw/feat/272/data/labour_law/forms/Grievance%20-%200%20-%20Instructions.pdf)
       - [Grievance - A - Grievor Case](https://github.com/DerekRoberts/vexilon/raw/feat/272/data/labour_law/forms/Grievance%20-%20A%20-%20Grievor%20Case.pdf)
       - [Grievance - B - Notify Designates](https://github.com/DerekRoberts/vexilon/raw/feat/272/data/labour_law/forms/Grievance%20-%20B%20-%20Notify%20Designates.pdf)
       - [Grievance - C - Steward Case](https://github.com/DerekRoberts/vexilon/raw/feat/272/data/labour_law/forms/Grievance%20-%20C%20-%20Steward%20Case.pdf)
   - YOU MUST ALSO mention the 'BCGEU Grievance Form Guide.md' for instructions.
   - DISCLAIMER: You MUST include this verbatim: "Note: Viability of this grievance will be assessed by the staff representative and/or arbitrator, not by the steward."
6. NO MERIT ASSESSMENT: Do NOT judge the merit, viability, or likelihood of success of a grievance. Your role is to identify potential violations and facilitate the filing process.
7. TONE: Professional, authoritative, and forensic.
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
    if path and path.is_file():
        content = path.read_text(encoding="utf-8")
    elif mode_name in fallbacks:
        content = fallbacks[mode_name]
    else:
        # Falls back to get_system_prompt which already handles get_mandatory_header()
        return get_system_prompt(DEVELOPER_MODE)
        
    return f"{get_mandatory_header()}{content}"



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

# ─── Refinement Prompt (The Synthesis) ─────────────────────────────────────────
REFINER_SYSTEM_PROMPT = """You are a Senior BCGEU Expert Rep. Your task is to take a draft response from a junior steward and a critique from a senior reviewer, then synthesize them into a SINGLE, high-quality, final response for the member.

INPUTS:
1. DRAFT: The initial response generated by the steward.
2. CRITIQUE: The senior reviewer's audit of that draft (including errors, gaps, or corrections).
3. GROUND TRUTH: Verified excerpts from the Collective Agreement.

YOUR RULES:
- RESOLVE: Follow every correction provided in the CRITIQUE.
- TONE: Maintain a supportive, professional, and authoritative 'BCGEU' tone.
- CITATION: Ensure all citations are verbatim and accurate per the GROUND TRUTH.
- NO META: Do not mention the review process or the bots. Only provide the final corrected answer.
- STRUCTURE: Use clear headings, bullet points, and citations.

Goal: Provide the absolute best, most accurate answer possible."""

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
            logger.warning(f"[registry] Warning: {directory} does not exist.")
            return

        with self._lock:
            self.tests = []
            for f in directory.glob("*.md"):
                if f.name == "index.md":
                    continue
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
                    logger.error(f"[registry] Failed to load {f.name}: {e}")
            logger.info(f"[registry] Loaded {len(self.tests)} tests from {directory.name}")

    def find_matches(self, query: str) -> list[TestDoctrine]:
        """Find all tests whose keywords appear in the lowercased query."""
        q_lower = query.lower()
        with self._lock:
            return [test for test in self.tests if any(k in q_lower for k in test.keywords)]

_test_registry = TestRegistry()

# ─── Chunking ─────────────────────────────────────────────────────────────────



def startup(force_rebuild: bool = False, skip_pdf_fetch: bool = False) -> None:
    """
    Initialise the vector index and load document chunks.
    Attempts cold-start from pre-computed index first.
    """
    global _chunks, _index, INTEGRITY_WARNING
    get_anthropic()
    
    # 1. Basic Metadata
    if os.getenv("VEXILON_QUIET", "").lower() not in ("1", "true"):
        logger.info(f"[startup] Starting Vexilon {VEXILON_VERSION}…")
    if DEVELOPER_MODE:
        logger.info("[startup] DEVELOPER_MODE is ACTIVE.")
    
    # Check for missing model cache
    hf_home = Path(os.getenv("HF_HOME", "./hf_cache"))
    if not hf_home.exists() and not (os.getenv("HF_SPACE_ID") or os.getenv("EXTERNAL_CI")):
        logger.warning(f"⚠️ [startup] Model cache directory '{hf_home}' not found. Vexilon is in offline mode and may fail to start.")
    
    # 2. Local Knowledge Bases
    _test_registry.load(TESTS_DIR)

    # 3. Load / Build Vector Index
    if not skip_pdf_fetch:
        _fetch_pdf_cache_if_missing()

    # Attempt to load precomputed first if not forcing
    if not force_rebuild:
        _index, _chunks = load_precomputed_index()
    
    # Rebuild only if forced OR if loading failed (missing/corrupt files)
    if _index is None or _chunks is None or force_rebuild:
        logger.info("[startup] Pre-computed index missing or forced rebuild. Refreshing from sources...")
        # If we're here as a fallback (not a manual force), we MUST force the rebuild 
        # to ensure it doesn't just re-read the same corrupt/empty files.
        _index, _chunks = build_index_from_sources(force=True)
    else:
        # We already successfully loaded it in the load_precomputed_index call
        pass

    if _index is not None and _chunks:
        logger.info("[startup] Ready.")
        # 4. Integrity Reporting
        report = get_integrity_report()
        if report.get("failed_files"):
            failed = report["failed_files"]
            INTEGRITY_WARNING = (
                f"⚠️ **INDEX INTEGRITY WARNING:** {len(failed)} document(s) failed to index "
                f"(e.g., {', '.join(failed[:2])}). Knowledge base may be incomplete."
            )
            logger.warning(f"[integrity] Found {len(failed)} failures.")
    else:
        logger.error("[startup] ERROR: Knowledge base failed to load.")


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
        logger.error(f"[rag] Query condensation failed: {exc}. Using raw message.")
        return message


async def generate_perspective_queries(message: str, history: list[dict]) -> list[str]:
    """
    Analyze if the query is complex and generate multiple perspectives if so.
    Returns a list of queries (at least one, the original/condensed one).
    Integrated Complexity Detection & Multi-Query Generation (Issue #132).
    """
    condensed = await condense_query(message, history)
    
    # Analyze complexity and generate perspectives in one go
    prompt = (
        "Analyze if the following search query for a BCGEU Steward Assistant is 'complex'. "
        "A query is complex if it involves potential conflicts between contract articles, "
        "exceptions, multiple different documents (statutes vs agreement), or "
        "nuanced topics like off-duty conduct or seniority disputes.\n\n"
        "If it is NOT complex, simply return the query itself without any changes, prefixes, or commentary.\n\n"
        "If it IS complex, generate 3-5 distinct search queries from different 'angles' or perspectives "
        "(e.g., employer rights, employee obligations, specific exceptions, related precedents). "
        "Provide each query on a new line started with a hyphen (-). "
        "Do NOT provide any opening or closing commentary, just the hyphenated queries if complex, "
        "or the single query if simple.\n\n"
        f"Query: {condensed}\n\n"
        "Response:"
    )

    client = get_anthropic()
    try:
        response = await client.messages.create(
            model=CONDENSE_MODEL,
            max_tokens=250,
            messages=[{"role": "user", "content": prompt}],
        )
        resp_text = response.content[0].text.strip()
        
        # Robust parsing: extract any line starting with a hyphen (Issue #132 feedback)
        perspective_queries = [
            line.strip().lstrip("-").strip().strip('"')
            for line in resp_text.split("\n")
            if line.strip().startswith("-")
        ]
        
        if not perspective_queries:
            return [condensed]
            
        logger.info(f"[rag] Complex query detected. Generated {len(perspective_queries)} perspectives.")
        return perspective_queries
    except Exception as exc:
        logger.error(f"[rag] Multi-perspective generation failed: {exc}. Using condensed query.")
        return [condensed]


async def get_multi_perspective_context(message: str, history: list[dict]) -> tuple[list[str], str]:
    """
    Generate multiple perspectives for complex queries, search the index,
    and aggregate deduplicated chunks. Returns (queries, context_string).
    Shared helper for rag_stream, rag_review_stream, and get_rag_context (Issue #132).
    """
    queries = await generate_perspective_queries(message, history)
    
    all_hits = []
    seen_texts = set()
    
    for q in queries:
        # Smaller k per query if multiple, to keep total context size reasonable.
        # We target about 1.5x the standard top_k total chunks across all queries.
        k = max(10, (SIMILARITY_TOP_K * 3) // (2 * len(queries))) if len(queries) > 1 else SIMILARITY_TOP_K
        relevant_chunks = search_index(_index, _chunks, q, top_k=k)

        for chunk in relevant_chunks:
            # Deduplicate by direct string comparison (Issue #132 feedback)
            text = chunk["text"]
            if text not in seen_texts:
                seen_texts.add(text)
                all_hits.append(chunk)

    # Limit total aggregated context to prevent token overflows
    relevant_chunks = all_hits[:50]
    
    context_parts = []
    for chunk in relevant_chunks:
        context_parts.append(
            f"[Source: {chunk.get('source', 'Unknown')}, Page: {chunk['page']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)
    
    return queries, context



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

    # Issue #132: Multi-perspective retrieval for complex topics (Shared Refactor)
    queries, context = await get_multi_perspective_context(message, history)

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
            logger.info(
                f"[rag] Tokens — input: {usage.input_tokens}, "
                f"cache_create: {cache_created or 0}, cache_read: {cache_read or 0}, "
                f"output: {usage.output_tokens}"
            )
            if final.stop_reason == "max_tokens":
                yield ("\n\n⚠️ Response truncated. Please ask for the rest of the answer.", "")
    except Exception as exc:
        yield (f"\n\n⚠️ API error: {exc}", "")


async def get_rag_context(message: str, history: list[dict]) -> tuple[str, str]:
    """Get context (excerpts) for a query without generating a response. Consistent with Issue #132."""
    queries, context = await get_multi_perspective_context(message, history)
    query_display = " | ".join(queries) if len(queries) > 1 else queries[0]
    return query_display, context


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
    raw_response: str, query: str, context: str, all_chunks: list[dict] = None
) -> AsyncIterator[str]:
    """
    Bot B: Senior BCGEU rep reviewing Bot A's (steward) output.
    Yields the review result with score and verified steps.
    """
    client = get_anthropic()
    # Independent re-retrieval for Bot B (Issue #183)
    # Extracts cited articles from Bot A's response and fetches ground truth context.
    ground_truth = get_ground_truth_for_review(raw_response, all_chunks or _chunks)
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
            logger.info(f"[review] Score: {score}/10")
    except Exception as exc:
        yield f"\n\n⚠️ Review error: {exc}"


async def refine_stream(
    draft: str, critique: str, ground_truth: str
) -> AsyncIterator[str]:
    """
    Step 3: Refiner Bot (Claude) synthesizes draft + critique into a final polished answer.
    """
    client = get_anthropic()
    
    prompt = textwrap.dedent(f"""
        DRAFT:
        {draft}

        CRITIQUE:
        {critique}

        GROUND TRUTH:
        {ground_truth}
    """).strip()

    try:
        async with client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=RAG_MAX_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": REFINER_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text_chunk in stream.text_stream:
                yield text_chunk
            
            final_refine = await stream.get_final_message()
            logger.info(f"[refine] Tokens — input: {final_refine.usage.input_tokens}, output: {final_refine.usage.output_tokens}")

            if final_refine.stop_reason == "max_tokens":
                yield "\n\n⚠️ Response truncated. Please ask for the rest of the answer."
    except Exception as exc:
        yield f"\n\n⚠️ Refinement error: {exc}"


# ─── Combined RAG + Review + Refine Stream (Collaboration Model) ──────────────
async def rag_review_stream(
    message: str,
    history: list[dict],
    use_reviewer: bool = False,
    persona_mode: str = "Explorer",
    all_chunks: list[dict] = None,
) -> AsyncIterator[str]:
    """
    Retrieve relevant chunks, build the prompt, perform silent draft/audit,
    and yield a single high-fidelity synthesized response.
    """

    if _index is None:
        yield "⚠️ The index is not ready yet. Please wait a moment and try again."
        return

    # Issue #132: Multi-perspective retrieval for complex topics
    queries, context = await get_multi_perspective_context(message, history)
    query = queries[0]

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
            base_prompt = get_system_prompt(DEVELOPER_MODE)
            
        formatted_prompt = base_prompt.replace("{manifest}", get_knowledge_manifest()).replace("{verify_message}", VERIFY_STEWARD_MESSAGE)

        # 2. Audit Logic (Issue #161 Refactor) - INJECT INTO PROMPT
        if persona_mode != "Explore":
            matched_tests = _test_registry.find_matches(message + " " + query)
            for test in matched_tests:
                show_request = any(k in message.lower() for k in ["show me", "what are the factors", "list the criteria", "what is the test for", "give me the test"])
                if show_request:
                    formatted_prompt += f"\n\n--- USER REQUESTED TEST: {test.name.upper()} ---\n"
                    formatted_prompt += f"The user asked to see this test. You MUST start your response by displaying the following factor list/criteria EXACTLY as written here:\n{test.content}\n"
                    formatted_prompt += f"After displaying it, ask the user if they want you to apply it to their specific facts.\n"
                
                formatted_prompt += f"\n\n--- MANDATORY LOGIC CHECK: {test.name.upper()} ---\n"
                formatted_prompt += f"This case involves potential {test.name}. You MUST follow this pattern:\n"
                formatted_prompt += "1. EXPLAIN: Briefly explain the test factors.\n"
                formatted_prompt += "2. QUESTION: If any facts are missing from the user's query to satisfy these factors, ASK those specific questions now.\n"
                formatted_prompt += "3. APPLY: Once facts are known, apply these factors to the scenario and identify which ones management HAS or HAS NOT proven.\n"
                formatted_prompt += "4. CITE: Point to the specific articles or documents that support or limit the employer's position.\n"
                formatted_prompt += f"CRITERIA:\n{test.content}\n"

            if not matched_tests and MILLHAVEN_FACTORS:
                msg_lower = message.lower()
                query_lower = query.lower()
                is_off_duty = any(k in msg_lower or k in query_lower for k in OFF_DUTY_KEYWORDS)
                if is_off_duty:
                    formatted_prompt += f"\n\n--- MANDATORY LOGIC CHECK: MILLHAVEN AUDIT ---\n"
                    formatted_prompt += f"This case involves potential off-duty conduct. You MUST audit the facts against these 5 factors:\n{MILLHAVEN_FACTORS}\n"
                    formatted_prompt += "In your response, identify which factors management HAS NOT PROVEN."

        # 3. Step 1: Steward Draft (Bot A) - SILENT
        yield "🧠 Vexilon is analyzing the collective agreement and drafting a response..."
        raw_response = ""
        async with client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=RAG_MAX_TOKENS,
            system=[
                {"type": "text", "text": formatted_prompt, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": ("--- AGREEMENT EXCERPTS ---\n\n" + context + "\n\n--- END EXCERPTS ---"), "cache_control": {"type": "ephemeral"}}
            ],
            messages=messages,
        ) as stream:
            async for text_chunk in stream.text_stream:
                raw_response += text_chunk
            
            final_draft = await stream.get_final_message()
            logger.info(f"[rag] Draft tokens — input: {final_draft.usage.input_tokens}, output: {final_draft.usage.output_tokens}")
            
            if final_draft.stop_reason == "max_tokens":
                raw_response += "\n\n⚠️ Response truncated during drafting phase. Synthesis results may be incomplete."
            
            # If no reviewer, we just stream Bot A's response (with a clean break)
            if not use_reviewer:
                yield "\n\n---\n\n"
                yield raw_response

        # 4. Step 2 & 3: Collaborative Refinement (Bot B + Bot A)
        if use_reviewer:
            yield "\n\n🕵️ Senior Rep is auditing the response for forensic accuracy..."
            
            # Silent Audit (Bot B)
            review_text = ""
            async for review_chunk in review_stream(raw_response, query, context, all_chunks=all_chunks):
                review_text += review_chunk
            
            # Fetch ground_truth from Bot B's logic to pass to refiner
            # Use provided chunks or fall back to global _chunks
            target_chunks = all_chunks if all_chunks is not None else _chunks
            ground_truth = get_ground_truth_for_review(raw_response, target_chunks) or context
            
            # Synthesis Phase (Step 3) - STREAMED
            yield "\n\n---\n\n✨ **VEXILON RECOMMENDATION**\n\n"
            async for refined_chunk in refine_stream(raw_response, review_text, ground_truth):
                yield refined_chunk

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
    "Show me the Harassment Threshold test.",
    "Does my employer have a social media policy?",
]






import html
import urllib.parse
_SAFE_VEXILON_VERSION = html.escape(VEXILON_VERSION)
_URL_VEXILON_VERSION = urllib.parse.quote(VEXILON_VERSION)

ATTRIBUTION_HTML = f"""
<div style='text-align: center; color: #6b7280; font-size: 0.85rem; margin-top: 1rem;'>
    <a href='{VEXILON_REPO_URL}' target='_blank' rel='noopener noreferrer' style='color: #005691; text-decoration: none;'>View code on GitHub</a>
    <span style='margin-left: 0.5rem; opacity: 0.7;'>•</span>
    <a href='{VEXILON_REPO_URL}/blob/main/docs/PRIVACY.md' target='_blank' rel='noopener noreferrer' style='color: #008542; text-decoration: none;'>Privacy Policy (PIPA)</a>
    <span style='margin-left: 0.5rem; opacity: 0.7;'>•</span>
    <a href='{VEXILON_REPO_URL}/pkgs/container/vexilon/versions?filters%5Bversion_type%5D=tagged&query={_URL_VEXILON_VERSION}' target='_blank' rel='noopener noreferrer' style='color: #005691; text-decoration: none;'>{_SAFE_VEXILON_VERSION}</a>
</div>
"""




_CUSTOM_JS = """
function() {
    // Use capture phase (true) so this fires before Gradio's element-level
    // textarea handler, preventing Enter from inserting a newline (#276).
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            const textarea = document.querySelector('#msg_input textarea');
            if (textarea && document.activeElement === textarea) {
                e.preventDefault();
                const sendBtn = document.querySelector('#send_btn');
                if (sendBtn) sendBtn.click();
            }
        }
    }, true);
}
"""

def build_ui() -> "gr.Blocks":
    """Assemble and return the Gradio Blocks application."""
    import gradio as gr

    with gr.Blocks(
        title="Collective Agreement Explorer",
    ) as demo:
        # ── Header ────────────────────────────────────────────────────────────
        gr.Markdown("# BCGEU Steward Assistant")
        if INTEGRITY_WARNING:
            gr.Markdown(INTEGRITY_WARNING)

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
                max_lines=6,
                scale=5,
                show_label=False,
                container=False,
                elem_id="msg_input",
                lines=1,
            )
            send_btn = gr.Button("Send ➤", scale=1, variant="primary", elem_id="send_btn")

        # ── Submit handlers ───────────────────────────────────────────────────
        async def submit(
            message: str,
            history: list[dict],
            use_reviewer: bool,
            persona_mode: str,
            **kwargs,
        ) -> AsyncIterator[tuple[list[dict], str, dict]]:
            import gradio as gr
            
            # Onboarding visibility logic



            request = kwargs.get("request")
            hide = gr.update(visible=False)
            show = gr.update(visible=True)
            if not message.strip():
                yield history, "", show
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
                message, prior_history, use_reviewer, persona_mode, _chunks
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
        logger.info("[startup] Index rebuild complete. Exiting (--rebuild-index specified).")
        sys.exit(0)

    # Standard startup sequence
    startup(force_rebuild=False)
    app = build_ui()
    # Enable authentication if a password is set in the environment.
    VEXILON_PASSWORD = os.getenv("VEXILON_PASSWORD")
    auth_creds = None
    if VEXILON_PASSWORD:
        auth_creds = [(VEXILON_USERNAME, VEXILON_PASSWORD)]
        logger.info(f"[startup] Authentication enabled for user '{VEXILON_USERNAME}'")

    # Build allowed_paths: allow labour_law directory for PDF downloads
    # Use relative path for cross-environment compatibility (local dev + Docker container)
    # Note: This allows the entire directory rather than specific files for cross-environment
    # compatibility. The directory only contains PDF files per project structure, so this is acceptable.
    allowed_paths = [str(LABOUR_LAW_DIR), str(Path("docs"))]
    
    # ── Final Build Report ──────────────────────────────────────────────────
    logger.info(f"[startup] Vexilon UI initialized. Ready to serve at port {os.getenv('PORT', 7860)}.")
    logger.info(f"[startup] Version: {VEXILON_VERSION} | Threads: {os.getenv('OMP_NUM_THREADS', 'Auto')}")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
        allowed_paths=allowed_paths,
        css=_CSS_PATH.read_text() if _CSS_PATH.exists() else "",
        auth=auth_creds,
        js=_CUSTOM_JS,
    )
