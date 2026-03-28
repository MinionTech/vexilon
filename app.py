"""
app.py — BCGEU Steward Assistant
--------------------------------------------
Tech stack:
  - pypdf                : PDF → pages with page number preservation
  - sentence-transformers: Local CPU embeddings (all-MiniLM-L6-v2, no API key)
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
# (numpy, pypdf, anthropic, faiss, sentence_transformers, gradio)
# are imported inside functions to keep startup and test-loading fast.
print("[boot] All boilerplate complete.", flush=True)

# ─── Configuration ───────────────────────────────────────────────────────────
PDF_CACHE_DIR = Path("./.pdf_cache")
LABOUR_LAW_DIR = Path("./data/labour_law")
TESTS_DIR = LABOUR_LAW_DIR / "tests"
INDEX_PATH = PDF_CACHE_DIR / "index.faiss"
CHUNKS_PATH = PDF_CACHE_DIR / "chunks.json"
MANIFEST_PATH = PDF_CACHE_DIR / "manifest.json"

# Public GitHub raw URL base for labour_law PDFs.
# Used for folder/file links in the UI.
GITHUB_LABOUR_LAW_URL = (
    "https://github.com/DerekRoberts/vexilon/tree/main/data/labour_law"
)

# Raw URL base for downloading pre-computed index from GitHub.
# Used by _fetch_pdf_cache_if_missing() for HF Spaces bootstrap.
_GITHUB_RAW_BASE = "https://raw.githubusercontent.com/DerekRoberts/vexilon/main"

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
CONDENSE_MODEL = os.getenv("CONDENSE_MODEL", "claude-haiku-4-5-20251001")
# Brain: Local Embeddings (Search) + Cloud LLM (Claude)
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")  # 512-token window
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 450))  # Sized for BGE-small
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", 40))  # More context depth

# Memory / Context Condensation
CONDENSE_QUERY_HISTORY_TURNS = int(os.getenv("CONDENSE_QUERY_HISTORY_TURNS", 3))
CONDENSE_QUERY_CONTENT_MAX_LEN = int(os.getenv("CONDENSE_QUERY_CONTENT_MAX_LEN", 200))

import re
import logging

# Input Sanitization (for prompt injection prevention)
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", 2000))
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
                    f"[security] Prompt injection pattern detected: {pattern.pattern[:30]}..."
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
VERIFY_MODEL = os.getenv("VERIFY_MODEL", "claude-haiku-4-5-20251001")
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
REVIEWER_MODEL = os.getenv("REVIEWER_MODEL", "claude-haiku-4-5-20251001")


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

# Embedding dimension for all-MiniLM-L6-v2
EMBED_DIM = 384

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
        print(f"[embed] Loading local embedding model '{EMBED_MODEL}'…")
        from sentence_transformers import SentenceTransformer

        _embed_model = SentenceTransformer(EMBED_MODEL)
        # Increase limits to handle full-page tokenization mapping without warnings.
        # We manually chunk to 256 tokens later, so this is just to keep the logs clean.
        _embed_model.max_seq_length = 100000
        if hasattr(_embed_model, "tokenizer"):
            _embed_model.tokenizer.model_max_length = 100000
        print("[embed] Embedding model ready.")
    return _embed_model


def get_anthropic() -> "anthropic.AsyncAnthropic":
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic

        # Reads ANTHROPIC_API_KEY from environment automatically; raises AuthenticationError if missing
        _anthropic_client = anthropic.AsyncAnthropic()
    return _anthropic_client


def _get_all_source_files() -> list[Path]:
    """
    Recursively scan LABOUR_LAW_DIR for PDF and Markdown files and return a
    combined list sorted by filename for consistent processing.
    The tests/ subdirectory is excluded (used by TestRegistry, not RAG index).
    """
    if not LABOUR_LAW_DIR.exists():
        return []
    tests_dir = LABOUR_LAW_DIR / "tests"
    pdfs = [p for p in LABOUR_LAW_DIR.rglob("*.pdf") if not p.is_relative_to(tests_dir)]
    mds = [p for p in LABOUR_LAW_DIR.rglob("*.md") if not p.is_relative_to(tests_dir)]
    return sorted(pdfs + mds, key=lambda p: str(p))


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
    files = _get_all_source_files()
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

    files = _get_all_source_files()
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


# ─── System Prompts ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Vexilon, a highly authoritative professional assistant for BCGEU union stewards.

--- HOW YOUR SEARCH WORKS ---
Your library contains the COMPLETE, full text of these documents:
{manifest}

IMPORTANT: For each question you receive, a semantic search retrieves the most relevant \
excerpts from this library. You see a SUBSET of the library per query — not the whole thing. \
Content that does not appear in the excerpts below may still exist in the library; it simply \
was not retrieved for THIS particular question. \
NEVER claim that an Article, section, or document is "missing" or "not in my documents" \
just because it is not in the current excerpts. Instead, say: \
"The specific text was not retrieved for this search. Try asking about [topic] directly."
--------------------------

Rules you must follow without exception:

1. ANSWER FROM EXCERPTS ONLY: Base your answer strictly on the excerpts provided below. \
If the excerpts contain only a reference to a section (e.g., "See Section 10.4") but not the \
actual text, say the specific language was not retrieved for this search and suggest the user \
ask about that section directly. NEVER guess or fabricate contract language.
2. Every claim must be supported by a verbatim quote from the provided excerpts, formatted as a markdown blockquote (> "...") followed by its citation: — [Document Name], Article/Section [X], [Title if available], p. [N]
3. Plain-language explanation comes BEFORE the verbatim quote, not after.
4. AUTHORITY HIERARCHY: ALWAYS lead with the Collective Agreement (Main Agreement) as your first-tier authority. Use Primary Statutes (e.g. BC Labour Relations Code) to reinforce the legal foundation or framework of your argument, but never as a replacement for contract language.
5. If consecutive sections appear with a gap (e.g., you see 10.1 and 10.3 but not 10.2), note the gap and suggest the user ask about the missing section specifically.
6. Do not predict outcomes or give legal opinions.
7. Tone: professional, forensic, and confident. Do NOT be apologetic about retrieval limitations — the library is comprehensive; the search just needs more specific queries.
8. Cite every relevant clause separately.
9. Maintain conversational continuity. Use the previous conversation context and the provided excerpts.
10. If the search results are contradictory or unclear, flag this ambiguity to the user immediately.
11. Every chunk is tagged with its Article or Appendix name for context.
12. If asked about your capabilities, knowledge gaps, or what documents you have: describe the library manifest above. Do NOT audit or list "missing" articles — you have the complete text of everything listed above.

Response format:

[Plain-language explanation]

> "[Verbatim quote]"
— [Document Name], Article/Section [X], p. [N]
"""

VERIFY_STEWARD_MESSAGE = "Verify w/ Area Office: 604-291-9611"

def get_persona_prompt(mode_name: str) -> str:
    """Helper to load system prompts for different operational modes."""
    paths = {
        "Direct": Path("./prompts/direct_staff_rep.txt"),
        "Defend": Path("./prompts/case_builder.txt"),
    }
    
    path = paths.get(mode_name)
    if path and path.is_file():
        return path.read_text(encoding="utf-8")
    
    # Fallbacks if files are missing or empty
    fallbacks = {
        "Direct": "You are a BCGEU Staff Rep providing DIRECT OPERATIONAL GUIDANCE.",
        "Defend": "You are a BCGEU Staff Rep specializing in Grievance Drafting.",
    }
    return fallbacks.get(mode_name, SYSTEM_PROMPT)

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
        start += CHUNK_SIZE - CHUNK_OVERLAP
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
    content = md_path.read_text(encoding="utf-8")
    source_name = _get_source_name(md_path.stem)
    print(f"[loader] Parsing Markdown '{source_name}'…")

    tokenizer = get_embed_model().tokenizer
    token_metadata = []
    
    # Very simple header detection for MD
    current_header = ""
    lines = content.split("\n")
    char_offset = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            current_header = stripped.lstrip("#").strip().upper()
        
        # Tokenize line and record metadata
        encoding = tokenizer(
            line,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        for start_off, end_off in encoding.offset_mapping:
            token_metadata.append(
                (char_offset + start_off, char_offset + end_off, 1, current_header)
            )
        
        char_offset += len(line) + 1 # +1 for newline

    return chunk_text(content, token_metadata, source_name)


def load_pdf_chunks(pdf_path: Path) -> list[dict]:
    """
    Parse the PDF at *pdf_path* into one continuous stream before chunking.
    This bridges page boundaries so sentences that span pages aren't decapitated.

    Navigational pages (Table of Contents, alphabetical Index) are skipped
    so they don't contaminate semantic search results.  URL artifacts from
    web-extracted statute PDFs are also stripped before embedding.
    """
    from pypdf import PdfReader
    import re

    reader = PdfReader(str(pdf_path))
    source_name = _get_source_name(pdf_path.stem)
    print(f"[loader] Parsing '{source_name}' ({len(reader.pages)} pages)…")

    tokenizer = get_embed_model().tokenizer
    full_text = ""
    token_metadata = []  # List of (char_start, char_end, page_num, header)

    current_header = ""
    header_pattern = re.compile(r"^\s*(ARTICLE|APPENDIX)\s+(\d+|[A-Z]+)", re.IGNORECASE)
    skipped_pages = 0

    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        page_text = page.extract_text() or ""
        if not page_text.strip():
            continue

        # Skip pure navigational pages (TOC / alphabetical Index).
        # These mention every article by name but add no substantive content;
        # indexing them causes TOC entries to crowd out real contract text in
        # semantic search results.
        if _is_toc_or_index_page(page_text):
            skipped_pages += 1
            continue

        # Strip web-extraction artifacts (URLs, timestamps) from statute PDFs.
        page_text = _clean_page_text(page_text)
        if not page_text.strip():
            continue

        # Update breadcrumb header context
        # Look through more lines - headers can appear anywhere on the page due to
        # complex PDF layouts (two-column, footnotes, etc.)
        page_lines = page_text.split("\n")
        lines_to_check = min(50, len(page_lines))
        for line in page_lines[:lines_to_check]:
            # Skip TOC-style entries and page numbers
            if ".........." in line or (
                line.strip().endswith(".") and re.search(r"\d+$", line.strip())
            ):
                continue
            match = header_pattern.search(line)
            if match:
                current_header = match.group(0).strip().upper()
                break  # Usually one primary header per page

        # Track offsets in the global full_text
        page_offset = len(full_text)
        full_text += page_text + "\n"

        # Tokenize this page and record metadata for every token
        encoding = tokenizer(
            page_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        for start, end in encoding.offset_mapping:
            token_metadata.append(
                (page_offset + start, page_offset + end, page_num, current_header)
            )

    if skipped_pages:
        print(
            f"[loader] Skipped {skipped_pages} navigational pages (TOC/index) in '{source_name}'."
        )

    return chunk_text(full_text, token_metadata, source_name)


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
        urllib.request.urlretrieve(url, dest_path)
        print(f"[fetch] Saved {dest_path}")


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

    all_files = _get_all_source_files()
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
        print(f"[rag] Condensed query: '{message}' -> '{condensed}'")
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
            max_tokens=1024,
            # Two cache breakpoints:
            # 1. Static instructions — identical every request; cached once per session.
            # 2. Dynamic excerpts — changes per query; cached separately.
            # This avoids re-caching the instructions block whenever the excerpts change.
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT.format(manifest=get_knowledge_manifest()),
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


# ─── Review Logging ───────────────────────────────────────────────────────────────
REVIEW_LOG_PATH = Path("./.pdf_cache/review_log.csv")


def log_review(query: str, raw_response: str, review_output: str, score: int) -> None:
    """Append a review record to the audit log CSV."""
    import csv
    import datetime

    REVIEW_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not REVIEW_LOG_PATH.exists()
    with open(REVIEW_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "timestamp",
                    "query",
                    "raw_response",
                    "review_output",
                    "score",
                    "steward",
                ]
            )
        writer.writerow(
            [
                datetime.datetime.now().isoformat(),
                query[:500],  # Truncate long queries
                raw_response[:1000],  # Truncate long responses
                review_output[:2000],
                score,
                VEXILON_USERNAME,
            ]
        )


# ─── Two-Bot Review Stream (Bot B) ─────────────────────────────────────────────
async def review_stream(
    raw_response: str, query: str, context: str
) -> AsyncIterator[str]:
    """
    Bot B: Senior BCGEU rep reviewing Bot A's (steward) output.
    Yields the review result with score and verified steps.
    """
    client = get_anthropic()
    review_prompt = f"""Review the following steward's response for accuracy and completeness:

QUERY: {query}

STEWARD'S RESPONSE:
{raw_response}

CONTEXT USED:
{context[:2000]}

{REVIEWER_SYSTEM_PROMPT}
"""

    try:
        async with client.messages.stream(
            model=REVIEWER_MODEL,
            max_tokens=512,
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
            log_review(query, raw_response, review_text, score)
            print(f"[review] Score: {score}/10 for query: '{query[:50]}...'")
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
        base_prompt = persona_mode if persona_mode != "Explore" else SYSTEM_PROMPT
        if base_prompt in ["Direct", "Defend"]:
            base_prompt = get_persona_prompt(base_prompt)

        formatted_prompt = base_prompt.format(
            manifest=get_knowledge_manifest(),
            verify_message=VERIFY_STEWARD_MESSAGE,
        )

        # 2. Audit Logic (Issue #161 Refactor)
        if persona_mode != "Explore":
            matched_tests = _test_registry.find_matches(message + " " + query)
            
            # 1. New Registry Tests
            for test in matched_tests:
                formatted_prompt += f"\n\n--- MANDATORY LOGIC CHECK: {test.name.upper()} ---\n"
                formatted_prompt += f"This case involves potential {test.name}. You MUST audit the facts against these criteria:\n{test.content}\n"
                formatted_prompt += f"In your response, identify which factors in the {test.name} management HAS NOT PROVEN."

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
            max_tokens=1024,
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
    "What is the nexus test for off-duty conduct?",
    "Does my employer have a social media policy?",
    "What happens if I'm disciplined for off-duty behavior?",
]

# Disclaimer rendered entirely with inline styles so Gradio theme cannot override text colour.
DISCLAIMER_HTML = (
    '<div style="'
    "background-color:#fff8e1;"
    "border-left:4px solid #f59e0b;"
    "color:#7c4a00;"
    "padding:10px 14px;"
    "border-radius:4px;"
    "font-size:0.85rem;"
    "margin-bottom:12px;"
    '">'
    "This project is not affiliated with the BCGEU. AI-generated responses may contain errors."
    "</div>"
)

DIRECT_MODE_HTML = """
<div style="background-color:#e0f2fe; border-left:4px solid #0ea5e9; color:#075985; padding:10px 14px; border-radius:4px; font-size:0.85rem; margin-bottom:12px;">
    <b>⚡ Direct Advice Mode Active:</b> Responses focus on operational steps and scripts.
</div>
"""

CASE_BUILDER_HTML = """
<div style="background-color:#f0fdf4; border-left:4px solid #22c55e; color:#14532d; padding:10px 14px; border-radius:4px; font-size:0.85rem; margin-bottom:12px;">
    <b>📝 Case Builder Mode Active:</b> Responses focus on drafting formal rebuttals and grievances.
</div>
"""

ATTRIBUTION_HTML = f"""
<div style='text-align: center; color: #6b7280; font-size: 0.85rem; margin-top: 1rem;'>
    <a href='https://github.com/DerekRoberts/vexilon' target='_blank' style='color: #005691; text-decoration: none;'>View code or contribute on GitHub</a>
    <span style='margin-left: 0.5rem; opacity: 0.7;'>•</span>
    <a href='https://github.com/DerekRoberts/vexilon/pkgs/container/vexilon' target='_blank' style='color: #005691; text-decoration: none;'>{VEXILON_VERSION}</a>
</div>
"""

# Custom CSS for a Unified, Single-Line Action Bar
CUSTOM_CSS = """
/* 1. Unified row alignment */
.compact-row {
    align-items: center !important;
    gap: 6px !important;
    flex-wrap: nowrap !important;
    overflow: visible !important;
}

/* 2. Persona Segmented Control (Pill style) */
#persona_selector.block {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    min-width: fit-content !important;
}
#persona_selector .wrap {
    display: flex !important;
    gap: 0 !important;
    flex-wrap: nowrap !important;
    padding: 0 !important;
}
#persona_selector label {
    flex: 1 !important;
    height: 32px !important;
    line-height: 32px !important;
    padding: 0 !important;
    border: 1px solid var(--border-color-primary) !important;
    font-size: 0.8rem !important;
    border-radius: 0 !important;
    background: var(--background-fill-secondary) !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
#persona_selector label span {
    margin: 0 !important;
    padding: 0 !important;
}
#persona_selector label:not(:last-child) {
    margin-right: -1px !important;
}
#persona_selector label:first-child {
    border-top-left-radius: 8px !important;
    border-bottom-left-radius: 8px !important;
}
#persona_selector label:last-child {
    border-top-right-radius: 8px !important;
    border-bottom-right-radius: 8px !important;
}
#persona_selector input[type="radio"], 
#persona_selector .radio-circle {
    display: none !important;
}
#persona_selector label.selected {
    background-color: var(--primary-500) !important;
    color: white !important;
    border-color: var(--primary-600) !important;
    z-index: 1;
}

/* 3. Reviewer Checkbox (Unified height) */
#reviewer_toggle.block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    min-width: 90px !important;
    overflow: visible !important;
}
#reviewer_toggle label {
    height: 32px !important;
    line-height: 32px !important;
    padding: 0 10px !important;
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    display: flex !important;
    align-items: center !important;
    white-space: nowrap !important;
    background: var(--background-fill-secondary) !important;
    cursor: pointer !important;
}
#reviewer_toggle input {
    margin: 0 6px 0 0 !important;
}

/* 4. Button normalization */
.sm-btn {
    height: 32px !important;
    min-height: 32px !important;
    padding: 0 10px !important;
    font-size: 0.8rem !important;
    min-width: 60px !important;
}
"""


def build_ui() -> "gr.Blocks":
    """Assemble and return the Gradio Blocks application."""
    import gradio as gr

    with gr.Blocks(title="Collective Agreement Explorer", css=CUSTOM_CSS) as demo:
        # ── Header ────────────────────────────────────────────────────────────
        gr.Markdown("## BCGEU Steward Assistant")

        with gr.Accordion("Knowledge Base & Priority", open=False):
            gr.Markdown(
                "**The Collective Agreement and Primary Statutes** are our primary references. Anything else provides additional context."
            )
            # Use gr.HTML() to preserve clickable links (gr.Markdown sanitizes HTML)
            gr.HTML(build_pdf_download_links())
            gr.Markdown(
                f"[📁 Browse Knowledge Base on GitHub]({GITHUB_LABOUR_LAW_URL})"
            )

        # ── Disclaimer (persistent, non-dismissible) ──────────────────────────
        disclaimer_box = gr.HTML(DISCLAIMER_HTML)

        with gr.Row(visible=True) as chip_row:
            chip_btns = [gr.Button(q, size="sm") for q in EXAMPLE_QUESTIONS]

        # ── Chat interface ────────────────────────────────────────────────────
        chatbot = gr.Chatbot(
            height=480,
            buttons=["copy"],
            render_markdown=True,
            show_label=False,
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
        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Ask about the collective agreement…",
                label="",
                lines=2,
                max_lines=6,
                scale=5,
                show_label=False,
                container=False,
            )
            send_btn = gr.Button("Send ➤", scale=1, variant="primary")

        # ── Submit handlers ───────────────────────────────────────────────────
        async def submit(
            message: str,
            history: list[dict],
            use_reviewer: bool,
            persona_mode: str,
            **kwargs,
        ) -> AsyncIterator[tuple[list[dict], str, dict, dict]]:
            import gradio as gr
            
            # 1. Identify Banner
            top_banner = DISCLAIMER_HTML
            if persona_mode == "Direct":
                top_banner = DIRECT_MODE_HTML
            elif persona_mode == "Defend":
                top_banner = CASE_BUILDER_HTML

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
                yield history, "", show, gr.update()
                return

            message, was_flagged = sanitize_input(message)
            if was_flagged:
                yield (
                    history,
                    "Your input was flagged for security review. Please try a different question.",
                    show,
                    gr.update(),
                )
                return
            prior_history = list(history)
            # Append user turn; seed an empty assistant bubble for streaming.
            # Hide onboarding components on first message.
            history = prior_history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": ""},
            ]
            yield history, "", hide, gr.update(value=top_banner)
            # Stream tokens from RAG; accumulate into the assistant bubble
            accumulated = ""
            async for chunk in rag_review_stream(
                message, prior_history, use_reviewer, persona_mode
            ):
                accumulated += chunk
                history[-1]["content"] = accumulated
                yield history, "", hide, gr.update(value=top_banner)

        submit_inputs = [msg_input, chatbot, reviewer_toggle, persona_selector]
        submit_outputs = [chatbot, msg_input, chip_row, disclaimer_box]

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
    startup()
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
    allowed_paths = [str(LABOUR_LAW_DIR)]

    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
        allowed_paths=allowed_paths,
        auth=auth_creds,
    )
