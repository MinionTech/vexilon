import os
from pathlib import Path

# CHAINLIT_FILES_DIR defensive fallback for non-container dev (must precede chainlit imports)
os.environ.setdefault("CHAINLIT_FILES_DIR", "/tmp/chainlit_files")
os.environ.setdefault("AGNAV_APP_NAME", "BCGEU Navigator")
os.environ.setdefault("AGNAV_APP_DESCRIPTION", "BCGEU Agreement Navigator")
Path(os.environ["CHAINLIT_FILES_DIR"]).mkdir(parents=True, exist_ok=True)

if os.getenv("SPACE_ID") or os.getenv("HF_SPACE_ID"):
    os.environ["CHAINLIT_COOKIE_SAMESITE"] = "none"
    space_host = os.getenv("SPACE_HOST")
    if space_host:
        os.environ["CHAINLIT_URL"] = f"https://{space_host}"

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import logging
from brand import AGNAV_APP_NAME, AGNAV_APP_DESCRIPTION
from indexing import DATA_DIR, CACHE_DIR, PDF_CACHE_DIR

# Single Source of Truth for local development models.
OLLAMA_MODEL_ID = "tinyllama"
# Allow environment override for CI (e.g. tinyllama for smoke tests)
CURRENT_MODEL_ID = os.getenv("OLLAMA_MODEL_ID", OLLAMA_MODEL_ID)
DEFAULT_HF_MODEL_ID = "google/gemma-4-31B-it"

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

AGNAV_VERSION = os.getenv("AGNAV_VERSION", "Dev mode")
IS_DEV = not (os.getenv("SPACE_ID") or os.getenv("HF_SPACE_ID"))
AGNAV_REPO_URL = os.getenv("AGNAV_REPO_URL", "https://github.com/MinionTech/vexilon")
GITHUB_DATA_URL = os.getenv(
    "AGNAV_KNOWLEDGE_URL", f"{AGNAV_REPO_URL}/tree/main/app/data"
)

# Models & Providers
def get_llm_provider() -> str:
    # 1. Explicit override (keeps the 'prod' profile working)
    val = os.getenv("AGNAV_LLM_PROVIDER")
    if val:
        return val.lower().strip()

    # 2. Explicit dev mode flag (defaults to PROD)
    if IS_DEV:
        return "ollama"  # We're coding locally!
    return "huggingface"  # We're in the clouds!

# Curated List of Supported Models
HF_PROVIDER = os.getenv("AGNAV_HF_PROVIDER", "fastest").strip()

def get_default_model_setting() -> str:
    provider = get_llm_provider()
    if provider == "ollama":
        return f"ollama:{CURRENT_MODEL_ID}"
    return f"huggingface:{DEFAULT_HF_MODEL_ID}"

def _get_default_model() -> str:
    provider = get_llm_provider()
    # Default to Hugging Face or Ollama
    if provider == "ollama":
        val = os.getenv("OLLAMA_MODEL")
        return val if (val and val.strip()) else CURRENT_MODEL_ID
    return DEFAULT_HF_MODEL_ID

DEFAULT_MODEL_LLM = os.getenv("AGNAV_DEFAULT_MODEL", _get_default_model())
CLAUDE_MODEL = os.getenv("AGNAV_CLAUDE_MODEL", DEFAULT_MODEL_LLM)
REVIEWER_MODEL = os.getenv("AGNAV_REVIEWER_MODEL", DEFAULT_MODEL_LLM)
CONDENSE_MODEL = os.getenv("AGNAV_CONDENSE_MODEL", DEFAULT_MODEL_LLM)
VERIFY_MODEL = os.getenv("AGNAV_VERIFY_MODEL", DEFAULT_MODEL_LLM)

RAG_MAX_TOKENS = 4096
REVIEWER_MAX_TOKENS = 4096

MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", 10000))
LOG_SUSPICIOUS_INPUTS = os.getenv("LOG_SUSPICIOUS_INPUTS", "true").lower() == "true"

RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "999999" if IS_DEV else "10"))
RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "999999" if IS_DEV else "100"))

VERIFY_ENABLED = os.getenv("VERIFY_ENABLED", "false" if IS_DEV else "true").lower() == "true"

LLM_MAX_RETRIES = int(os.getenv("AGNAV_LLM_MAX_RETRIES", "3"))
LLM_RETRY_BASE_DELAY = float(os.getenv("AGNAV_LLM_RETRY_BASE_DELAY", "0.5"))
LLM_RETRY_MAX_DELAY = float(os.getenv("AGNAV_LLM_RETRY_MAX_DELAY", "8.0"))

PERSONAS = ["Lookup", "Grieve", "Manage"]
DEFAULT_PERSONA = "Lookup"
VEXILON_SAVE_SENTINEL = "__VEXILON_SAVE__"

EXAMPLES = [
    "What are the Article 14 (Discipline) requirements for just cause?",
    "What are my rights as a steward during an investigation meeting?",
    "What is the nexus test for establishing a link in off-duty conduct cases?",
    "Show me the Harassment Threshold test.",
    "I need to file a grievance for a member. What steps should I take?",
]

TESTS_DIR = DATA_DIR / "test_fixtures"
PUBLIC_DOCS_DIR = Path(__file__).parent.parent / "public" / "docs"

# ─── RAG Pipeline Constants ─────────────────────────────────────────────────
_SIMPLE_KEYWORDS = {"phone", "number", "address", "email", "contact", "list", "who", "are", "you", "hello", "hi"}
_JOKE_KEYWORDS = {"joke", "funny", "nose", "pick", "mad", "angry", "boss", "dumb", "stupid"}
_ALL_SIMPLE_KEYWORDS = _SIMPLE_KEYWORDS | _JOKE_KEYWORDS

HIGH_TRAFFIC_MESSAGE = "⏳ The AI service is currently experiencing high traffic. Please wait a moment and try again."
GENERIC_ERROR_MESSAGE = "⚠️ An unexpected error occurred while processing your request. Please try again."

def format_rag_error_message(exc: Exception) -> str:
    """Map exceptions to user-facing error messages, hiding internal error details."""
    from services.llm import is_transient_llm_error
    if is_transient_llm_error(exc):
        return HIGH_TRAFFIC_MESSAGE

    return GENERIC_ERROR_MESSAGE
