---
title: BCGEU Steward Assistant
emoji: 📋
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.9.0"
app_file: app.py
pinned: false
license: mit
short_description: Look up the BCGEU 19th Main Public Service Agreement
---

# BCGEU Steward Assistant

AI chatbot built to empower BCGEU union stewards with instant, cited answers from a broad library
of labour law and contract documents.

> See [SPEC.md](SPEC.md) for the full product specification.

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Anthropic Claude (`claude-haiku-4-5-20251001`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` — local CPU, no API key |
| Vector Store | FAISS (in-memory, rebuilt at startup) |
| PDF Parsing | pypdf — preserves page numbers |
| Web UI | Gradio 6 — `http://localhost:7860` |
| Knowledge Base | Multi-source PDFs in `data/labour_law/` |
| Deployment | Hugging Face Spaces + GitHub Actions |

## Knowledge Base

Vexilon is currently indexed with the following core documents:

- **BCGEU 19th Main Public Service Agreement (Priority 1)**: The core collective agreement. This is the **authoritative source** for union stewards; all other documents provide context.
- **BC Employment Standards Act (Priority 2)**: Statutory minimums for wages, overtime, and notice.
- **BC Labour Relations Code (Priority 3)**: The legal framework for union-management relations and LRB precedents.
- **BC Human Rights Code (Priority 4)**: Protections against discrimination and the duty to accommodate.
- **BCGEU Steward Fundamentals Handbook**: Practical union guidance for grievances and meeting scripts.
- **Standards of Conduct (Public Service Ethics)**: Policy framework for employee behavior and social media use.
- **BC Social Media Guidance for Public Service Employees**: Specific guidelines for personal and professional social media conduct.

### Priority & Weighting Logic
Vexilon is programmed to prioritize the **Collective Agreement** above all else. When a query overlaps multiple sources:
1. The **Agreement** is used for primary enforcement.
2. **Statutes** (ESA, Labour Code, HRC) are cited as secondary legal context.
3. If no contract language exists, the assistant identifies relevant statutory protections.

### Adding or Updating Documents

Drop PDFs into `data/labour_law/` using our **Smart Renaming** convention to ensure proper indexing and UI labeling:

`[Index]_[Category]_[Human Readable Title].pdf`
*(Example: `7_Guidance_Social Media Policy.pdf`)*

Then rebuild and commit the index:

```bash
python -c "from app import build_index_from_pdfs; build_index_from_pdfs()"
```

When done, commit `pdf_cache/index.faiss` and `pdf_cache/chunks.json` if you want to update
the GitHub fallback that HF Spaces downloads on first boot.  The container image always
rebuilds the index automatically during `docker build` via the `Containerfile` `RUN` step — no
manual commit is required for Docker deployments.

> [!TIP]
> **Suggested Additions:**
> - WorkSafeBC (WCB) Occupational Health & Safety Policies
> - Specific Component/Local Agreements
> - Public Service Benefit Plan Details

## Hosted

🚀 **TEST:** https://huggingface.co/spaces/DerekRoberts/landru

🚀 **PROD:** https://huggingface.co/spaces/DerekRoberts/vexilon

## Quick Start

### Prerequisites

- [Podman](https://podman.io/docs/installation) + [Podman Compose](https://github.com/containers/podman-compose)
- An [Anthropic API key](https://console.anthropic.com/) (`ANTHROPIC_API_KEY`)

### Run

**Run the production-optimized container:**

```bash
export ANTHROPIC_API_KEY=sk-ant-...
podman-compose up --build
```

The container uses a multi-stage build and pre-indexes the agreement at build time for zero-downtime startup.

**Alternatives:**

```bash
# Run locally (useful for quick iteration without a container):
uv run --with-requirements requirements.txt python app.py
```

Open <http://localhost:7860> in your browser.

> ✅ **Startup is fast** — the embedding model and FAISS index are both baked into the
> container image at build time (via the `Containerfile` `RUN` step). No rebuild on first run.

### Troubleshooting

**`anthropic.AuthenticationError`** — `ANTHROPIC_API_KEY` is not set or not exported.
Check with `echo $ANTHROPIC_API_KEY` and re-export before running:

````bash
export ANTHROPIC_API_KEY=sk-ant-...
podman-compose up
````

## Usage

The app is ready immediately on page load — no dropdown, no Load button.

1. Type a question in the input field and press **Enter** or tap **Send**
2. Or click one of the suggested question chips on the welcome screen
3. Responses include a plain-language explanation followed by verbatim quotes with citations
4. **Direct Advice Mode**: Toggle this to receive tactical, operational guidance from a "Senior Staff Rep" persona. This includes:
   - **Immediate Action Steps**: Numbered instructions for your next moves.
   - **Meeting Scripts**: Verbatim language to use with management.
   - **Nexus Analysis**: Specific guidance for off-duty conduct conduct cases.

> **Note:** Informational purposes only. Consult your BCGEU representative or a legal advisor as appropriate.

## Configuration

All settings are optional — defaults match the product specification.

### Core Settings

| Variable | Default | Description |
|---|---|---|
| `VEXILON_USERNAME` | `admin` | Username for basic authentication |
| `VEXILON_PASSWORD` | *(optional)* | Password for basic authentication. If unset, auth is disabled. |
| `ANTHROPIC_API_KEY` | *(required)* | Anthropic API key |
| `CLAUDE_MODEL` | `claude-haiku-4-5-20251001` | Claude model for responses |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers embedding model |
| `PORT` | `7860` | Gradio listen port |
| `SIMILARITY_TOP_K` | `40` | Chunks retrieved per query |
| `CHUNK_SIZE` | `450` | Tokens per chunk |
| `CHUNK_OVERLAP` | `100` | Token overlap between chunks |

### Verification Bot

Vexilon includes a second AI bot that verifies responses against source citations to reduce hallucinations:

| Variable | Default | Description |
|---|---|---|
| `VERIFY_ENABLED` | `true` | Enable verification bot to check claims against citations |
| `VERIFY_MODEL` | `claude-haiku-4-5-20251001` | Claude model for verification |

When enabled, the verification bot reviews each response and checks if quoted text actually supports the claims made. If claims are disputed, a "Verification" note is appended to the response. Verified responses remain clean with no added note.

> **Evaluation:** After deploying, monitor responses for "Verification:" notes.
> - If notes appear frequently → verification is catching real issues; keep enabled
> - If notes never appear → the main bot is reliable; consider disabling to save ~30% on API costs
> - This evaluation approach assumes VERIFY_ENABLED=true by default to assess value over time

### Rate Limiting

Rate limiting prevents abuse and controls API costs by throttling requests per client IP:

| Variable | Default | Description |
|---|---|---|
| `RATE_LIMIT_PER_MINUTE` | `10` | Maximum requests per minute per client IP |
| `RATE_LIMIT_PER_HOUR` | `100` | Maximum requests per hour per client IP |

When a rate limit is exceeded, users receive a clear error message indicating which limit was hit and when they can retry.

### Input Sanitization

Input sanitization prevents prompt injection attacks by detecting and blocking malicious inputs:

| Variable | Default | Description |
|---|---|---|
| `MAX_INPUT_LENGTH` | `2000` | Maximum characters per message |
| `LOG_SUSPICIOUS_INPUTS` | `true` | Log flagged inputs for security review |

The sanitization checks for 16+ prompt injection patterns including:
- `ignore all/previous/system instructions`
- `forget your/the instructions`
- `you are now ... instead`
- `jailbreak`, `developer mode`, `sudo mode`
- And other common injection techniques

When a potentially malicious input is detected, users receive a friendly message: "Your input was flagged for security review. Please try a different question."

## Hugging Face Spaces Deployment

The Space runs as **`sdk: docker`** in production — the deploy script pushes a stub
`Dockerfile` pointing to the pre-built container image on `ghcr.io/derekroberts/vexilon`.
The FAISS index is already baked into that image (built via the `Containerfile` `RUN` step),
so the Space starts instantly.

### FAISS index fallback (Gradio SDK / bare startup)

If the app ever runs without the pre-built index (e.g. during development or on a fresh
Gradio-SDK Space), [`_fetch_pdf_cache_if_missing()`](app.py) downloads
`pdf_cache/index.faiss` and `pdf_cache/chunks.json` from this public GitHub repo.
Those files are **not** committed by default (`pdf_cache/` is gitignored).
To publish an updated fallback after rebuilding the index locally:

```bash
python -c "from app import build_index_from_pdfs; build_index_from_pdfs()"
git add -f pdf_cache/index.faiss pdf_cache/chunks.json
git commit -m "chore(index): rebuild fallback cache"
git push
```

### Automated deploy (GitHub Actions)

The deployment process (`.github/workflows/deploy-*.yml`) pushes a stub `Dockerfile` to
the HF Space.

- **TEST:** Every push to `main` triggers [`.github/workflows/deploy-test.yml`](.github/workflows/deploy-test.yml), deploying to the `DerekRoberts/landru` Space.
- **PROD:** Every published GitHub release triggers [`.github/workflows/deploy-prod.yml`](.github/workflows/deploy-prod.yml), deploying to the `DerekRoberts/vexilon` Space.

**Required GitHub secret:**

| Secret | Value |
|---|---|
| `HF_TOKEN` | Hugging Face write-scoped access token ([settings/tokens](https://huggingface.co/settings/tokens)) |

**Required HF Space secret** (set in [Space settings](https://huggingface.co/spaces/DerekRoberts/vexilon/settings)):

| Secret | Value |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key |

### Manual deploy (one-time setup or re-deploy)

The `.github/scripts/deploy.sh` script handles this end-to-end.  To run manually:

````bash
HF_TOKEN=YOUR_HF_TOKEN ./.github/scripts/deploy.sh sha-$(git rev-parse --short HEAD) --prod
````

### Running tests

Vexilon uses a **Quality Gate** pattern in the `compose.yml`. By default, the app will not start unless the test suite passes.

````bash
# 1. Run the gated startup (Tests must pass before Vexilon launches)
export ANTHROPIC_API_KEY=sk-ant-...
podman-compose up

# 2. Skip the gate (Useful for rapid UI iteration)
podman-compose up vexilon

# 3. Run containerized tests manually
podman-compose run --rm tests

# 4. Run 'Smoke Tests' against the real Anthropic API (inside container)
podman-compose run --rm tests sh -c "uv run pytest tests/smoke/ -v"
````

> 💡 **Tip:** Local tests use mocked responses by default to save API credits. Use the Smoke Test command above to verify real API connectivity.

## Project Structure

````
vexilon/
├── app.py            # Main application (RAG pipeline + Gradio UI)
├── conftest.py       # pytest root path configuration
├── requirements.txt  # Python dependencies
├── Containerfile     # Production-optimized multi-stage image definition
├── compose.yml       # Podman Compose config (production parity)
├── SPEC.md           # Product specification
├── data/             # Knowledge base source files
│   └── labour_law/   # Directory for PDF documents to be indexed
├── tests/            # pytest test suite
│   ├── test_chunking.py    # chunk_text() unit tests
│   ├── test_index.py       # FAISS build/search unit tests
│   ├── test_pdf_loader.py  # load_pdf_chunks() unit tests
│   ├── test_rag_stream.py  # rag_stream() unit tests + model name blocklist
│   └── smoke/
│       └── test_model_valid.py  # live API model validation
└── pdf_cache/        # Pre-built FAISS index and chunk metadata
````
