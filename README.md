---
title: BCGEU Steward Assistant
emoji: 📋
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860 # Must match PORT=7860 in app.py to prevent status-sync drift
startup_duration_timeout: 10m
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
| Embeddings | `BAAI/bge-small-en-v1.5` — local CPU, no API key |
| Vector Store | FAISS (in-memory, rebuilt at startup) |
| Web UI | Gradio 6 — `http://localhost:7860` |
| Knowledge Base | Multi-source Markdown in `data/labour_law/` |
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

Vexilon indexes **Markdown files** (`.md`), not PDFs. PDFs are kept only for the "Download Original" links in the UI.

Add or replace Markdown files in `data/labour_law/` using the naming convention:

`[Index]_[Category]_[Human Readable Title].md`
*(Example: `7_Guidance_Social Media Policy.md`)*

To convert a PDF to Markdown first, see [docs/MARKDOWN_CONVERSION.md](docs/MARKDOWN_CONVERSION.md).

Then rebuild the index:

```bash
python app.py --rebuild-index
```

When done, commit `.pdf_cache/index.faiss` and `.pdf_cache/chunks.json` if you want to update
the GitHub fallback that HF Spaces downloads on first boot. The container image always
rebuilds the index automatically during `docker build` — no manual commit is required for
Docker deployments.

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
export ANTHROPIC_API_KEY=<YOUR_ANTHROPIC_API_KEY>
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
export ANTHROPIC_API_KEY=<YOUR_ANTHROPIC_API_KEY>
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

## Maintenance & Tools

### PDF to Markdown Conversion
For optimal RAG performance, we recommend converting core PDFs into high-fidelity Markdown. This improves header-aware chunking and removes "noise" like web-to-PDF artifacts.

See [docs/MARKDOWN_CONVERSION.md](docs/MARKDOWN_CONVERSION.md) for full rationale and instructions.

**Run the converter:**
```bash
export ANTHROPIC_API_KEY=<YOUR_ANTHROPIC_API_KEY>
python scripts/pdf_to_md.py path/to/document.pdf
```

## Configuration

All settings are optional — defaults match the product specification.

### Core Settings

| Variable | Default | Description |
|---|---|---|
| `VEXILON_USERNAME` | `admin` | Username for basic authentication |
| `VEXILON_PASSWORD` | *(optional)* | Password for basic authentication. If unset, auth is disabled. |
| `ANTHROPIC_API_KEY` | *(required)* | Anthropic API key |
| `CLAUDE_MODEL` | `claude-haiku-4-5-20251001` | Claude model for responses |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | sentence-transformers embedding model (512-token window) |
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
| `MAX_INPUT_LENGTH` | `10000` | Maximum characters per message |
| `LOG_SUSPICIOUS_INPUTS` | `true` | Log flagged inputs for security review |

The sanitization checks for 16+ prompt injection patterns including:
- `ignore all/previous/system instructions`
- `forget your/the instructions`
- `you are now ... instead`
- `jailbreak`, `developer mode`, `sudo mode`
- And other common injection techniques

### Privacy & Data Retention

Vexilon is a "content-blind" application designed for maximum privacy and to support compliance with the British Columbia **Personal Information Protection Act (PIPA)**.

- **Ephemeral Conversations**: Chats are tied only to your current browser session and are permanently deleted upon refresh or closure.
- **No Content Logging**: We **never** log user queries, bot responses, or search reasoning.
- **Minimal Metadata**: Non-sensitive data (token counts, quality scores) is tracked for system health but never reaches persistent storage.

For full technical disclosure and mapping to the 10 PIPA Fair Information Principles, see our [Privacy Policy (PIPA)](docs/PRIVACY.md) and the [Privacy & Data Retention](SPEC.md#privacy--data-retention-updated-215-216) section of the project specification. 

## Hugging Face Spaces Deployment

The Space runs as **`sdk: docker`** in production — the deploy script pushes a stub
`Dockerfile` pointing to the pre-built container image on `ghcr.io/derekroberts/vexilon`.
The FAISS index is already baked into that image (built via the `Containerfile` `RUN` step),
so the Space starts instantly.

### FAISS index fallback (Gradio SDK / bare startup)

If the app ever runs without the pre-built index (e.g. during development or on a fresh
Gradio-SDK Space), [`_fetch_pdf_cache_if_missing()`](app.py) downloads
`.pdf_cache/index.faiss` and `.pdf_cache/chunks.json` from this public GitHub repo.
Those files are **not** committed by default (`.pdf_cache/` is gitignored).
To publish an updated fallback after rebuilding the index locally:

```bash
python app.py --rebuild-index
git add -f .pdf_cache/index.faiss .pdf_cache/chunks.json
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

Vexilon uses a **Quality Gate** pattern in `compose.yml` — the app will not start unless the test suite passes.

#### Test tiers

| Tier | Location | Model | When to run |
|---|---|---|---|
| **Unit** | `tests/test_*.py` | Mocked (no download) | Every commit — fast, zero RAM cost |
| **Integration** | `tests/integration/` | Real `BAAI/bge-small-en-v1.5` (~800 MB) | In container — memory-capped at 2 GB |
| **Smoke** | `tests/smoke/` | Real Anthropic API | Manually, to verify live API connectivity |

#### Commands

````bash
# Run unit tests only — fast, safe locally
uv run pytest tests/ --ignore=tests/integration --ignore=tests/smoke

# Run full suite (unit + integration) inside the memory-capped container
podman-compose run --rm tests

# Gated startup — tests must pass before Vexilon launches
export ANTHROPIC_API_KEY=<YOUR_ANTHROPIC_API_KEY>
podman-compose up

# Skip the gate — useful for rapid UI iteration
podman-compose up vexilon

# Smoke tests — verifies real Anthropic API connectivity
podman-compose run --rm tests sh -c "uv run --no-sync pytest tests/smoke/ -v"
````

> [!NOTE]
> Integration tests intentionally load the real embedding model. Run them locally only if you have
> ~1.5 GB of free RAM headroom. The Compose `tests` service caps usage at 2 GB.

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
│   └── labour_law/   # Hierarchical document library
│       ├── primary/       # Collective Agreement, Labour Relations Code
│       ├── statutory/     # Employment Standards Act, Human Rights Code
│       ├── resources/     # Steward manuals, ethics guides
│       ├── jurisprudence/ # Arbitration awards, case precedents
│       └── tests/         # Test/doctrine registry (Millhaven, KVP)
├── tests/            # pytest test suite
│   ├── conftest.py         # root: mock embedding model + mock Anthropic client
│   ├── test_chunking.py    # chunk_text() unit tests
│   ├── test_condense_query.py  # query condensation unit tests
│   ├── test_fetch.py       # index bootstrap / HTTP fetch unit tests
│   ├── test_index.py       # FAISS build/search unit tests
│   ├── test_knowledge_base.py  # PDF/MD parity integrity check
│   ├── test_md_ingestion.py    # TOC detection, artifact cleaning, MD loader
│   ├── test_md_loader.py   # load_md_chunks() unit tests
│   ├── test_persistence.py # index save/load round-trip tests
│   ├── test_rag_stream.py  # rag_stream() unit tests
│   ├── test_rate_limit.py  # rate limiter unit tests
│   ├── test_sanitize_input.py  # prompt injection detection tests
│   ├── test_verify_response.py # verification bot unit tests
│   ├── integration/        # real model — run via: podman-compose run --rm tests
│   │   ├── test_app_flow.py        # full startup → index → RAG stream flow
│   │   ├── test_embed_pipeline.py  # sentence-transformers + FAISS interop
│   │   └── test_gradio_ui.py       # Gradio Blocks construction check
│   └── smoke/
│       └── test_model_valid.py  # live Anthropic API model validation
└── .pdf_cache/       # Pre-built FAISS index and chunk metadata
````
