---
title: Agreement Navigator (AgNav)
emoji: 📋
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: Look up the BCGEU 19th Main Public Service Agreement
---

# Agreement Navigator (AgNav)

AI chatbot built to empower BCGEU union stewards with instant, cited answers from a broad library
of labour law and contract documents.

> [!IMPORTANT]
> **Agreement Navigator is NOT a replacement for your Staff Representative.** It is a research tool
> designed to help you find contract language and legal context quickly. Always verify
> important findings with your steward team or union leadership.

---

## Knowledge Base

Agreement Navigator is currently indexed with the following core documents:

- **BCGEU 19th Main Public Service Agreement (Priority 1)**: The core collective agreement. This is the **authoritative source** for union stewards; all other documents provide context.
- **BC Employment Standards Act (Priority 2)**: Statutory minimums for wages, overtime, and notice.
- **BC Labour Relations Code**: The framework for collective bargaining and union operations.
- **BC Human Rights Code**: Protection against discrimination and harassment.
- **BC Workers Compensation Act**: Occupational health and safety (OHS) and injury claims.
- **BC Social Media Guidance for Public Service Employees**: Specific guidelines for personal and professional social media conduct.

### Priority & Weighting Logic
Agreement Navigator is programmed to prioritize the **Collective Agreement** above all else. When a query overlaps multiple sources:
1. The **Agreement** is used for primary enforcement.
2. **Statutes** (ESA, Labour Code, HRC) are cited as secondary legal context.
3. If no contract language exists, the assistant identifies relevant statutory protections.

### Adding or Updating Documents

Agreement Navigator indexes **Markdown files** (`.md`), not PDFs. PDFs are kept only for the "Download Original" links in the UI.

Add or replace Markdown files in `data/labour_law/` using the naming convention:

| Category | Path | Use case |
|---|---|---|
| `primary` | `01_primary/` | The Main Agreement and specific Component agreements |
| `statutory` | `02_statutory/` | Provincial or Federal laws (ESA, Labour Code) |
| `resources` | `03_resources/` | Steward manuals, ethics guides, "How-to" docs |
| `jurisprudence` | `04_jurisprudence/` | Arbitration awards and case precedents |

---

## Deployment Status

Agreement Navigator is deployed as a Docker container. We maintain two environments for
Docker deployments.

🚀 **TEST:** https://huggingface.co/spaces/DerekRoberts/landru

🚀 **PROD:** https://huggingface.co/spaces/MinionTech/vexilon

## Quick Start

### Prerequisites
- **Podman** (or Docker)
- **Podman Compose** (or Docker Compose)
- **Python 3.12+** (for local script execution)

### Run

Agreement Navigator is "Secure by Default" but optimized for a zero-config developer experience via Podman Compose.

**1. Local Development (Zero-Config)**
This is the default mode. It starts a local **Ollama** instance, pulls the required model weights, and launches the app. No API keys or tokens are required.

```bash
podman compose up --build
```

> [!NOTE]
> **Performance:** Local LLM execution speed depends on your CPU/GPU. The first run will be slower as it pulls the model weights defined in `app.py`.

**2. Production / Cloud Mode**
Uses the **Hugging Face Inference API** for high-speed "Flash" responses. Requires an internet connection and a valid token.

```bash
# Add your HF_TOKEN to .env or export it
export HF_TOKEN=your_token_here
podman compose up prod --build
```

> [!TIP]
> Using `--no-deps` prevents the local Ollama services from starting, allowing for an instant cloud-connected session.

The container uses a multi-stage build and pre-indexes the agreement at build time for zero-downtime startup.

---

## Using the Assistant

The app is ready immediately on page load — no dropdown, no Load button.

1. Type a question in the input field and press **Enter** or tap **Send**
2. Or click one of the suggested question chips on the welcome screen
3. Responses include a plain-language explanation followed by verbatim quotes with citations
4. **Persona Mode**: Toggle between operational modes to receive tactical guidance:
   - **Lookup**: Standard research mode for finding specific clauses.
   - **Grieve**: Forensic Auditor mode to build air-tight grievance cases with "Staff Rep" level scrutiny.
   - **Manage**: Strategic Consultant mode for risk mitigation and compliance analysis.

> **Note:** Informational purposes only. Consult your BCGEU representative or a legal advisor as appropriate.

---

## Forensic Integrity Pipeline

To ensure the AI never "hallucinates" contract language, we use a forensic conversion pipeline:

1. **Precision Extraction**: PDFs are converted to Markdown using `scripts/pdf_to_md.py`.
2. **Dual-Pass Verification**: The converter uses two different LLM passes to verify structural integrity.
3. **Word Fingerprinting**: We verify that every substantive word in the Markdown exists in the original PDF.

```bash
# Convert a new PDF to high-integrity Markdown
python scripts/pdf_to_md.py path/to/document.pdf
```

---

## Security & Reliability

### Rate Limiting

To prevent API abuse and ensure fair usage, Agreement Navigator implements a multi-tier rate limiter:

| Tier | Default Limit | Description |
|---|---|---|
| `RATE_LIMIT_PER_MINUTE` | `5` | Burst protection for chat messages |
| `RATE_LIMIT_PER_HOUR` | `100` | Maximum requests per hour per client IP |

When a rate limit is exceeded, users receive a clear error message indicating which limit was hit and when they can retry.

### Input Sanitization

Input sanitization prevents prompt injection attacks by detecting and blocking malicious inputs:

| Setting | Value | Description |
|---|---|---|
| `MAX_INPUT_LENGTH` | `10000` | Maximum characters per message |
| `LOG_SUSPICIOUS_INPUTS` | `true` | Log flagged inputs for security review |

The sanitization checks for 16+ prompt injection patterns including:
- `ignore all/previous/system instructions`
- `forget your/the instructions`
- `you are now ... instead`
- `jailbreak`, `developer mode`, `sudo mode`

### Privacy & Data Retention

Agreement Navigator is a "content-blind" application designed for maximum privacy and to support compliance with the British Columbia **Personal Information Protection Act (PIPA)**.

- **Ephemeral Conversations**: Chats are tied only to your current browser session and are permanently deleted upon refresh or closure.
- **No Content Logging**: We **never** log user queries, bot responses, or search reasoning.
- **Anonymized Metrics**: We only track non-sensitive technical metadata (token counts, query frequency) to monitor system health.

For full technical disclosure and mapping to the 10 PIPA Fair Information Principles, see [PRIVACY.md](docs/PRIVACY.md).

---

## Hugging Face Spaces Deployment

The Space runs as **`sdk: docker`** in production — the deploy script pushes a stub
`Dockerfile` pointing to the pre-built container image on `ghcr.io/miniontech/agnav`.
The FAISS index is already baked into that image (built via the `Containerfile` `RUN` step),
so the Space starts instantly.

### Automated deploy (GitHub Actions)

The deployment process (`.github/workflows/deploy-*.yml`) pushes a stub `Dockerfile` to
the HF Space.

- **TEST:** Every push to `main` triggers [`.github/workflows/deploy-test.yml`](.github/workflows/deploy-test.yml), deploying to the `DerekRoberts/landru` Space.
- **PROD:** Every published GitHub release triggers [`.github/workflows/deploy-prod.yml`](.github/workflows/deploy-prod.yml), deploying to the `DerekRoberts/vexilon` Space.

**Required GitHub secret:**

| Secret | Value |
|---|---|
| `HF_TOKEN` | Hugging Face write-scoped access token ([settings/tokens](https://huggingface.co/settings/tokens)) |

---

## Running Tests

Agreement Navigator uses a **Quality Gate** pattern in `compose.yml` — the app will not start unless the test suite passes.

### Test Tiers

| Tier | Location | Model | When to run |
|---|---|---|---|
| **Unit** | `tests/test_*.py` | Mocked (no download) | Every commit — fast, zero RAM cost |
| **Integration** | `tests/integration/` | Real `BAAI/bge-small-en-v1.5` (~800 MB) | In container — memory-capped at 2 GB |
| **Smoke** | `tests/smoke/` | Real HF/Ollama API | Manually, to verify live API connectivity |

### Commands

```bash
# Run unit tests only — fast, safe locally
uv run pytest tests/ --ignore=tests/integration --ignore=tests/smoke

# Run full suite (unit + integration) inside the memory-capped container
podman compose run --rm tests

# Gated startup — tests must pass before AgNav launches
podman compose up
```

---

## Project Structure

```
agnav/
├── app.py            # Main application (RAG pipeline + Gradio UI)
├── conftest.py       # pytest root path configuration
├── pyproject.toml    # Python dependencies (managed by uv)
├── compose.yml       # Podman Compose config (production parity)
├── SPEC.md           # Product specification
├── data/             # Knowledge base source files
│   └── labour_law/   # Hierarchical document library
│       ├── primary/       # Collective Agreement, Labour Relations Code
│       ├── statutory/     # Employment Standards Act, Human Rights Code
│       ├── resources/     # Steward manuals, ethics guides
│       ├── jurisprudence/ # Arbitration awards, case precedents
│       └── tests/         # Test/doctrine registry (Millhaven, KVP)
└── tests/            # pytest test suite
    ├── conftest.py         # root: mock embedding model + mock OpenAI client
    ├── test_chunking.py    # chunk_text() unit tests
    ├── test_sanitize_input.py  # prompt injection detection tests
    └── integration/        # real model — run via: podman-compose run --rm tests
```
