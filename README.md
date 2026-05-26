# Agreement Navigator

AI chatbot built to empower BCGEU union stewards with instant, cited answers from a broad library
of labour law and contract documents.

> [!IMPORTANT]
> **Agreement Navigator is NOT a replacement for your Staff Representative.** It is a research tool
> designed to help you find contract language and legal context quickly. Always verify
> important findings with your steward team or union leadership.

---

## Knowledge Base

Agreement Navigator dynamically ingests collective agreements and statutory/procedural resource documents. By default, the retrieval pipeline is optimized to prioritize primary collective agreements, using statutes, statutory codes, and organizational manuals to provide secondary legal and operational context.

For the active list of source texts and regulatory documents loaded into this instance, refer to the files organized under the `app/data/` directories.

### Adding or Updating Documents

Agreement Navigator indexes **Markdown files** (`.md`), not PDFs. PDFs are kept only for the "Download Original" links in the UI.

Add or replace Markdown files in `app/data/` using the naming convention:

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

- **TEST**: https://bcgeu-navigator-test.hf.space

- **PROD**: https://bcgeu-navigator.hf.space

## Quick Start

### Prerequisites
- **Podman** (or Docker)
- **Podman Compose** (or Docker Compose)
- **Python 3.12+** (for local script execution)

### Container Orchestration

Agreement Navigator is fully configured for container execution. You can spin up development, testing, and staging environments directly from the repository root using native Compose commands.

Exposed services (run from the repository root):
- **Boot Local Development**: `docker compose up dev`
- **Graceful Tear Down**: `docker compose down -v`
- **Run Full Containerized Test Suite**: `docker compose up test-everything`
- **Build Container Images**: `docker compose build`

---

### Running the Application Natively

**1. Local Development (Zero-Config)**
This is the default mode. It starts a local **Ollama** instance, pulls the required model weights, and launches the app with hot-reload. No API keys or tokens are required.

```bash
docker compose up --build dev
```

> [!NOTE]
> **Performance:** Local LLM execution speed depends on your CPU/GPU. The first run will be slower as it pulls the model weights defined in `app/main.py`.

**2. Production / Cloud Simulation**
Uses the **Hugging Face Inference API** for high-speed "Flash" responses. Requires an internet connection and a valid token. This simulates the exact environment of the Hugging Face Space.

```bash
# Add your HF_TOKEN to .env or export it
export HF_TOKEN=your_token_here
docker compose up --build staging
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

## Multi-Perspective Retrieval

To ensure follow-up questions work reliably (e.g., "What about for part-time?"), Agreement Navigator uses a **Multi-Perspective Context** pattern:

1. **Query Condensing**: A fast LLM pass reconstructs the user's intent into a standalone search query based on conversation history.
2. **Perspective Generation**: For complex queries, the system generates 3 different search angles (legal, procedural, factual) to ensure maximum retrieval coverage.
3. **Multi-Perspective Search**: The FAISS index is searched using all generated queries, and the results are ranked and deduplicated.

## Forensic Integrity Pipeline

To ensure the AI never "hallucinates" contract language, we use a forensic conversion pipeline:

1. **Precision Extraction**: PDFs are converted to Markdown using `app/scripts/pdf_to_md.py` via **PyMuPDF**.
2. **Dual-Pass Verification**: The converter uses two different LLM passes to verify structural integrity.
3. **Word Fingerprinting**: We verify that every substantive word in the Markdown exists in the original PDF.

```bash
# Convert a new PDF to high-integrity Markdown
python app/scripts/pdf_to_md.py path/to/document.pdf
```

---

## Security & Reliability

### Rate Limiting & Abuse Prevention

To prevent API abuse and ensure high service availability, the application implements active IP-based rate limiting. Tiered request limits are applied per minute and per hour. When limits are exceeded, the API returns user-friendly rate-limit warning cards with retry-duration indicators.

### Input Sanitization & Jailbreak Protection

Input sanitization safeguards the LLM context from prompt injection attacks and malicious overrides. The security engine dynamically checks and filters user inputs for maximum length constraints and known system instruction override patterns to prevent jailbreaking, adversarial roleplaying, or instruction leakage.

### Privacy & Data Retention

Agreement Navigator is a "content-blind" application designed for maximum privacy and to support compliance with the British Columbia **Personal Information Protection Act (PIPA)**.

- **Ephemeral Conversations**: Chats are tied only to your current browser session and are permanently deleted upon refresh or closure.
- **Surgical Query Masking**: We log the occurrence of queries (including technical metadata like word/character counts) to monitor system health, but the **actual content** of user messages and bot responses is never logged.
- **Anonymized Metrics**: We only track non-sensitive technical metadata to monitor system performance and rate-limiting compliance.

For full technical disclosure and mapping to the 10 PIPA Fair Information Principles, see [PRIVACY.md](./PRIVACY.md).

---

## Hugging Face Spaces Deployment

The Space runs as **`sdk: docker`** in production — the deploy script pushes a stub
`Dockerfile` pointing to the pre-built container image on `ghcr.io/miniontech/vexilon/agnav`.
The FAISS index is already baked into that image (built via the `Containerfile` `RUN` step),
so the Space starts instantly.

### Automated deploy (GitHub Actions)

The deployment process (`.github/workflows/deploy-*.yml`) pushes a stub `Dockerfile` to
the HF Space.

- **TEST:** Every push to `main` triggers [`.github/workflows/deploy-test.yml`](.github/workflows/deploy-test.yml), deploying to the `bcgeu/navigator-test` Space.
- **PROD:** Every published GitHub release triggers [`.github/workflows/deploy-prod.yml`](.github/workflows/deploy-prod.yml), deploying to the `bcgeu/navigator` Space.

**Required GitHub secret:**

| Secret | Value |
|---|---|
| `HF_TOKEN` | Hugging Face write-scoped access token ([settings/tokens](https://huggingface.co/settings/tokens)) |

---

## Running Tests

Agreement Navigator features a strict **Quality Gate** deployment pattern—all automated unit and integration tests must pass to verify the application's integrity before local or staging environments boot.

### Commands

```bash
# Run unit tests only — fast, safe locally
uv run pytest app/tests/ --ignore=app/tests/integration --ignore=app/scripts/smoke_multi.py

# Run containerized unit tests (Mocked, zero-AI)
docker compose up --build test-unit

# Run model integration tests (FAISS + Embedding Model)
docker compose up --build test-integration-model

# Run app integration tests (Functional RAG flow)
docker compose up --build test-integration-app

# Verify everything at once (The "Grand Slam") and launch the dev app if successful
docker compose up --build test-everything && docker compose up dev
```

---

## Contributing

We encourage contributions to Agreement Navigator via **pull requests**. 

- **Workflow**: Create a branch from `main`, commit your changes, and submit a PR.
- **Merge**: PRs are evaluated by the maintainers and are typically squash-merged to `main`.
- **Licensing**: By contributing, you grant all users the right to use, modify, and distribute your contributions under the terms of the [GNU GPLv3 License](./LICENSE).

---

*Agreement Navigator — Empowering Stewards through Forensic RAG.*
