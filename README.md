---
title: Agreement Navigator (AgNav)
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

# Agreement Navigator (AgNav)

AI chatbot built to empower BCGEU union stewards with instant, cited answers from a broad library
of labour law and contract documents.

> See [SPEC.md](SPEC.md) for the full product specification.

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Hugging Face (`Qwen2.5-7B`) / Ollama (`qwen3`) |
| Embeddings | `BAAI/bge-small-en-v1.5` — local CPU, no API key |
| Vector Store | FAISS (in-memory, rebuilt at startup) |
| Web UI | Gradio 6 — `http://localhost:7860` |
| Knowledge Base | Multi-source Markdown in `data/labour_law/` |
| Deployment | Hugging Face Spaces + GitHub Actions |

## Knowledge Base

Agreement Navigator is currently indexed with the following core documents:

- **BCGEU 19th Main Public Service Agreement (Priority 1)**: The core collective agreement. This is the **authoritative source** for union stewards; all other documents provide context.
- **BC Employment Standards Act (Priority 2)**: Statutory minimums for wages, overtime, and notice.
- **BC Labour Relations Code (Priority 3)**: The legal framework for union-management relations and LRB precedents.
- **BC Human Rights Code (Priority 4)**: Protections against discrimination and the duty to accommodate.
- **BCGEU Steward Fundamentals Handbook**: Practical union guidance for grievances and meeting scripts.
- **Standards of Conduct (Public Service Ethics)**: Policy framework for employee behavior and social media use.
- **BC Social Media Guidance for Public Service Employees**: Specific guidelines for personal and professional social media conduct.

### Priority & Weighting Logic
Agreement Navigator is programmed to prioritize the **Collective Agreement** above all else. When a query overlaps multiple sources:
1. The **Agreement** is used for primary enforcement.
2. **Statutes** (ESA, Labour Code, HRC) are cited as secondary legal context.
3. If no contract language exists, the assistant identifies relevant statutory protections.

### Adding or Updating Documents

Agreement Navigator indexes **Markdown files** (`.md`), not PDFs. PDFs are kept only for the "Download Original" links in the UI.

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

🚀 **PROD:** https://huggingface.co/spaces/MinionTech/vexilon

## Quick Start

### Prerequisites

- **Container Engine**: [Podman](https://podman.io/docs/installation) (recommended) or [Docker](https://docs.docker.com/get-docker/)
- **Compose**: `podman compose` (built-in) or [Docker Compose V2](https://docs.docker.com/compose/) plugin
- **Hugging Face Token**: Required only for Production mode (HF Inference API)

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
export HF_TOKEN=<YOUR_HF_TOKEN>
podman compose up prod --build
```

> [!TIP]
> Using `--no-deps` prevents the local Ollama services from starting, allowing for an instant cloud-connected session.

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

**`openai.PermissionDeniedError`** — Your `HF_TOKEN` lacks "Inference" permissions. Update your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

**`openai.APIConnectionError`** — The app cannot reach the LLM provider.
- In **DEV**: Ensure the `ollama` container is running (`podman ps`).
- In **PROD**: Check your internet connection to `router.huggingface.co`.

## Usage

The app is ready immediately on page load — no dropdown, no Load button.

1. Type a question in the input field and press **Enter** or tap **Send**
2. Or click one of the suggested question chips on the welcome screen
3. Responses include a plain-language explanation followed by verbatim quotes with citations
4. **Persona Mode**: Toggle between Lookup, Grieve, and Manage to receive tactical guidance.

> **Note:** Informational purposes only. Consult your BCGEU representative or a legal advisor as appropriate.

## Maintenance & Tools

### PDF to Markdown Conversion
For optimal RAG performance, we recommend converting core PDFs into high-fidelity Markdown. This improves header-aware chunking and removes "noise" like web-to-PDF artifacts.

See [docs/MARKDOWN_CONVERSION.md](docs/MARKDOWN_CONVERSION.md) for full rationale and instructions.

**Run the converter:**
```bash
export HF_TOKEN=<YOUR_HF_TOKEN>
python scripts/pdf_to_md.py path/to/document.pdf
```

## 🚀 Contributing

We welcome contributions from everyone! Whether you are interested in development, infrastructure, or documentation, your help is appreciated.

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for our full guidelines. To get started:
- Browse the [Issue Tracker](https://github.com/MinionTech/vexilon/issues).
- Look for the **`good first issue`** label—these are specifically curated tasks for those new to the project.

## Configuration

All settings are optional — defaults match the product specification.

### Core Settings

Agreement Navigator uses **App-Authority** for model versioning. The primary source of truth is `app.py`.

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL_ID` | `qwen3:1.7b` | *(Code Constant)* Defined in `app.py`. Infrastructure automatically pulls this. |
| `HF_TOKEN` | *(required for PROD)* | Hugging Face access token with Inference permissions |
| `AGNAV_LLM_PROVIDER` | `ollama` | Deployment mode (`ollama` or `huggingface`). Set via Compose profiles. |
| `PORT` | `7860` | Gradio listen port |

### Verification Bot

Agreement Navigator includes a second AI bot that verifies responses against source citations to reduce hallucinations:

| Variable | Default | Description |
|---|---|---|
| `VERIFY_ENABLED` | `true` | Enable verification bot to check claims against citations |
| `VERIFY_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Model for verification |

### Input Sanitization

Input sanitization prevents prompt injection attacks by detecting and blocking malicious inputs:

| Variable | Default | Description |
|---|---|---|
| `MAX_INPUT_LENGTH` | `10000` | Maximum characters per message |
| `LOG_SUSPICIOUS_INPUTS` | `true` | Log flagged inputs for security review |

### Privacy & Data Retention

Agreement Navigator is a "content-blind" application designed for maximum privacy and to support compliance with the British Columbia **Personal Information Protection Act (PIPA)**.

- **Ephemeral Conversations**: Chats are tied only to your current browser session and are permanently deleted upon refresh or closure.
- **No Content Logging**: We **never** log user queries, bot responses, or search reasoning.
- **Minimal Metadata**: Non-sensitive data (token counts, quality scores) is tracked for system health but never reaches persistent storage.

For full technical disclosure and mapping to the 10 PIPA Fair Information Principles, see our [Privacy Policy (PIPA)](docs/PRIVACY.md) and the [Privacy & Data Retention](SPEC.md#privacy--data-retention-updated-215-216) section of the project specification. 

## Hugging Face Spaces Deployment

The Space runs as **`sdk: docker`** in production — the deploy script pushes a stub
`Dockerfile` pointing to the pre-built container image on `ghcr.io/miniontech/agnav`.
The FAISS index is already baked into that image, so the Space starts instantly.

## Project Structure

```
agnav/
├── app.py            # Main application (RAG pipeline + Gradio UI)
├── conftest.py       # pytest root path configuration
├── requirements.txt  # Python dependencies
├── Containerfile     # Production-optimized multi-stage image definition
├── compose.yml       # Podman Compose config (production parity)
├── SPEC.md           # Product specification
├── data/             # Knowledge base source files
└── tests/            # pytest test suite
```
