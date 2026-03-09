---
title: Vexilon — BCGEU Agreement Assistant
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

# Vexilon — BCGEU Agreement Assistant

AI chatbot for BCGEU union stewards to look up the 19th Main Public Service Agreement
(Social, Information & Health). Ask questions in plain language; get verbatim quotes with
article and page citations.

> See [SPEC.md](SPEC.md) for the full product specification.

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Anthropic Claude (`claude-haiku-4-5`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` — local CPU, no API key |
| Vector Store | FAISS (in-memory, rebuilt at startup) |
| PDF Parsing | pypdf — preserves page numbers |
| Web UI | Gradio 5 — `http://localhost:7860` |
| PDF Source | Bundled in `pdf_cache/` |
| Container | Podman + Podman Compose (`compose.yml`) |

## Hosted

🚀 **Live:** https://huggingface.co/spaces/DerekRoberts/vexilon

## Quick Start

### Prerequisites

- [Podman](https://podman.io/docs/installation) + [Podman Compose](https://github.com/containers/podman-compose)
- An [Anthropic API key](https://console.anthropic.com/) (`ANTHROPIC_API_KEY`)

### Run

**Recommended for local development — container with live reload:**

````bash
export ANTHROPIC_API_KEY=sk-ant-...

podman-compose watch
````

`app.py` edits are synced instantly into the running container — no rebuild needed.
`requirements.txt` or `Containerfile` changes trigger a full rebuild automatically.
This matches the production environment while keeping iteration fast.

**Alternatives:**

````bash
# One-shot container build (no live reload):
podman-compose up --build

# No container — useful for quick iteration or CI:
uv run --with-requirements requirements.txt python app.py
````

Open <http://localhost:7860> in your browser.

> ✅ **Startup is fast** — the embedding model is baked into the container image at build time,
> and the FAISS index is pre-built and committed in `pdf_cache/`. No rebuild on first run.

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

> **Note:** Responses cite the agreement text only. This is not legal advice.
> Consult your BCGEU staff representative for complex matters.

## Configuration

All settings are optional — defaults match the product specification.

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Anthropic API key |
| `CLAUDE_MODEL` | `claude-haiku-4-5` | Claude model for responses |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers embedding model |
| `PORT` | `7860` | Gradio listen port |
| `SIMILARITY_TOP_K` | `5` | Chunks retrieved per query |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `100` | Token overlap between chunks |

## Hugging Face Spaces Deployment

The Space uses the Gradio SDK (`sdk: gradio`). HF installs [`requirements.txt`](requirements.txt)
and runs [`app.py`](app.py) directly. [`Containerfile`](Containerfile) is used for local
development only and is ignored by HF Spaces.

### Binary files (PDF, FAISS index)

HF Spaces does not accept binary files via git push. Instead, [`app.py`](app.py) downloads
`pdf_cache/` assets from this public GitHub repo at startup if they are absent:

- `pdf_cache/main_public_service_19th.pdf` — the collective agreement
- `pdf_cache/index.faiss` — pre-built FAISS index
- `pdf_cache/chunks.json` — pre-built chunk metadata

This is a no-op when running locally (files are already present).

### Automated deploy (GitHub Actions)

Every published GitHub release triggers [`.github/workflows/deploy-hf-spaces.yml`](.github/workflows/deploy-hf-spaces.yml),
which strips `pdf_cache/` from the commit and pushes code-only to the HF Space.

**Required GitHub secret:**

| Secret | Value |
|---|---|
| `HF_TOKEN` | Hugging Face write-scoped access token ([settings/tokens](https://huggingface.co/settings/tokens)) |

**Required HF Space secret** (set in [Space settings](https://huggingface.co/spaces/DerekRoberts/vexilon/settings)):

| Secret | Value |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key |

### Manual deploy (one-time setup or re-deploy)

````bash
# Create a fresh orphan snapshot with no history (required — HF scans full git history
# for binary files, so amending is not enough)
git checkout --orphan hf-snapshot
git rm --cached -r pdf_cache/
git commit -m "deploy: manual"

# Push to HF Space (token as password)
git remote add hf "https://DerekRoberts:YOUR_HF_TOKEN@huggingface.co/spaces/DerekRoberts/vexilon"
git push hf hf-snapshot:main --force --no-verify

# Return to your working branch
git switch -
git branch -D hf-snapshot
````

### Running tests

````bash
# Fast suite — no API keys needed, runs in ~5 s
uv run --with-requirements requirements.txt pytest tests/ --ignore=tests/smoke -v

# Smoke test — validates CLAUDE_MODEL against the real Anthropic API
pytest tests/smoke/ -v
````

## Project Structure

````
vexilon/
├── app.py            # Main application (RAG pipeline + Gradio UI)
├── conftest.py       # pytest root path configuration
├── requirements.txt  # Python dependencies (includes pytest)
├── manifest.json     # PWA manifest
├── Containerfile     # Container image definition
├── compose.yml       # Podman Compose — single vexilon service with live-reload watch config
├── SPEC.md           # Product specification
├── tests/            # pytest test suite
│   ├── test_chunking.py    # chunk_text() unit tests
│   ├── test_index.py       # FAISS build/search unit tests
│   ├── test_pdf_loader.py  # load_pdf_chunks() unit tests
│   ├── test_rag_stream.py  # rag_stream() unit tests + model name blocklist
│   └── smoke/
│       └── test_model_valid.py  # live API model validation (requires ANTHROPIC_API_KEY)
└── pdf_cache/        # Bundled PDFs + pre-built FAISS index (committed to repo)
````
