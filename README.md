# Vexilon — BCGEU Collective Agreement RAG Chatbot

AI chatbot for answering questions about BC Government / BCGEU collective agreements,
powered by a local LLM via Ollama and a persistent vector index via Chroma DB.

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Ollama `llama3.2:3b` (local) |
| Embeddings | Ollama `nomic-embed-text` (local) |
| RAG Framework | LlamaIndex v0.10+ |
| Vector Store | Chroma DB — persisted in `chroma_db` volume |
| Web UI | Gradio — `http://localhost:7860` |
| PDF Source | [www2.gov.bc.ca](https://www2.gov.bc.ca) / bundled in `pdf_cache/` |
| Container | Podman + Podman Compose (`compose.yml`) |

## Quick Start

### Prerequisites

- [Podman](https://podman.io/docs/installation) + [Podman Compose](https://github.com/containers/podman-compose)

### Run

```bash
podman-compose up
```

Open <http://localhost:7860> in your browser.

> ⏳ **First run:** `model-puller` will download `llama3.2:3b` (~2 GB) and
> `nomic-embed-text` (~274 MB) into the `ollama_data` volume. This only happens
> once; subsequent starts are fast.

### Troubleshooting

**Port 11434 already in use** — a previous failed start left stale containers. Clean up and retry:
```bash
podman-compose down && podman rm -f ollama model-puller vexilon
podman-compose up
```

## Usage

1. Select an agreement from the dropdown
   (e.g. *"19th Main Public Service Agreement (Social, Information & Health)"*)
2. Click **📥 Load Agreement**
   (~10 s on first load; instant on subsequent loads from the Chroma cache)
3. Ask questions in the chat:

| Example question | What you get |
|------------------|--------------|
| `Article 12 overtime rules?` | Article text + page citation |
| `What is the probationary period?` | Relevant clause + page |
| `health victoria` | Island Health HR: 250-519-3500 |
| `health northern` | Northern Health HR: 250-565-2000 |

> **Note:** All responses are prefixed with *"From document only—not legal advice."*

## BC Contacts

Type any of the following keywords (case-insensitive) to look up HR contacts
without querying the document:

| Keyword | Result |
|---------|--------|
| `health island` / `health victoria` | Island Health HR — 250-519-3500 |
| `health northern` | Northern Health HR — 250-565-2000 |
| `health interior` | Interior Health HR — 1-800-707-8550 |
| `health fraser` | Fraser Health HR — 604-587-4600 |
| `health coastal` | Vancouver Coastal Health HR — 604-875-4111 |
| `health providence` | Providence Health Care HR — 604-682-2344 |
| `health phsa` | PHSA HR — 604-875-2000 |
| `health first nations` | FNHA — 604-693-6500 |
| `bcgeu` | BCGEU Provincial Office — 604-291-9611 |
| `corrections` | BC Corrections Labour Relations — 250-387-5041 |
| `cssea` | CSSEA — 604-942-0505 |

## Manual PDF Upload

If the automatic PDF download fails (e.g. the document URL changes), upload
the PDF directly via the **Upload PDF** file picker in the UI.

## Supported Agreements

| Display Name | Source |
|---|---|
| 19th Main Public Service Agreement (Social, Information & Health) | [www2.gov.bc.ca](https://www2.gov.bc.ca/assets/gov/careers/managers-supervisors/managing-employee-labour-relations/bcgeu_19th_main_agreement_38fa.pdf) / bundled in `pdf_cache/` |

## Hugging Face Spaces Deployment

Ollama is not available on HF Spaces out of the box. Options:

- Set `OLLAMA_BASE_URL` to an external Ollama server endpoint
- Swap the LLM / embedding imports for `llama_index.llms.huggingface`
  and `llama_index.embeddings.huggingface`

```bash
# HF Spaces secret / environment variable
OLLAMA_BASE_URL=https://your-ollama-server.example.com
```

## Configuration

| Variable | Compose default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama server URL (service name in compose) |
| `OLLAMA_MODEL` | `llama3.2:3b` | Ollama LLM model |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `PORT` | `7860` | Gradio listen port |

## Project Structure

```
vexilon/
├── app.py            # Main application (LlamaIndex + Gradio)
├── requirements.txt  # Python dependencies
├── manifest.json     # PWA manifest
├── Containerfile     # Container image definition
├── compose.yml       # Podman Compose — vexilon + ollama + model-puller
├── pdf_cache/        # Bundled PDFs (committed); runtime downloads are git-ignored
└── chroma_db/        # Chroma vector store (named volume in compose; git-ignored)
```
