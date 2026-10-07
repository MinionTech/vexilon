---
description: RAG pipeline architecture, chunking rules, and FAISS governance
---

# RAG Pipeline Specification

This document defines the constraints for the Agreement Navigator (AgNav) RAG pipeline. **Follow these rules to maintain forensic accuracy and system performance.**

## 1. Data Ingestion (The Forensic Pipeline)

### "Paranoid Determinism"
All source documents (PDF/MD) must be processed into Markdown before indexing.
- **Verbatim Requirement**: Do NOT change, summarize, or "fix" contract language.
- **Structural Integrity**: Preserve Articles (`#`), Sections (`##`), and Clauses (`###`) as headers.
- **Substantive Content Hashing**: Substantive content is extracted via `extract_substantive_body()` (stripping provenance headers and trailing whitespace) and baselined via SHA-256 in `app/data/sources.yaml` and `app/data/manifest.json`.

### Declarative Registry & Authority Hierarchy (`sources.yaml`)
Every document in the Knowledge Base is classified under a domain category that drives RAG retrieval weights and drawer navigation:
- **`agreement`**: Tier 1 (`1.2x` boost; Primary Authority drawer section).
- **`conduct`**: Tier 1 (`1.2x` boost; elevated to top of Policy & Jurisprudence drawer section).
- **`statutory`**: Tier 3 (`0.8x` weight; Legislation & Regulations drawer section).
- **`resources`**: Tier 2 (`1.0x` baseline; Policy & Jurisprudence drawer section).
- **`forms`**: Tier 2 (`1.0x` baseline; Forms drawer section).

Ingestion strategies:
- **`manual`**: Locally maintained files (e.g., `BCGEU_20th_Main_Agreement.md`). URL optional. Drift sentinel validates local SHA-256 hash without HTTP calls.
- **`html_selector` | `bclaws` | `pdf`**: Upstream network sources. URL mandatory.

### Ingestion Lifecycle Runbook
1. Place source Markdown into `app/data/<tier_folder>/`.
2. Register the entry in `app/data/sources.yaml` with `path`, `category`, and `type`.
3. Compute baseline hash & sync manifest:
   ```bash
   uv run python scripts/sync_sources.py --sync-all
   ```
4. Regenerate Knowledge Base drawer navigation:
   ```bash
   uv run python scripts/generate_knowledge_base.py
   ```
5. Run test verification suite:
   ```bash
   pytest tests/test_sync_sources.py tests/test_index.py tests/test_generate_knowledge_base.py tests/deploy_integrity/
   ```

### Chunking Logic
- **Default Size**: `CHUNK_SIZE = 450` tokens.
- **Default Overlap**: `CHUNK_OVERLAP = 100` tokens.
- **Prefixing**: Every chunk MUST be prefixed with its source and header context:
  `[Source: Document Name, Header: ARTICLE 14] ... verbatim text ...`

## 2. Vector Store (FAISS) Governance

### Local Execution Only
- **CPU-Only**: FAISS must run on the CPU (`faiss-cpu`) to avoid CUDA dependency bloat.
- **In-Memory**: The index is ephemeral and exists only in RAM at runtime.

### Persistence & Security
- **No Pickle**: Never use `.pkl` for chunk storage (RCE risk). Use `chunks.json`.
- **Manifest Validation**: The index must only be loaded if the `manifest.json` hashes match the current source files.

## 3. Embedding Model

### Model ID
- **Standard**: `BAAI/bge-small-en-v1.5`
- **Constraint**: Must use "Fast" tokenizers for reliable character-offset mapping.

### Offline Enforcement
- **Transformers/HF Offline**: `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1` must be set during RAG operations to prevent accidental model downloads in production.

## 4. Query Condensing

Follow-up questions must always pass through the `condense_query` LLM step to reconstruct a standalone search query from conversation history. **Never search FAISS using raw follow-up messages like "What about part-time?".**
