---
description: RAG pipeline architecture, chunking rules, and FAISS governance
---

# RAG Pipeline Specification

This document defines the constraints for the Agreement Navigator (AgNav) RAG pipeline. **Follow these rules to maintain forensic accuracy and system performance.**

## 1. Data Ingestion (The Forensic Pipeline)

### "Paranoid Determinism"
All source documents (PDF/MD) must be processed into Markdown before indexing.
- **Verbatim Requirement**: Do NOT change, summarize, or "fix" contract language.
- **Structural Integrity**: preserve Articles (#), Sections (##), and Clauses (###) as headers.
- **Integrity Reports**: Every `.md` file must have a corresponding `.integrity.md` diff report.

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
