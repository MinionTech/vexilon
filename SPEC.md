# Agreement Navigator (AgNav): Product Specification

> **Version:** 0.1.0
> **Status:** Draft — approved by product owner


---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Users](#2-users)
3. [Goals and Non-Goals](#3-goals-and-non-goals)
4. [User Stories](#4-user-stories)
5. [Security](#5-security)
6. [Response Format](#6-response-format)
7. [UI/UX Requirements](#7-uiux-requirements)
8. [Tech Stack](#8-tech-stack)
9. [Architecture](#9-architecture)
10. [Deployment](#10-deployment)
11. [Success Criteria](#11-success-criteria)
12. [Out of Scope (MVP)](#12-out-of-scope-mvp)
13. [Future Roadmap](#13-future-roadmap)
14. [Open Questions](#14-open-questions)

---

## 1. Problem Statement

BCGEU union stewards are expected to consult the collective agreement constantly — in grievance meetings, during discipline discussions, when advising members, and when learning their own rights. The agreement is a dense legal document. Looking things up manually is slow, error-prone, and inaccessible to stewards who are not trained in legal or labour relations language.

95%+ of stewards come from unrelated fields (administration, records, archaeology, etc.) and have no background in reading collective agreements. Under pressure in a meeting, they need an answer in seconds, not minutes.

This specification defines **Agreement Navigator (AgNav)**, an AI tool built to solve this problem reliably for stewards.

---

## 2. Users

### Primary User: Union Steward

- **Background:** Typically non-technical.
- **Context of use:**
  - In a grievance or discipline meeting (mobile, urgent, under pressure)
  - Preparing a grievance or response (desktop, research-mode, more time)
- **Technical comfort:** Low to none.
- **Device:** Both mobile and desktop.

### Secondary User: Developer / Maintainer

- The product owner is a DevOps specialist and new union steward.

---

## 3. Goals and Non-Goals

### Goals (MVP)

- Answer questions about the collective agreement and related labour laws in plain language
- Always cite the exact verbatim clause(s) with article number and page reference
- **Prioritize the Collective Agreement** (Priority 1) as the authoritative source
- Support multi-turn conversation
- Respond in under 10 seconds
- Provide basic authentication (`AGNAV_USERNAME`, `AGNAV_PASSWORD`)

### Non-Goals (MVP)

- Provide legal advice
- Remember conversations between browser sessions

---

## 4. User Stories

### US-01: Ask a Basic Question
(Steward types a question, gets an answer with citations within 10 seconds).

### US-02: Persona Modes
(Steward can toggle between **Lookup**, **Grieve**, and **Manage** modes to get tailored tactical advice).

---

## 5. Security

### Authentication
Controlled via environment variables (`AGNAV_USERNAME`, `AGNAV_PASSWORD`).

### Privacy & Data Retention
AgNav is a "content-blind" application. Conversations are ephemeral and permanently deleted upon refresh. No user queries are logged to disk.

---

## 8. Tech Stack

| Component | Choice | Rationale |
|---|---|---|
| **LLM** | Hugging Face Router / Ollama (Qwen2.5) | Open-standard client, high-fidelity citations |
| **Embeddings** | `BAAI/bge-small-en-v1.5` | Local CPU, no API key required |
| **Vector Store** | FAISS | In-memory, pre-computed for fast cold starts |
| **Web UI** | Gradio 6.x | Mobile-responsive, native HF Spaces support |

---

## 9. Architecture

### RAG Pipeline

```
User sends message
  └── Condense Query (LLM)
  └── Multi-perspective Query Generation (optional)
  └── FAISS similarity search → top-5 chunks
  └── Persona-specific prompt construction
  └── Stream response via Open-Standard client
```

---

## 10. Deployment

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `AGNAV_USERNAME` | `admin` | Username for basic authentication |
| `AGNAV_PASSWORD` | *(optional)* | Password for basic authentication |
| `HF_TOKEN` | *(required for PROD)* | Hugging Face access token |
| `AGNAV_LLM_PROVIDER` | `ollama` | `ollama` or `huggingface` |
| `AGNAV_DEFAULT_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Primary LLM model |
