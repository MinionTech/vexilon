# Vexilon — BCGEU Agreement Assistant: Product Specification

> **Version:** 0.1.0
> **Status:** Draft — approved by product owner
> **Last updated:** 2026-03-08

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Users](#2-users)
3. [Goals and Non-Goals](#3-goals-and-non-goals)
4. [User Stories](#4-user-stories)
5. [Response Format](#5-response-format)
6. [UI/UX Requirements](#6-uiux-requirements)
7. [Tech Stack](#7-tech-stack)
8. [Architecture](#8-architecture)
9. [Deployment](#9-deployment)
10. [Success Criteria](#10-success-criteria)
11. [Out of Scope (MVP)](#11-out-of-scope-mvp)
12. [Future Roadmap](#12-future-roadmap)
13. [Open Questions](#13-open-questions)

---

## 1. Problem Statement

BCGEU union stewards are expected to consult the collective agreement constantly — in grievance meetings, during discipline discussions, when advising members, and when learning their own rights. The agreement is a dense legal document. Looking things up manually is slow, error-prone, and inaccessible to stewards who are not trained in legal or labour relations language.

95%+ of stewards come from unrelated fields (administration, records, archaeology, etc.) and have no background in reading collective agreements. Under pressure in a meeting, they need an answer in seconds, not minutes.

The previous attempt at solving this problem produced a tool that:

- Took 2+ minutes to respond, and often returned nothing
- Used a local LLM with no GPU — structurally incapable of performing well
- Invented features nobody asked for (hardcoded phone lookup tables)
- Provided no discovery or requirements gathering before development
- Built something that worked occasionally for a developer, not reliably for a steward

This specification exists to prevent that from happening again.

---

## 2. Users

### Primary User: Union Steward

- **Background:** Typically non-technical. Comes from fields like administration, records management, public health, archeology, corrections, or social services.
- **Context of use:**
  - In a grievance or discipline meeting (mobile, urgent, under pressure)
  - Preparing a grievance or response (desktop, research-mode, more time)
  - Learning the agreement as a new steward (desktop, self-directed study)
- **Technical comfort:** Low to none. If it requires configuration, explanation, or more than one click to start, it has failed.
- **Device:** Both mobile (phone in meetings) and desktop (prep and research). Both are first-class.

### Secondary User: Developer / Maintainer

- The product owner is a DevOps specialist and new union steward.
- At least one full-stack developer steward may contribute.
- Developer experience matters but is secondary to end-user experience.

### Target Agreement

- **19th Main Public Service Agreement (Social, Information & Health)**
- Covers BCGEU members in the Social, Information and Health bargaining unit of the BC Public Service.
- Future iterations will support additional BCGEU collective agreements.

---

## 3. Goals and Non-Goals

### Goals (MVP)

- Answer questions about the collective agreement in plain language
- Always cite the exact verbatim clause(s) with article number and page reference
- Support multi-turn conversation so stewards can ask follow-up questions
- Work on mobile and desktop without configuration
- Respond in under 10 seconds on a standard internet connection
- Be honest when the agreement does not address a question
- Be deployable to Hugging Face Spaces with a public URL

### Non-Goals (MVP)

- Provide legal advice or interpret how a grievance will be decided
- Support multiple collective agreements in a single session
- Remember conversations between browser sessions
- Authenticate or restrict access to specific users
- Replace a trained union representative or labour relations officer

---

## 4. User Stories

### US-01: Ask a Basic Question

> **As a** union steward,
> **I want to** type a plain-language question about the collective agreement,
> **so that** I can quickly find out what the agreement says without reading the whole document.

**Acceptance criteria:**
- The app is ready to accept input immediately on page load — no loading step, no dropdown to select, no button to click before the first question
- The steward types a question and presses Enter or taps Send
- A response appears within 10 seconds
- The response includes a plain-language explanation followed by the verbatim clause(s) and citation(s)

---

### US-02: Get Verbatim Quotes with Citations

> **As a** steward in a grievance meeting,
> **I want to** see the exact words from the agreement with the article number and page,
> **so that** I can read the clause aloud or point to it without being challenged on paraphrasing.

**Acceptance criteria:**
- Every response that draws from the agreement includes at least one verbatim quote
- Each quote is visually distinct (e.g. blockquote styling)
- Each quote is followed by its citation: Article number and page number
- If no relevant clause exists, the response says so clearly — no fabrication

---

### US-03: Ask Follow-Up Questions

> **As a** steward doing research,
> **I want to** ask follow-up questions that build on the previous answer,
> **so that** I can explore related clauses without starting over each time.

**Acceptance criteria:**
- The chat interface maintains conversation history within the current session
- A follow-up question like "Does that apply to part-time employees?" correctly uses prior context
- Conversation history is cleared when the page is refreshed (session persistence is out of scope for MVP)

---

### US-04: Onboarding for New Stewards

> **As a** new steward opening the tool for the first time,
> **I want to** understand what this tool does and see example questions,
> **so that** I know how to use it without reading documentation.

**Acceptance criteria:**
- The empty chat state (before any message is sent) shows a brief welcome message explaining the tool's purpose and limitations
- 3–5 example questions are displayed as clickable chips or buttons (e.g. *"What are my overtime rights?"*, *"What is the probationary period?"*, *"Can my employer change my schedule without notice?"*)
- Clicking an example question populates the input and submits it
- The welcome message and example questions disappear once the first message is sent

---

### US-05: "Not Found" Handling

> **As a** steward asking about something the agreement doesn't cover,
> **I want to** get an honest answer that the agreement doesn't address it,
> **so that** I don't act on fabricated information.

**Acceptance criteria:**
- If no relevant clause is found (similarity score below threshold), the response clearly states the agreement does not appear to address the question
- The response does not invent clauses or speculate
- The response may suggest related topics that are covered, if applicable

---

### US-06: Legal Disclaimer

> **As a** steward using this tool,
> **I want to** be reminded that this is not legal advice,
> **so that** I understand the tool's limitations before acting.

**Acceptance criteria:**
- A persistent, non-dismissible disclaimer is visible at all times: *"This tool references the collective agreement text only. It is not legal advice. Consult your BCGEU staff representative for complex matters."*
- The disclaimer does not obscure the chat interface on mobile

---

### US-07: Mobile Use in a Meeting

> **As a** steward on my phone in a meeting,
> **I want** the tool to be usable on a small screen without zooming or horizontal scrolling,
> **so that** I can look up information discreetly and quickly.

**Acceptance criteria:**
- The interface renders correctly in portrait orientation on a standard smartphone
- Text is legible without zooming
- The input field and send button are reachable with one thumb
- Verbatim quotes do not overflow the screen horizontally
- Response time meets the 10-second target on a mobile (LTE) connection

---

## 5. Response Format

Each response must follow this structure:

```
[Plain-language explanation of what the agreement says]

> "[Verbatim quote from the agreement]"
> — Article [X], [Title], p. [N]

> "[Second quote if applicable]"
> — Article [X.Y], [Title], p. [N]

[Optional: "This may also be relevant:" + follow-up suggestion]
```

**Rules enforced via system prompt:**

1. Never state something the agreement doesn't say
2. Always include at least one verbatim quote when a relevant clause exists
3. If no relevant clause is found, say so — do not speculate
4. Do not offer legal opinions, predict outcomes, or advise on strategy
5. Plain language explanation comes before the quote, not after
6. Citations must include both article number and page number

---

## 6. UI/UX Requirements

### Layout

- Single-page chat interface
- Input field at the bottom (mobile-standard positioning)
- Response area fills remaining vertical space, scrollable
- Persistent disclaimer footer visible at all times

### Empty State

- Welcome message: brief (2–3 sentences max), explains what Vexilon does and its limitation
- 3–5 suggested question chips/buttons, clickable to auto-submit
- Disappears after first message is sent

### Response Rendering

- Plain-language text renders as normal prose
- Verbatim quotes render as styled blockquotes (visually distinct — border, background tint, or similar)
- Citations render in a muted/smaller style beneath each quote
- Markdown rendering enabled in responses

### Branding

- Colour palette: BCGEU brand colours — Primary Blue `#005691`, Green `#008542`, Dark Navy `#003366`, white, light grey
- Clear, professional appearance — not a developer demo aesthetic
- Application title: **Vexilon — BCGEU Agreement Assistant**

### Accessibility

- Sufficient colour contrast for WCAG AA compliance
- Works with browser zoom up to 200%
- Input field has a descriptive placeholder (e.g. *"Ask about the collective agreement..."*)

### No Configuration Required

- The agreement is pre-loaded at app startup
- There is no "Select Agreement" dropdown in the MVP
- There is no "Load" button
- The app is ready when the page loads

---

## 7. Tech Stack

| Component | Choice | Rationale |
|---|---|---|
| **LLM** | Anthropic Claude (`claude-3-5-haiku-20241022` — MVP only) | Best-in-class instruction following; reliable citation behaviour; pay-per-use; Haiku sufficient for citation-grounded retrieval |
| **Embeddings** | `text-embedding-3-small` (OpenAI) | Cheap ($0.02/million tokens), accurate, stable API |
| **Vector Store** | FAISS (in-memory) | No server process, no persistence overhead, trivially replaceable with ChromaDB when multi-agreement support is added |
| **PDF Parsing** | `pypdf` | Lightweight, already available; preserves page numbers |
| **RAG Framework** | Direct implementation (no LlamaIndex) | LlamaIndex added complexity without value for this use case; direct control over chunking, retrieval, and prompting is preferable |
| **Web UI** | Gradio 5.x | Same framework as current codebase; HF Spaces native; responsive CSS possible |
| **Hosting** | Hugging Face Spaces | Free tier; Gradio-native; public URL with no infrastructure to manage |
| **Local Dev** | Podman + `compose.yml` | Existing setup retained |
| **Language** | Python 3.11+ | Existing codebase language |

### Removed from Current Stack

| Removed | Reason |
|---|---|
| Ollama | Replaced by Anthropic API |
| `llama3.2:3b` / `llama3.1:8b` | Replaced by Claude |
| `nomic-embed-text` (Ollama) | Replaced by OpenAI text-embedding-3-small |
| LlamaIndex | Replaced by direct RAG implementation |
| ChromaDB | Replaced by FAISS for MVP |
| Hardcoded phone number lookup | Not a requested feature; removed entirely |

---

## 8. Architecture

### RAG Pipeline

```
App startup
  └── Load PDF from pdf_cache/
  └── Parse pages with pypdf (preserve page numbers)
  └── Chunk text (512 tokens, 100 token overlap)
  └── Embed all chunks with text-embedding-3-small
  └── Build FAISS index in memory
  └── Ready

User sends message
  └── Embed user query with text-embedding-3-small
  └── FAISS similarity search → top-5 chunks (with page numbers)
  └── Build prompt:
        system: [citation-enforcement rules + agreement context]
        user: [conversation history + new query]
        context: [retrieved chunks with page numbers]
  └── Send to Claude API (claude-3-5-haiku-20241022)
  └── Stream response to Gradio chat interface
  └── Append to conversation history
```

### Chunking Strategy

- Chunk size: 512 tokens (fits within embedding model context; aligns with typical article length)
- Overlap: 100 tokens (~20%) to avoid splitting mid-clause
- Metadata per chunk: source filename, page number, chunk index
- Page number preserved and passed to the LLM as part of chunk metadata

### System Prompt (outline)

The system prompt will enforce:

1. Role: "You are Vexilon, an assistant for BCGEU union stewards..."
2. Grounding: "You may only answer using the provided agreement excerpts. Do not draw on outside knowledge."
3. Citation requirement: "Every claim must be supported by a verbatim quote from the provided excerpts, formatted as a blockquote with article and page citation."
4. Not-found handling: "If the excerpts do not address the question, say so clearly."
5. No legal advice: "Do not predict outcomes, advise on strategy, or offer legal opinions."
6. Tone: "Plain language. Explain clause-by-clause. New stewards are your audience."

### Cost Estimate

| Component | Rate | Estimated Monthly (moderate use) |
|---|---|---|
| `claude-3-5-haiku-20241022` | $0.80/M input tokens, $4.00/M output | ~$5–15 CAD |
| text-embedding-3-small | $0.02/M tokens (queries only; index built once) | <$1 CAD |
| **Total** | | **~$6–16 CAD/month** |

Well within the $100 CAD/month budget. The index is built once at startup; embedding costs are query-only after that.

---

## 9. Deployment

### Local Development

```bash
# Set your API keys
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...   # for embeddings only

# Run with Podman Compose
podman-compose up
```

Open `http://localhost:7860`.

### Hugging Face Spaces

- App type: Gradio
- Secrets: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`
- The `pdf_cache/` directory is committed to the repo and available at runtime
- No persistent volume required (FAISS index rebuilt on each cold start — acceptable for this scale)

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Anthropic API key |
| `OPENAI_API_KEY` | *(required)* | OpenAI API key (embeddings only) |
| `CLAUDE_MODEL` | `claude-3-5-haiku-20241022` | Claude model for responses |
| `EMBED_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `PORT` | `7860` | Gradio listen port |
| `SIMILARITY_TOP_K` | `5` | Chunks retrieved per query |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `100` | Token overlap between chunks |

---

## 10. Success Criteria

The MVP is complete and successful when:

1. **Performance:** Average response time < 10 seconds on LTE mobile
2. **Accuracy:** Responses cite verbatim quotes with article and page for all relevant clauses
3. **Honesty:** Responses correctly say "not found" when no relevant clause exists (verified by testing 5 questions that are not in the agreement)
4. **Mobile:** Works correctly in portrait on iPhone Safari and Android Chrome without horizontal scrolling
5. **Onboarding:** A non-technical steward (not the developer) can use the tool correctly within 10 seconds of first opening it, without reading any instructions
6. **Stability:** No crashes or blank responses during a 10-question test session

---

## 11. Out of Scope (MVP)

These are explicitly deferred and must NOT be built until the MVP criteria above are met:

- Multiple collective agreements in a single session
- Conversation history persistence across browser sessions
- User authentication or login
- Bookmarking or saving specific clauses
- Comparing clauses across agreements
- Admin interface for managing agreements or users
- Analytics or usage tracking
- Rate limiting (revisit if costs exceed budget)
- Native mobile app

---

## 12. Future Roadmap

In rough priority order after MVP:

1. **Multi-agreement support** — steward selects which agreement applies to their workplace
2. **ChromaDB persistence** — replace FAISS with ChromaDB when multiple agreements are loaded
3. **Auth** — restrict access to verified BCGEU stewards (likely BCGEU member number or email domain)
4. **Bookmarking** — save and share relevant clauses
5. **Grievance helper** — structured workflow for building a grievance argument from relevant clauses
6. **Suggested related clauses** — "You may also want to look at Article X" surfaced by the retrieval layer

---

## 13. Open Questions

These have been discussed and resolved:

| Question | Decision |
|---|---|
| Quote verbatim or paraphrase? | Verbatim, with plain-language explanation first |
| Which LLM provider? | Anthropic Claude |
| Which vector store? | FAISS for MVP; ChromaDB when multi-agreement support added |
| Session persistence? | Fresh session per page load (MVP) |
| Auth required? | No for MVP; must be easy to add later |
| Mobile required? | Yes — first-class, not an afterthought |
| Phone number lookup? | Removed entirely |
| Hosting? | Hugging Face Spaces |
