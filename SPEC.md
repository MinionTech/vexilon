# BCGEU Steward Assistant: Product Specification

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

- **BCGEU 19th Main Public Service Agreement (Social, Information & Health)**: The core collective agreement. **(Primary authoritative source)**.
- **BC Employment Standards Act [RSBC 1996]**: Statutory minimums.
- **BC Labour Relations Code [RSBC 1996]**: Union-management statute.
- **BC Human Rights Code [RSBC 1996]**: Discrimination/accommodation framework.
- **BCGEU Steward Fundamentals Handbook**: Union guidance.
- **Standards of Conduct**: BC Public Service ethics/behavior policy.

---

## 3. Goals and Non-Goals

### Goals (MVP)

- Answer questions about the collective agreement and related labour laws in plain language
- Always cite the exact verbatim clause(s) with article number and page reference
- **Prioritize the Collective Agreement** (Priority 1) as the authoritative source for stewards
- Support multi-turn conversation so stewards can ask follow-up questions
- Work on mobile and desktop without configuration
- Respond in under 10 seconds on a standard internet connection
- Be honest when the provided documents do not address a question
- Be deployable to Hugging Face Spaces with a public URL
- Provide basic authentication to prevent unauthorized public access

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
- A persistent, non-dismissible disclaimer is visible at all times: *"Informational purposes only. Consult your BCGEU representative or a legal advisor as appropriate."*
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

### US-08: Direct Advice Mode for Experienced Stewards

> **As an** experienced union steward in a high-pressure meeting,
> **I want to** toggle a "Direct Advice" mode,
> **so that** I get numbered action steps, meeting scripts, and aggressive nexus-test arguments instead of general educational summaries.

**Acceptance criteria:**
- A checkbox/toggle labeled "Direct Advice Mode" is available in the UI.
- When enabled, the assistant's persona shifts to a "Senior BCGEU Staff Rep" (10+ years experience).
- Responses follow a strict operational format:
  1. **Immediate Actions** (numbered steps)
  2. **Meeting Scripts** (verbatim language for management)
  3. **Contract Authority** (verbatim quotes)
  4. **Strategy/Nexus Notes** (defensive analysis)
- A clear visual indicator (banner) appears when this mode is active to prevent accidental use.

---

## 5. Security

### Authentication

To prevent unintended public access while running on Hugging Face Spaces or other public platforms, Vexilon implements optional basic authentication.

- **Mechanism:** Gradio's built-in basic authentication (`auth` parameter).
- **Configuration:** Controlled via environment variables (`VEXILON_USERNAME`, `VEXILON_PASSWORD`).
- **Behavior:**
  - If `VEXILON_PASSWORD` is set, users must log in to access the interface.
  - If unset, the app remains public (intended for local development).
  - Credentials are checked on every session start.

### Privacy & Data Retention (Updated: #215, #216)

Vexilon is a "content-blind" application designed to protect the privacy of BCGEU stewards and their members. It is specifically built to comply with the British Columbia **Personal Information Protection Act (PIPA)** and its **10 Fair Information Principles**.

- **PIPA Compliance**: For detailed mapping of Vexilon features to PIPA principles, see [docs/PRIVACY.md](docs/PRIVACY.md).
- **NO Conversation History**: Conversations are ephemeral. Once a browser tab is refreshed or closed, all history is permanently deleted. No conversation data is persisted across sessions.
- **NO Content Logging**: User queries, condensed search queries, and bot responses are **never** written to disk or any persistent database. 
- **Minimal Metadata Tracking**: For the purpose of monitoring system health and API costs, Vexilon only logs the following "lite" metadata:
    - **Timestamp**: When the interaction occurred.
    - **Score**: The 1-10 "Quality Score" (only recorded when **Senior Rep Review** is enabled).
    - **Steward ID**: The authenticated username (if `VEXILON_PASSWORD` is set).
    - **Token Counts**: Input, output, and cache effectiveness tokens (for billing/performance).
- **Transparency**: These metrics are stored in an ephemeral CSV file (`./.pdf_cache/review_log.csv`) and are **completely wiped** every time the application redeploys or restarts.

### Input Sanitization

To prevent prompt injection attacks, Vexilon implements input sanitization:

- **Detection:** Regex-based pattern matching for 16+ known prompt injection patterns
- **Patterns include:** `ignore all instructions`, `forget your rules`, `jailbreak`, `developer mode`, `sudo mode`, roleplay attempts, and other common injection techniques
- **Length limits:** Maximum input length (default 10000 characters) to prevent buffer overflow attacks
- **Logging:** Flagged inputs are logged for security monitoring (configurable)
- **User feedback:** When input is flagged, users receive: "Your input was flagged for security review. Please try a different question."

---

## 6. Response Format

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
6. Citations must include both the document name, article/section number, and page number
7. **Prioritization**: If a question involves both the contract (Agreement) and a statute (Code/ESA), the response must lead with the contract language.

---

## 7. UI/UX Requirements

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
- Application title: **BCGEU Steward Assistant**

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

## 8. Tech Stack

| Component | Choice | Rationale |
|---|---|---|
| **LLM** | Anthropic Claude (`claude-haiku-4-5-20251001`) | Best-in-class instruction following; reliable citation behaviour; pay-per-use; Haiku sufficient for citation-grounded retrieval |
| **Embeddings** | `BAAI/bge-small-en-v1.5` via `sentence-transformers` (local CPU) | No API key; no per-query cost; ~90 MB model; runs on CPU; index pre-computed and committed to repo for fast cold starts |
| **Vector Store** | FAISS (in-memory, pre-computed index on disk) | No server process; index loaded from disk at startup (<1s); pre-computed once per agreement update |
| **Markdown-First RAG** | Native Markdown | High-precision extraction via `pdf_to_md.py`; structured MD ensures the highest grounding accuracy and eliminates runtime PDF parsing overhead. |
| **Forensic Pipeline** | `pdf_to_md.py` | Markdown-First architecture: PDFs are pre-converted to structured Markdown with dual-pass AI verification and integrity auditing. |
| **Web UI** | Gradio 6.x | HF Spaces native; supports asynchronous handlers for high concurrency |
| **Hosting** | Hugging Face Spaces | Free tier; Gradio-native; public URL with no infrastructure to manage |
| **Local Dev** | Podman + `compose.yml` | Existing setup retained |
| **Language** | Python 3.11+ | Existing codebase language |

### Removed from Current Stack

| Removed | Reason |
|---|---|
| Ollama | Replaced by Anthropic API |
| `llama3.2:3b` / `llama3.1:8b` | Replaced by Claude |
| `nomic-embed-text` (Ollama) | Replaced by `sentence-transformers` local model |
| OpenAI `text-embedding-3-small` | Replaced by local `BAAI/bge-small-en-v1.5` — eliminates second API dependency |
| LlamaIndex | Replaced by direct RAG implementation |
| ChromaDB | Replaced by FAISS for MVP |
| Hardcoded phone number lookup | Not a requested feature; removed entirely |

---

## 9. Architecture

### RAG Pipeline

```
App startup
  └── Scan data/labour_law/ for all Markdown (.md) "shadow" files
  └── (CI Gate: Ensures every .pdf has a validated .md partner)
  └── Parse Markdown with source metadata and page-tags
  └── Chunk text (256 tokens, 50 token overlap)
  └── Embed all chunks with BAAI/bge-small-en-v1.5
  └── Build FAISS index in memory
  └── Ready

User sends message
  └── Condense Query (Claude)
        └── [conversation history + new message] → standalone search query
  └── Embed condensed query with BAAI/bge-small-en-v1.5
  └── FAISS similarity search → top-5 chunks (with page numbers)
  └── Build final prompt:
        system: [citation-rules + agreement context + continuity rule]
        user: [conversation history + new query]
        context: [retrieved chunks with page numbers]
  └── Send to Claude API (claude-haiku-4-5-20251001) via AsyncAnthropic
  └── Stream response to Gradio chat interface (asynchronous generator)
  └── Append to conversation history
```

### Concurrency and Asynchrony

To support multiple simultaneous users without thread pool exhaustion, Vexilon uses:
- **`AsyncAnthropic`**: The asynchronous variant of the Anthropic client.
- **`async def` handlers**: Gradio handlers are implemented as asynchronous generators.
- **Deferred Imports**: Heavy libraries (`torch`, `sentence_transformers`, `faiss`, `gradio`) are imported lazily within functions to ensure fast startup and responsive CLI/test environments.

### Chunking Strategy

- Chunk size: 256 tokens (matches embedding model's max sequence length; prevents silent truncation)
- Overlap: 50 tokens (~20%) to avoid splitting mid-clause
- Metadata per chunk: source filename, page number, chunk index
- Page number preserved and passed to the LLM as part of chunk metadata

### System Prompt (outline)

The system prompt will enforce:

1. Role: "You are the BCGEU Steward Assistant, an assistant for BCGEU union stewards..."
2. Grounding: "You may only answer using the provided agreement excerpts. Do not draw on outside knowledge."
3. Citation requirement: "Every claim must be supported by a verbatim quote from the provided excerpts, formatted as a blockquote with article and page citation."
4. Not-found handling: "If the excerpts do not address the question, say so clearly."
5. No legal advice: "Do not predict outcomes, advise on strategy, or offer legal opinions."
6. Tone: "Plain language. Explain clause-by-clause. New stewards are your audience."

### Cost Estimate

| Component | Rate | Estimated Monthly (moderate use) |
|---|---|---|
| `claude-haiku-4-5-20251001` | $0.80/M input tokens, $4.00/M output | ~$6–18 CAD |
| `BAAI/bge-small-en-v1.5` embeddings | $0 — runs locally on CPU | $0 |
| **Total** | | **~$6–18 CAD/month** |

Note: The "Query Condenser" adds one extra fast LLM call per multi-turn message, increasing costs by ~10% compared to single-turn RAG.

### 9.5 Context Awareness (Query Condensing)

To ensure multi-turn conversations are reliable, the system uses the **Query Condensing** pattern:

1.  **Reasoning**: Vague follow-up questions (e.g., "What about for part-time?") yield poor similarity results if searched directly.
2.  **Implementation**: A fast LLM pass (Claude) reconstructs the user's intent into a standalone query using the conversation history.
3.  **Benefit**: Decouples the "conversational brain" from the "retrieval search," ensuring the FAISS index always receives high-fidelity queries even for vague follow-ups.

### 9.6 Verification Bot

To reduce hallucinations, Vexilon includes an optional verification bot that reviews responses against source citations:

1.  **Trigger**: Runs after the main response completes streaming
2.  **Process**: A second LLM call checks if quoted text actually supports the claims made
3.  **Output**: 
   - If claims are verified → clean response (no note added)
   - If claims are disputed → "Verification:" note appended with issues
4.  **Configuration**:
   | Variable | Default | Description |
   |---|---|---|
   | `VERIFY_ENABLED` | `true` | Enable verification bot |
   | `VERIFY_MODEL` | `claude-haiku-4-5-20251001` | Model for verification (can use cheaper model) |

**Note:** The verification bot provides limited additional value since it uses the same context as the main bot. It may catch obvious issues (wrong page numbers, misquoted text) but cannot detect when relevant text was simply not retrieved. Future improvements may include multi-perspective retrieval for complex topics.

### 9.8 Forensic Markdown Pipeline

To ensure the highest possible grounding and citation accuracy, Vexilon uses a "Markdown-First" ingestion strategy. This decouples the messy PDF parsing from the RAG retrieval logic.

1.  **Atomic Engine (`pdf_to_md.py`)**: Uses **PyMuPDF** for geometric word reconstruction, followed by a **Claude 4.6 (Sonnet)** pass to restructure the text into clean, hierarchical Markdown.
2.  **Dual-Pass Verification**: A second model (Haiku) performs a parallel conversion. The script flags substantive word discrepancies (hallucinations) between the two models and the raw source.
3.  **Integrity Audit**: Generates a sidecar `.integrity.md` report showing exactly which lines were flagged, allowing for a rapid human audit of 200+ page documents.
4.  **Shadow File Architecture**: Side-by-side storage (`filename.pdf` for users, `filename.md` for AI) ensures the official source is always available for human verification while the RAG index uses high-fidelity text.

---

---

## 10. Deployment

### Local Development

```bash
# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# Run with Podman Compose
# The build stage automatically bakes the PDF index for zero-downtime startup
podman-compose up --build
```

Open `http://localhost:7860`.

### Hugging Face Spaces

- App type: Gradio
- Secrets: `ANTHROPIC_API_KEY`
- The `.pdf_cache/` directory is committed to the repo and available at runtime
- No persistent volume required (FAISS index rebuilt on each cold start — acceptable for this scale)

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VEXILON_USERNAME` | `admin` | Username for basic authentication |
| `VEXILON_PASSWORD` | *(optional)* | Password for basic authentication. If unset, auth is disabled. |
| `ANTHROPIC_API_KEY` | *(required)* | Anthropic API key |
| `CLAUDE_MODEL` | `claude-haiku-4-5-20251001` | Claude model for responses |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Local sentence-transformers embedding model |
| `PORT` | `7860` | Gradio listen port |
| `SIMILARITY_TOP_K` | `40` | Chunks retrieved per query |
| `CHUNK_SIZE` | `450` | Tokens per chunk |
| `CHUNK_OVERLAP` | `100` | Token overlap between chunks (~22%) |
| `CONDENSE_QUERY_HISTORY_TURNS` | `3` | Number of previous turns used for context condensation |
| `CONDENSE_QUERY_CONTENT_MAX_LEN` | `200` | Max character length of historical messages in condensation prompt |
| `VERIFY_ENABLED` | `true` | Enable verification bot |
| `VERIFY_MODEL` | `claude-haiku-4-5-20251001` | Claude model for verification |
| `RATE_LIMIT_PER_MINUTE` | `10` | Max requests per minute per client IP |
| `RATE_LIMIT_PER_HOUR` | `100` | Max requests per hour per client IP |
| `MAX_INPUT_LENGTH` | `10000` | Max characters per user message |
| `LOG_SUSPICIOUS_INPUTS` | `true` | Log flagged inputs for security review |

---

## 11. Success Criteria

The MVP is complete and successful when:

1. **Performance:** Average response time < 10 seconds on LTE mobile
2. **Accuracy:** Responses cite verbatim quotes with article and page for all relevant clauses
3. **Honesty:** Responses correctly say "not found" when no relevant clause exists (verified by testing 5 questions that are not in the agreement)
4. **Mobile:** Works correctly in portrait on iPhone Safari and Android Chrome without horizontal scrolling
5. **Onboarding:** A non-technical steward (not the developer) can use the tool correctly within 10 seconds of first opening it, without reading any instructions
6. **Stability:** No crashes or blank responses during a 10-question test session

---

## 12. Out of Scope (MVP)

These are explicitly deferred and must NOT be built until the MVP criteria above are met:

- Multiple collective agreements in a single session
- Conversation history persistence across browser sessions
- Bookmarking or saving specific clauses
- Comparing clauses across agreements
- Admin interface for managing agreements or users
- Analytics or usage tracking
- ~~Rate limiting~~ (IMPLEMENTED)
- Native mobile app

---

## 13. Future Roadmap

In rough priority order after MVP:

1. **Multi-agreement support** — steward selects which agreement applies to their workplace
2. **ChromaDB persistence** — replace FAISS with ChromaDB when multiple agreements are loaded
3. **Auth** — restrict access to verified BCGEU stewards (likely BCGEU member number or email domain)
4. **Bookmarking** — save and share relevant clauses
5. **Grievance helper** — structured workflow for building a grievance argument from relevant clauses
6. **Suggested related clauses** — "You may also want to look at Article X" surfaced by the retrieval layer

---

## 14. Open Questions

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
