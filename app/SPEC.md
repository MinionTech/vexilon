# Product Specification: Agreement Navigator (AgNav)

Agreement Navigator (AgNav) is a high-integrity RAG (Retrieval-Augmented Generation) chatbot designed specifically for BCGEU union stewards. It provides instant, cited, and verbatim answers from collective agreements and labour law documents, optimized for high-speed mobile use on the shop floor.

---

## 1. Product Goal
To reduce the "Information Gap" for union stewards by providing a mobile-first, zero-config tool that surfaces the correct contract language in seconds, ensuring stewards are as well-informed as management during meetings and grievances.

## 2. Core Value Proposition
- **High-Integrity Citations:** Every claim must be backed by a verbatim quote and a specific article/page citation.
- **Mobile-First:** A streamlined, "instant-on" interface that works on low-bandwidth LTE connections.
- **Forensic Accuracy:** A Markdown-first ingestion pipeline that ensures the AI "sees" the contract structure (Articles, Sections, Clauses) exactly as a human does.
- **Privacy-by-Design:** No persistent storage of user queries; PIPA-compliant ephemeral sessions.

---

## 3. Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| **Interface** | Gradio 6 | Rapid, high-performance web UI with native streaming support. |
| **Logic** | Python 3.12 | Standard for LLM orchestration and RAG pipelines. |
| **LLM (PROD)** | Hugging Face Router | Access to Qwen/Qwen3-4B-Instruct-2507 via high-speed "Flash" API. |
| **LLM (DEV)** | Ollama | Local execution of qwen3:4b-instruct for zero-config, offline development. |
| **Embeddings** | `BAAI/bge-small-en-v1.5` | State-of-the-art local CPU embeddings; avoids 3GB CUDA dependencies. |
| **Vector Store** | FAISS | Ultra-fast, in-memory CPU vector index; requires no database server. |
| **Deployment** | Podman / HF Spaces | Containerized deployment with immutable production parity. |

---

## 4. Persona Modes (Direct Advice)

AgNav features three operational personas that adapt the AI's "brain" to the steward's current situation:

1. **Lookup (Default):** The "Forensic Navigator." Focuses on literal interpretation, clause-finding, and cross-referencing.
2. **Grieve:** The "Forensic Auditor." Acts as a Senior Staff Rep, building air-tight grievance cases by identifying contract violations and suggesting specific evidence to gather.
3. **Manage:** The "Strategic Consultant." Focuses on compliance, risk mitigation, and management-steward meeting preparation.

---

## 5. Security & Reliability

- **Prompt Injection Protection:** A multi-layered defense using a 16-pattern regex-based sanitizer. It blocks:
    - Intent overrides (*"ignore previous instructions"*, *"forget the rules"*)
    - Roleplay attempts (*"you are now a..."*, *"pretend to be..."*)
    - Technical bypasses (*"jailbreak"*, *"developer mode"*, *"sudo mode"*)
    - Structural attacks (*"[[SYSTEM]]"*, *"### system instructions"*)
- **Rate Limiting:** Multi-tier protection (5/min, 100/hour) to prevent API abuse and cost overruns.
- **Ephemeral RAM-Only Storage:** The FAISS index and conversation history exist only in memory; no records are written to disk.
- **PIPA Compliance:** Content-blind architecture ensures no user queries or bot responses are logged.

---

## 6. The Forensic Markdown Pipeline

AgNav moves beyond messy PDF-to-text extraction by using a specialized "Forensic" ingestion strategy:
1. **Geometric Reconstruction:** PyMuPDF extracts text based on physical page coordinates, ensuring correct word ordering for complex multi-column layouts.
2. **Zero-Reasoning Transcription:** An LLM pass adds hierarchical Markdown headers (# Article, ## Section) under strict "PARANOID DETERMINISM" rules:
    - **VERBATIM ONLY:** Forbidden from changing, adding, or removing a single substantive word.
    - **NO IMPROVEMENT:** Do not fix typos; if the raw text is broken, preserve it.
    - **NO NOISE:** Removal of page numbers, URLs, and footers is the only structural change allowed.
3. **Integrity Audit:** A side-by-side verification report (`.integrity.md`) uses diffing to ensure the resulting Markdown preserves 100% of the substantive source content.

---

## 7. Cost Estimate

| Component | Rate | Estimated Monthly (moderate use) |
|---|---|---|
| **Hugging Face Router** | Qwen3-4B-Instruct-2507 (Flash) | ~$5–10 CAD |
| **Embeddings** | `bge-small-en-v1.5` | $0 (Runs locally on CPU) |
| **Total** | | **~$5–15 CAD/month** |

---

## 8. Multi-Turn Context (Query Condensing)

To ensure follow-up questions work reliably (e.g., "What about for part-time?"), AgNav uses a Query Condensing pass:
1. **Input:** User query + Conversation History.
2. **Pass:** A fast LLM call reconstructs the user's intent into a standalone search query.
3. **Search:** The FAISS index is searched using the condensed query, not the vague follow-up.

---

AgNav includes an automated "Adversarial Reviewer" that double-checks its own answers:
1. **Trigger:** Runs as an asynchronous background task immediately after a response finishes streaming.
2. **Verification:** A second LLM pass (configured via `VERIFY_MODEL`) compares the generated answer against the retrieved quotes.
3. **Categories:**
    - **VERIFIED:** The claim is explicitly supported by the verbatim quote.
    - **DISPUTED:** The claim is unsupported or contradicts the retrieved text.
    - **UNCERTAIN:** The citation is unclear or incomplete.
4. **Flagging:** If any claim is disputed, a "Verification Note" is appended to the response, warning the steward to check the source.

---

## 10. Deployment & Infrastructure

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `AGNAV_USERNAME` | `admin` | Basic Auth username |
| `AGNAV_PASSWORD` | *(None)* | Basic Auth password (enables auth if set) |
| `AGNAV_LLM_PROVIDER` | `huggingface` | `huggingface` or `ollama` |
| `AGNAV_DEFAULT_MODEL` | `Qwen/Qwen3-4B-Instruct-2507` | Primary LLM for responses |
| `HF_TOKEN` | *(Required for HF)* | Hugging Face API token |
| `PORT` | `7860` | Gradio port |
| `SIMILARITY_TOP_K` | `40` | Number of chunks retrieved |
| `VERIFY_ENABLED` | `true` | Enable the verification bot |
| `RATE_LIMIT_PER_HOUR` | `100` | Max requests per hour per client IP |

---

## 11. Success Criteria

The Agreement Navigator is successful when:
1. **Speed:** Answers begin streaming in < 3 seconds on an LTE connection.
2. **Accuracy:** 100% of claims are accompanied by a verbatim quote from the knowledge base.
3. **Honesty:** The bot correctly identifies "Not found in the agreement" for questions outside its scope.
4. **Mobile Utility:** A steward can navigate to a specific clause while standing in a meeting with management.

---

## 12. Future Roadmap

1. **Multi-Agreement Support:** Allow stewards to select their specific Component agreement.
2. **Grievance Builder:** A wizard-style interface for generating a "Grievance Fact Sheet" based on retrieved clauses.
3. **Arbitration Awards:** Indexing a library of BC labour arbitration awards for jurisprudence context.
4. **Member-Only Auth:** Integration with union member portals for secure access.

---

## 13. Open Questions (Resolved)

- **PDF or Markdown?** Markdown is the source of truth for the AI; PDF is for human download.
- **Provider?** Provider-agnostic via unified OpenAI-compatible client (HF/Ollama).
- **Security?** Secure-by-default with rate limiting and input sanitization.
