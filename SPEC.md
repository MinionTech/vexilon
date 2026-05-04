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
| **LLM (PROD)** | Hugging Face Router | Access to Qwen/Qwen3-7B-Instruct via high-speed "Flash" API. |
| **LLM (DEV)** | Ollama | Local execution of Qwen3-7B-Instruct for zero-config, offline development. |
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

- **Basic Auth:** Optional layer for protecting private HF Spaces deployments.
- **Prompt Injection Protection:** A 16-pattern regex-based sanitizer that blocks "jailbreak" attempts and system-prompt overrides.
- **Rate Limiting:** Multi-tier (Minute/Hour) protection to prevent API abuse and cost overruns.
- **Ephemeral RAM-Only Storage:** FAISS index and conversation history exist only in memory; no records are written to disk.

---

## 6. The Forensic Markdown Pipeline

AgNav moves beyond messy PDF-to-text extraction by using a specialized Markdown ingestion strategy:
1. **Geometric Reconstruction:** PyMuPDF extracts text based on physical page coordinates.
2. **Structural Restructuring:** An LLM pass adds hierarchical Markdown headers (# Article, ## Section) based on the original document structure.
3. **Integrity Audit:** A side-by-side verification report (`.integrity.md`) ensures no words were lost or added during conversion.

---

## 7. Cost Estimate

| Component | Rate | Estimated Monthly (moderate use) |
|---|---|---|
| **Hugging Face Router** | Qwen3-7B-Instruct (Flash) | ~$5–15 CAD |
| **Embeddings** | `bge-small-en-v1.5` | $0 (Runs locally on CPU) |
| **Total** | | **~$5–15 CAD/month** |

---

## 8. Multi-Turn Context (Query Condensing)

To ensure follow-up questions work reliably (e.g., "What about for part-time?"), AgNav uses a Query Condensing pass:
1. **Input:** User query + Conversation History.
2. **Pass:** A fast LLM call reconstructs the user's intent into a standalone search query.
3. **Search:** The FAISS index is searched using the condensed query, not the vague follow-up.

---

## 9. Verification Bot

AgNav includes an optional automated reviewer that double-checks its own answers:
1. **Trigger:** Runs immediately after a response finishes streaming.
2. **Verification:** A second LLM pass compares the generated answer against the retrieved quotes.
3. **Flagging:** If a claim is unsupported by the text, a "Verification Note" is appended to the response.

---

## 10. Deployment & Infrastructure

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `AGNAV_USERNAME` | `admin` | Basic Auth username |
| `AGNAV_PASSWORD` | *(None)* | Basic Auth password (enables auth if set) |
| `AGNAV_LLM_PROVIDER` | `huggingface` | `huggingface` or `ollama` |
| `AGNAV_DEFAULT_MODEL` | `Qwen/Qwen3-7B-Instruct` | Primary LLM for responses |
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
