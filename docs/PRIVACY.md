# Vexilon Privacy Policy: PIPA Compliance Framework

Vexilon is built with a "Privacy-by-Design" architecture, specifically aligned with the British Columbia **Personal Information Protection Act (PIPA)**. As a tool designed for union stewards, we recognize the high sensitivity of labour relations data and the legal requirements for protecting personal information.

## The 10 PIPA Principles in Vexilon

### 1. Accountability
*   **Designation**: The Vexilon maintainer serves as the Privacy Officer.
*   **Policies**: This document serves as our formal privacy policy.
*   **Third-Parties**: Vexilon uses Anthropic (Claude) for processing. All data sent to Anthropic is subject to their standard data processing agreements, which prohibit the use of customer data for training their models.

### 2. Identifying Purposes
*   **Purpose**: Vexilon processes user queries for the sole purpose of providing context-aware answers in real-time. 
*   **Tracking**: We only track non-sensitive metadata (query counts and token consumption) for system health and API billing.
*   **No Secondary Use**: Data is never used for marketing, profiling, or tracking individual users.

### 3. Consent
*   **Implied Consent**: By using the chat interface and submitting queries, users provide implied consent for the processing of those queries.
*   **Notification**: The UI footer explicitly states that chats are ephemeral and not saved.

### 4. Limiting Collection
*   **Minimal Metrics**: Our collection is strictly limited to non-sensitive performance metrics: **query count** and **token consumption**.
*   **Content-Blind**: Vexilon does not collect user IP addresses (except for active session rate-limiting in memory), device fingerprints, or location data.
*   **No PII**: We do not require registration or any Personally Identifiable Information (PII) to function.

### 5. Limiting Use, Disclosure, and Retention
*   **Ephemeral History**: Chat history exists only in the user's browser session. Refreshing the page permanently deletes the conversation.
*   **No Persistence**: Vexilon does *not* write user queries or bot responses to a database. 
*   **Log Sanitization**: Technical logs (for debugging) do not include the content of user queries.

### 6. Accuracy
*   **Source Integrity**: Vexilon uses a "Forensic Markdown Pipeline" to ensure that the collective agreements used for grounding are accurate representations of the official source documents.
*   **Verification Bot**: An optional reviewer bot checks responses against source text to prevent hallucinations.

### 7. Safeguards
*   **Input Sanitization**: We use regex-based pattern matching to prevent prompt injection and unauthorized access to system instructions.
*   **Rate Limiting**: Protects the service from abuse and potential data-scraping attempts.
*   **Hosting**: Deployed on Hugging Face Spaces with standard TLS encryption for all data in transit.

### 8. Openness
*   **Transparency**: Our technical implementation (including the retrieval logic and system prompts) is open-source and available for audit on GitHub.
*   **Policy Access**: This privacy policy is linked directly from the application interface.

### 9. Individual Access
*   **Immediate Access**: Users see all data processed (their query) and the resulting output immediately.
*   **No "Records"**: Because we do not retain data, there are no persistent records for a user to request or correct.

### 10. Challenging Compliance
*   **Reporting**: Users can report privacy concerns or potential vulnerabilities through GitHub Issues or by contacting the project maintainer.
*   **Recourse**: As a tool for union stewards, we encourage users to consult their BCGEU representative if they believe privacy standards are not being met.

---

## Data Flow Summary

1.  **User Input** → Sanitized locally → Sent to Anthropic (Claude) via TLS.
2.  **Processing** → Claude generates a response based *only* on the provided agreement excerpts.
3.  **Output** → Streamed back to the user's browser.
4.  **Closure** → Tab closes → RAM is cleared → No record remains on Vexilon servers.
