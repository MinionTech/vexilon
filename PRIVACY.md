# Privacy Policy: PIPA Compliance Framework

This app is built with a "Privacy-by-Design" architecture, specifically aligned with the British Columbia [**Personal Information Protection Act (PIPA)**](https://www.bclaws.gov.bc.ca/civix/document/id/complete/statreg/03063_01). As a tool designed for union stewards, we recognize the high sensitivity of labour relations data and the legal requirements for protecting personal information.

## The 10 PIPA Principles in this Application

### 1. Accountability
*   **Designation**: The project maintainer serves as the Privacy Officer.
*   **Policies**: This document serves as our formal privacy policy.
*   **Third-Parties**: This application uses Hugging Face Router for processing. All data sent to the LLM provider is subject to their standard data processing agreements.

### 2. Identifying Purposes
*   **Purpose**: This application processes user queries for the sole purpose of providing context-aware answers in real-time. 
*   **Tracking**: We only track non-sensitive metadata (query counts and token consumption) for system health and API billing.
*   **Pseudonymous Client ID**: The browser generates a random UUID (via `crypto.randomUUID()`) and stores it in that browser's local storage. On each session the browser sends this ID to the backend, where it is held in server-side session state (in-memory only, tied to that session — no database) to distinguish discrete sessions for rate-limiting integrity, and it may appear in the operational log lines described in Principle 5. It is generated entirely client-side from randomness — never derived from IP address, device fingerprint, or any other real-world identifying information — and is not linked to a user's real-world identity. This satisfies PIPA's allowance for identifying *discrete* users without identifying *who* they are.
*   **No Secondary Use**: Data is never used for marketing, profiling, or tracking individual users.

### 3. Consent
*   **Implied Consent**: By using the chat interface and submitting queries, users provide implied consent for the processing of those queries.
*   **Notification**: The UI footer explicitly states that chats are ephemeral and not saved.

### 4. Limiting Collection
*   **Minimal Metrics**: Our collection is strictly limited to non-sensitive performance metrics: **query count** and **token consumption**.
*   **Content-Blind**: The application does not collect user IP addresses (except for active session rate-limiting in memory), device fingerprints, or location data. The pseudonymous client ID described in Principle 2 does not change this — its *value* is randomly generated, not derived from any IP address, device fingerprint, or other data collected from the user's device (though the ID token itself is sent to and briefly held by the server, as described in Principle 2).
*   **No PII**: We do not require registration or any Personally Identifiable Information (PII) to function.

### 5. Limiting Use, Disclosure, and Retention
*   **Ephemeral History**: Chat history exists only in the user's browser session. Refreshing the page permanently deletes the conversation.
*   **No Persistence**: This application does *not* write user queries or bot responses to a database. 
*   **Surgical Log Masking**: Technical logs for forensic health monitoring include non-PII metadata (word counts, character counts, persona modes, and the pseudonymous client ID from Principle 2), but **never** the actual content of user queries or bot responses. These logs are plain console output written by the application itself; the application does not write them to a database, though the hosting platform's own log capture/retention may apply.

### 6. Accuracy
*   **Source Integrity**: This application uses a "Forensic Markdown Pipeline" to ensure that the collective agreements used for grounding are accurate representations of the official source documents.
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
*   **User Control of Pseudonymous ID**: The client ID is stored in the browser's local storage for this site, not its HTTP cache — clearing the browser's *cache* alone will not remove it. To reset it, use your browser's "Clear site data" / "Cookies and other site data" option for this site (a plain "clear cache" action is not sufficient). Doing so deletes the browser's copy and severs correlation with future sessions; a new random ID is generated on next use. It does not retroactively alter operational log lines already written using the old ID (see Principle 5).

### 10. Challenging Compliance
*   **Reporting**: Users can report privacy concerns or potential vulnerabilities through GitHub Issues or by contacting the project maintainer.
*   **Recourse**: As a tool for union stewards, we encourage users to consult their BCGEU representative if they believe privacy standards are not being met.

---

## Data Flow Summary

1.  **User Input** → Sanitized locally → Sent to LLM Provider via TLS.
2.  **Processing** → LLM generates a response based *only* on the provided agreement excerpts.
3.  **Output** → Streamed back to the user's browser.
4.  **Closure** → Tab closes → RAM is cleared → No record remains on the servers.
