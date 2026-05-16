# Design: PR #496 Finalization

## Architecture Overview
The finalization focuses on three main areas: logic hardening, UI sanitization, and persistence.

## Component Changes

### 1. Forensic UI Sanitization
- **Emoji Removal**: Scour `app/main.py`, `app/.chainlit/config.toml`, and `app/chainlit.md` for any decorative emojis. Replace with ASCII-based symbols or plain text where necessary.
- **Header Alignment**: Update `app/public/style.css` to ensure the persona selector and header links are perfectly aligned and responsive.
- **Custom Footer**: Refine `app/public/index.js` `injectFooter` to use a template-based approach and ensure it doesn't overlap with Chainlit's dynamic elements.

### 2. Forensic Logic & RAG
- **Multi-hop Reasoning**: Update `rag_review_stream` to allow for iterative context gathering if the first pass is insufficient.
- **Audit Rules**: Expand `UNION_MANDATORY_RULES` and `MANAGER_MANDATORY_RULES` with more specific forensic auditing patterns.

### 3. Save/Load Persistence
- **Serialization**: Ensure `history_to_markdown` captures all metadata (persona, reasoning state).
- **Deserialization**: Fix `markdown_to_history` to correctly reconstruct the session state upon load.

### 4. Link & Policy Integrity
- **Unified Links**: Centralize link definitions in `app/main.py` or `config.toml` to avoid duplication between the backend and injected frontend components.
- **Privacy Policy**: Update the destination to the correct absolute URL or relative path within the repo.

## Testing Strategy
- **Integration Tests**: Expand `app/tests/integration/test_chainlit_ui.py` to test:
    - Persona switching via injected selector.
    - Footer presence and link validity.
    - Export/Import cycle.
    - Upload rejection.
- **Verification Receipts**: All tasks MUST produce a `.log` file in `reports/` with verification output.
