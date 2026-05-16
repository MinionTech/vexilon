# Tasks: PR #496 Finalization

## Phase 1: Aesthetic & Link Integrity
- [x] **T1: Emoji Purge & Header Alignment**
    - Remove all emojis from `main.py`, `config.toml`, and `chainlit.md`.
    - Fix alignment of the persona selector in `style.css`.
- [x] **T2: Link & Footer Restoration**
    - Correct Source Code and Privacy Policy links in `config.toml` and `index.js`.
    - Restore the project-standard AgNav footer in `index.js`.
- [ ] **T3: Settings Optimization**
    - Refine the native Chat Settings gear icon content in `main.py`.

## Phase 2: Functional Hardening
- [ ] **T4: Upload Governance**
    - Ensure spontaneous file uploads are disabled and verify in the UI.
- [ ] **T5: Knowledge Base & Manifest**
    - Finalize agreement excerpts and manifest navigation in `chainlit.md`.
- [ ] **T6: Save/Load Persistence**
    - Implement robust conversation history management in `main.py`.

## Phase 3: Logic & Verification
- [ ] **T7: Advanced Forensic Logic**
    - Implement multi-hop reasoning and expanded audit rule-sets in `main.py`.
- [ ] **T8: 100% Test Coverage**
    - Add surgical fixes for remaining edge cases in `test_chainlit_ui.py`.
- [ ] **V1: Final Verification**
    - Run `test-everything` and generate final report.
