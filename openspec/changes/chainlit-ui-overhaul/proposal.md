# Proposal: Chainlit UI Overhaul

## Summary
Transform the existing Chainlit implementation into a premium, "Forensic" UI that aligns with the BCGEU Navigator's mission of providing high-integrity, fact-based labor law assistance. This overhaul focuses on using native Chainlit 2.x components for a more integrated feel, removing custom UI hacks, and improving accessibility to key features like personas and knowledge base navigation.

## Motivation
The current UI, while functional, still carries some "legacy" feel from the Gradio transition. Personas are hidden in a settings menu, and knowledge base links are tucked away in a "Readme" tab. By moving to a more "native" Chainlit approach, we can provide a smoother, more professional experience that feels like a standalone "Forensic" tool.

## Proposed Changes

### 1. Forensic Starters
- Update `cl.set_starters` to provide more structured, high-value entry points for stewards (e.g., "Analyze Article 14", "Build Just Cause Case").

### 2. Native Sidebar Navigation
- Use `cl.Text` or `cl.Sidebar` to provide direct links to the Knowledge Base (labour law documents) in the left sidebar.
- This replaces the "Readme" tab approach, making references always accessible.

### 3. Integrated Persona Toggles
- Move Persona selection from the "Gear" icon (`cl.ChatSettings`) to a more visible location, possibly using `cl.Action` buttons in the sidebar or a custom header profile.

### 4. Ephemeral History Actions
- Implement PIPA-compliant manual history handling.
- Provide `cl.Action` buttons for "Export History" (Markdown) and "Clear Session" to ensure user data privacy.

### 5. Patch Stability
- Maintain and document the critical Python 3.14 AnyIO patches to ensure the system remains stable on bleeding-edge environments.

## Non-goals
- Deep backend RAG pipeline changes (we focus on the UI layer).
- Changing the underlying model (stay on Qwen 3).
- Moving away from Chainlit (this is an *overhaul*, not a *migration*).

## Success Criteria
- [ ] UI feels "premium" and "native" (no custom CSS/JS hacks).
- [ ] Personas are switchable with fewer clicks.
- [ ] Knowledge base is persistently accessible.
- [ ] App remains fully functional on Python 3.14.
