---
description: Gradio 6 UI architecture, mobile-first constraints, and CSS/JS overrides
---

# UI & UX Standards

This document defines the interface standards for Vexilon. **Maintain these aesthetics and behaviors for all UI modifications.**

## 1. Gradio 6 Architectural Constraints

### Stability First
- **Version**: Enforce `gradio==6.14.0`. Do not upgrade without verifying the custom CSS/JS injection stability.
- **Fill Height**: Always use `fill_height=True` in the main `gr.Blocks` to ensure the chatbot expands correctly on mobile.

### Suppressed Elements
To maintain a clean, high-integrity "Forensic" look, several default Gradio elements are suppressed via CSS:
- **Buttons**: `share-button`, `undo-button`, `retry-button`, and `clear-button` are hidden.
- **Footers**: Standard Gradio footers are disabled.

## 2. Mobile-First Design

### Viewport Protection
- **iFrame Detection**: Custom JS in `_HEAD` adds an `.is-iframe` class to handle embedding in union portals.
- **Chatbot Height**: Maintain `height="70vh"` with a `min_height=400` to prevent keyboard overlap on mobile devices.

### Interaction Logic
- **Enter Key**: Custom JS enables "Enter-to-Submit" behavior while allowing "Shift-Enter" for new lines.
- **Persona Selector**: The persona dropdown (`Lookup`, `Grieve`, `Manage`) must be pinned to the top for easy mode-switching.

## 3. Streaming & Feedback

### Streaming UI
- **Immediate Response**: The UI must yield the user message and a "Thinking..." placeholder immediately to ensure the app feels "instant-on."
- **Context Display**: Retrieval context (excerpts) must be streamed or displayed as a separate data artifact if possible, but never clutter the main chat flow unless requested.

## 4. Custom Styling (Vanilla CSS)

All custom styles are located in the `_CSS` constant in `app/main.py`. 
- **Typography**: Use standard system fonts for maximum loading speed on LTE.
- **Citations**: Ensure the `[Source, Page]` format is rendered clearly in Markdown.
