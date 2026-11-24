# Accessibility (WCAG 2.1 AA)

Vexilon renders on Chainlit (`app/main.py`) with accessibility patches and contrast rules applied in `app/public/index.js` and `app/public/style.css`.

## Implemented Measures

### Landmarks, Skip Link, and Labels (`app/public/index.js`)
A `MutationObserver` (with a 500ms fallback poll) keeps DOM accessibility attributes intact across React mounts and re-renders:

- **Landmarks & Skip Link**: Assigns `role="banner"` to `#header` and `role="main"` (`id="main-content"`, `tabindex="-1"`) to its main sibling, and prepends a visually-hidden-until-focused `#skip-to-main` link inside `#header`.
- **Chat Controls**: Sets `aria-label` on `#chat-profiles`, `#upload-button`, `#chat-settings-open-modal`, `#chat-submit`, `#stop-button`, and unnamed message copy buttons (`svg.lucide-copy` -> `"Copy message"`), strips Chainlit's `role="presentation"` / `aria-hidden` from `#upload-button`, hides non-interactive `#upload-button-loading` placeholders, and names interactive upload controls `"Uploading"`.
- **Composer Input**: Sets `aria-label="Type your message"` on `#chat-input` when unlabeled.
- **Knowledge Base Button**: Keeps `#readme-button`'s `aria-label` and inner `<span>` text synced to `"Knowledge Base"` (WCAG 2.5.3 Label in Name).
- **Decorative Welcome Placeholder**: Suppresses the 1×1 transparent `logo_file_url` image (`alt=""`, `aria-hidden="true"`, and `display: none`).
- **Drawer Attributes**: Tags the Knowledge Base dialog with `data-knowledge-base-drawer` and clears dangling Radix `aria-describedby` references when no matching description node exists.

### Focus, Contrast, and Sizing (`app/public/style.css`)
- **Composer Focus Ring**: Restores a 2px/4px `#f5145f` `box-shadow` focus indicator on `#chat-input:focus-visible` in light and dark themes, with a `3px solid Highlight` outline fallback under `@media (forced-colors: active)` (WCAG 2.4.7).
- **Primary Action Contrast**: Overrides `#confirm` and `button.bg-primary` background from `#f5145f` to `#e7135a` (hover `#c8104e`) with `#ffffff` text for 4.53:1 contrast (WCAG 1.4.3).
- **Starter Contrast (Dark Theme)**: Raises `[id^="starter-"] p` text from `#7c7c7c` (3.85:1) to `#a3a3a3` (6.34:1 on `#212121`, WCAG 1.4.3).
- **Knowledge Base Drawer & Message Links**: Uses `#d10d50` in light theme and `#ff6b9d` in dark theme with `text-decoration: underline` for `a.text-primary` links in both the drawer and `[role="main"]` messages (WCAG 1.4.1 / 1.4.3), hides the duplicate `"Knowledge Base"` span visually (`font-size: 0`) so `::after` is the single visible label, and pins a `44×44px` close button with safe-area padding on mobile.

## Verification

Automated Playwright + `axe-core` tests live in `app/tests/e2e/`:

- `tests/e2e/test_axe_audit.py`: Full WCAG 2.1 A/AA `axe-core` scans across the welcome screen, settings dialog, Knowledge Base drawer, and post-message conversation view (light and dark themes), computed contrast on `#confirm`, `#chat-input` focus ring and `forced-colors` outline fallback, skip link, hidden placeholder logo, and `#upload-button-loading` transitions.
- `tests/e2e/test_knowledge_base_button_a11y.py`: Verifies `#readme-button` accessible name, `::after` label, and hidden inner `<span>`.
- `tests/e2e/test_chat_controls_a11y.py`: Verifies composer control labels and `#stop-button` label during streaming.
- `tests/e2e/test_knowledge_base_drawer_contrast_a11y.py`: Verifies dark-theme link contrast inside the Knowledge Base drawer.

Run the full containerized UI E2E suite from the repository root:

```bash
podman compose up --build --abort-on-container-exit --exit-code-from test-ui-e2e test-ui-e2e
```
