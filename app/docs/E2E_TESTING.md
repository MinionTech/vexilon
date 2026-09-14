# UI End-to-End Testing (Playwright)

Browser-based regression tests for Chainlit UI behavior live under `app/tests/e2e/`.
They run against a real Chainlit app instance (Compose `app-ci`) with Chromium via
[`pytest-playwright`](https://github.com/microsoft/playwright-pytest).

## CI ownership

The **`UI E2E Tests (Playwright)`** job in [`.github/workflows/pr.yml`](../../.github/workflows/pr.yml)
(`test-ui-e2e`) builds the dev image, starts `app-ci` + Ollama, and runs:

```bash
docker compose up --abort-on-container-exit --exit-code-from test-ui-e2e test-ui-e2e
```

That job is a required gate for the `test-everything` aggregate.

## Run locally (Compose — recommended)

From the repository root, with Podman or Docker Compose:

```bash
podman compose up --build --abort-on-container-exit --exit-code-from test-ui-e2e test-ui-e2e
```

This starts:

| Service | Role |
|---|---|
| `ollama` / `ollama-init` | Pulls `tinyllama` for chat smokes |
| `app-ci` | Chainlit app on port 7860 (baked image, no bind mount) |
| `test-ui-e2e` | Installs Chromium, runs `pytest tests/e2e/ -v` with `APP_URL=http://app-ci:7860` |

On failure, inspect logs:

```bash
podman compose logs app-ci test-ui-e2e
```

## Run against a local dev server (optional)

If `podman compose up dev` is already running on port 7860:

```bash
cd app
APP_URL=http://localhost:7860 uv run pytest tests/e2e/ -v
```

Chat response smokes require a working LLM backend (Ollama or configured provider).

## Isolation from unit/integration tests

| Suite | Path | Compose service | Local pytest command |
|---|---|---|---|
| Unit | `app/tests/test_*.py` | `test-unit` | `uv run pytest app/tests/test_*.py` |
| Integration | `app/tests/integration/` | `test-integration-*` | `uv run pytest app/tests/integration/` |
| UI E2E | `app/tests/e2e/` | `test-ui-e2e` | `uv run pytest app/tests/e2e/` |

The `test-unit` Compose service runs `pytest tests/test_*.py` only. For local
unit runs, target `app/tests/test_*.py` explicitly (or add
`--ignore=app/tests/e2e`) — the recursive `app/tests/` command in README also
collects `app/tests/e2e/` unless ignored. Each tier uses a separate Compose
service and explicit pytest path.

## Coverage map

| File | Behavior verified |
|---|---|
| `test_ui_enter_submit.py` | Enter submits; Shift+Enter newline; empty submit guard (#644) |
| `test_chat_controls_a11y.py` | Composer aria-labels (#641); stop button during streaming |
| `test_knowledge_base_button_a11y.py` | Knowledge Base button label in name (#643) |
| `test_knowledge_base_drawer.py` | Drawer opens with expected sections and document links |
| `test_knowledge_base_drawer_contrast_a11y.py` | Drawer link contrast in dark theme (#642) |
| `test_chat_response_smoke.py` | User + assistant message bubbles after send |

## Selectors and fixtures

Stable Chainlit element IDs (`#chat-submit`, `#readme-button`, `#stop-button`) and
custom JS hooks (`textarea.dataset.listenerAttached`, `[data-knowledge-base-drawer]`)
are preferred over brittle CSS class chains.

Shared helpers live in `tests/e2e/helpers.py`; pytest fixtures (`app_url`)
live in `tests/e2e/conftest.py`.

## Related issues

- #672 — harness introduction
- #673 — suite expansion (this document)
- #641, #642, #643, #401 — accessibility backlog
