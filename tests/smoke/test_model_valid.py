"""
tests/smoke/test_model_valid.py — Smoke test: verify CLAUDE_MODEL is reachable

❗ This test calls the real Anthropic API and requires ANTHROPIC_API_KEY to be set.
   Run manually or in a gated CI step:

       pytest tests/smoke/ -v

   It is intentionally excluded from the default test run (no auto-discovery
   from the root `pytest` invocation unless you pass `--smoke`).

   To skip:      pytest tests/ --ignore=tests/smoke
   To run only:  pytest tests/smoke/ -v
"""

import os
import pytest
import anthropic

# Skip gracefully if no key is present so the default suite stays green
pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping smoke test",
)


def test_claude_model_exists_and_responds():
    """
    Send a minimal 1-token request to CLAUDE_MODEL.
    Fails with a clear message if the model name is invalid (404).
    This is the exact failure mode that hit production on 2026-03-08.
    """
    from app import CLAUDE_MODEL

    client = anthropic.Anthropic()
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )
    except anthropic.NotFoundError as exc:
        pytest.fail(
            f"CLAUDE_MODEL='{CLAUDE_MODEL}' does not exist in the Anthropic API.\n"
            f"Update the default in app.py or set the CLAUDE_MODEL env var.\n"
            f"Original error: {exc}"
        )

    # Any valid response confirms the model is reachable
    assert response.content is not None
