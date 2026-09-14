"""
tests/e2e/test_chat_response_smoke.py — E2E: chat message round-trip smoke

Verifies a user message is submitted and an assistant response completes.
Uses the real Ollama stack in Compose; asserts lifecycle signals, not LLM quality.
"""

import pytest
from playwright.sync_api import Page, expect

from helpers import wait_for_custom_js

USER_PROMPT = "What is a steward?"


def test_user_message_and_assistant_response_appear(page: Page, app_url: str):
    """Sending a chat message starts generation and produces user + assistant bubbles."""
    page.goto(app_url, wait_until="domcontentloaded")
    wait_for_custom_js(page)

    textarea = page.locator("textarea")
    textarea.press_sequentially(USER_PROMPT, delay=10)
    textarea.press("Enter")

    page.wait_for_function(
        "document.querySelector('textarea').value === ''",
        timeout=15000,
    )

    expect(page.get_by_text(USER_PROMPT, exact=True)).to_be_visible(timeout=30000)

    stop_button = page.locator("#stop-button")
    stop_button.wait_for(state="visible", timeout=60000)
    stop_button.wait_for(state="hidden", timeout=120000)

    assistant_message = page.locator(".message-assistant").first
    assistant_message.wait_for(timeout=30000)
    page.wait_for_function(
        """() => {
            const el = document.querySelector('.message-assistant');
            return el && el.textContent.trim().length > 0;
        }""",
        timeout=30000,
    )
