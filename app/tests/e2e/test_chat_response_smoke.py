"""
tests/e2e/test_chat_response_smoke.py — E2E: chat message round-trip smoke

Verifies a user message is submitted and an assistant response completes.
Uses the real Ollama stack in Compose; asserts lifecycle signals, not LLM quality.
"""

import pytest
from playwright.sync_api import Page

from helpers import wait_for_custom_js


def test_user_message_and_assistant_response_appear(page: Page, app_url: str):
    """Sending a chat message starts generation and produces a response bubble."""
    page.goto(app_url, wait_until="domcontentloaded")
    wait_for_custom_js(page)

    textarea = page.locator("textarea")
    textarea.press_sequentially("What is a steward?", delay=10)
    textarea.press("Enter")

    page.wait_for_function(
        "document.querySelector('textarea').value === ''",
        timeout=15000,
    )

    stop_button = page.locator("#stop-button")
    stop_button.wait_for(state="visible", timeout=60000)
    stop_button.wait_for(state="hidden", timeout=120000)

    page.wait_for_function(
        """() => {
            const bubbles = document.querySelectorAll(
                '.message, [class*="message"]:not(.message-list):not([class*="actions"])'
            );
            return Array.from(bubbles).some(
                (m) => m.textContent.trim().length > 0 && m.textContent.trim() !== 'What is a steward?'
            );
        }""",
        timeout=60000,
    )
