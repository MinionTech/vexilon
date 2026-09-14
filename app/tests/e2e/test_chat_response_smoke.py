"""
tests/e2e/test_chat_response_smoke.py — E2E: chat message round-trip smoke

Verifies a user message is submitted and an assistant response bubble appears.
Uses the real Ollama stack in Compose; asserts presence of content, not LLM quality.
"""

import pytest
from playwright.sync_api import Page, expect

from tests.e2e.helpers import wait_for_custom_js


def test_user_message_and_assistant_response_appear(page: Page, app_url: str):
    """Sending a chat message produces user and assistant message bubbles."""
    page.goto(app_url, wait_until="domcontentloaded")
    wait_for_custom_js(page)

    textarea = page.locator("textarea")
    textarea.press_sequentially("What is a steward?", delay=10)
    textarea.press("Enter")

    user_message = page.locator(".message-user").first
    user_message.wait_for(timeout=15000)
    expect(user_message).to_contain_text("What is a steward?")

    assistant_message = page.locator(".message-assistant").first
    assistant_message.wait_for(timeout=60000)
    page.wait_for_function(
        """() => {
            const el = document.querySelector('.message-assistant');
            return el && el.textContent.trim().length > 0;
        }""",
        timeout=120000,
    )
