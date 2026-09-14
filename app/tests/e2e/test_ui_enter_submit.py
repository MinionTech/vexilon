"""
tests/e2e/test_ui_enter_submit.py — E2E: Enter-to-submit keyboard shortcut

Verifies that the Enter-to-submit handler works correctly:
- Enter (without Shift) submits the message
- Shift+Enter creates a newline without submitting
"""

import pytest
from playwright.sync_api import Page

from tests.e2e.helpers import wait_for_custom_js


def test_enter_submits_message(page: Page, app_url: str):
    """Pressing Enter (without Shift) in the chat input submits the message."""
    page.goto(app_url, wait_until="domcontentloaded")
    wait_for_custom_js(page)
    page.wait_for_selector("#chat-submit", timeout=5000)

    test_message = "Test message from Enter key"
    textarea = page.locator("textarea")
    textarea.press_sequentially(test_message, delay=10)

    filled_value = textarea.input_value()
    assert filled_value == test_message, f"Failed to type into textarea. Got: {repr(filled_value)}"

    textarea.press("Enter")

    page.wait_for_function(
        "document.querySelector('textarea').value === ''",
        timeout=15000,
    )

    final_textarea_value = textarea.input_value()
    assert final_textarea_value == "", f"Textarea should be cleared after submission. Got: {repr(final_textarea_value)}"


def test_shift_enter_creates_newline(page: Page, app_url: str):
    """Pressing Shift+Enter in the chat input creates a newline without submitting."""
    page.goto(app_url, wait_until="domcontentloaded")
    page.wait_for_selector("textarea", timeout=15000)

    textarea = page.locator("textarea")
    textarea.press_sequentially("Line 1", delay=10)

    messages_before = page.locator(".message, [class*='message']").count()

    textarea.press("Shift+Enter")
    textarea.type("Line 2")

    page.wait_for_timeout(500)

    messages_after = page.locator(".message, [class*='message']").count()
    assert messages_after == messages_before, "Shift+Enter should not submit the message"

    textarea_value = textarea.input_value()
    assert "Line 1" in textarea_value, "First line should be present"
    assert "Line 2" in textarea_value, "Second line should be present"
    assert "\n" in textarea_value, "Textarea should contain a newline"


def test_enter_with_empty_textarea(page: Page, app_url: str):
    """Enter with empty textarea (naturally disabled button) does not cause errors."""
    page.goto(app_url, wait_until="domcontentloaded")
    wait_for_custom_js(page)
    page.wait_for_selector("#chat-submit", timeout=5000)

    textarea = page.locator("textarea")

    textarea_value = textarea.input_value()
    assert textarea_value == "", "Textarea should start empty"

    button_disabled = page.evaluate("document.querySelector('#chat-submit').disabled")
    assert button_disabled == True, "Button should be disabled when empty"

    textarea.press("Enter")

    page.wait_for_timeout(500)

    final_textarea_value = textarea.input_value()
    assert final_textarea_value == "", "Textarea should remain empty"

    final_button_disabled = page.evaluate("document.querySelector('#chat-submit').disabled")
    assert final_button_disabled == True, "Button should remain disabled"
