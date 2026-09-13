"""
tests/e2e/test_ui_enter_submit.py — E2E: Enter-to-submit keyboard shortcut

Verifies that the Enter-to-submit handler works correctly:
- Enter (without Shift) submits the message
- Shift+Enter creates a newline without submitting
"""

import os
import pytest
from playwright.sync_api import Page, expect


@pytest.fixture(scope="module")
def app_url():
    """Base URL for the running Chainlit app."""
    return os.getenv("APP_URL", "http://localhost:7860")


def test_enter_submits_message(page: Page, app_url: str):
    """Pressing Enter (without Shift) in the chat input submits the message."""
    page.goto(app_url, wait_until="domcontentloaded")
    
    # Wait for the chat interface to load
    page.wait_for_selector("textarea", timeout=15000)
    page.wait_for_selector("#chat-submit", timeout=5000)
    
    # Type a test message
    test_message = "Test message from Enter key"
    textarea = page.locator("textarea")
    textarea.fill(test_message)
    
    # Press Enter (without Shift) to submit
    textarea.press("Enter")
    
    # Wait for the message to appear - Chainlit uses specific markup
    # Wait for either the user message element or the textarea to clear
    try:
        # Try waiting for the message text to appear anywhere on the page
        page.wait_for_selector(f"text={test_message}", timeout=15000)
    except:
        # If that fails, check if textarea was cleared (another success signal)
        page.wait_for_timeout(2000)
        if textarea.input_value() == "":
            # Textarea cleared, message was likely sent
            pass
        else:
            raise AssertionError(f"Message '{test_message}' did not appear and textarea was not cleared")
    
    # Verify the textarea was cleared after submission (Chainlit behavior)
    page.wait_for_timeout(1000)
    assert textarea.input_value() == "", "Textarea should be cleared after submission"


def test_shift_enter_creates_newline(page: Page, app_url: str):
    """Pressing Shift+Enter in the chat input creates a newline without submitting."""
    page.goto(app_url, wait_until="domcontentloaded")
    
    # Wait for the chat interface to load
    page.wait_for_selector("textarea", timeout=15000)
    
    # Type first line
    textarea = page.locator("textarea")
    textarea.fill("Line 1")
    
    # Count existing messages before Shift+Enter
    messages_before = page.locator(".message, [class*='message']").count()
    
    # Press Shift+Enter to create a newline
    textarea.press("Shift+Enter")
    
    # Type second line
    textarea.type("Line 2")
    
    # Wait a bit to ensure no submission happened
    page.wait_for_timeout(500)
    
    # Verify no new message was submitted
    messages_after = page.locator(".message, [class*='message']").count()
    assert messages_after == messages_before, "Shift+Enter should not submit the message"
    
    # Verify the textarea contains both lines (with newline between them)
    textarea_value = textarea.input_value()
    assert "Line 1" in textarea_value, "First line should be present"
    assert "Line 2" in textarea_value, "Second line should be present"
    assert "\n" in textarea_value, "Textarea should contain a newline"


def test_enter_respects_disabled_button(page: Page, app_url: str):
    """Enter keypress does not submit when the submit button is disabled."""
    page.goto(app_url, wait_until="domcontentloaded")
    
    # Wait for the chat interface to load
    page.wait_for_selector("textarea", timeout=15000)
    page.wait_for_selector("#chat-submit", timeout=5000)
    
    textarea = page.locator("textarea")
    
    # Type a message
    textarea.fill("Test message")
    
    # Force-disable the submit button to make this test deterministic
    page.evaluate("document.querySelector('#chat-submit').disabled = true")
    
    # Wait a bit to ensure the state is set
    page.wait_for_timeout(300)
    
    # Press Enter with the button disabled
    # Because the handler only preventDefault()s when button is enabled and not disabled,
    # disabled Enter should preserve native textarea behavior (newline)
    textarea.press("Enter")
    
    # Wait for any potential submission to occur
    page.wait_for_timeout(1000)
    
    # With the button disabled, Enter should have created a newline (native behavior)
    # The handler checks if button is disabled and only prevents default if enabled
    textarea_value = textarea.input_value()
    assert textarea_value == "Test message\n", \
        f"Enter with disabled button should create newline. Got: {repr(textarea_value)}"
