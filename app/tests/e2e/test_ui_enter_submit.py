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
    
    # Wait for the custom event handler to attach (runs via setInterval every 500ms)
    page.wait_for_function(
        "document.querySelector('textarea')?.dataset?.listenerAttached === 'true'",
        timeout=3000
    )
    
    # Type a test message using pressSequentially to trigger React onChange
    test_message = "Test message from Enter key"
    textarea = page.locator("textarea")
    textarea.press_sequentially(test_message, delay=10)
    
    # Verify message was typed
    filled_value = textarea.input_value()
    assert filled_value == test_message, f"Failed to type into textarea. Got: {repr(filled_value)}"
    
    # Press Enter (without Shift) to submit
    textarea.press("Enter")
    
    # Wait for the textarea to clear (Chainlit clears it after submission)
    page.wait_for_function(
        "document.querySelector('textarea').value === ''",
        timeout=15000
    )
    
    # Verify the textarea was cleared after submission (Chainlit behavior)
    final_textarea_value = textarea.input_value()
    assert final_textarea_value == "", f"Textarea should be cleared after submission. Got: {repr(final_textarea_value)}"


def test_shift_enter_creates_newline(page: Page, app_url: str):
    """Pressing Shift+Enter in the chat input creates a newline without submitting."""
    page.goto(app_url, wait_until="domcontentloaded")
    
    # Wait for the chat interface to load
    page.wait_for_selector("textarea", timeout=15000)
    
    # Type first line using pressSequentially to trigger React onChange
    textarea = page.locator("textarea")
    textarea.press_sequentially("Line 1", delay=10)
    
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


def test_enter_with_empty_textarea(page: Page, app_url: str):
    """Enter with empty textarea (naturally disabled button) does not cause errors."""
    page.goto(app_url, wait_until="domcontentloaded")
    
    # Wait for the chat interface to load
    page.wait_for_selector("textarea", timeout=15000)
    page.wait_for_selector("#chat-submit", timeout=5000)
    
    # Wait for the custom event handler to attach (runs via setInterval every 500ms)
    page.wait_for_function(
        "document.querySelector('textarea')?.dataset?.listenerAttached === 'true'",
        timeout=3000
    )
    
    textarea = page.locator("textarea")
    
    # Verify textarea starts empty and button is disabled
    textarea_value = textarea.input_value()
    assert textarea_value == "", "Textarea should start empty"
    
    button_disabled = page.evaluate("document.querySelector('#chat-submit').disabled")
    assert button_disabled == True, "Button should be disabled when empty"
    
    # Press Enter with empty textarea
    # Handler should preventDefault but there's nothing to submit
    # This shouldn't cause any errors or unexpected behavior
    textarea.press("Enter")
    
    # Wait a moment for any potential side effects
    page.wait_for_timeout(500)
    
    # Verify textarea is still empty and no errors occurred
    final_textarea_value = textarea.input_value()
    assert final_textarea_value == "", "Textarea should remain empty"
    
    final_button_disabled = page.evaluate("document.querySelector('#chat-submit').disabled")
    assert final_button_disabled == True, "Button should remain disabled"
