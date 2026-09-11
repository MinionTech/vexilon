"""
tests/test_ui_enter_submit.py — E2E: Enter-to-submit keyboard shortcut

Verifies that the Enter-to-submit handler works correctly:
- Enter (without Shift) submits the message
- Shift+Enter creates a newline without submitting
"""

import re
import pytest
from playwright.sync_api import Page, expect


@pytest.fixture(scope="module")
def app_url():
    """Base URL for the running Chainlit app."""
    return "http://localhost:7860"


def test_enter_submits_message(page: Page, app_url: str):
    """Pressing Enter (without Shift) in the chat input submits the message."""
    page.goto(app_url)
    
    # Wait for the chat interface to load
    page.wait_for_selector("textarea", timeout=10000)
    
    # Type a test message
    test_message = "Test message from Enter key"
    textarea = page.locator("textarea")
    textarea.fill(test_message)
    
    # Count existing messages before submission
    messages_before = page.locator(".message, [class*='message']").count()
    
    # Press Enter (without Shift) to submit
    textarea.press("Enter")
    
    # Wait for a new message to appear (timeout after 10 seconds)
    # The new message should contain our test text
    page.wait_for_selector(f"text={test_message}", timeout=10000)
    
    # Verify a new message was added
    messages_after = page.locator(".message, [class*='message']").count()
    assert messages_after > messages_before, "Expected new message after Enter keypress"
    
    # Verify the textarea was cleared after submission
    expect(textarea).to_have_value("")


def test_shift_enter_creates_newline(page: Page, app_url: str):
    """Pressing Shift+Enter in the chat input creates a newline without submitting."""
    page.goto(app_url)
    
    # Wait for the chat interface to load
    page.wait_for_selector("textarea", timeout=10000)
    
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
    # Check for newline (either \n or multiple lines)
    assert "\n" in textarea_value or len(textarea_value.split("\n")) > 1, \
        "Textarea should contain a newline"


def test_enter_respects_disabled_button(page: Page, app_url: str):
    """Enter keypress does not submit when the submit button is disabled."""
    page.goto(app_url)
    
    # Wait for the chat interface to load
    page.wait_for_selector("textarea", timeout=10000)
    
    textarea = page.locator("textarea")
    submit_button = page.locator("#chat-submit")
    
    # Type a message
    textarea.fill("Test message")
    
    # If the button is disabled, Enter should not submit
    # This test is conditional based on the initial state
    if submit_button.is_disabled():
        messages_before = page.locator(".message, [class*='message']").count()
        
        # Try to submit with Enter
        textarea.press("Enter")
        
        # Wait a moment
        page.wait_for_timeout(500)
        
        # Verify no message was submitted
        messages_after = page.locator(".message, [class*='message']").count()
        assert messages_after == messages_before, \
            "Enter should not submit when button is disabled"
        
        # Message should still be in textarea
        assert textarea.input_value() == "Test message", \
            "Text should remain when submission is blocked"
