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
    # Capture console messages to verify index.js loads
    console_messages = []
    page.on("console", lambda msg: console_messages.append(f"{msg.type}: {msg.text}"))
    
    page.goto(app_url, wait_until="domcontentloaded")
    
    # Wait for the chat interface to load
    page.wait_for_selector("textarea", timeout=15000)
    page.wait_for_selector("#chat-submit", timeout=5000)
    
    # Print captured console messages
    print(f"DEBUG console messages: {console_messages[:10]}")  # First 10 messages
    
    # Debug: Check if index.js loaded and handler attached
    debug_info = page.evaluate("""() => {
        const textarea = document.querySelector("textarea");
        const submitBtn = document.querySelector("#chat-submit");
        return {
            textareaExists: !!textarea,
            submitBtnExists: !!submitBtn,
            submitBtnDisabled: submitBtn ? submitBtn.disabled : null,
            listenerAttached: textarea ? textarea.dataset.listenerAttached : null,
            textareaValue: textarea ? textarea.value : null
        };
    }""")
    print(f"DEBUG before submit: {debug_info}")
    
    # Type a test message using pressSequentially to trigger React onChange
    test_message = "Test message from Enter key"
    textarea = page.locator("textarea")
    textarea.press_sequentially(test_message, delay=10)
    
    # Verify message was typed
    filled_value = textarea.input_value()
    print(f"DEBUG typed textarea value: {repr(filled_value)}")
    assert filled_value == test_message, f"Failed to type into textarea. Got: {repr(filled_value)}"
    
    # Debug: Check state before Enter
    pre_enter_state = page.evaluate("""() => {
        const textarea = document.querySelector("textarea");
        const submitBtn = document.querySelector("#chat-submit");
        return {
            textareaValue: textarea.value,
            submitBtnDisabled: submitBtn.disabled,
            listenerAttached: textarea.dataset.listenerAttached
        };
    }""")
    print(f"DEBUG pre-enter state: {pre_enter_state}")
    
    # Press Enter (without Shift) to submit
    textarea.press("Enter")
    
    # Wait a moment for the handler to process
    page.wait_for_timeout(500)
    
    # Debug: Check state after Enter
    post_enter_state = page.evaluate("""() => {
        const textarea = document.querySelector("textarea");
        return {
            textareaValue: textarea.value,
            textareaLength: textarea.value.length
        };
    }""")
    print(f"DEBUG post-enter state: {post_enter_state}")
    
    # Wait for the textarea to clear (Chainlit clears it after submission)
    # Or for the message to appear in the chat
    try:
        # Wait for textarea to clear (primary signal)
        page.wait_for_function(
            "document.querySelector('textarea').value === ''",
            timeout=15000
        )
    except:
        # If textarea didn't clear, check if message appeared
        try:
            page.wait_for_selector(f"text={test_message}", timeout=2000)
        except:
            final_value = textarea.input_value()
            raise AssertionError(
                f"Enter handler did not submit message. "
                f"Textarea value after Enter: {repr(final_value)}"
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


def test_enter_respects_disabled_button(page: Page, app_url: str):
    """Enter keypress does not submit when the submit button is disabled."""
    page.goto(app_url, wait_until="domcontentloaded")
    
    # Wait for the chat interface to load
    page.wait_for_selector("textarea", timeout=15000)
    page.wait_for_selector("#chat-submit", timeout=5000)
    
    textarea = page.locator("textarea")
    
    # Type a message using pressSequentially to trigger React onChange
    textarea.press_sequentially("Test message", delay=10)
    
    # Verify message was filled
    filled_value = textarea.input_value()
    print(f"DEBUG filled value before disable: {repr(filled_value)}")
    assert filled_value == "Test message", f"Failed to fill textarea. Got: {repr(filled_value)}"
    
    # Test realistic scenario: button is naturally disabled when textarea is empty
    # Clear textarea to trigger React's natural button-disable behavior
    textarea.fill("")
    page.wait_for_timeout(200)  # Let React process the change
    
    # Debug: Verify button is naturally disabled with empty textarea
    pre_clear_state = page.evaluate("""() => {
        const textarea = document.querySelector("textarea");
        const submitBtn = document.querySelector("#chat-submit");
        return {
            textareaValue: textarea.value,
            submitBtnDisabled: submitBtn.disabled,
            listenerAttached: textarea.dataset.listenerAttached
        };
    }""")
    print(f"DEBUG pre-clear state (empty textarea): {pre_clear_state}")
    assert pre_clear_state["submitBtnDisabled"] == True, "Button should be disabled when textarea is empty"
    assert pre_clear_state["textareaValue"] == "", "Textarea should be empty"
    
    # Type single character to have something to test with
    textarea.type("X")
    
    # Now forcibly disable the button while there's content
    # (simulates edge case like rate limiting or backend unavailable)
    page.evaluate("document.querySelector('#chat-submit').disabled = true")
    
    # Immediately press Enter before React can re-enable
    # With button disabled, our handler should preventDefault but NOT click
    # This means native behavior (newline) should NOT occur either (preventDefault blocks it)
    textarea.press("Enter")
    
    # Wait a moment
    page.wait_for_timeout(500)
    
    # Debug: Check state after Enter
    post_enter_state = page.evaluate("""() => {
        const textarea = document.querySelector("textarea");
        return {
            textareaValue: textarea.value
        };
    }""")
    print(f"DEBUG post-enter (disabled) state: {post_enter_state}")
    
    # With button disabled, our handler calls preventDefault but doesn't click
    # Since we preventDefault, native newline behavior is also blocked
    # So textarea should still have just "X" (no submission, no newline)
    textarea_value = textarea.input_value()
    assert textarea_value == "X", \
        f"With disabled button, Enter should be blocked (no submit, no newline). Got: {repr(textarea_value)}"
