"""
tests/e2e/test_chat_controls_a11y.py — E2E: core chat control accessible names

Verifies WCAG 4.1.2: icon-only Chainlit composer controls receive aria-label
via public/index.js so axe button-name passes and screen readers announce purpose.
"""

import pytest
from playwright.sync_api import Page, expect

from helpers import wait_for_aria_label, wait_for_custom_js

WELCOME_SCREEN_CONTROLS = {
    "#chat-profiles": "Choose persona",
    "#upload-button": "Attach file",
    "#chat-settings-open-modal": "Open chat settings",
    "#chat-submit": "Send message",
}


def test_welcome_screen_chat_controls_accessible_names(page: Page, app_url: str):
    """Welcome-screen composer controls have descriptive accessible names."""
    page.goto(app_url, wait_until="domcontentloaded")

    for selector, expected_name in WELCOME_SCREEN_CONTROLS.items():
        page.locator(selector).wait_for(timeout=15000)
        wait_for_aria_label(page, selector, expected_name)
        expect(page.locator(selector)).to_have_accessible_name(expected_name)


def test_stop_button_accessible_name_during_generation(page: Page, app_url: str):
    """Stop control receives an accessible name while a response is streaming."""
    page.goto(app_url, wait_until="domcontentloaded")
    wait_for_custom_js(page)

    textarea = page.locator("textarea")
    textarea.press_sequentially("What is the nexus test?", delay=10)
    textarea.press("Enter")

    stop_button = page.locator("#stop-button")
    stop_button.wait_for(state="visible", timeout=30000)

    wait_for_aria_label(page, "#stop-button", "Stop generation")
    expect(stop_button).to_have_accessible_name("Stop generation")
