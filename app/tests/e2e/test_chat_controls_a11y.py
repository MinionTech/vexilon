"""
tests/e2e/test_chat_controls_a11y.py — E2E: core chat control accessible names

Verifies WCAG 4.1.2: icon-only Chainlit composer controls receive aria-label
via public/index.js so axe button-name passes and screen readers announce purpose.
"""

import os

import pytest
from playwright.sync_api import Page, expect

WELCOME_SCREEN_CONTROLS = {
    "#chat-profiles": "Choose persona",
    "#upload-button": "Attach file",
    "#chat-settings-open-modal": "Open chat settings",
    "#chat-submit": "Send message",
}


@pytest.fixture(scope="module")
def app_url():
    """Base URL for the running Chainlit app."""
    return os.getenv("APP_URL", "http://localhost:7860")


def _wait_for_aria_label(page: Page, selector: str, label: str) -> None:
    """Wait until custom JS has applied the expected aria-label."""
    page.wait_for_function(
        f"document.querySelector('{selector}')?.getAttribute('aria-label') === '{label}'",
        timeout=5000,
    )


def test_welcome_screen_chat_controls_accessible_names(page: Page, app_url: str):
    """Welcome-screen composer controls have descriptive accessible names."""
    page.goto(app_url, wait_until="domcontentloaded")

    for selector, expected_name in WELCOME_SCREEN_CONTROLS.items():
        page.locator(selector).wait_for(timeout=15000)
        _wait_for_aria_label(page, selector, expected_name)
        expect(page.locator(selector)).to_have_accessible_name(expected_name)


def test_stop_button_accessible_name_during_generation(page: Page, app_url: str):
    """Stop control receives an accessible name while a response is streaming."""
    page.goto(app_url, wait_until="domcontentloaded")

    page.wait_for_selector("textarea", timeout=15000)
    page.wait_for_selector("#chat-submit", timeout=5000)
    page.wait_for_function(
        "document.querySelector('textarea')?.dataset?.listenerAttached === 'true'",
        timeout=3000,
    )

    textarea = page.locator("textarea")
    textarea.press_sequentially("What is the nexus test?", delay=10)
    textarea.press("Enter")

    stop_button = page.locator("#stop-button")
    stop_button.wait_for(state="visible", timeout=30000)

    _wait_for_aria_label(page, "#stop-button", "Stop generation")
    expect(stop_button).to_have_accessible_name("Stop generation")
