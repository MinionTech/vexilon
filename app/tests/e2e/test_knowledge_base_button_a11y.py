"""
tests/e2e/test_knowledge_base_button_a11y.py — E2E: Knowledge Base button a11y

Verifies WCAG 2.5.3 Label in Name: the visible "Knowledge Base" label matches
the button's accessible name (aria-label), not Chainlit's stock "Readme" text.
"""

import os

import pytest
from playwright.sync_api import Page, expect


@pytest.fixture(scope="module")
def app_url():
    """Base URL for the running Chainlit app."""
    return os.getenv("APP_URL", "http://localhost:7860")


def test_knowledge_base_button_accessible_name(page: Page, app_url: str):
    """#readme-button accessible name is 'Knowledge Base', not 'Readme'."""
    page.goto(app_url, wait_until="domcontentloaded")

    readme_button = page.locator("#readme-button")
    readme_button.wait_for(timeout=15000)

    # Custom JS labels the button via setInterval (500ms) and on first paint.
    page.wait_for_function(
        "document.querySelector('#readme-button')?.getAttribute('aria-label') === 'Knowledge Base'",
        timeout=3000,
    )

    expect(readme_button).to_have_accessible_name("Knowledge Base")

    visible_label = page.evaluate(
        """() => getComputedStyle(document.querySelector('#readme-button'), '::after').content"""
    )
    assert "Knowledge Base" in visible_label
