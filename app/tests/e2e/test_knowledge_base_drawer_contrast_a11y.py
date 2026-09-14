"""
tests/e2e/test_knowledge_base_drawer_contrast_a11y.py — E2E: Knowledge Base drawer link contrast

Verifies WCAG 2.1 AA color-contrast (1.4.3): primary links in the Readme drawer
meet 4.5:1 against the dark-theme background (fixes #642).
"""

import pytest
from playwright.sync_api import Page

from helpers import (
    WCAG_AA_CONTRAST_MIN,
    contrast_ratio_js,
    open_knowledge_base_drawer,
)

# Chainlit persists theme preference under this Vite UI key; `.dark` on <html> is the assertion.
CHAINLIT_THEME_STORAGE_KEY = "vite-ui-theme"


def test_knowledge_base_drawer_link_contrast_dark_theme(page: Page, app_url: str):
    """Drawer primary links meet WCAG AA 4.5:1 contrast in dark theme."""
    page.goto(app_url, wait_until="domcontentloaded")

    page.evaluate(f"localStorage.setItem('{CHAINLIT_THEME_STORAGE_KEY}', 'dark')")
    page.reload(wait_until="domcontentloaded")

    open_knowledge_base_drawer(page)

    page.wait_for_function("document.documentElement.classList.contains('dark')", timeout=5000)

    result = page.evaluate(contrast_ratio_js())

    assert "error" not in result, result.get("error")
    assert result["ratio"] >= WCAG_AA_CONTRAST_MIN, (
        f"Contrast ratio {result['ratio']:.2f} < {WCAG_AA_CONTRAST_MIN} "
        f"(fg={result['fg']}, bg={result['bg']})"
    )
