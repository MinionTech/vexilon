"""
tests/e2e/test_knowledge_base_drawer.py — E2E: Knowledge Base drawer content smoke

Verifies the Readme drawer opens and renders expected markdown sections/links
from chainlit.md without requiring LLM interaction.
"""

import pytest
from playwright.sync_api import Page, expect

from helpers import open_knowledge_base_drawer

# chainlit.md renders 17 drawer links with Chainlit's text-primary class (see #642).
EXPECTED_DRAWER_PRIMARY_LINKS = 17


def test_knowledge_base_drawer_opens_with_expected_sections(page: Page, app_url: str):
    """Opening Knowledge Base shows the indexed document sections."""
    page.goto(app_url, wait_until="domcontentloaded")

    dialog = open_knowledge_base_drawer(page)

    expect(dialog.get_by_role("heading", name="Knowledge Base")).to_be_visible()
    expect(dialog.get_by_text("Data Privacy (PIPA Compliance)")).to_be_visible()
    expect(dialog.get_by_text("Primary Authority")).to_be_visible()
    expect(dialog.get_by_text("Legislation & Regulations")).to_be_visible()


def test_knowledge_base_drawer_contains_document_links(page: Page, app_url: str):
    """Drawer links point at known public docs from chainlit.md."""
    page.goto(app_url, wait_until="domcontentloaded")

    dialog = open_knowledge_base_drawer(page)

    expect(dialog.get_by_role("link", name="BCGEU 19th Main Agreement")).to_be_visible()
    expect(dialog.get_by_role("link", name="Privacy Policy")).to_be_visible()
    expect(dialog.get_by_role("link", name="BC Labour Relations Code")).to_be_visible()

    primary_links = dialog.locator("a.text-primary")
    assert primary_links.count() == EXPECTED_DRAWER_PRIMARY_LINKS, (
        f"Expected {EXPECTED_DRAWER_PRIMARY_LINKS} indexed document links in drawer"
    )
