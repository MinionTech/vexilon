"""
tests/e2e/test_knowledge_base_drawer.py — E2E: Knowledge Base drawer content smoke

Verifies the Readme drawer opens and renders expected markdown sections/links
from chainlit.md without requiring LLM interaction.
"""

import pytest
from playwright.sync_api import Page, expect

from helpers import open_knowledge_base_drawer

# chainlit.md renders 19 drawer links with Chainlit's text-primary class.
EXPECTED_DRAWER_PRIMARY_LINKS = 19


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


def test_knowledge_base_drawer_close_button_remains_visible_on_mobile_scroll(
    page: Page, app_url: str
):
    """Close button stays pinned, visible, and interactive when drawer content scrolls on mobile."""
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(app_url, wait_until="domcontentloaded")

    dialog = open_knowledge_base_drawer(page)
    close_btn = dialog.locator("> button")
    expect(close_btn).to_be_visible()
    # Allow 200ms dialog entrance animation to settle before measuring geometry
    page.wait_for_timeout(300)

    # Verify dialog is anchored to the top of the viewport (no negative offset from modal centering)
    dialog_box = dialog.bounding_box()
    assert dialog_box is not None, "Dialog bounding box should exist"
    assert (
        abs(dialog_box["y"]) < 1
    ), f"Dialog should be anchored at viewport top (y=0), got y={dialog_box['y']}"
    assert (
        abs(dialog_box["x"]) < 1
    ), f"Dialog should be anchored at viewport left (x=0), got x={dialog_box['x']}"

    # Verify touch target size meets the 44 CSS pixel target
    initial_box = close_btn.bounding_box()
    assert initial_box is not None, "Close button bounding box should exist"
    assert initial_box["width"] >= 44, f"Expected width >= 44, got {initial_box['width']}"
    assert initial_box["height"] >= 44, f"Expected height >= 44, got {initial_box['height']}"

    # Verify close button is aligned with the Knowledge Base heading line (not pushed above it)
    kb_heading = dialog.get_by_role("heading", name="Knowledge Base")
    expect(kb_heading).to_be_visible()
    heading_box = kb_heading.bounding_box()
    assert heading_box is not None, "Knowledge Base heading bounding box should exist"
    btn_center_y = initial_box["y"] + initial_box["height"] / 2
    heading_center_y = heading_box["y"] + heading_box["height"] / 2
    assert abs(btn_center_y - heading_center_y) < 15, (
        f"Close button (center {btn_center_y}) should vertically align with "
        f"Knowledge Base title (center {heading_center_y})"
    )

    # Scroll content down substantially and verify the inner container scrolled
    scroll_container = dialog.locator("> div").first
    scroll_top = scroll_container.evaluate(
        "el => { el.scrollTop = 400; return el.scrollTop; }"
    )
    assert scroll_top > 0, f"Inner content container must be scrollable and have moved, got {scroll_top}"

    # Close button must remain inside the mobile viewport and visible
    scrolled_box = close_btn.bounding_box()
    assert scrolled_box is not None, "Close button bounding box should exist after scroll"
    assert scrolled_box["y"] >= 0, f"Close button scrolled off top: y={scrolled_box['y']}"
    assert (
        scrolled_box["y"] + scrolled_box["height"] <= 844
    ), f"Close button below viewport: y={scrolled_box['y']}"
    assert (
        scrolled_box["x"] + scrolled_box["width"] <= 390
    ), f"Close button past right viewport edge: x={scrolled_box['x']}"
    expect(close_btn).to_be_visible()

    # Clicking close button successfully dismisses drawer
    close_btn.click()
    expect(dialog).not_to_be_visible()
