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

    # Verify dialog override is active: top-anchored without modal centering transform
    dialog_style = dialog.evaluate(
        "el => { const s = getComputedStyle(el); return { top: s.top, left: s.left, transform: s.transform }; }"
    )
    assert dialog_style["top"] == "0px", (
        f"Dialog should be top-anchored (top: 0px), got {dialog_style['top']}"
    )
    assert dialog_style["left"] == "0px", (
        f"Dialog should be left-anchored (left: 0px), got {dialog_style['left']}"
    )
    assert dialog_style["transform"] == "none", (
        f"Dialog should have no centering transform (transform: none), got {dialog_style['transform']}"
    )

    # Verify dialog fills the mobile viewport dimensions
    dialog_box = dialog.bounding_box()
    assert dialog_box is not None, "Dialog bounding box should exist"
    assert abs(dialog_box["width"] - 390) < 1, f"Expected width 390, got {dialog_box['width']}"
    assert abs(dialog_box["height"] - 844) < 1, f"Expected height 844, got {dialog_box['height']}"

    # Verify touch target size meets the 44 CSS pixel target
    initial_box = close_btn.bounding_box()
    assert initial_box is not None, "Close button bounding box should exist"
    assert initial_box["width"] >= 44, f"Expected width >= 44, got {initial_box['width']}"
    assert initial_box["height"] >= 44, f"Expected height >= 44, got {initial_box['height']}"

    # Verify hit target: close button itself (or an element inside it) owns the hit point at its center
    initial_hit = close_btn.evaluate(
        """(btn, pt) => {
            const el = document.elementFromPoint(pt.x, pt.y);
            return el !== null && (el === btn || btn.contains(el));
        }""",
        {"x": initial_box["x"] + initial_box["width"] / 2, "y": initial_box["y"] + initial_box["height"] / 2},
    )
    assert initial_hit, "Expected close button or its child to own the hit target at its center before scroll"

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

    # Verify the scroll container spans the full viewport width so its scrollbar hugs the screen edge
    scroll_box = scroll_container.bounding_box()
    assert scroll_box is not None, "Scroll container bounding box should exist"
    assert abs(scroll_box["x"] + scroll_box["width"] - 390) < 1, (
        f"Scroll container right edge should reach viewport boundary (390px), got {scroll_box['x'] + scroll_box['width']}"
    )

    # Close button must remain inside the mobile viewport and visible
    scrolled_box = close_btn.bounding_box()
    assert scrolled_box is not None, "Close button bounding box should exist after scroll"
    assert scrolled_box["y"] >= 0, f"Close button scrolled off top: y={scrolled_box['y']}"
    assert (
        scrolled_box["y"] + scrolled_box["height"] <= 844
    ), f"Close button below viewport: y={scrolled_box['y']}"
    # Verify close button has 24px gutter clearance from right edge to avoid scrollbar track collision
    assert (
        scrolled_box["x"] + scrolled_box["width"] <= 390 - 24
    ), f"Close button lacks 24px gutter clearance from right edge: right edge={scrolled_box['x'] + scrolled_box['width']}"
    expect(close_btn).to_be_visible()

    # Verify hit target remains unobstructed by any overlay or container after scrolling
    scrolled_hit = close_btn.evaluate(
        """(btn, pt) => {
            const el = document.elementFromPoint(pt.x, pt.y);
            return el !== null && (el === btn || btn.contains(el));
        }""",
        {"x": scrolled_box["x"] + scrolled_box["width"] / 2, "y": scrolled_box["y"] + scrolled_box["height"] / 2},
    )
    assert scrolled_hit, "Expected close button or its child to own the hit target at its center after scroll"

    # Clicking close button successfully dismisses drawer
    close_btn.click()
    expect(dialog).not_to_be_visible()


def test_knowledge_base_drawer_compressed_viewport_address_bar_simulation(
    page: Page, app_url: str
):
    """Simulates mobile dynamic address bar compression (e.g. 720px height on a 390px wide display).

    Verifies that dialog sizing adapts dynamically, remains top-anchored without centering drift,
    and close button remains interactive and dismisses the drawer.
    """
    page.set_viewport_size({"width": 390, "height": 720})
    page.goto(app_url, wait_until="domcontentloaded")

    dialog = open_knowledge_base_drawer(page)
    close_btn = dialog.locator("> button")
    expect(close_btn).to_be_visible()
    page.wait_for_timeout(300)

    # Dialog adapts to compressed viewport
    dialog_box = dialog.bounding_box()
    assert dialog_box is not None, "Dialog bounding box should exist in compressed viewport"
    assert abs(dialog_box["y"]) < 1, f"Dialog should be anchored at y=0, got {dialog_box['y']}"
    assert abs(dialog_box["height"] - 720) < 1, (
        f"Dialog height should match compressed viewport 720px, got {dialog_box['height']}"
    )

    # Verify the scroll container spans the full viewport width
    scroll_container = dialog.locator("> div").first
    scroll_box = scroll_container.bounding_box()
    assert scroll_box is not None, "Scroll container bounding box should exist in compressed viewport"
    assert abs(scroll_box["x"] + scroll_box["width"] - 390) < 1, (
        f"Scroll container right edge should reach viewport boundary (390px), got {scroll_box['x'] + scroll_box['width']}"
    )

    btn_box = close_btn.bounding_box()
    assert btn_box is not None, "Close button bounding box should exist"
    assert btn_box["y"] >= 0, f"Close button should not be clipped at top: y={btn_box['y']}"
    assert btn_box["x"] + btn_box["width"] <= 390 - 24, (
        f"Close button right edge should have 24px gutter clearance, got {btn_box['x'] + btn_box['width']}"
    )

    # Verify close button hit target in compressed viewport
    hit_target = close_btn.evaluate(
        """(btn, pt) => {
            const el = document.elementFromPoint(pt.x, pt.y);
            return el !== null && (el === btn || btn.contains(el));
        }""",
        {"x": btn_box["x"] + btn_box["width"] / 2, "y": btn_box["y"] + btn_box["height"] / 2},
    )
    assert hit_target, "Expected close button or its child to own the hit target in compressed viewport"

    # Dismiss drawer
    close_btn.click()
    expect(dialog).not_to_be_visible()

