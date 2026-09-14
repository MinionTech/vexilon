"""
tests/e2e/test_knowledge_base_drawer_contrast_a11y.py — E2E: Knowledge Base drawer link contrast

Verifies WCAG 2.1 AA color-contrast (1.4.3): primary links in the Readme drawer
meet 4.5:1 against the dark-theme background (fixes #642).
"""

import os

import pytest
from playwright.sync_api import Page


@pytest.fixture(scope="module")
def app_url():
    """Base URL for the running Chainlit app."""
    return os.getenv("APP_URL", "http://localhost:7860")


def _contrast_ratio_js() -> str:
    """Return a JS function body that computes WCAG contrast for the first drawer link."""
    return """
    () => {
        function luminance(r, g, b) {
            const [rs, gs, bs] = [r, g, b].map((c) => {
                c = c / 255;
                return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
            });
            return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
        }
        function contrast(fg, bg) {
            const l1 = luminance(...fg);
            const l2 = luminance(...bg);
            const lighter = Math.max(l1, l2);
            const darker = Math.min(l1, l2);
            return (lighter + 0.05) / (darker + 0.05);
        }
        function parseRgba(rgb) {
            const m = rgb.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)(?:,\\s*([\\d.]+))?\\)/);
            if (!m) return null;
            return {
                r: +m[1],
                g: +m[2],
                b: +m[3],
                a: m[4] !== undefined ? parseFloat(m[4]) : 1,
            };
        }
        function opaqueBackground(el) {
            let node = el;
            while (node) {
                const rgba = parseRgba(getComputedStyle(node).backgroundColor);
                if (rgba && rgba.a > 0) {
                    return [rgba.r, rgba.g, rgba.b];
                }
                node = node.parentElement;
            }
            return null;
        }

        const dialog = document.querySelector(
            '[role="dialog"][data-knowledge-base-drawer]'
        );
        if (!dialog) return { error: "Knowledge Base drawer not found" };

        const link = dialog.querySelector("a.text-primary");
        if (!link) return { error: "no drawer link found" };

        const fgRgba = parseRgba(getComputedStyle(link).color);
        if (!fgRgba) return { error: "could not parse link color" };
        const fg = [fgRgba.r, fgRgba.g, fgRgba.b];

        const bg = opaqueBackground(link.parentElement) || [33, 33, 33];

        return {
            ratio: contrast(fg, bg),
            fg: getComputedStyle(link).color,
            bg: `rgb(${bg.join(",")})`,
        };
    }
    """


def test_knowledge_base_drawer_link_contrast_dark_theme(page: Page, app_url: str):
    """Drawer primary links meet WCAG AA 4.5:1 contrast in dark theme."""
    page.goto(app_url, wait_until="domcontentloaded")

    page.evaluate("localStorage.setItem('vite-ui-theme', 'dark')")
    page.reload(wait_until="domcontentloaded")

    readme_button = page.locator("#readme-button")
    readme_button.wait_for(timeout=15000)
    readme_button.click()

    dialog = page.locator('[role="dialog"][data-knowledge-base-drawer]')
    dialog.wait_for(timeout=5000)
    dialog.locator("a.text-primary").first.wait_for(timeout=5000)

    page.wait_for_function("document.documentElement.classList.contains('dark')", timeout=5000)

    result = page.evaluate(_contrast_ratio_js())

    assert "error" not in result, result.get("error")
    assert result["ratio"] >= 4.5, (
        f"Contrast ratio {result['ratio']:.2f} < 4.5 "
        f"(fg={result['fg']}, bg={result['bg']})"
    )
