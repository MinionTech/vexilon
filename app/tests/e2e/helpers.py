"""Shared helpers for Playwright UI e2e tests."""

from playwright.sync_api import Page, Locator

WCAG_AA_CONTRAST_MIN = 4.5


def wait_for_custom_js(page: Page) -> None:
    """Wait until the chat textarea and Enter-to-submit handler are ready."""
    page.wait_for_selector("textarea", timeout=15000)
    page.wait_for_function(
        "document.querySelector('textarea')?.dataset?.listenerAttached === 'true'",
        timeout=3000,
    )


def wait_for_aria_label(page: Page, selector: str, label: str) -> None:
    """Wait until custom JS has applied the expected aria-label."""
    page.wait_for_function(
        f"document.querySelector('{selector}')?.getAttribute('aria-label') === '{label}'",
        timeout=5000,
    )


def wait_for_knowledge_base_button(page: Page) -> Locator:
    """Wait until the Knowledge Base button is labeled and return its locator."""
    button = page.locator("#readme-button")
    button.wait_for(timeout=15000)
    wait_for_aria_label(page, "#readme-button", "Knowledge Base")
    return button


def open_knowledge_base_drawer(page: Page) -> Locator:
    """Open the Knowledge Base drawer and return the dialog locator."""
    wait_for_knowledge_base_button(page).click()
    dialog = page.locator('[role="dialog"][data-knowledge-base-drawer]')
    dialog.wait_for(timeout=5000)
    dialog.locator("a.text-primary").first.wait_for(timeout=5000)
    return dialog


def contrast_ratio_js() -> str:
    """Return a JS function body that computes the lowest WCAG contrast among drawer links."""
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

        const links = dialog.querySelectorAll("a.text-primary");
        if (!links.length) return { error: "no drawer link found" };

        let worst = null;
        for (const link of links) {
            const fgRgba = parseRgba(getComputedStyle(link).color);
            if (!fgRgba) return { error: "could not parse link color" };
            const fg = [fgRgba.r, fgRgba.g, fgRgba.b];
            const bg = opaqueBackground(link);
            if (!bg) return { error: "could not resolve opaque background for link" };

            const ratio = contrast(fg, bg);
            if (!worst || ratio < worst.ratio) {
                worst = {
                    ratio,
                    fg: getComputedStyle(link).color,
                    bg: `rgb(${bg.join(",")})`,
                    text: link.textContent.trim(),
                };
            }
        }

        return worst;
    }
    """
