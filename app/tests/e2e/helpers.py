"""Shared helpers for Playwright UI e2e tests."""

import os
from pathlib import Path

from playwright.sync_api import Page, Locator, TimeoutError as PlaywrightTimeoutError, expect

WCAG_AA_CONTRAST_MIN = 4.5


def wait_for_custom_js(page: Page) -> None:
    """Wait until the chat textarea and Enter-to-submit handler are ready."""
    page.wait_for_selector("textarea", timeout=15000)
    page.wait_for_function(
        "document.querySelector('textarea')?.dataset?.listenerAttached === 'true'",
        timeout=15000,
    )


def submit_chat_with_enter(page: Page, textarea: Locator) -> None:
    """Press Enter to submit. Click #chat-submit if the textarea does not clear."""
    textarea.press("Enter")
    try:
        expect(textarea).to_have_value("", timeout=5000)
        return
    except (AssertionError, PlaywrightTimeoutError):
        page.locator("#chat-submit").click()
    expect(textarea).to_have_value("", timeout=30000)


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


AXE_CORE_VERSION = "4.14.1"
AXE_CORE_SHA256 = "29348999450cdf2849a8e70b7912fc2da85a849da39c0d0d5394f51d4a9ed95c"
AXE_CORE_URL = f"https://unpkg.com/axe-core@{AXE_CORE_VERSION}/axe.min.js"
CHAINLIT_THEME_STORAGE_KEY = "vite-ui-theme"


def axe_core_path() -> Path:
    """Return a local axe-core script, downloading it once if needed."""
    import hashlib
    import urllib.request

    cache_dir = Path(os.environ.get("AXE_CACHE_DIR", "/tmp/axe-core"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"axe.min.{AXE_CORE_VERSION}.js"
    if not dest.exists() or hashlib.sha256(dest.read_bytes()).hexdigest() != AXE_CORE_SHA256:
        with urllib.request.urlopen(AXE_CORE_URL, timeout=30) as response:
            dest.write_bytes(response.read())
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if digest != AXE_CORE_SHA256:
        raise AssertionError(
            f"axe-core {AXE_CORE_VERSION} sha256 mismatch: {digest}"
        )
    return dest


def set_chainlit_theme(page: Page, theme: str) -> None:
    """Persist a Chainlit theme and reload so `.dark` / `.light` is applied."""
    page.evaluate(
        f"localStorage.setItem('{CHAINLIT_THEME_STORAGE_KEY}', '{theme}')"
    )
    page.reload(wait_until="domcontentloaded")
    wait_for_custom_js(page)
    if theme == "dark":
        page.wait_for_function(
            "document.documentElement.classList.contains('dark')",
            timeout=5000,
        )


def run_axe(page: Page, context_selector: str | None = None, rules: list[str] | None = None) -> dict:
    """Run axe-core on the page or a context element and return the raw results."""
    page.add_script_tag(path=str(axe_core_path()))
    return page.evaluate(
        """async ([contextSelector, ruleIds]) => {
            const context = contextSelector
                ? document.querySelector(contextSelector)
                : document;
            if (!context) {
                return { error: `context not found: ${contextSelector}` };
            }
            const options = {
                runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"] },
            };
            if (ruleIds && ruleIds.length) {
                options.runOnly = { type: "rule", values: ruleIds };
            }
            return await axe.run(context, options);
        }""",
        [context_selector, list(rules) if rules else None],
    )


def axe_violations(results: dict, rule_ids: tuple[str, ...] | None = None) -> list[dict]:
    """Return axe violations, optionally filtered to selected rule ids."""
    if "error" in results:
        raise AssertionError(results["error"])
    violations = results.get("violations") or []
    if rule_ids is None:
        return violations
    wanted = set(rule_ids)
    return [v for v in violations if v.get("id") in wanted]


def format_axe_violations(violations: list[dict]) -> str:
    """Format axe violations for assertion messages."""
    lines = []
    for v in violations:
        lines.append(f"[{v.get('impact')}] {v.get('id')}: {v.get('help')}")
        for node in (v.get("nodes") or [])[:8]:
            lines.append(f"  {node.get('target')} {node.get('failureSummary', '')[:200]}")
    return "\n".join(lines)
