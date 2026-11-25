"""
tests/e2e/test_axe_audit.py — E2E: axe-core scan of the main UI surfaces

Runs a pinned axe-core (WCAG 2.1 A/AA) audit against the welcome screen,
settings dialog, Knowledge Base drawer, and post-message conversation view.
"""

from playwright.sync_api import Page, expect

from helpers import (
    WCAG_AA_CONTRAST_MIN,
    axe_violations,
    format_axe_violations,
    open_knowledge_base_drawer,
    run_axe,
    set_chainlit_theme,
    submit_chat_with_enter,
    wait_for_aria_label,
    wait_for_custom_js,
    wait_for_knowledge_base_button,
)


def _wait_for_settled(page: Page, selector: str = "[role='dialog']") -> None:
    """Wait until an element's opacity transition completes before sampling colors."""
    page.wait_for_function(
        """(sel) => {
            const el = document.querySelector(sel);
            if (!el || getComputedStyle(el).opacity !== '1') return false;
            return el.getAnimations().every((a) => a.playState === 'finished');
        }""",
        arg=selector,
        timeout=5000,
    )


def test_welcome_screen_axe_light_and_dark(page: Page, app_url: str):
    """Welcome screen has no WCAG 2.1 A/AA axe violations in light or dark theme."""
    page.goto(app_url, wait_until="domcontentloaded")
    wait_for_custom_js(page)
    wait_for_knowledge_base_button(page)

    for theme in ("light", "dark"):
        set_chainlit_theme(page, theme)
        wait_for_knowledge_base_button(page)
        wait_for_aria_label(page, "#upload-button", "Attach file")
        wait_for_aria_label(page, "#chat-submit", "Send message")
        page.wait_for_function(
            """() => {
                const el = document.getElementById('upload-button-loading');
                return !el || el.getAttribute('aria-hidden') === 'true' || el.getAttribute('role') !== 'presentation';
            }""",
            timeout=5000,
        )
        _wait_for_settled(page, ".welcome-screen")
        results = run_axe(page)
        violations = axe_violations(results)
        assert violations == [], (
            f"{theme} theme axe violations:\n{format_axe_violations(violations)}"
        )


def test_knowledge_base_drawer_axe_light_and_dark(page: Page, app_url: str):
    """Knowledge Base drawer passes WCAG 2.1 A/AA axe in light and dark themes."""
    page.goto(app_url, wait_until="domcontentloaded")
    wait_for_custom_js(page)

    for theme in ("light", "dark"):
        set_chainlit_theme(page, theme)
        open_knowledge_base_drawer(page)
        drawer_sel = '[role="dialog"][data-knowledge-base-drawer]'
        _wait_for_settled(page, drawer_sel)

        assert page.locator(drawer_sel).get_attribute("aria-describedby") is None
        results = run_axe(page, context_selector=drawer_sel)
        violations = axe_violations(results)
        assert violations == [], (
            f"{theme} drawer axe violations:\n{format_axe_violations(violations)}"
        )
        page.keyboard.press("Escape")


def test_conversation_view_axe_light_and_dark(page: Page, app_url: str):
    """Active conversation view passes WCAG 2.1 A/AA axe after sending a message."""
    page.goto(app_url, wait_until="domcontentloaded")
    wait_for_custom_js(page)

    for theme in ("light", "dark"):
        set_chainlit_theme(page, theme)
        wait_for_knowledge_base_button(page)
        textarea = page.locator("#chat-input")
        textarea.press_sequentially("What is a steward?", delay=10)
        page.wait_for_selector("#chat-submit:not([disabled])", timeout=15000)
        submit_chat_with_enter(page, textarea)

        page.wait_for_function(
            """() => {
                if (document.getElementById('stop-button')) return false;
                const bubbles = document.querySelectorAll(
                    '.message, [class*="message"]:not(.message-list):not([class*="actions"])'
                );
                return Array.from(bubbles).some(
                    (m) => m.textContent.trim().length > 0 && m.textContent.trim() !== 'What is a steward?'
                );
            }""",
            timeout=120000,
        )
        wait_for_aria_label(page, "#chat-submit", "Send message")
        wait_for_aria_label(page, "#upload-button", "Attach file")
        results = run_axe(page)
        violations = axe_violations(results)
        assert violations == [], (
            f"{theme} conversation axe violations:\n{format_axe_violations(violations)}"
        )


def test_chat_input_has_visible_focus_indicator(page: Page, app_url: str):
    """#message-composer shows a rounded focus ring in both themes, plus forced-colors outline."""
    page.goto(app_url, wait_until="domcontentloaded")
    wait_for_custom_js(page)

    expected_ring_rgb = {
        "light": "rgb(115, 115, 115)",
        "dark": "rgb(138, 138, 138)",
    }
    for theme, ring_rgb in expected_ring_rgb.items():
        set_chainlit_theme(page, theme)
        textarea = page.locator("#chat-input")
        textarea.wait_for(timeout=15000)
        textarea.focus()

        styles = page.evaluate(
            """() => {
                const composer = getComputedStyle(document.querySelector('#message-composer'));
                const input = getComputedStyle(document.querySelector('#chat-input'));
                return {
                    composerShadow: composer.boxShadow,
                    inputShadow: input.boxShadow,
                    inputOutline: input.outlineStyle,
                };
            }"""
        )
        # Custom CSS uses a 2px page-background gap plus a 4px neutral ring on the pill container.
        assert "4px" in (styles["composerShadow"] or ""), (
            f"{theme}: expected a visible 4px focus ring on #message-composer, got boxShadow={styles['composerShadow']}"
        )
        assert ring_rgb in (styles["composerShadow"] or ""), (
            f"{theme}: expected {ring_rgb} in #message-composer boxShadow, got {styles['composerShadow']}"
        )
        assert styles["inputShadow"] == "none", (
            f"{theme}: expected no inner focus box-shadow on #chat-input, got {styles['inputShadow']}"
        )
        assert styles["inputOutline"] == "none", (
            f"{theme}: expected no inner outline on #chat-input, got {styles['inputOutline']}"
        )

    page.emulate_media(forced_colors="active")
    forced = page.evaluate(
        """() => {
            const s = getComputedStyle(document.querySelector('#message-composer'));
            return { outlineStyle: s.outlineStyle, outlineWidth: s.outlineWidth, boxShadow: s.boxShadow };
        }"""
    )
    assert forced == {"outlineStyle": "solid", "outlineWidth": "3px", "boxShadow": "none"}


def test_welcome_landmarks_and_labels(page: Page, app_url: str):
    """Header/main landmarks and composer labels are present after custom JS."""
    page.goto(app_url, wait_until="domcontentloaded")
    wait_for_custom_js(page)
    wait_for_knowledge_base_button(page)

    header = page.locator("#header")
    expect(header).to_have_attribute("role", "banner")
    main = page.locator("[role='main']")
    expect(main).to_have_count(1)

    skip = page.locator("#skip-to-main")
    expect(skip).to_have_text("Skip to main content")
    skip.focus()
    skip.press("Enter")
    assert page.evaluate("() => document.activeElement?.getAttribute('role') === 'main'")

    hidden_logo = page.evaluate(
        """() => {
            const img = document.querySelector('.welcome-screen img');
            if (!img) return null;
            return {
                display: getComputedStyle(img).display,
                alt: img.getAttribute('alt'),
                ariaHidden: img.getAttribute('aria-hidden'),
            };
        }"""
    )
    if hidden_logo is not None:
        assert hidden_logo == {"display": "none", "alt": "", "ariaHidden": "true"}

    upload = page.locator("#upload-button")
    expect(upload).to_have_accessible_name("Attach file")
    role = upload.get_attribute("role")
    assert role != "presentation", f"#upload-button still has role={role}"

    expect(page.locator("#chat-input")).to_have_attribute("aria-label", "Type your message")
    expect(page.locator("#readme-button")).to_have_accessible_name("Knowledge Base")
    assert "Knowledge Base" in page.locator("#readme-button").inner_text()


def test_upload_loading_transitions_expose_interactive_controls(page: Page, app_url: str):
    """Non-interactive upload loader is hidden; interactive controls clear aria-hidden."""
    page.goto(app_url, wait_until="domcontentloaded")
    wait_for_custom_js(page)

    page.evaluate(
        """() => {
            const loader = document.createElement('div');
            loader.id = 'upload-button-loading';
            loader.setAttribute('role', 'presentation');
            document.body.appendChild(loader);
        }"""
    )
    page.wait_for_function(
        """() => {
            const el = document.getElementById('upload-button-loading');
            return el && el.getAttribute('aria-hidden') === 'true' && !el.hasAttribute('role');
        }""",
        timeout=5000,
    )

    page.evaluate(
        """() => {
            const loader = document.getElementById('upload-button-loading');
            loader.appendChild(document.createElement('button'));
        }"""
    )
    page.wait_for_function(
        """() => {
            const el = document.getElementById('upload-button-loading');
            const btn = el && el.querySelector('button');
            return el && !el.hasAttribute('aria-hidden') && btn && btn.getAttribute('aria-label') === 'Uploading';
        }""",
        timeout=5000,
    )
