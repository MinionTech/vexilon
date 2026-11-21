"""
tests/e2e/test_axe_audit.py — E2E: axe-core scan of the main UI surfaces

Runs a pinned axe-core (WCAG 2.1 A/AA) audit against the welcome screen
and settings dialog, and asserts the #401 quick-fix rules stay clean.
"""

from playwright.sync_api import Page, expect

from helpers import (
    AXE_ENFORCED_RULES,
    WCAG_AA_CONTRAST_MIN,
    axe_violations,
    format_axe_violations,
    run_axe,
    set_chainlit_theme,
    wait_for_custom_js,
    wait_for_knowledge_base_button,
)


def test_welcome_screen_axe_enforced_rules_light_and_dark(page: Page, app_url: str):
    """Welcome screen has no enforced-rule axe violations in light or dark theme."""
    page.goto(app_url, wait_until="domcontentloaded")
    wait_for_custom_js(page)
    wait_for_knowledge_base_button(page)

    for theme in ("light", "dark"):
        set_chainlit_theme(page, theme)
        wait_for_knowledge_base_button(page)
        page.wait_for_function(
            """() => {
                const el = document.getElementById('upload-button-loading');
                return !el || el.getAttribute('aria-hidden') === 'true' || el.getAttribute('role') !== 'presentation';
            }""",
            timeout=5000,
        )
        results = run_axe(page, rules=list(AXE_ENFORCED_RULES))
        violations = axe_violations(results, AXE_ENFORCED_RULES)
        assert violations == [], (
            f"{theme} theme axe violations:\n{format_axe_violations(violations)}"
        )


def test_settings_dialog_primary_button_contrast(page: Page, app_url: str):
    """Settings Confirm button meets WCAG AA 4.5:1 contrast in both themes."""
    page.goto(app_url, wait_until="domcontentloaded")
    wait_for_custom_js(page)

    for theme in ("light", "dark"):
        set_chainlit_theme(page, theme)
        page.locator("#chat-settings-open-modal").click()
        confirm = page.locator("#confirm")
        confirm.wait_for(timeout=8000)

        result = page.evaluate(
            """() => {
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
                    return (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
                }
                function parseRgb(rgb) {
                    const m = rgb.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);
                    return m ? [+m[1], +m[2], +m[3]] : null;
                }
                const btn = document.querySelector("#confirm");
                if (!btn) return { error: "Confirm button not found" };
                const span = btn.querySelector("span") || btn;
                const fg = parseRgb(getComputedStyle(span).color);
                const bg = parseRgb(getComputedStyle(btn).backgroundColor);
                if (!fg || !bg) return { error: "could not parse Confirm colours" };
                return { ratio: contrast(fg, bg), fg, bg };
            }"""
        )
        assert "error" not in result, result.get("error")
        assert result["ratio"] >= WCAG_AA_CONTRAST_MIN, (
            f"{theme} Confirm contrast {result['ratio']:.2f} < {WCAG_AA_CONTRAST_MIN} "
            f"(fg={result['fg']}, bg={result['bg']})"
        )
        page.keyboard.press("Escape")


def test_chat_input_has_visible_focus_indicator(page: Page, app_url: str):
    """#chat-input shows a visible focus ring when focused from the keyboard."""
    page.goto(app_url, wait_until="domcontentloaded")
    wait_for_custom_js(page)

    textarea = page.locator("#chat-input")
    textarea.wait_for(timeout=15000)
    textarea.focus()

    shadow = page.evaluate(
        "() => getComputedStyle(document.querySelector('#chat-input')).boxShadow"
    )
    # Custom CSS uses a 2px page-background ring plus a 4px brand-color ring.
    assert "4px" in (shadow or ""), (
        f"Expected a visible 4px focus ring on #chat-input, got boxShadow={shadow}"
    )


def test_welcome_landmarks_and_labels(page: Page, app_url: str):
    """Header/main landmarks and composer labels are present after custom JS."""
    page.goto(app_url, wait_until="domcontentloaded")
    wait_for_custom_js(page)
    wait_for_knowledge_base_button(page)

    header = page.locator("#header")
    expect(header).to_have_attribute("role", "banner")
    expect(page.locator("[role='main']")).to_have_count(1)

    upload = page.locator("#upload-button")
    expect(upload).to_have_accessible_name("Attach file")
    role = upload.get_attribute("role")
    assert role != "presentation", f"#upload-button still has role={role}"

    expect(page.locator("#chat-input")).to_have_attribute("aria-label", "Type your message")
    expect(page.locator("#readme-button")).to_have_accessible_name("Knowledge Base")
    assert "Knowledge Base" in page.locator("#readme-button").inner_text()
