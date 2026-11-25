"""
tests/e2e/test_chat_profiles_regression.py — E2E: welcome-screen starters & no persona switcher

Verifies that the persona switcher (#chat-profiles) is removed and that
welcome-screen starter chips are served and visible.
"""

from playwright.sync_api import Page, Playwright, expect


def test_project_settings_omits_chat_profiles(
    playwright: Playwright, app_url: str
) -> None:
    """Running app must not expose a persona switcher in /project/settings."""
    request_context = playwright.request.new_context()
    try:
        response = request_context.get(f"{app_url}/project/settings")
        assert response.ok, (
            f"GET /project/settings failed: {response.status} {response.text()}"
        )
        data = response.json()
        assert not data.get("chatProfiles")
    finally:
        request_context.dispose()


def test_welcome_screen_hides_chat_profiles_and_renders_starters(page: Page, app_url: str) -> None:
    """Welcome screen must render starter chips without a persona picker."""
    page.goto(app_url, wait_until="domcontentloaded")
    starter = page.locator("#starter-grievance-builder")
    starter.wait_for(state="visible", timeout=15000)
    expect(starter).to_be_visible()
    expect(page.locator("#chat-profiles")).to_have_count(0)
