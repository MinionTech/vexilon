"""
tests/e2e/test_chat_profiles_regression.py — E2E: welcome-screen chat profiles regression

Guards against Chainlit dev reload leaving /project/settings with empty chatProfiles
and hiding #chat-profiles plus starter chips on the welcome screen.
"""

from playwright.sync_api import Page, Playwright, expect

EXPECTED_PERSONAS = ("Lookup", "Grieve", "Manage")


def test_project_settings_exposes_registered_chat_profiles(
    playwright: Playwright, app_url: str
) -> None:
    """Running app must serve Lookup, Grieve, and Manage from /project/settings."""
    request_context = playwright.request.new_context()
    try:
        response = request_context.get(f"{app_url}/project/settings")
        assert response.ok, (
            f"GET /project/settings failed: {response.status} {response.text()}"
        )
        data = response.json()
        profiles = data.get("chatProfiles")
        assert profiles, (
            "chatProfiles is empty; the welcome screen will hide #chat-profiles "
            "(Chainlit watch reload regression, 2026-11-01)"
        )
        names = [profile["name"] for profile in profiles]
        assert names == list(EXPECTED_PERSONAS), names
        for profile in profiles:
            starters = profile.get("starters") or []
            assert starters, (
                f"Profile {profile['name']!r} must register starters on the welcome screen"
            )
        default_names = [profile["name"] for profile in profiles if profile.get("default")]
        assert default_names == ["Lookup"]
    finally:
        request_context.dispose()


def test_welcome_screen_chat_profiles_control_visible(page: Page, app_url: str) -> None:
    """Welcome screen must render the persona picker (#chat-profiles)."""
    page.goto(app_url, wait_until="domcontentloaded")
    profiles_control = page.locator("#chat-profiles")
    profiles_control.wait_for(state="visible", timeout=15000)
    expect(profiles_control).to_be_visible()


def test_welcome_screen_chat_profiles_lists_personas(page: Page, app_url: str) -> None:
    """Persona picker must offer Lookup, Grieve, and Manage by name."""
    page.goto(app_url, wait_until="domcontentloaded")
    profiles_control = page.locator("#chat-profiles")
    profiles_control.wait_for(state="visible", timeout=15000)
    profiles_control.click()
    for name in EXPECTED_PERSONAS:
        expect(page.get_by_role("option", name=name)).to_be_visible()
