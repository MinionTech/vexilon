"""
tests/e2e/test_goatcounter.py — E2E: optional GoatCounter script loading

CI leaves GOATCOUNTER_CODE unset, so the real-server test expects the script
to be absent. A local app with the code set is also accepted: the test reads
the live /api/goatcounter response and checks the matching page state. The
"set" case stubs that endpoint so the same app can still exercise script
injection. Requests to GoatCounter are aborted so CI never sends hits.
"""

import json

from playwright.sync_api import Page, Route, expect

GOATCOUNTER_SCRIPT = "https://gc.zgo.at/count.js"
TEST_CODE = "vexilon-ci"
TEST_ENDPOINT = f"https://{TEST_CODE}.goatcounter.com/count"


def _is_goatcounter_request(url: str) -> bool:
    lowered = url.lower()
    return "gc.zgo.at" in lowered or "goatcounter.com" in lowered


def _block_goatcounter_hits(page: Page) -> list[str]:
    blocked: list[str] = []

    def handle(route: Route) -> None:
        blocked.append(route.request.url)
        route.abort()

    page.route("**/gc.zgo.at/**", handle)
    page.route("**/*goatcounter.com/**", handle)
    return blocked


def _wait_for_goatcounter_state(page: Page, state: str) -> None:
    page.wait_for_function(
        "document.documentElement.dataset.goatcounter === %s" % json.dumps(state),
        timeout=15000,
    )


def test_goatcounter_script_follows_server_config(page: Page, app_url: str):
    """The script tag follows the real /api/goatcounter response (not stubbed)."""
    blocked = _block_goatcounter_hits(page)
    with page.expect_response("**/api/goatcounter") as resp_info:
        page.goto(app_url, wait_until="domcontentloaded")
    response = resp_info.value
    assert response.status == 200
    payload = response.json()
    assert isinstance(payload, dict)
    assert "goatcounter_code" in payload
    code = payload["goatcounter_code"]

    script = page.locator(f'script[src="{GOATCOUNTER_SCRIPT}"]')
    if code:
        _wait_for_goatcounter_state(page, "on")
        expect(script).to_have_count(1)
        expect(script).to_have_attribute(
            "data-goatcounter",
            f"https://{code}.goatcounter.com/count",
        )
        assert json.loads(script.get_attribute("data-goatcounter-settings") or "{}") == {"allow_frame": True}
        assert all(_is_goatcounter_request(url) for url in blocked)
    else:
        _wait_for_goatcounter_state(page, "off")
        expect(script).to_have_count(0)
        assert blocked == []


def test_goatcounter_script_present_when_set(page: Page, app_url: str):
    """A valid site code injects the official script with the matching count URL."""

    def fulfill_config(route: Route) -> None:
        if _is_goatcounter_request(route.request.url):
            route.abort()
            return
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"goatcounter_code": TEST_CODE}),
        )

    blocked = _block_goatcounter_hits(page)
    page.route("**/api/goatcounter", fulfill_config)
    page.goto(app_url, wait_until="domcontentloaded")
    _wait_for_goatcounter_state(page, "on")

    script = page.locator(f'script[src="{GOATCOUNTER_SCRIPT}"]')
    expect(script).to_have_count(1)
    expect(script).to_have_attribute("data-goatcounter", TEST_ENDPOINT)
    assert json.loads(script.get_attribute("data-goatcounter-settings") or "{}") == {"allow_frame": True}
    assert all(_is_goatcounter_request(url) for url in blocked)
