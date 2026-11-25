"""
tests/e2e/test_goatcounter.py — E2E: optional GoatCounter loading & feature events

CI leaves GOATCOUNTER_CODE unset, so the real-server test expects the script
to be absent. A local app with the code set is also accepted: the test reads
the live /api/goatcounter response and checks the matching page state. The
"set" cases stub /api/goatcounter so the same app can exercise script injection
and PIPA-safe UI feature-usage events. Requests to GoatCounter are aborted so
CI never sends hits.
"""

import json

from playwright.sync_api import Page, Route, expect

from helpers import open_knowledge_base_drawer, submit_chat_with_enter, wait_for_custom_js

GOATCOUNTER_SCRIPT = "https://gc.zgo.at/count.js"
TEST_CODE = "vexilon-ci"
TEST_ENDPOINT = f"https://{TEST_CODE}.goatcounter.com/count"

ALLOWED_EVENT_PATHS = {
    "event/chat-submit",
    "event/persona-lookup",
    "event/persona-grieve",
    "event/knowledge-base-open",
    "event/session-save",
    "event/error-ui",
}


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


def _stub_goatcounter_config(page: Page, code: str | None) -> None:
    def fulfill_config(route: Route) -> None:
        if _is_goatcounter_request(route.request.url):
            route.abort()
            return
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"goatcounter_code": code}),
        )

    page.route("**/api/goatcounter", fulfill_config)


def _install_goatcounter_spy(page: Page, *, throw_on_count: bool = False) -> None:
    if throw_on_count:
        page.add_init_script(
            """
            window.__gcEvents = [];
            window.goatcounter = {
                count() {
                    throw new Error("Simulated analytics failure");
                },
            };
            """
        )
    else:
        page.add_init_script(
            """
            window.__gcEvents = [];
            window.goatcounter = {
                count(payload) {
                    window.__gcEvents.push(payload);
                },
            };
            """
        )


def _wait_for_goatcounter_state(page: Page, state: str) -> None:
    page.wait_for_function(
        "document.documentElement.dataset.goatcounter === %s" % json.dumps(state),
        timeout=15000,
    )


def _recorded_events(page: Page) -> list[dict]:
    return page.evaluate("() => window.__gcEvents || []")


def _wait_for_event_path(page: Page, event_path: str) -> None:
    page.wait_for_function(
        "(p) => (window.__gcEvents || []).some((e) => e && e.path === p)",
        arg=event_path,
        timeout=10000,
    )


def _assert_pipa_safe_events(
    events: list[dict], forbidden_substrings: tuple[str, ...] = ()
) -> None:
    assert events, "Expected at least one recorded GoatCounter event"
    for ev in events:
        assert set(ev.keys()) == {"path", "title", "referrer", "event"}, ev
        assert ev["event"] is True, ev
        assert ev["referrer"] == "", ev
        assert ev["path"] in ALLOWED_EVENT_PATHS, ev
        assert isinstance(ev["title"], str) and ev["title"], ev
        combined = f"{ev['path']} {ev['title']}".lower()
        for forbidden in forbidden_substrings:
            assert forbidden.lower() not in combined, (
                f"Forbidden substring {forbidden!r} leaked into event {ev!r}"
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
    blocked = _block_goatcounter_hits(page)
    _stub_goatcounter_config(page, TEST_CODE)
    page.goto(app_url, wait_until="domcontentloaded")
    _wait_for_goatcounter_state(page, "on")

    script = page.locator(f'script[src="{GOATCOUNTER_SCRIPT}"]')
    expect(script).to_have_count(1)
    expect(script).to_have_attribute("data-goatcounter", TEST_ENDPOINT)
    assert json.loads(script.get_attribute("data-goatcounter-settings") or "{}") == {"allow_frame": True}
    assert all(_is_goatcounter_request(url) for url in blocked)


def test_goatcounter_feature_events_knowledge_base_and_personas(page: Page, app_url: str):
    """Knowledge Base open and persona switches emit allowlisted PIPA-safe events."""
    blocked = _block_goatcounter_hits(page)
    _stub_goatcounter_config(page, TEST_CODE)
    _install_goatcounter_spy(page)

    page.goto(app_url, wait_until="domcontentloaded")
    _wait_for_goatcounter_state(page, "on")
    wait_for_custom_js(page)
    page.locator("#starter-grievance-builder").wait_for(state="visible", timeout=15000)

    assert _recorded_events(page) == []

    # Knowledge Base open fires event/knowledge-base-open without recording doc names.
    dialog = open_knowledge_base_drawer(page)
    _wait_for_event_path(page, "event/knowledge-base-open")
    first_doc_text = (dialog.locator("a.text-primary").first.text_content() or "").strip()
    assert first_doc_text
    dialog.locator("> button").click()
    expect(dialog).not_to_be_visible()

    profile_events = _recorded_events(page)
    _assert_pipa_safe_events(profile_events, forbidden_substrings=(first_doc_text, ".pdf", ".md"))
    profile_paths = [ev["path"] for ev in profile_events]
    assert "event/knowledge-base-open" in profile_paths
    assert all(_is_goatcounter_request(url) for url in blocked)


def test_goatcounter_chat_submit_and_session_save_events(page: Page, app_url: str):
    """Chat submit and Save Session emit allowlisted events without query text or filenames."""
    blocked = _block_goatcounter_hits(page)
    _stub_goatcounter_config(page, TEST_CODE)
    _install_goatcounter_spy(page)

    page.goto(app_url, wait_until="domcontentloaded")
    _wait_for_goatcounter_state(page, "on")
    wait_for_custom_js(page)

    textarea = page.locator("textarea")
    textarea.press("Enter")
    textarea.type("Line 1")
    textarea.press("Shift+Enter")
    page.wait_for_timeout(300)
    assert _recorded_events(page) == []

    textarea.fill("")
    secret_query = "CONFIDENTIAL_STEWARD_QUERY_98765"
    textarea.press_sequentially(secret_query, delay=10)
    page.wait_for_selector("#chat-submit:not([disabled])", timeout=15000)
    submit_chat_with_enter(page, textarea)

    _wait_for_event_path(page, "event/chat-submit")
    _wait_for_event_path(page, "event/persona-lookup")

    save_btn = page.get_by_role("button", name="💾 Save Session")
    save_btn.wait_for(state="visible", timeout=120000)
    save_btn.click()
    _wait_for_event_path(page, "event/session-save")

    events = _recorded_events(page)
    paths = [ev["path"] for ev in events]
    assert "event/chat-submit" in paths
    assert "event/persona-lookup" in paths
    assert "event/session-save" in paths
    _assert_pipa_safe_events(
        events,
        forbidden_substrings=(secret_query, "CONFIDENTIAL", "conversation_", ".md"),
    )
    assert all(_is_goatcounter_request(url) for url in blocked)


def test_goatcounter_starter_and_error_toast_events(page: Page, app_url: str):
    """Mode-changing starter updates active persona; error toast emits new event/error-ui without text."""
    blocked = _block_goatcounter_hits(page)
    _stub_goatcounter_config(page, TEST_CODE)
    _install_goatcounter_spy(page)

    page.goto(app_url, wait_until="domcontentloaded")
    _wait_for_goatcounter_state(page, "on")
    wait_for_custom_js(page)

    # Grievance Builder starter runs inside Lookup mode.
    starter = page.locator("#starter-grievance-builder")
    starter.wait_for(state="visible", timeout=15000)
    starter.click()
    _wait_for_event_path(page, "event/chat-submit")
    _wait_for_event_path(page, "event/persona-lookup")

    # Wait for informational reply, then verify a normal submit stays in Lookup.
    page.get_by_text("Grievance Builder", exact=False).first.wait_for(state="visible", timeout=15000)
    page.locator("textarea").wait_for(state="visible", timeout=15000)
    page.evaluate("() => { window.__gcEvents = []; }")
    textarea = page.locator("textarea")
    textarea.press_sequentially("Follow up grievance question", delay=10)
    page.wait_for_selector("#chat-submit:not([disabled])", timeout=15000)
    submit_chat_with_enter(page, textarea)
    _wait_for_event_path(page, "event/chat-submit")
    _wait_for_event_path(page, "event/persona-lookup")

    stop_button = page.locator("#stop-button")
    try:
        stop_button.wait_for(state="visible", timeout=5000)
        stop_button.click()
        stop_button.wait_for(state="hidden", timeout=10000)
    except Exception:
        pass

    followup_events = _recorded_events(page)
    assert [ev["path"] for ev in followup_events] == [
        "event/chat-submit",
        "event/persona-lookup",
    ]

    # Require a strictly new error-ui event after injecting the sensitive toast.
    before_error_count = sum(1 for ev in followup_events if ev["path"] == "event/error-ui")
    page.evaluate(
        """() => {
            const toast = document.createElement("li");
            toast.setAttribute("data-sonner-toast", "");
            toast.setAttribute("data-type", "error");
            toast.textContent = "SECRET_ERROR_DETAIL_/tmp/grievance_123.pdf";
            document.body.appendChild(toast);
        }"""
    )
    page.wait_for_function(
        "(n) => (window.__gcEvents || []).filter((e) => e && e.path === 'event/error-ui').length > n",
        arg=before_error_count,
        timeout=10000,
    )

    events = _recorded_events(page)
    assert any(ev["path"] == "event/error-ui" for ev in events)
    _assert_pipa_safe_events(
        events,
        forbidden_substrings=(
            "Grievance Builder",
            "Follow up grievance question",
            "SECRET_ERROR_DETAIL",
            "grievance_123.pdf",
        ),
    )
    assert all(_is_goatcounter_request(url) for url in blocked)


def test_goatcounter_feature_events_omit_url_query(page: Page, app_url: str):
    """Feature events omit page URL query (`q`) and referrer (`r`) even when count.js get_data runs."""
    blocked = _block_goatcounter_hits(page)
    _stub_goatcounter_config(page, TEST_CODE)
    page.add_init_script(
        f"""
        window.goatcounter = {{
            get_data(vars) {{
                vars = vars || {{}};
                const r = vars.referrer === undefined ? "https://example.test/?q=SECRET_REFERRER" : vars.referrer;
                return {{
                    p: vars.path,
                    t: vars.title,
                    r: r,
                    e: !!vars.event,
                    q: location.search,
                }};
            }},
            url(vars) {{
                const data = window.goatcounter.get_data(vars || {{}});
                const params = new URLSearchParams();
                for (const [k, v] of Object.entries(data)) {{
                    if (v !== "" && v !== null && v !== undefined && v !== false) {{
                        params.set(k, String(v));
                    }}
                }}
                return "{TEST_ENDPOINT}?" + params.toString();
            }},
            count(vars) {{
                const hitUrl = window.goatcounter.url(vars);
                const img = new Image();
                img.src = hitUrl;
            }},
        }};
        """
    )

    page.goto(f"{app_url}/?q=CONFIDENTIAL_URL_QUERY_999", wait_until="domcontentloaded")
    _wait_for_goatcounter_state(page, "on")
    wait_for_custom_js(page)

    with page.expect_request("**/*goatcounter.com/count*") as req_info:
        dialog = open_knowledge_base_drawer(page)
    expect(dialog).to_be_visible()

    hit_url = req_info.value.url
    assert "p=event%2Fknowledge-base-open" in hit_url, hit_url
    assert "q=" not in hit_url, hit_url
    assert "r=" not in hit_url, hit_url
    assert "CONFIDENTIAL_URL_QUERY_999" not in hit_url, hit_url
    assert "SECRET_REFERRER" not in hit_url, hit_url
    assert any("/count?" in url for url in blocked)


def test_goatcounter_error_toast_retries_until_count_callable(page: Page, app_url: str):
    """An error toast shown before count.js finishes loading is retried once count() becomes callable."""
    _block_goatcounter_hits(page)
    _stub_goatcounter_config(page, TEST_CODE)
    page.add_init_script("window.__gcEvents = [];")

    page.goto(app_url, wait_until="domcontentloaded")
    _wait_for_goatcounter_state(page, "on")
    wait_for_custom_js(page)

    # Toast appears while window.goatcounter.count is still absent.
    page.evaluate(
        """() => {
            const toast = document.createElement("li");
            toast.setAttribute("data-sonner-toast", "");
            toast.setAttribute("data-type", "error");
            toast.textContent = "Early startup error";
            document.body.appendChild(toast);
        }"""
    )
    page.wait_for_timeout(600)
    assert page.evaluate("() => document.querySelector('[data-sonner-toast]')?.dataset?.gcTracked") is None

    # Once count() becomes callable, the next scan counts the toast and marks it tracked.
    page.evaluate(
        """() => {
            window.goatcounter = {
                count(payload) {
                    window.__gcEvents.push(payload);
                },
            };
        }"""
    )
    _wait_for_event_path(page, "event/error-ui")
    assert page.evaluate("() => document.querySelector('[data-sonner-toast]')?.dataset?.gcTracked") == "true"


def _non_secure_context_filtered(errors: list[str]) -> list[str]:
    """Exclude pre-existing crypto.randomUUID error on non-HTTPS Compose URLs (http://app-ci:7860)."""
    return [err for err in errors if "randomUUID" not in err]


def test_goatcounter_disabled_or_blocked_never_breaks_ui(page: Page, app_url: str):
    """Unset code, blocked script, or throwing count() never produces page errors or UI regressions."""
    # 1. Code unset: even if window.goatcounter exists, no events fire.
    page_errors: list[str] = []
    page.on("pageerror", lambda err: page_errors.append(str(err)))
    _block_goatcounter_hits(page)
    _stub_goatcounter_config(page, None)
    _install_goatcounter_spy(page)
    page.goto(app_url, wait_until="domcontentloaded")
    _wait_for_goatcounter_state(page, "off")
    wait_for_custom_js(page)
    dialog = open_knowledge_base_drawer(page)
    dialog.locator("> button").click()
    expect(dialog).not_to_be_visible()
    assert _recorded_events(page) == []
    assert _non_secure_context_filtered(page_errors) == []

    # 2. Code set, gc.zgo.at aborted (no window.goatcounter): UI interactions still succeed.
    blocked_page = page.context.new_page()
    try:
        blocked_errors: list[str] = []
        blocked_page.on("pageerror", lambda err: blocked_errors.append(str(err)))
        _block_goatcounter_hits(blocked_page)
        _stub_goatcounter_config(blocked_page, TEST_CODE)
        blocked_page.goto(app_url, wait_until="domcontentloaded")
        _wait_for_goatcounter_state(blocked_page, "on")
        wait_for_custom_js(blocked_page)
        dialog = open_knowledge_base_drawer(blocked_page)
        dialog.locator("> button").click()
        expect(dialog).not_to_be_visible()
        blocked_page.locator("#starter-grievance-builder").click()
        blocked_page.get_by_text("Grievance Builder", exact=False).first.wait_for(state="visible", timeout=15000)
        assert _non_secure_context_filtered(blocked_errors) == []
    finally:
        blocked_page.close()

    # 3. Throwing window.goatcounter.count never propagates or blocks drawer opening.
    throwing_page = page.context.new_page()
    try:
        throwing_errors: list[str] = []
        throwing_page.on("pageerror", lambda err: throwing_errors.append(str(err)))
        _block_goatcounter_hits(throwing_page)
        _stub_goatcounter_config(throwing_page, TEST_CODE)
        _install_goatcounter_spy(throwing_page, throw_on_count=True)
        throwing_page.goto(app_url, wait_until="domcontentloaded")
        _wait_for_goatcounter_state(throwing_page, "on")
        wait_for_custom_js(throwing_page)
        dialog = open_knowledge_base_drawer(throwing_page)
        expect(dialog).to_be_visible()
        assert _non_secure_context_filtered(throwing_errors) == []
    finally:
        throwing_page.close()
