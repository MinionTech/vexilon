import pytest
# Skip this entire module if playwright is not installed (e.g. in minimal CI containers)
pytest.importorskip("playwright")

import threading
import time
import socket
from playwright.sync_api import Page, expect

# Import the app components
import app

def get_free_port():
    """Find an available port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))  # Bind to localhost specifically (Security Fix)
        return s.getsockname()[1]

@pytest.fixture(scope="module")
def live_server():
    """Starts the Gradio app in a background thread."""
    port = get_free_port()
    
    # Ensure index is loaded (use mock or existing index)
    # For UI tests, we don't strictly need a full index, but app startup needs to finish.
    app.startup(force_rebuild=False)
    
    demo = app.build_ui()
    
    # Launch in a thread so it doesn't block the tests
    # We use prevent_thread_lock=True to ensure it doesn't hang
    thread = threading.Thread(
        target=demo.launch,
        kwargs={
            "server_name": "127.0.0.1",
            "server_port": port,
            "prevent_thread_lock": True,
            "share": False
        },
        daemon=True
    )
    thread.start()
    
    # Wait for the server to be ready (Reliability Fix)
    timeout = 10
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                break
        except (OSError, ConnectionRefusedError):
            time.sleep(0.2)
    else:
        pytest.fail("Server failed to start within 10s timeout")
    
    url = f"http://127.0.0.1:{port}"
    yield url
    
    # Shutdown
    demo.close()

def test_chatbot_layout_desktop(page: Page, live_server: str):
    """Verifies that the chatbot has sufficient height on desktop viewports."""
    # Set a standard desktop resolution
    page.set_viewport_size({"width": 1280, "height": 800})
    page.goto(live_server)
    
    chatbot = page.locator("#chatbot")
    expect(chatbot).to_be_visible()
    
    # Wait for layout to settle
    page.wait_for_load_state("networkidle")
    
    box = chatbot.bounding_box()
    if not box:
        pytest.fail("Could not find bounding box for #chatbot")
        
    print(f"DEBUG: Desktop Chatbot Height: {box['height']}px")
    
    # ASSERT: High-fidelity layout check.
    # With a 100vh - 21rem calculation on an 800px screen (~50rem),
    # the height should be around 29rem (~460px). 
    # If it's less than 400px, something is squished.
    assert box['height'] > 400, f"Chatbot is too squashed on Desktop! Height: {box['height']}px"

def test_chatbot_layout_mobile(page: Page, live_server: str):
    """Verifies that the chatbot switches to a usable height on mobile."""
    # Set a common mobile resolution (iPhone 12/13)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(live_server)
    
    chatbot = page.locator("#chatbot")
    expect(chatbot).to_be_visible()
    
    # Wait for layout to settle
    page.wait_for_load_state("networkidle")
    
    box = chatbot.bounding_box()
    if not box:
        pytest.fail("Could not find bounding box for #chatbot")
        
    print(f"DEBUG: Mobile Chatbot Height: {box['height']}px")
    
    # ASSERT: On mobile (<540px), we set height to 65vh.
    # 65% of 844px is ~548px.
    assert box['height'] > 450, f"Chatbot is too squashed on Mobile! Height: {box['height']}px"

def test_ui_components_presence(page: Page, live_server: str):
    """Ensures critical buttons and inputs are actually rendered and reachable."""
    page.goto(live_server)
    
    # Check for Persona Selector
    expect(page.locator("#persona_selector")).to_be_visible()
    
    # Check for the Input area
    expect(page.locator("#msg_input textarea")).to_be_visible()
    
    # Check for the Send button
    expect(page.locator("#send_btn")).to_be_visible()
    
    # Check for Example Chips
    chips = page.locator("button", has_text="What are the just cause requirements")
    expect(chips).to_be_visible()
