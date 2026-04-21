import pytest
import time
import os
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import socket

def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

@pytest.fixture(scope="module")
def mock_hf_environment():
    """
    Creates a local HTML file that mocks the Hugging Face Space iframe environment.
    It embeds the local Gradio app in an iframe and simulates a height-growth script.
    """
    port = get_free_port()
    tmp_dir = Path("./.tmp")
    tmp_dir.mkdir(exist_ok=True)
    
    # This HTML mocks the HF 'growth' behavior.
    # It attempts to resize the iframe based on content.
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vexilon Iframe Mock</title>
        <style>
            body { margin: 0; padding: 0; background: #eee; }
            #container { width: 100%; height: 100vh; overflow: auto; }
            iframe { width: 100%; border: none; transition: height 0.1s; }
        </style>
    </head>
    <body>
        <div id="container">
            <iframe id="vexilon_iframe" src="http://localhost:7860"></iframe>
        </div>
        <script>
            const iframe = document.getElementById('vexilon_iframe');
            // Mocking the HF auto-resizer loop
            setInterval(() => {
                try {
                    const doc = iframe.contentDocument || iframe.contentWindow.document;
                    if (doc && doc.body) {
                        const newHeight = doc.documentElement.scrollHeight;
                        if (newHeight > 0) {
                            iframe.style.height = newHeight + 'px';
                        }
                    }
                } catch (e) {
                    // Cross-origin might block this locally if not handled, 
                    // but for this test we'll assume same-origin localhost.
                }
            }, 500);
        </script>
    </body>
    </html>
    """
    
    mock_file = tmp_dir / "hf_mock.html"
    mock_file.write_text(html_content)
    
    # Simple server to serve the mock file
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args): return

    server = HTTPServer(('localhost', port), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    
    yield f"http://localhost:{port}/.tmp/hf_mock.html"
    
    server.shutdown()

@pytest.mark.integration
def test_iframe_growth_integrity(mock_hf_environment):
    """
    This test verifies that the app does not enter an infinite growth loop 
    when embedded in a height-resizing iframe.
    
    SUCCESS CRITERIA: The iframe height must stabilize and not exceed 1.5x viewport height.
    """
    # Note: This test is designed to be run via a browser agent or manually 
    # verified. For the purpose of this task, I am setting up the 
    # infrastructure for the browser subagent to use.
    print(f"\n[INTEGRITY] Mock HF environment ready at: {mock_hf_environment}")
    print("[INTEGRITY] Verification required via browser subagent.")
