import http.server
import socketserver
import os
from pathlib import Path

PORT = int(os.getenv("MIRROR_PORT", 8888))
APP_URL = os.getenv("TARGET_URL", "http://localhost:7860")

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Vexilon: Local HF Mirror</title>
    <style>
        body {{ 
            margin: 0; 
            padding: 0; 
            background: #111; 
            color: #eee; 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }}
        #header {{ 
            padding: 10px 20px; 
            background: #222; 
            border-bottom: 1px solid #333; 
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        #container {{ 
            flex-grow: 1;
            width: 100%;
            overflow: auto;
            background: #000;
        }}
        iframe {{ 
            width: 100%; 
            border: none; 
            background: white;
            display: block;
        }}
        .status-badge {{
            padding: 4px 8px;
            background: #ff9d00;
            color: black;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div id="header">
        <div>
            <strong>Vexilon Local HF Mirror</strong>
            <span style="margin-left: 10px; color: #888; font-size: 13px;">Simulating Hugging Face Iframe Environment</span>
        </div>
        <div class="status-badge">IFRAME ACTIVE</div>
    </div>
    <div id="container">
        <iframe id="hf_iframe" src="{APP_URL}"></iframe>
    </div>
    <script>
        const iframe = document.getElementById('hf_iframe');
        
        // Mocking the HF auto-resizer loop
        // This is the specific logic that causes the 'Infinite Growth' bug
        setInterval(() => {{
            try {{
                const doc = iframe.contentDocument || iframe.contentWindow.document;
                if (doc && doc.documentElement) {{
                    const newHeight = doc.documentElement.scrollHeight;
                    // If the app is not 'sealed', this will grow infinitely
                    if (newHeight > 10) {{
                        iframe.style.height = newHeight + 'px';
                    }}
                }}
            }} catch (e) {{
                // This will fail if localhost:7860 is not running
                console.warn("Cannot access iframe content. Ensure {APP_URL} is running.", e);
            }}
        }}, 200);
    </script>
</body>
</html>
"""

def run():
    # Ensure scripts directory exists
    Path("scripts").mkdir(exist_ok=True)
    
    # Create the mirror file
    with open("scripts/hf_mirror.html", "w") as f:
        f.write(html_content)
    
    os.chdir("scripts")
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"\n🚀 HF Local Mirror ready at: http://localhost:{PORT}/hf_mirror.html")
        print(f"👉 Pointing to your app at: {APP_URL}")
        print("Press Ctrl+C to stop.")
        httpd.serve_forever()

if __name__ == "__main__":
    run()
