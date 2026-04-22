import os
import html
import urllib.parse
import gradio as gr

def get_vexilon_info():
    version = os.getenv("VEXILON_VERSION", "Dev mode")
    return version

def chat_fn(message, history, persona):
    # A generic placeholder for the future RAG functionality
    return f"BCGEU Navigator ({persona} Mode) received: {message}"

VEXILON_VERSION = get_vexilon_info()
VEXILON_REPO_URL = os.getenv("VEXILON_REPO_URL", "https://github.com/DerekRoberts/vexilon")
_container_url = f"{VEXILON_REPO_URL}/pkgs/container/vexilon/versions"
_version_url = _container_url
if VEXILON_VERSION != "Dev mode":
    _version_url += f"/versions?filters%5Bversion_type%5D=tagged&query={urllib.parse.quote(VEXILON_VERSION)}"

# Standard examples
EXAMPLES = [
    ["What are the steps for a Step 1 grievance?", "Lookup"],
    ["How do I report a safety hazard?", "Lookup"],
    ["What are the shift premium rates?", "Lookup"],
    ["Tell me about the sick leave policy?", "Lookup"]
]

# The "Vanilla Startup Guide" approach: Pure ChatInterface
demo = gr.ChatInterface(
    fn=chat_fn,
    title="BCGEU Navigator",
    description=f"""
        <div style="text-align: center; color: #6b7280; font-size: 0.85rem;">
            <a href="{VEXILON_REPO_URL}" target="_blank" style="color: #3b82f6; text-decoration: none;">GitHub</a>
            &nbsp;&nbsp;•&nbsp;&nbsp;
            <a href="{VEXILON_REPO_URL}/blob/main/docs/PRIVACY.md" target="_blank" style="color: #3b82f6; text-decoration: none;">Privacy</a>
            &nbsp;&nbsp;•&nbsp;&nbsp;
            <a href="{_version_url}" target="_blank" style="color: #3b82f6; text-decoration: none;">{html.escape(VEXILON_VERSION)}</a>
        </div>
    """,
    additional_inputs=[
        gr.Dropdown(
            choices=["Lookup", "Grieve", "Manage"],
            value="Lookup",
            label="Operational Role"
        )
    ],
    examples=EXAMPLES,
    fill_height=True
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
    )
