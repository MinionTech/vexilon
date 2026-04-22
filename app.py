import os
import sys
import html
import urllib.parse
import gradio as gr

def get_vexilon_info():
    version = os.getenv("VEXILON_VERSION", "Dev mode")
    return version

def chat_fn(message, history, persona):
    # A generic placeholder for the future RAG functionality
    return f"Vexilon ({persona} Mode) received: {message}"

VEXILON_VERSION = get_vexilon_info()
VEXILON_REPO_URL = os.getenv("VEXILON_REPO_URL", "https://github.com/DerekRoberts/vexilon")
_container_url = f"{VEXILON_REPO_URL}/pkgs/container/vexilon/versions"
_version_url = _container_url
if VEXILON_VERSION != "Dev mode":
    _version_url += f"/versions?filters%5Bversion_type%5D=tagged&query={urllib.parse.quote(VEXILON_VERSION)}"

# Standard examples from the original Vexilon UI
EXAMPLES = [
    ["What are the steps for a Step 1 grievance?", "Lookup"],
    ["How do I report a safety hazard?", "Lookup"],
    ["What are the shift premium rates?", "Lookup"],
    ["Tell me about the sick leave policy?", "Lookup"]
]

with gr.Blocks(title="BCGEU Navigator", fill_height=True) as demo:
    with gr.Row():
        gr.HTML("<div style='display: flex; height: 100%; align-items: center;'><h3 style='margin: 0;'>BCGEU Navigator</h3></div>")
        persona = gr.Dropdown(
            choices=["Lookup", "Grieve", "Manage"],
            value="Lookup",
            show_label=False,
            container=False,
            min_width=100
        )
    
    # We use a height-constrained chatbot inside the interface to force expansion
    gr.ChatInterface(
        fn=chat_fn,
        chatbot=gr.Chatbot(show_label=False, height=700),
        additional_inputs=[persona],
        examples=EXAMPLES,
        fill_height=True
    )

    gr.HTML(f"""
        <div style="text-align: center; color: #6b7280; font-size: 0.85rem; padding-top: 10px;">
            <a href="{VEXILON_REPO_URL}" target="_blank" style="color: #3b82f6; text-decoration: none;">GitHub</a>
            &nbsp;&nbsp;•&nbsp;&nbsp;
            <a href="{VEXILON_REPO_URL}/blob/main/docs/PRIVACY.md" target="_blank" style="color: #3b82f6; text-decoration: none;">Privacy</a>
            &nbsp;&nbsp;•&nbsp;&nbsp;
            <a href="{_version_url}" target="_blank" style="color: #3b82f6; text-decoration: none;">{html.escape(VEXILON_VERSION)}</a>
        </div>
    """)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
    )
