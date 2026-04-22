# BCGEU Navigator - UI Version: 2026-04-22_13-03
import os
import html
import urllib.parse
import gradio as gr

def get_vexilon_info():
    version = os.getenv("VEXILON_VERSION", "Dev mode")
    return version

def chat_fn(message, history, persona):
    if history is None:
        history = []
    persona_mode = persona if persona else "Lookup"
    history.append({"role": "user", "content": message})
    response = f"BCGEU Navigator ({persona_mode} Mode) received: {message}"
    history.append({"role": "assistant", "content": response})
    return "", history, gr.update(open=False)

VEXILON_VERSION = get_vexilon_info()
VEXILON_REPO_URL = os.getenv("VEXILON_REPO_URL", "https://github.com/DerekRoberts/vexilon")
_version_url = f"{VEXILON_REPO_URL}/pkgs/container/vexilon/versions"

EXAMPLES = [
    "What are the steps for a Step 1 grievance?",
    "How do I report a safety hazard?",
    "What are the shift premium rates?",
    "Tell me about the sick leave policy?"
]

CLOSE_ACCORDION_JS = """
() => {
    const accordion = document.querySelector('#quick-questions-accordion');
    if (accordion) {
        accordion.open = false;
    }
}
"""

_CSS = "footer {display: none !important;}"

with gr.Blocks(title="BCGEU Navigator", fill_height=True) as demo:
    with gr.Row():
        gr.HTML("<div style='display: flex; height: 100%; align-items: center;'><h3 style='margin: 0;'>BCGEU Navigator</h3></div>")
        persona = gr.Dropdown(
            choices=["Lookup", "Grieve", "Manage"],
            value="Lookup",
            show_label=False,
            container=False,
            min_width=100,
            interactive=True
        )
    
    chatbot = gr.Chatbot(show_label=False, scale=1, min_height=400)
    
    with gr.Row():
        msg = gr.Textbox(show_label=False, placeholder="Type a message...", container=False, scale=7)
        submit = gr.Button("Send", variant="primary", scale=1)

    with gr.Accordion("Quick Questions", open=False, elem_id="quick-questions-accordion") as examples_accordion:
        with gr.Row():
            for q in EXAMPLES:
                example_btn = gr.Button(q, size="sm", variant="secondary")
                example_btn.click(
                    chat_fn, 
                    [gr.State(q), chatbot, persona], 
                    [msg, chatbot, examples_accordion],
                    js=CLOSE_ACCORDION_JS
                )

    gr.HTML(f"""
        <div style="text-align: center; color: #6b7280; font-size: 0.85rem; padding: 10px 0;">
            <a href="{VEXILON_REPO_URL}" target="_blank" style="color: #3b82f6; text-decoration: none;">GitHub</a>
            &nbsp;&nbsp;•&nbsp;&nbsp;
            <a href="{VEXILON_REPO_URL}/blob/main/docs/PRIVACY.md" target="_blank" style="color: #3b82f6; text-decoration: none;">Privacy</a>
            &nbsp;&nbsp;•&nbsp;&nbsp;
            <a href="{_version_url}" target="_blank" style="color: #3b82f6; text-decoration: none;">{html.escape(VEXILON_VERSION)}</a>
        </div>
    """)

    msg.submit(chat_fn, [msg, chatbot, persona], [msg, chatbot, examples_accordion], js=CLOSE_ACCORDION_JS)
    submit.click(chat_fn, [msg, chatbot, persona], [msg, chatbot, examples_accordion], js=CLOSE_ACCORDION_JS)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, css=_CSS)
