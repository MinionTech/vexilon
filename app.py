import os
import gradio as gr

def chat_fn(message, history, persona):
    # A generic placeholder for the future RAG functionality
    return f"Vexilon ({persona} Mode) received: {message}"

with gr.Blocks(title="Vexilon", fill_height=True) as demo:
    with gr.Sidebar():
        gr.Markdown("### Vexilon Settings")
        persona = gr.Dropdown(
            choices=["Lookup", "Grieve", "Manage"],
            value="Lookup",
            label="Operational Role"
        )
        gr.Markdown("Choose a role to change how Vexilon responds.")

    chatbot = gr.Chatbot(fill_height=True, label="Vexilon")
    
    # Standard ChatInterface logic in a manual wrapper
    gr.ChatInterface(
        fn=chat_fn,
        chatbot=chatbot,
        additional_inputs=[persona],
        fill_height=True
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
    )
