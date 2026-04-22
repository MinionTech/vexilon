import os
import gradio as gr

def chat_fn(message, history, persona):
    # A generic placeholder for the future RAG functionality
    return f"Vexilon ({persona} Mode) received: {message}"

with gr.Blocks(title="Vexilon", fill_height=True) as demo:
    with gr.Row():
        gr.Markdown("### Vexilon", scale=3)
        persona = gr.Dropdown(
            choices=["Lookup", "Grieve", "Manage"],
            value="Lookup",
            show_label=False,
            container=False,
            scale=1
        )
    
    gr.ChatInterface(
        fn=chat_fn,
        additional_inputs=[persona],
        fill_height=True
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
    )
