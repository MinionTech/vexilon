"""
app.py — BCGEU Steward Assistant (Modular Version)
--------------------------------------------
Tech stack:
  - src.vexilon: Modular RAG components (loaders, vector store, manifest)
  - Gradio 6   : Web UI at http://localhost:7860
"""

import os
import json
import time
import datetime
import threading
import logging
from pathlib import Path
from collections.abc import AsyncIterator

from src.vexilon import config, utils, loader, vector, manifest, prompts

# ─── Initialization ───────────────────────────────────────────────────────────
_info = utils.get_vexilon_info()
VEXILON_VERSION = _info["ver"]
utils.print_banner(_info)

_rate_limiter = utils.RateLimiter(
    max_per_minute=config.RATE_LIMIT_PER_MINUTE, 
    max_per_hour=config.RATE_LIMIT_PER_HOUR
)

_test_registry = prompts.TestRegistry()

# ─── Clients ─────────────────────────────────────────────────────────────────
_anthropic_client = None

def get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.AsyncAnthropic()
    return _anthropic_client

# ─── RAG App State ────────────────────────────────────────────────────────────
_chunks: list[dict] = []
_index = None

def startup(force_rebuild: bool = False) -> None:
    """Initialise the vector index and load document chunks."""
    global _chunks, _index
    get_anthropic()
    print(f"[startup] Starting Vexilon {VEXILON_VERSION}…")
    
    # Hidden cache setup
    config.PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _test_registry.load(config.TESTS_DIR)

    if not force_rebuild:
        index, chunks = vector.load_precomputed_index()
        if index is not None and chunks is not None:
            _index, _chunks = index, chunks
            loader.get_embed_model() # Warm model
            print("[startup] Ready.")
            return

    _index, _chunks = vector.build_index_from_sources(force=force_rebuild)
    print("[startup] Ready.")

# ─── RAG Query Logic ──────────────────────────────────────────────────────────
async def condense_query(message: str, history: list[dict]) -> str:
    """Condense history into a standalone search query."""
    if not history: return message
    client = get_anthropic()
    
    context_lines = []
    for turn in history[-config.CONDENSE_QUERY_HISTORY_TURNS:]:
        role = "User" if turn["role"] == "user" else "Assistant"
        raw_content = str(turn["content"])
        content = raw_content[:config.CONDENSE_QUERY_CONTENT_MAX_LEN] + ("..." if len(raw_content) > config.CONDENSE_QUERY_CONTENT_MAX_LEN else "")
        context_lines.append(f"{role}: {content}")
        
    prompt = (
        "Rephrase user follow-up as a standalone search query based on history. "
        "Only provide query text.\n\n"
        f"History:\n{chr(10).join(context_lines)}\n\n"
        f"User: {message}"
    )

    try:
        response = await client.messages.create(
            model=config.CONDENSE_MODEL,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip().strip('"')
    except Exception as exc:
        print(f"[rag] Condense failed: {exc}")
        return message

async def rag_review_stream(
    message: str,
    history: list[dict],
    use_reviewer: bool = False,
    persona_mode: str = "Explorer",
) -> AsyncIterator[tuple[str, str]]:
    """Execute RAG + optional review stream."""
    if _index is None:
        yield "⚠️ Index not ready.", ""
        return

    query = await condense_query(message, history)
    relevant_chunks = vector.search_index(_index, _chunks, query)
    context = "\n\n---\n\n".join([
        f"[Source: {c.get('source', 'Unknown')}, Page: {c['page']}]\n{c['text']}" 
        for c in relevant_chunks
    ])

    messages = [{"role": t["role"], "content": t["content"]} for t in history if t["role"] in ("user", "assistant")]
    messages.append({"role": "user", "content": message})

    client = get_anthropic()
    try:
        base_prompt = persona_mode if persona_mode != "Explore" else prompts.SYSTEM_PROMPT
        if base_prompt in ["Direct", "Defend"]:
            base_prompt = prompts.get_persona_prompt(base_prompt)

        formatted_prompt = base_prompt.format(
            manifest=manifest.get_knowledge_manifest(),
            verify_message=prompts.VERIFY_STEWARD_MESSAGE,
        )

        # Audit Logic (matched tests)
        if persona_mode != "Explore":
            for test in _test_registry.find_matches(message + " " + query):
                formatted_prompt += f"\n\n--- MANDATORY LOGIC CHECK: {test.name.upper()} ---\n{test.content}\n"

        # Yield context once for the verification bot
        yield "", context
        raw_response = ""

        async with client.messages.stream(
            model=config.CLAUDE_MODEL,
            max_tokens=1024,
            system=[
                {"type": "text", "text": formatted_prompt, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": f"--- EXCERPTS ---\n{context}\n--- END ---", "cache_control": {"type": "ephemeral"}},
            ],
            messages=messages,
        ) as stream:
            async for chunk in stream.text_stream:
                raw_response += chunk
                yield chunk, context
            final = await stream.get_final_message()
            print(f"[rag] Tokens: {final.usage.input_tokens} in / {final.usage.output_tokens} out")

        # Automatically verify if enabled
        if config.VERIFY_ENABLED:
            verify_res = await verify_response(raw_response, context)
            if verify_res:
                yield f"\n\n---\n\n**🔍 Verification:**\n{verify_res}", context

        if use_reviewer:
            yield "\n\n---\n\n**🔍 Senior Rep Review:**\n", context
            async for review_chunk in review_stream(raw_response, query, context):
                yield review_chunk, context
    except Exception as exc:
        yield f"⚠️ API error: {exc}", ""

async def review_stream(raw_response: str, query: str, context: str) -> AsyncIterator[str]:
    client = get_anthropic()
    prompt = f"Review steward output.\nQUERY: {query}\nRESPONSE: {raw_response}\nCONTEXT: {context[:2000]}\n{prompts.REVIEWER_SYSTEM_PROMPT}"
    try:
        review_text = ""
        async with client.messages.stream(
            model=config.REVIEWER_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            async for chunk in stream.text_stream: 
                review_text += chunk
                yield chunk
            final = await stream.get_final_message()
            
            # Parse score from response
            import re
            score = 5
            score_match = re.search(r"SCORE:\s*(\d+)", review_text, re.IGNORECASE)
            if score_match: score = int(score_match.group(1))
            
            # Log the review
            utils.log_review(query, raw_response, review_text, score)
            print(f"[review] Score: {score}/10 for query '{query[:20]}...'")
    except Exception as exc: yield f"⚠️ Review error: {exc}"

async def verify_response(response_text: str, context: str) -> str:
    """Verification Bot: Check claims against context."""
    if not config.VERIFY_ENABLED: return ""
    client = get_anthropic()
    prompt = f"Verify claims.\nCONTEXT: {context[:4000]}\nRESPONSE: {response_text}\n{prompts.VERIFIER_SYSTEM_PROMPT}"
    try:
        msg = await client.messages.create(
            model=config.VERIFY_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text
    except Exception as exc:
        print(f"[verify] Failed: {exc}")
        return f"⚠️ Verification unavailable: {exc}"

# ─── Gradio UI ────────────────────────────────────────────────────────────────
def history_to_markdown(history: list[dict]) -> str:
    md = f"# Vexilon Export - {datetime.datetime.now()}\n\n"
    for turn in history:
        content = turn["content"]
        if isinstance(content, list): 
            content = "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in content])
        md += f"### {turn['role'].capitalize()}\n{content}\n\n"
    return md

def build_ui():
    import gradio as gr
    from src.vexilon import ui_styles
    
    with gr.Blocks(title="Vexilon: Steward Assistant") as demo:
        gr.Markdown("## BCGEU Steward Assistant")
        
        with gr.Accordion("Knowledge Base & Priority", open=False):
            gr.HTML(manifest.build_pdf_download_links())
        
        chatbot = gr.Chatbot(type="messages", label="Vexilon RAG", show_label=False, height=550)
        
        with gr.Row(elem_classes=["compact-row"]):
            with gr.Column(scale=8):
                msg = gr.Textbox(placeholder="Ask about collective agreement articles, nexus, or discipline...", container=False)
            with gr.Column(scale=1, min_width=80):
                send = gr.Button("Send", variant="primary")

        with gr.Row(elem_classes=["compact-row"]):
            persona = gr.Radio(["Explore", "Direct", "Defend"], value="Explore", label="Assistant Persona", elem_id="persona_selector")
            reviewer = gr.Checkbox(label="Senior Review", value=config.USE_REVIEWER, elem_id="reviewer_toggle")
            clear = gr.Button("🗑️ Clear", elem_classes=["sm-btn"])
            export_btn = gr.Button("📥 Export", elem_classes=["sm-btn"])

        export_file = gr.File(label="Exported Conversation", visible=False)

        def user_msg(user_input, history):
            sanitized, flagged = utils.sanitize_input(user_input)
            if flagged: return "", history + [{"role": "user", "content": sanitized}, {"role": "assistant", "content": "⚠️ Content flagged."}]
            return "", history + [{"role": "user", "content": sanitized}]

        async def bot_stream(history, use_rev, persona_mode):
            if history[-1]["content"] == "⚠️ Content flagged.": yield history; return
            user_message = history[-1]["content"]
            history.append({"role": "assistant", "content": ""})
            async for chunk, context_val in rag_review_stream(user_message, history[:-1], use_rev, persona_mode):
                history[-1]["content"] += chunk
                yield history

        msg.submit(user_msg, [msg, chatbot], [msg, chatbot]).then(bot_stream, [chatbot, reviewer, persona], chatbot)
        send.click(user_msg, [msg, chatbot], [msg, chatbot]).then(bot_stream, [chatbot, reviewer, persona], chatbot)
        clear.click(lambda: [], None, chatbot)
        
        export_btn.click(lambda h: gr.File(value=utils.save_temp_md(history_to_markdown(h)), visible=True), chatbot, export_file)
        
    return demo

if __name__ == "__main__":
    startup()
    from src.vexilon import ui_styles
    build_ui().launch(server_name="0.0.0.0", server_port=7860, css=ui_styles.CUSTOM_CSS, allowed_paths=[config.LABOUR_LAW_DIR.absolute()])
