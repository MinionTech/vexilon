import os
import sys
import asyncio
import contextlib
import logging
from pathlib import Path
from collections.abc import AsyncIterator

import chainlit as cl
import faiss

from core.config import (
    IS_DEV,
    AGNAV_VERSION,
    AGNAV_APP_NAME,
    PUBLIC_DOCS_DIR,
    TESTS_DIR,
    CLAUDE_MODEL,
    REVIEWER_MODEL,
    CONDENSE_MODEL,
    RAG_MAX_TOKENS,
    REVIEWER_MAX_TOKENS,
    DEFAULT_MODEL_LLM,
    format_rag_error_message,
    get_llm_provider,
)
from services.llm import (
    has_chainlit_context,
    unified_chat_create,
    unified_chat_stream,
    get_system_prompt,
    get_persona_prompt,
)
from services.registry import _test_registry
from indexing import (
    _get_source_name,
    _get_rag_source_files,
    build_index_from_sources,
    get_integrity_report,
    load_precomputed_index,
    search_index_batch,
    _fetch_pdf_cache_if_missing,
    DATA_DIR,
    CACHE_DIR,
    PDF_CACHE_DIR,
    get_embed_model,
    EMBED_DIM,
    chunk_text,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

logger = logging.getLogger(__name__)

# RAG module state
_chunks: list[dict] = []
_index: "faiss.IndexFlatIP | None" = None
INTEGRITY_WARNING: str | None = None
_source_path_map: dict[str, Path] = {}

def _get_active_index():
    main_mod = sys.modules.get("main")
    if main_mod and hasattr(main_mod, "_index") and main_mod._index is not _index:
        return main_mod._index
    return _index

def _get_active_chunks():
    main_mod = sys.modules.get("main")
    if main_mod and hasattr(main_mod, "_chunks") and main_mod._chunks is not _chunks:
        return main_mod._chunks
    return _chunks

def _get_active_search_batch():
    main_mod = sys.modules.get("main")
    if main_mod and hasattr(main_mod, "search_index_batch"):
        return main_mod.search_index_batch
    curr_mod = sys.modules.get(__name__)
    if curr_mod and hasattr(curr_mod, "search_index_batch") and curr_mod.search_index_batch != search_index_batch:
        return curr_mod.search_index_batch
    return search_index_batch

def _get_active_test_registry():
    main_mod = sys.modules.get("main")
    if main_mod and hasattr(main_mod, "_test_registry"):
        return main_mod._test_registry
    return _test_registry

@contextlib.asynccontextmanager
async def status_step(name: str, remove_on_exit: bool = False):
    """Safe context manager to show Chainlit steps only when a UI context exists."""
    if has_chainlit_context():
        async with cl.Step(name=name) as step:
            steps_list = cl.user_session.get("steps_to_remove")
            if isinstance(steps_list, list):
                steps_list.append(step)
            try:
                yield step
            finally:
                if remove_on_exit:
                    await step.remove()
    else:
        class DummyStep:
            def __init__(self):
                self.output = ""
            async def update(self):
                pass
            async def remove(self):
                pass
        yield DummyStep()

async def clear_active_status_steps() -> None:
    """Wipe any registered intermediate UI steps to keep chat history clean and autoscroll smooth."""
    if not has_chainlit_context():
        return
    steps_to_remove = cl.user_session.get("steps_to_remove") or []
    for s in steps_to_remove:
        try:
            await s.remove()
        except Exception as e:
            logger.error(f"[chat] Failed to remove step: {e}")
    cl.user_session.set("steps_to_remove", [])

def _format_history(history: list[dict]) -> str:
    """Format conversation history list into a standardized string for LLM prompts."""
    history_text = ""
    for turn in history[-5:]:
        role = (turn["role"] if isinstance(turn, dict) else turn.role).capitalize()
        content = turn["content"] if isinstance(turn, dict) else turn.content
        if isinstance(content, list):
            content = "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in content])
        history_text += f"{role}: {content}\n"
    return history_text

async def condense_query(message: str, history: list[dict]) -> str:
    """Turn the conversation history and new message into a standalone search query."""
    if not history:
        return message

    history_text = _format_history(history)
    prompt = f"CONVERSATION HISTORY:\n{history_text}\nUSER MESSAGE: {message}\n\nTask: Condense into a standalone search query."
    try:
        resp_text = await unified_chat_create(
            model=CONDENSE_MODEL,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp_text.strip().strip('"')
    except Exception:
        return message

async def get_rag_context(message: str, history: list[dict]) -> tuple[list[str], str, list[dict]]:
    main_mod = sys.modules.get("main")
    force_condense = os.getenv("AGNAV_FORCE_CONDENSE") == "true"
    is_dev = getattr(main_mod, "IS_DEV", IS_DEV)

    if history and (not is_dev or force_condense):
        async with status_step("context synthesis...") as step:
            condense_fn = getattr(main_mod, "condense_query", condense_query)
            condensed = await condense_fn(message, history)
            step.output = f"Condensed query: \"{condensed}\""
    else:
        condensed = message

    queries = [condensed]

    async with status_step("retrieval...") as step:
        top_k_count = 3 if is_dev else 5
        search_fn = _get_active_search_batch()
        current_index = _get_active_index()
        current_chunks = _get_active_chunks()
        all_res = await asyncio.to_thread(search_fn, current_index, current_chunks, queries, [top_k_count] * len(queries))
        seen = set()
        context_parts = []
        unique_snippets = []
        for res_list in all_res:
            for c in res_list:
                if c["text"] not in seen:
                    seen.add(c["text"])
                    unique_snippets.append(c)
                    source = c.get("source", "Unknown")
                    page = c.get("page", "?")
                    context_parts.append(f"<<< SOURCE: {source} | Page: {page} >>>\n{c['text']}")

        sources_found = set(c.get("source", "Unknown") for c in unique_snippets)
        step.output = f"Retrieved {len(unique_snippets)} matching excerpts from {len(sources_found)} reference documents."

    return queries, "\n\n".join(context_parts), unique_snippets

async def rag_stream(message: str, history: list[dict]) -> AsyncIterator[tuple[str, str]]:
    """Yields (chunk, context) for tests and legacy callers."""
    current_index = _get_active_index()
    if current_index is None:
        yield "⚠️ Knowledge base not loaded.", ""
        return
    try:
        main_mod = sys.modules.get("main")
        context_fn = getattr(main_mod, "get_rag_context", get_rag_context)
        queries, context, snippets = await context_fn(message, history)

        system = get_system_prompt().format(manifest="", verify_message="") + f"\n\nContext:\n{context}"

        capped = []
        for h in history[-2:]:
            c = h["content"] if isinstance(h["content"], str) else str(h["content"])
            capped.append({"role": h["role"], "content": c[:300] + "..." if len(c) > 300 else c})
        messages = capped + [{"role": "user", "content": message}]

        has_yielded_context = False
        chat_stream_fn = getattr(main_mod, "unified_chat_stream", unified_chat_stream)
        async for chunk in chat_stream_fn(
            model=CLAUDE_MODEL,
            max_tokens=RAG_MAX_TOKENS,
            system=system,
            messages=messages,
        ):
            if not has_yielded_context:
                yield "", context
                has_yielded_context = True
            yield chunk, ""
    except Exception as exc:
        yield f"⚠️ API error: {exc}", ""

async def rag_review_stream(
    message: str,
    history: list[dict],
    persona_mode: str = "Lookup",
    context: str | None = None,
    queries: list[str] | None = None,
) -> AsyncIterator[str]:
    try:
        main_mod = sys.modules.get("main")
        if not context or not queries:
            context_fn = getattr(main_mod, "get_rag_context", get_rag_context)
            q_new, c_new, s_new = await context_fn(message, history)
            context = context or c_new
            queries = queries or q_new

        base_persona = get_persona_prompt(persona_mode)
        audit_rules = ""
        if persona_mode in ("Grieve", "Manage"):
            registry = _get_active_test_registry()
            matched_tests = registry.find_matches(message + " " + queries[0])
            for test in matched_tests:
                audit_rules += f"\n\n--- MANDATORY LOGIC CHECK: {test.name.upper()} ---\n"
                audit_rules += (
                    f"This case involves potential issues related to {test.name}. "
                    "You MUST follow the EXPLAIN/QUESTION/APPLY/CITE pattern "
                    "(Explain the principle, ask the relevant Question, Apply it to the facts, and Cite the source).\n"
                )
                audit_rules += f"Criteria:\n{test.content}\n"

        master_rules = get_system_prompt().format(manifest="", verify_message="")
        system = f"{master_rules}\n\n{base_persona}\n\n{audit_rules}\n\n--- KNOWLEDGE BASE CONTEXT ---\n{context}"

        capped = []
        for h in history[-2:]:
            c = h["content"] if isinstance(h["content"], str) else str(h["content"])
            capped.append({"role": h["role"], "content": c[:300] + "..." if len(c) > 300 else c})
        messages = capped + [{"role": "user", "content": message}]

        chat_stream_fn = getattr(main_mod, "unified_chat_stream", unified_chat_stream)
        async for text in chat_stream_fn(
            model=REVIEWER_MODEL,
            max_tokens=REVIEWER_MAX_TOKENS,
            system=system,
            messages=messages,
        ):
            yield text
    except Exception as exc:
        logger.error(f"[rag] Pipeline error: {exc}", exc_info=True)
        yield format_rag_error_message(exc)

def _get_download_source_files() -> list[Path]:
    """Scan DATA_DIR for PDF and MD files. Excludes test_fixtures/."""
    if not DATA_DIR.exists():
        return []
    fixtures_dir = DATA_DIR / "test_fixtures"
    files = [
        p for p in DATA_DIR.rglob("*")
        if p.suffix.lower() in (".pdf", ".md")
        and not p.is_relative_to(fixtures_dir)
        and not p.name.endswith(".integrity.md")
    ]
    return sorted(list(set(files)), key=lambda p: str(p))

def resolve_pdf_path(md_path: Path) -> Path:
    """Resolve the matching PDF file path for a given Markdown source path."""
    if md_path.suffix.lower() == ".pdf":
        return md_path

    # 1. Try same directory PDF
    pdf_same_dir = md_path.with_suffix(".pdf")
    if pdf_same_dir.exists():
        return pdf_same_dir

    from core.config import PUBLIC_DOCS_DIR as DEFAULT_PUBLIC_DOCS_DIR
    curr_mod = sys.modules.get(__name__)
    main_mod = sys.modules.get("main")
    effective_docs_dir = DEFAULT_PUBLIC_DOCS_DIR
    if curr_mod and getattr(curr_mod, "PUBLIC_DOCS_DIR", None) != DEFAULT_PUBLIC_DOCS_DIR:
        effective_docs_dir = curr_mod.PUBLIC_DOCS_DIR
    elif main_mod and getattr(main_mod, "PUBLIC_DOCS_DIR", None) != DEFAULT_PUBLIC_DOCS_DIR:
        effective_docs_dir = main_mod.PUBLIC_DOCS_DIR

    # 2. Try public docs directory
    exact_pdf = effective_docs_dir / f"{md_path.stem}.pdf"
    if exact_pdf.exists():
        return exact_pdf

    # 3. Try subdirectories in public docs (e.g. forms/)
    for found_pdf in effective_docs_dir.rglob(f"{md_path.stem}.pdf"):
        if found_pdf.is_file():
            return found_pdf

    if "_-_" in md_path.stem:
        base_stem = md_path.stem.split("_-_")[0]
        prefix_pdf = effective_docs_dir / f"{base_stem}.pdf"
        if prefix_pdf.exists():
            return prefix_pdf

        for found_pdf in effective_docs_dir.rglob(f"{base_stem}.pdf"):
            if found_pdf.is_file():
                return found_pdf

    return md_path

def build_reference_links(snippets: list[dict], source_path_map: dict | None = None) -> list[str]:
    """Generate markdown citation links for retrieved reference documents."""
    ref_links = []
    seen_sources = set()
    active_map = source_path_map if source_path_map is not None else _source_path_map
    main_mod = sys.modules.get("main")
    effective_docs_dir = getattr(
        main_mod, "PUBLIC_DOCS_DIR", getattr(sys.modules[__name__], "PUBLIC_DOCS_DIR", PUBLIC_DOCS_DIR)
    )
    for s in snippets:
        source_name = s.get("source", "")
        if source_name and source_name not in seen_sources and source_name != "Unknown":
            md_path = active_map.get(source_name)
            if md_path:
                download_path = resolve_pdf_path(md_path)
                if download_path.exists():
                    try:
                        rel_url = f"/public/docs/{download_path.relative_to(effective_docs_dir)}"
                    except ValueError:
                        rel_url = f"/public/docs/{download_path.name}"
                    clean_title = source_name.replace("_", " ")
                    ref_links.append(f"- [{clean_title}]({rel_url})")
            seen_sources.add(source_name)
    return ref_links

def startup(force_rebuild: bool = False):
    global _index, _chunks, INTEGRITY_WARNING, _source_path_map

    main_mod = sys.modules.get("main")
    provider = get_llm_provider()
    logger.info(f"[startup] AgNav {AGNAV_VERSION} starting...")
    logger.info(f"[startup] Provider: {provider}")
    logger.info(f"[startup] Default Model: {DEFAULT_MODEL_LLM}")
    logger.info(f"[startup] Build Integrity: {AGNAV_VERSION}")

    registry = _get_active_test_registry()
    registry.load(TESTS_DIR)

    import indexing
    indexing.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        test_file = indexing.CACHE_DIR / "permissions_test"
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        logger.warning(f"[startup] {indexing.CACHE_DIR} is not writable: {e}. Indexing may fail.")

    fetch_fn = getattr(main_mod, "_fetch_pdf_cache_if_missing", _fetch_pdf_cache_if_missing)
    fetch_fn()

    load_fn = getattr(main_mod, "load_precomputed_index", load_precomputed_index)
    _index, _chunks = load_fn()

    build_fn = getattr(main_mod, "build_index_from_sources", build_index_from_sources)
    if _index is None or force_rebuild:
        _index, _chunks = build_fn(force=True)

    if _index is not None:
        all_files = _get_rag_source_files()
        _source_path_map = {_get_source_name(p.stem): p for p in all_files}
        report = get_integrity_report()

    doc_list = _get_download_source_files()
    logger.info(f"[startup] {len(doc_list)} reference documents found.")

    if main_mod:
        main_mod._index = _index
        main_mod._chunks = _chunks
        main_mod.INTEGRITY_WARNING = INTEGRITY_WARNING
        main_mod._source_path_map = _source_path_map

_startup_done = False
_startup_lock = asyncio.Lock()

async def _ensure_startup() -> None:
    global _startup_done
    if _startup_done:
        return
    async with _startup_lock:
        if _startup_done:
            return
        await asyncio.to_thread(startup)
        _startup_done = True
