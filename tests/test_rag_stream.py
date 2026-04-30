"""
tests/test_rag_stream.py — Unit tests for rag_stream()

All external API calls (LLM, OpenAI/FAISS search) are mocked.
Tests verify guard-clause behaviour and correct prompt construction.
"""

import re
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import openai
import pytest

import app

def test_compose_yml_does_not_hardcode_model_name():
    """
    compose.yml must not hardcode a model name default.
    """
    compose_text = (Path(__file__).parent.parent / "compose.yml").read_text()
    match = re.search(r"DEFAULT_MODEL_LLM", compose_text)
    assert not match

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fake_chunks() -> list[dict]:
    return [
        {"text": "Article 1 says something important.", "page": 5, "chunk_index": 0},
        {"text": "Article 2 says something else.", "page": 6, "chunk_index": 0},
    ]

# ── Guard clauses ─────────────────────────────────────────────────────────────

async def test_rag_stream_no_index_yields_not_ready(monkeypatch):
    """When _index is None, yield the 'not ready' message."""
    monkeypatch.setattr(app, "_index", None)

    output = []
    async for chunk, ctx in app.rag_stream("What are my rights?", []):
        output.append(chunk)
    assert len(output) == 1
    assert "knowledge base not loaded" in output[0].lower()

# ── Happy path ────────────────────────────────────────────────────────────────

async def test_rag_stream_yields_tokens_from_llm(monkeypatch):
    """Happy path: tokens yielded by the LLM stream reach the caller."""
    fake_index = MagicMock()
    monkeypatch.setattr(app, "_index", fake_index)
    monkeypatch.setattr(app, "_chunks", _fake_chunks())

    def mock_search_batch(index, chunks, queries, top_ks):
        return [_fake_chunks() for _ in queries]

    monkeypatch.setattr(app, "search_index_batch", mock_search_batch)

    async def _mock_openai_stream(*args, **kwargs):
        if kwargs.get("stream"):
            async def _gen():
                for t in ["Hello", " there", "!"]:
                    chunk = MagicMock()
                    chunk.choices = [MagicMock(delta=MagicMock(content=t))]
                    yield chunk
            return _gen()
        return MagicMock()

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=_mock_openai_stream)
    monkeypatch.setattr(app, "get_llm_client", lambda: mock_client)

    output = []
    async for chunk, ctx in app.rag_stream("Any question", []):
        if chunk:  # Skip context-only yields
            output.append(chunk)
    assert output == ["Hello", " there", "!"]

async def test_rag_stream_includes_page_context_in_system_prompt(monkeypatch):
    """The system prompt sent to LLM must include agreement excerpts."""
    fake_index = MagicMock()
    monkeypatch.setattr(app, "_index", fake_index)
    monkeypatch.setattr(app, "_chunks", _fake_chunks())

    def mock_search_batch(index, chunks, queries, top_ks):
        return [_fake_chunks() for _ in queries]

    monkeypatch.setattr(app, "search_index_batch", mock_search_batch)

    captured = {}

    async def _capture_stream(**kwargs):
        captured.update(kwargs)
        if kwargs.get("stream"):
            async def _gen():
                chunk = MagicMock()
                chunk.choices = [MagicMock(delta=MagicMock(content="ok"))]
                yield chunk
            return _gen()
        return MagicMock()

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=_capture_stream)
    monkeypatch.setattr(app, "get_llm_client", lambda: mock_client)

    async for chunk, ctx in app.rag_stream("What about overtime?", []):
        pass

    messages = captured.get("messages", [])
    system_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
    
    assert "[Source: Unknown, Page: 5]" in system_prompt
    assert "[Source: Unknown, Page: 6]" in system_prompt
    assert "Article 1 says something important." in system_prompt

async def test_rag_stream_api_error_yields_error_message(monkeypatch):
    """An API error during streaming should yield an error string, not raise."""
    fake_index = MagicMock()
    monkeypatch.setattr(app, "_index", fake_index)
    monkeypatch.setattr(app, "_chunks", _fake_chunks())

    def mock_search_batch(index, chunks, queries, top_ks):
        return [_fake_chunks() for _ in queries]

    monkeypatch.setattr(app, "search_index_batch", mock_search_batch)

    async def _raising_stream(**kwargs):
        raise openai.APIStatusError(
            message="model: bad-model-name",
            response=MagicMock(status_code=404),
            body={"type": "error"},
        )

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=_raising_stream)
    monkeypatch.setattr(app, "get_llm_client", lambda: mock_client)

    output = []
    async for chunk, ctx in app.rag_stream("Any question", []):
        output.append(chunk)
    assert len(output) == 1
    assert "⚠️" in output[0]
    assert "API error" in output[0]
