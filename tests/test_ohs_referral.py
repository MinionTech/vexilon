import pytest
from app import _test_registry, TESTS_DIR, rag_review_stream
from unittest.mock import MagicMock, patch, AsyncMock
from contextlib import asynccontextmanager

def test_ohs_registry_loading():
    """Verify that the OHS referral test is loaded into the registry."""
    _test_registry.load(TESTS_DIR)
    ohs_test = next((t for t in _test_registry.tests if "Ohs" in t.name), None)
    assert ohs_test is not None

@pytest.mark.asyncio
async def test_rag_review_stream_triggers_ohs_logic(monkeypatch):
    """Verify that OHS keywords in the query trigger the OHS referral in the system prompt."""
    # Ensure registry is loaded
    _test_registry.load(TESTS_DIR)
    
    # Mock dependencies
    fake_index = MagicMock()
    monkeypatch.setattr("app._index", fake_index)
    
    fake_chunks = [{"text": "Agreement text", "page": 1, "source": "Main", "chunk_index": 0}]
    monkeypatch.setattr("app._chunks", fake_chunks)
    
    def mock_search_batch(*a, **kw):
        return [fake_chunks]
    monkeypatch.setattr("app.search_index_batch", mock_search_batch)
    
    all_captured_kwargs = []
    
    async def mock_create(**kwargs):
        all_captured_kwargs.append(kwargs)
        if kwargs.get("stream"):
            async def _gen():
                chunk = MagicMock()
                chunk.choices = [MagicMock(delta=MagicMock(content="Response"))]
                yield chunk
            return _gen()
        return MagicMock(choices=[MagicMock(message=MagicMock(content="unsafe work"))])

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)
    monkeypatch.setattr("app.get_async_openai_client", lambda: mock_client)
    
    # Mock generate_perspective_queries to avoid hitting the API in this test
    monkeypatch.setattr("app.generate_perspective_queries", AsyncMock(return_value=["I need to refuse unsafe work"]))
    
    # Run rag_review_stream with an OHS keyword
    # persona_mode must be 'Grieve' to trigger Audit Logic
    async for chunk in rag_review_stream("I need to refuse unsafe work", [], persona_mode="Grieve"):
        pass

    # Verify the system prompt in ANY of the calls contains the OHS referral
    found_ohs = False
    for call_kwargs in all_captured_kwargs:
        messages = call_kwargs.get("messages", [])
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        if "--- MANDATORY LOGIC CHECK: OHS SAFETY REFERRAL ---" in system:
            found_ohs = True
            assert "1-888-621-7233" in system
            assert "YOU MUST immediately advise" in system
            break
    
    assert found_ohs, "OHS Safety Referral was not found in any system prompt"
