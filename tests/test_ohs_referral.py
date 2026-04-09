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
    
    def mock_search(*a, **kw):
        return fake_chunks
    monkeypatch.setattr("app.search_index", mock_search)
    
    captured_kwargs = {}
    
    @asynccontextmanager
    async def mock_stream(**kwargs):
        captured_kwargs.update(kwargs)
        mock_s = MagicMock()
        
        async def _gen():
            yield "Response"
        mock_s.text_stream = _gen()
        
        async def _get_final():
            return MagicMock(usage=MagicMock(
                input_tokens=1, 
                output_tokens=1, 
                cache_creation_input_tokens=0, 
                cache_read_input_tokens=0
            ))
        mock_s.get_final_message = _get_final
        
        yield mock_s

    mock_client = MagicMock()
    mock_client.messages.stream = mock_stream
    mock_client.messages.create = AsyncMock(return_value=MagicMock(content=[MagicMock(text="unsafe work")]))
    # Force the mock to behave like a string for the .text attribute
    mock_client.messages.create.return_value.content[0].text = "unsafe work"
    monkeypatch.setattr("app.get_anthropic", lambda: mock_client)
    
    # Run rag_review_stream with an OHS keyword
    # persona_mode must NOT be 'Explore' to trigger Audit Logic
    async for chunk in rag_review_stream("I need to refuse unsafe work", [], persona_mode="Direct"):
        pass

    # Verify the system prompt contains the OHS referral
    system_prompt = captured_kwargs["system"][0]["text"]
    assert "--- MANDATORY LOGIC CHECK: OHS SAFETY REFERRAL ---" in system_prompt
    assert "1-888-621-7233" in system_prompt
    assert "You MUST follow this pattern:" in system_prompt
