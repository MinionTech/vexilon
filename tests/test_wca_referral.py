import pytest
from app import _test_registry, TESTS_DIR, rag_review_stream
from unittest.mock import MagicMock, patch, AsyncMock
from contextlib import asynccontextmanager

def test_wca_registry_loading():
    """Verify that the WCA claims referral test is loaded into the registry."""
    _test_registry.load(TESTS_DIR)
    wca_test = next((t for t in _test_registry.tests if "Wca" in t.name), None)
    assert wca_test is not None

@pytest.mark.asyncio
async def test_rag_review_stream_triggers_wca_logic(monkeypatch):
    """Verify that WCA keywords in the query trigger the WCA claims referral in the system prompt."""
    # Ensure registry is loaded
    _test_registry.load(TESTS_DIR)
    
    # Mock dependencies
    fake_index = MagicMock()
    monkeypatch.setattr("app._index", fake_index)
    
    fake_chunks = [{"text": "WCA text", "page": 1, "source": "Act", "chunk_index": 0}]
    monkeypatch.setattr("app._chunks", fake_chunks)
    
    def mock_search_batch(*a, **kw):
        return [fake_chunks]
    monkeypatch.setattr("app.search_index_batch", mock_search_batch)
    
    all_captured_kwargs = []
    
    @asynccontextmanager
    async def mock_stream(**kwargs):
        all_captured_kwargs.append(kwargs)
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
    monkeypatch.setattr("app.get_anthropic", lambda: mock_client)
    
    # Mock generate_perspective_queries to avoid hitting the API in this test
    monkeypatch.setattr("app.generate_perspective_queries", AsyncMock(return_value=["I have a back injury and need to file a WCB claim"]))
    
    # Run rag_review_stream with a WCA keyword
    async for chunk in rag_review_stream("I have a back injury and need to file a WCB claim", [], persona_mode="Grieve", all_chunks=fake_chunks):
        pass

    # Verify the system prompt in ANY of the calls contains the WCA referral
    found_wca = False
    for call_kwargs in all_captured_kwargs:
        system = call_kwargs.get("system", [])
        if system and isinstance(system, list) and "WCA CLAIMS REFERRAL" in system[0]["text"]:
            found_wca = True
            assert "90 days" in system[0]["text"]
            assert "Review Division" in system[0]["text"]
            assert "Request for Review" in system[0]["text"]
            break
    
    assert found_wca, "WCA Claims Referral was not found in any system prompt"
