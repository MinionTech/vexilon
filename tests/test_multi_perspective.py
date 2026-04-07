import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from contextlib import asynccontextmanager
import app

@pytest.mark.asyncio
async def test_generate_perspective_queries_simple():
    """Verify that a simple query returns just the condensed query."""
    message = "How many days of vacation do I get?"
    history = []
    
    mock_client = AsyncMock()
    mock_response = MagicMock()
    # Mocking a response that doesn't start with a hyphen (simple query)
    mock_response.content = [MagicMock(text="vacation days amount")]
    mock_client.messages.create.return_value = mock_response
    
    with patch("app.get_anthropic", return_value=mock_client):
        # We need to mock condense_query to return a known value
        with patch("app.condense_query", return_value="vacation days amount"):
            queries = await app.generate_perspective_queries(message, history)
            
    assert queries == ["vacation days amount"]
    # Verify the mock was called
    assert mock_client.messages.create.called

@pytest.mark.asyncio
async def test_generate_perspective_queries_complex():
    """Verify that a complex query returns multiple hyphenated queries."""
    message = "I was arrested for a DUI while off-duty. Can I be fired?"
    history = []
    
    mock_client = AsyncMock()
    mock_response = MagicMock()
    # Mocking a multi-perspective response
    mock_response.content = [MagicMock(text="- off-duty conduct case law\n- Millhaven factors DUI\n- employer rights off-site arrest")]
    mock_client.messages.create.return_value = mock_response
    
    with patch("app.get_anthropic", return_value=mock_client):
        with patch("app.condense_query", return_value="DUI arrest off-duty termination"):
            queries = await app.generate_perspective_queries(message, history)
            
    assert len(queries) == 3
    assert "off-duty conduct case law" in queries
    assert "Millhaven factors DUI" in queries
    assert "employer rights off-site arrest" in queries

@pytest.mark.asyncio
async def test_rag_stream_aggregates_multiple_queries(monkeypatch):
    """Verify that rag_stream calls search_index for each query and deduplicates."""
    
    fake_index = MagicMock()
    monkeypatch.setattr(app, "_index", fake_index)
    
    # Mock chunks from different queries, some overlapping
    chunk1 = {"text": "Original Article 1 text.", "page": 1, "source": "DocA"}
    chunk2 = {"text": "Detailed Article 2 text.", "page": 2, "source": "DocB"}
    
    monkeypatch.setattr(app, "_chunks", [chunk1, chunk2])

    def mock_generate_perspectives(*a, **kw):
        # Return two queries
        return ["query1", "query2"]
    
    monkeypatch.setattr(app, "generate_perspective_queries", AsyncMock(side_effect=mock_generate_perspectives))

    search_calls = []
    def mock_search(index, chunks, query, top_k):
        search_calls.append(query)
        if query == "query1":
            return [chunk1]
        else:
            return [chunk1, chunk2] # overlapping chunk1

    monkeypatch.setattr(app, "search_index", mock_search)

    # Mock Anthropic stream
    mock_client = MagicMock()
    
    @asynccontextmanager
    async def _mock_stream(**kwargs):
        # Capture the system prompt to check context
        system_prompt = kwargs.get("system", [])
        mock_stream = MagicMock()
        mock_stream.text_stream = AsyncMock()
        # Mocking an empty stream for simplicity
        async def _empty_gen():
            if False: yield
        mock_stream.text_stream.__aiter__.return_value = _empty_gen()
        
        # Mock get_final_message
        fake_usage = MagicMock(input_tokens=0, output_tokens=0, cache_creation_input_tokens=0, cache_read_input_tokens=0)
        fake_message = MagicMock(usage=fake_usage, stop_reason="stop")
        mock_stream.get_final_message = AsyncMock(return_value=fake_message)
        
        # Check context deduplication (only one instance of chunk1 should be in the system prompt)
        context_block = system_prompt[1]["text"]
        assert context_block.count("Original Article 1 text.") == 1
        assert "Detailed Article 2 text." in context_block
        
        yield mock_stream

    mock_client.messages.stream = _mock_stream
    monkeypatch.setattr(app, "get_anthropic", lambda: mock_client)

    async for chunk, ctx in app.rag_stream("Complex question", []):
        pass

    assert len(search_calls) == 2
    assert "query1" in search_calls
    assert "query2" in search_calls
