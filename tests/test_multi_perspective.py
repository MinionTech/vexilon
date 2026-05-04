import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from contextlib import asynccontextmanager
import app

@pytest.mark.asyncio
async def test_generate_perspective_queries_simple():
    """Verify that a simple query returns just the condensed query."""
    message = "How many days of vacation do I get?"
    history = []
    
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content='["vacation days amount"]'))]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
    
    with patch("app.get_async_openai_client", return_value=mock_client):
        # We need to mock condense_query to return a known value
        with patch("app.condense_query", return_value="vacation days amount"):
            queries = await app.generate_perspective_queries(message, history)
            
    assert queries == ["vacation days amount"]
    # Verify the mock was called
    assert mock_client.chat.completions.create.called

@pytest.mark.asyncio
async def test_generate_perspective_queries_complex():
    """Verify that a complex query returns multiple hyphenated queries."""
    message = "I was arrested for a DUI while off-duty. Can I be fired?"
    history = []
    
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content='["off-duty conduct case law", "Millhaven factors DUI", "employer rights off-site arrest"]'))]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
    
    with patch("app.get_async_openai_client", return_value=mock_client):
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
    monkeypatch.setattr(app, "IS_DEV", False)
    
    # Mock chunks from different queries, some overlapping
    chunk1 = {"text": "Original Article 1 text.", "page": 1, "source": "DocA"}
    chunk2 = {"text": "Detailed Article 2 text.", "page": 2, "source": "DocB"}
    
    monkeypatch.setattr(app, "_chunks", [chunk1, chunk2])

    def mock_generate_perspectives(*a, **kw):
        # Return two queries
        return ["query1", "query2"]
    
    monkeypatch.setattr(app, "generate_perspective_queries", AsyncMock(side_effect=mock_generate_perspectives))
    monkeypatch.setattr(app, "condense_query", AsyncMock(return_value="this is a very long string that has more than ten words in it to trigger the logic"))

    search_calls = []
    def mock_search_batch(index, chunks, queries, top_ks):
        for q in queries:
            search_calls.append(q)
        # Return a list of lists (one list of chunks per query)
        return [[chunk1], [chunk1, chunk2]]

    monkeypatch.setattr(app, "search_index_batch", mock_search_batch)

    # Mock OpenAI stream
    mock_client = MagicMock()
    
    async def _mock_openai_stream(**kwargs):
        # Capture the system prompt to check context
        system_prompt = kwargs.get("system", "")
        if kwargs.get("stream"):
            async def _gen():
                # Check context deduplication
                assert system_prompt.count("Original Article 1 text.") == 1
                assert "Detailed Article 2 text." in system_prompt
                chunk = MagicMock()
                chunk.choices = [MagicMock(delta=MagicMock(content="Mocked response content."))]
                yield chunk
            return _gen()
        return MagicMock()

    mock_client.chat.completions.create = AsyncMock(side_effect=_mock_openai_stream)
    monkeypatch.setattr(app, "get_async_openai_client", lambda: mock_client)

    async for chunk, ctx in app.rag_stream("This is a very complex question that needs multiple perspectives to answer correctly about the agreement", []):
        pass

    assert len(search_calls) == 2
    assert "query1" in search_calls
    assert "query2" in search_calls
