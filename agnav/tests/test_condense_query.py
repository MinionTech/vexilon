import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import app

@pytest.mark.asyncio
async def test_condense_query_with_gradio_blocks():
    """
    Verify that condense_query handles Gradio 6's list-based message blocks
    without raising a TypeError.
    """
    history = [
        {
            "role": "user",
            "content": [
                {"text": "Hello, ", "type": "text"},
                {"text": "world!", "type": "text"}
            ]
        }
    ]
    message = "How are you?"
    
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="Rephrased Query"))]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
    
    with patch("app.get_llm_client", return_value=mock_client):
        # We need to mock get_embed_model because app.py might try to load it
        with patch("app.get_embed_model"):
            condensed = await app.condense_query(message, history)
            
    assert condensed == "Rephrased Query"
    # Verify the mock was called with a string context
    args, kwargs = mock_client.chat.completions.create.call_args
    prompt = kwargs["messages"][0]["content"]
    assert "User: Hello, world!" in prompt

@pytest.mark.asyncio
async def test_condense_query_with_string_content():
    """Standard string content should still work."""
    history = [{"role": "user", "content": "Previous message"}]
    message = "Follow-up"
    
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="Condensed String"))]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
    
    with patch("app.get_llm_client", return_value=mock_client):
        with patch("app.get_embed_model"):
            condensed = await app.condense_query(message, history)
            
    assert condensed == "Condensed String"
    args, kwargs = mock_client.chat.completions.create.call_args
    prompt = kwargs["messages"][0]["content"]
    assert "User: Previous message" in prompt

@pytest.mark.asyncio
async def test_condense_query_handles_api_failure_gracefully():
    """If the API fails, it should return the raw message instead of crashing."""
    history = [{"role": "user", "content": "Anything"}]
    message = "The raw question"
    
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Down"))
    
    with patch("app.get_llm_client", return_value=mock_client):
        with patch("app.get_embed_model"):
            condensed = await app.condense_query(message, history)
            
    assert condensed == message
