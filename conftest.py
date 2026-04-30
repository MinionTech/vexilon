"""
conftest.py — pytest root configuration

Adds the project root to sys.path so `import app` works from tests/
regardless of how pytest is invoked.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest
import os
from unittest.mock import MagicMock, AsyncMock
from contextlib import asynccontextmanager

# Force Anthropic provider for tests to maintain compatibility with existing mocks
os.environ["VEXILON_LLM_PROVIDER"] = "anthropic"

@pytest.fixture(autouse=True)
def mock_embedding_model(request, monkeypatch):
    """
    Mocks the embedding model and tokenizer for unit tests to prevent
    large model downloads. Integration tests are skipped.
    """
    if "integration" in str(request.path):
        return
    import app
    # Only mock if we are not explicitly doing an integration test that needs the real model
    # We can check the test name or path, but it's safer to just provide a lightweight mock
    # and let integration tests skip the mock if they want.
    
    mock_model = MagicMock()
    
    # Mock tokenizer behavior for chunk_text
    def mock_tokenize(text, **kwargs):
        # Return something that looks like encoding.input_ids and encoding.offset_mapping
        tokens = [1] * (len(text) // 4 + 1) # dummy tokens
        offsets = [(i*4, min((i+1)*4, len(text))) for i in range(len(tokens))]
        encoding = MagicMock()
        encoding.input_ids = tokens
        encoding.offset_mapping = offsets
        return encoding
        
    mock_model.tokenizer = mock_tokenize
    mock_model.encode = MagicMock(return_value=[[0.1]*384])
    
    # We only patch if the test isn't an integration test that specifically wants the real deal
    # (By default we mock, integration tests can un-mock if needed)
    monkeypatch.setattr(app, "get_embed_model", lambda: mock_model)
    return mock_model

@pytest.fixture
def mock_anthropic():
    """Provides a mocked LLM client supporting both Anthropic and OpenAI APIs."""
    mock_client = MagicMock()
    
    # ── Anthropic Style ──
    @asynccontextmanager
    async def _mock_stream(*args, **kwargs):
        mock_stream = MagicMock()
        async def _async_gen():
            yield "Mocked response content."
        mock_stream.text_stream = _async_gen()
        fake_message = MagicMock(usage=MagicMock(input_tokens=10, output_tokens=5, cache_creation_input_tokens=0, cache_read_input_tokens=0))
        mock_stream.get_final_message = AsyncMock(return_value=fake_message)
        yield mock_stream

    mock_client.messages.stream = _mock_stream
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text="Mocked response content")]
    mock_client.messages.create = AsyncMock(return_value=mock_resp)

    # ── OpenAI / HF Style ──
    mock_chat = MagicMock()
    
    # Mock completions.create (non-streaming)
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="Mocked response content"))]
    mock_chat.completions.create = AsyncMock(return_value=mock_completion)
    
    # Mock completions.create (streaming)
    async def _mock_openai_stream(*args, **kwargs):
        if kwargs.get("stream"):
            async def _gen():
                chunk = MagicMock()
                chunk.choices = [MagicMock(delta=MagicMock(content="Mocked response content."))]
                yield chunk
            return _gen()
        return mock_completion
    
    mock_chat.completions.create = AsyncMock(side_effect=_mock_openai_stream)
    mock_client.chat = mock_chat
    
    return mock_client
