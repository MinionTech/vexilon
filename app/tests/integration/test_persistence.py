import pytest
import main as app
import chainlit as cl
from pathlib import Path

@pytest.mark.asyncio
async def test_history_to_markdown_and_back():
    """Verify that conversation history can be serialized and deserialized accurately."""
    original_history = [
        {"role": "user", "content": "How do I file a grievance?"},
        {"role": "assistant", "content": "Follow Article 8 process."},
        {"role": "user", "content": "What about timelines?"}
    ]
    
    md = app.history_to_markdown(original_history)
    assert "### User" in md
    assert "Follow Article 8 process" in md
    
    # Save to temp file and read back
    temp_path = Path("test_history.md")
    temp_path.write_text(md, encoding="utf-8")
    
    try:
        restored_history = app.markdown_to_history(str(temp_path))
        assert len(restored_history) == 3
        assert restored_history[0]["role"] == "user"
        assert restored_history[1]["content"] == "Follow Article 8 process."
    finally:
        if temp_path.exists():
            temp_path.unlink()

@pytest.mark.asyncio
async def test_import_history_action(monkeypatch, mocker):
    """Verify the import_history callback parses files and updates session."""
    mock_file = mocker.Mock()
    mock_file.path = "dummy_history.md"
    
    # Create dummy history file
    dummy_md = "### User\nTest query\n### Assistant\nTest response"
    Path(mock_file.path).write_text(dummy_md, encoding="utf-8")
    
    try:
        # Mock AskFileMessage to return our dummy file
        mock_ask = mocker.Mock()
        mock_ask.send = mocker.AsyncMock(return_value=[mock_file])
        monkeypatch.setattr(cl, "AskFileMessage", lambda **kwargs: mock_ask)
        
        # Mock cl.Message to avoid context checks during init
        mock_msg = mocker.Mock()
        mock_msg.send = mocker.AsyncMock()
        monkeypatch.setattr(cl, "Message", lambda **kwargs: mock_msg)
        
        # Chainlit's user_session is a proxy; we need to mock it as an object
        mock_session = mocker.Mock()
        mock_session.get = mocker.Mock(return_value=[])
        mock_session.set = mocker.Mock()
        monkeypatch.setattr(cl, "user_session", mock_session)
        
        # Execute import callback
        from main import on_import
        await on_import(mocker.Mock())
        
        # Verify session was updated
        mock_session.set.assert_any_call("history", [
            {"role": "user", "content": "Test query"},
            {"role": "assistant", "content": "Test response"}
        ])
        
        # Verify messages were "sent"
        assert mock_msg.send.called
    finally:
        if Path(mock_file.path).exists():
            Path(mock_file.path).unlink()
