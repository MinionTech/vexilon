import pytest
from vexilon.indexing import get_embed_model

def test_tokenizer_is_fast():
    """Vexilon requires 'Fast' tokenizers for character-offset mapping."""
    model = get_embed_model()
    assert hasattr(model, "tokenizer"), "Model should have a tokenizer"
    assert getattr(model.tokenizer, "is_fast", False), "Tokenizer must be a 'Fast' tokenizer"

def test_tokenizer_offset_mapping_valid():
    """Ensure offsets returned by the tokenizer align with the original text."""
    model = get_embed_model()
    tokenizer = model.tokenizer
    text = "The quick brown fox jumps over the lazy dog."
    
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False
    )
    
    assert "offset_mapping" in encoding, "Encoding must contain offset_mapping"
    mapping = encoding["offset_mapping"]
    assert len(mapping) > 0, "Mapping should not be empty"
    
    for start, end in mapping:
        token_text = text[start:end]
        assert len(token_text) > 0, f"Token at {start}:{end} is empty"
        # We don't assert it matches the token exactly because things like 
        # subword prefixes (##) can be present in some tokenizers, 
        # but the character range should be valid.

def test_tokenizer_offset_mapping_multiline():
    """Test offset mapping with multiline strings to ensure char_offset logic in indexing.py is sound."""
    model = get_embed_model()
    tokenizer = model.tokenizer
    lines = ["Line one", "Second line", "Third line"]
    full_text = "\n".join(lines)
    
    char_offset = 0
    for line in lines:
        encoding = tokenizer(
            line,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False
        )
        mapping = encoding["offset_mapping"]
        for start, end in mapping:
            # Absolute offsets
            abs_start = char_offset + start
            abs_end = char_offset + end
            assert full_text[abs_start:abs_end].strip() != "", f"Empty token at {abs_start}:{abs_end}"
        
        char_offset += len(line) + 1
