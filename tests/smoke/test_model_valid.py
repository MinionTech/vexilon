"""
tests/smoke/test_model_valid.py — Smoke test: verify HF Router is reachable
"""

import os
import pytest
from openai import OpenAI

# Skip gracefully if no key is present
pytestmark = pytest.mark.skipif(
    not os.getenv("HF_TOKEN"),
    reason="HF_TOKEN not set — skipping smoke test",
)

def test_hf_model_exists_and_responds():
    """
    Send a minimal 1-token request to the default model via HF Router.
    """
    from app import DEFAULT_MODEL_LLM

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.getenv("HF_TOKEN")
    )
    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL_LLM,
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )
    except Exception as exc:
        pytest.fail(
            f"DEFAULT_MODEL_LLM='{DEFAULT_MODEL_LLM}' failed via HF Router.\n"
            f"Original error: {exc}"
        )

    assert response.choices[0].message.content is not None
