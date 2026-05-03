"""
tests/integration/test_embed_pipeline.py — Integration: sentence-transformers + FAISS

Tests the real embedding pipeline end-to-end with NO external API calls.
Requires sentence-transformers to load the model from HuggingFace Hub (~90 MB,
cached after first run — see HF_HOME in analysis.yml).

Purpose: catch breaking changes in sentence-transformers or faiss-cpu after a
Renovate dependency bump. Mocked unit tests (test_index.py) won't catch these
because they never call the real model:

    If sentence-transformers changes encode() to return a different shape/dtype → caught here.
    If faiss-cpu changes IndexFlatIP.search() output format → caught here.
    If the two break each other's numpy interop → caught here.

Run time: ~5–15 s on model first load; ~1–2 s after the model is cached.
"""

import numpy as np
import pytest

from vexilon import indexing as app


# ── embed_texts() API shape ───────────────────────────────────────────────────

def test_embed_texts_returns_correct_shape_and_dtype():
    """
    embed_texts() must return (N, EMBED_DIM) float32.
    If sentence-transformers changes encode() output shape or dtype this fails — loudly.
    """
    texts = ["Hello, union steward.", "Overtime rates are one and a half times regular pay."]
    result = app.embed_texts(texts)

    assert result.ndim == 2, f"Expected 2D output, got {result.ndim}D"
    assert result.shape == (len(texts), app.EMBED_DIM), (
        f"Expected shape ({len(texts)}, {app.EMBED_DIM}), got {result.shape}"
    )
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"


def test_embed_texts_single_input():
    """Single-element list must return (1, EMBED_DIM) — not (EMBED_DIM,)."""
    result = app.embed_texts(["Just one sentence about vacation leave."])

    assert result.shape == (1, app.EMBED_DIM), (
        f"Expected (1, {app.EMBED_DIM}), got {result.shape}. "
        "Did sentence-transformers change squeeze behaviour for single inputs?"
    )


def test_embed_texts_different_inputs_produce_different_vectors():
    """Semantically different sentences must not produce identical embeddings."""
    a = app.embed_texts(["Vacation leave accrual rates."])
    b = app.embed_texts(["Overtime pay is one and a half times the hourly rate."])

    assert not np.allclose(a, b), (
        "Two semantically unrelated sentences produced identical embeddings — "
        "something is very wrong with the model."
    )


# ── Full pipeline: chunk → embed → index → search ────────────────────────────

def _make_chunks_from_text(text: str, page_num: int, source: str = "Test") -> list[dict]:
    """
    Helper: build the token_data list and call chunk_text() directly, mimicking what
    load_pdf_chunks() does for a single page. Uses the real embedding tokenizer so
    these tests exercise the real pipeline.
    """
    tokenizer = app.get_embed_model().tokenizer
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    token_data = [(start, end, page_num, "") for start, end in encoding.offset_mapping]
    return app.chunk_text(text, token_data, source)


def test_full_pipeline_returns_semantically_relevant_chunk():
    """
    Full end-to-end integration: chunk_text() → real embed_texts() → build_index() → search_index().

    Two semantically distinct topics are indexed. The query is about one of them.
    The top result must be the matching topic — not the unrelated one.

    This is the Renovate safety net: it validates that sentence-transformers, faiss-cpu,
    and numpy all still interoperate after a version bump.
    """
    vacation_text = (
        "Employees are entitled to annual vacation leave. "
        "The number of vacation days increases with years of service. "
        "Vacation must be scheduled by mutual agreement."
    )
    overtime_text = (
        "Overtime hours must be compensated at one and a half times the regular hourly rate. "
        "Overtime is defined as any hours worked beyond the standard workday. "
        "Employees must be notified of overtime requirements in advance."
    )

    vacation_chunks = _make_chunks_from_text(vacation_text, page_num=1)
    overtime_chunks = _make_chunks_from_text(overtime_text, page_num=2)
    all_chunks = vacation_chunks + overtime_chunks

    index = app.build_index(all_chunks)
    results = app.search_index(index, all_chunks, query="How many vacation days am I entitled to?", top_k=1)

    assert len(results) == 1
    assert results[0]["page"] == 1, (
        f"Expected top result from page 1 (vacation), got page {results[0]['page']} (overtime). "
        "Semantic similarity broke — check sentence-transformers or faiss-cpu version."
    )


def test_full_pipeline_query_matches_overtime_chunk():
    """Symmetrical test: an overtime query must match the overtime chunk, not vacation."""
    vacation_text = (
        "Annual vacation leave is calculated based on years of continuous service. "
        "Employees may carry over unused vacation days."
    )
    overtime_text = (
        "All overtime work shall be compensated at one and a half times the regular rate. "
        "Overtime rates apply to any hours worked beyond seven and a half hours per day."
    )

    chunks = _make_chunks_from_text(vacation_text, page_num=1) + _make_chunks_from_text(overtime_text, page_num=2)
    index = app.build_index(chunks)

    results = app.search_index(index, chunks, query="What is the overtime pay rate?", top_k=1)

    assert len(results) == 1
    assert results[0]["page"] == 2, (
        f"Expected top result from page 2 (overtime), got page {results[0]['page']} (vacation)."
    )
