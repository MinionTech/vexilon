"""
tests/test_index.py — Unit tests for build_index() and search_index()

Uses synthetic vectors and a mocked OpenAI client — zero API calls.
"""

import faiss
import numpy as np
import pytest

import indexing


def _make_chunks(n: int) -> list[dict]:
    """Create n fake chunks with distinct text."""
    return [{"text": f"chunk number {i}", "page": i + 1, "chunk_index": 0} for i in range(n)]


def _make_embed_fn(vectors: np.ndarray):
    """Return a replacement for embed_texts() that yields rows from *vectors*."""
    call_count = {"n": 0}

    def _embed(texts: list[str]) -> np.ndarray:
        start = call_count["n"]
        call_count["n"] += len(texts)
        return vectors[start : start + len(texts)].astype(np.float32)

    return _embed


# ── build_index ───────────────────────────────────────────────────────────────

def test_build_index_returns_faiss_index(monkeypatch):
    """build_index should return a FAISS IndexFlatIP with ntotal == len(chunks)."""
    n = 5
    chunks = _make_chunks(n)
    # Use random unit vectors (shape matches EMBED_DIM)
    vecs = np.random.randn(n, indexing.EMBED_DIM).astype(np.float32)
    faiss.normalize_L2(vecs)

    monkeypatch.setattr(indexing, "embed_texts", _make_embed_fn(vecs))
    index = indexing.build_index(chunks)

    assert isinstance(index, faiss.IndexFlatIP)
    assert index.ntotal == n


def test_build_index_normalises_vectors(monkeypatch):
    """After build_index, searching with an identical vector should produce score ≈ 1.0."""
    n = 3
    chunks = _make_chunks(n)
    vecs = np.random.randn(n, indexing.EMBED_DIM).astype(np.float32)
    faiss.normalize_L2(vecs)

    monkeypatch.setattr(indexing, "embed_texts", _make_embed_fn(vecs))
    index = indexing.build_index(chunks)

    query = vecs[0:1].copy()
    scores, _ = index.search(query, 1)
    assert scores[0][0] == pytest.approx(1.0, abs=1e-5)


# ── search_index ──────────────────────────────────────────────────────────────

def test_search_index_returns_top_k(monkeypatch):
    """search_index should return exactly top_k results (when index has >= top_k vectors)."""
    n = 10
    top_k = 3
    chunks = _make_chunks(n)
    vecs = np.random.randn(n, indexing.EMBED_DIM).astype(np.float32)
    faiss.normalize_L2(vecs)

    monkeypatch.setattr(indexing, "embed_texts", _make_embed_fn(vecs))
    index = indexing.build_index(chunks)

    # For search, embed_texts is called with the single query string
    query_vec = vecs[0:1].copy()

    def _embed_search(texts):
        return query_vec.copy()

    monkeypatch.setattr(indexing, "embed_texts", _embed_search)
    results = indexing.search_index(index, chunks, "any query", top_k=top_k)

    assert len(results) == top_k


def test_search_index_finds_most_similar(monkeypatch):
    """The top result should be the chunk whose vector is closest to the query."""
    n = 4
    chunks = _make_chunks(n)
    vecs = np.random.randn(n, indexing.EMBED_DIM).astype(np.float32)
    faiss.normalize_L2(vecs)

    monkeypatch.setattr(indexing, "embed_texts", _make_embed_fn(vecs))
    index = indexing.build_index(chunks)

    # Query is identical to chunk 2 — it must be the top hit
    target_vec = vecs[2:3].copy()
    monkeypatch.setattr(indexing, "embed_texts", lambda _texts: target_vec.copy())

    results = indexing.search_index(index, chunks, "irrelevant", top_k=1)
    assert results[0] == chunks[2]


def test_get_document_tier_weight():
    """Verify that get_document_tier_weight returns correct weights for all tiers."""
    # Tier 1 documents
    assert indexing.get_document_tier_weight("BCGEU 20th Main Agreement", "01_primary/BCGEU_20th_Main_Agreement.md") == 1.2
    assert indexing.get_document_tier_weight("Gov BC Standards of Conduct", "01_primary/Gov_BC_Standards_of_Conduct.md") == 1.2
    assert indexing.get_document_tier_weight("BC Criminal Notification Procedures", "03_resources/BC_Criminal_Notification_Procedures.md") == 1.2
    assert indexing.get_document_tier_weight("Gov BC Social Media Guidelines for Personal Use", "01_primary/Gov_BC_Social_Media_Guidelines_for_Personal_Use.md") == 1.2
    
    # Matching by source name only or path only (robustness checks)
    assert indexing.get_document_tier_weight("Gov BC Standards of Conduct", "") == 1.2
    assert indexing.get_document_tier_weight("", "01_primary/Gov_BC_Standards_of_Conduct.md") == 1.2
    assert indexing.get_document_tier_weight("BCGEU 20th Main Agreement", "") == 1.2
    assert indexing.get_document_tier_weight("", "01_primary/BCGEU_20th_Main_Agreement.md") == 1.2
    
    # Tier 2 documents (unmodified)
    assert indexing.get_document_tier_weight("Nexus Test and Off-Duty Conduct", "04_jurisprudence/Nexus_Test_and_Off-Duty_Conduct.md") == 1.0
    assert indexing.get_document_tier_weight("BCGEU Grievance Form Guide", "forms/BCGEU_Grievance_Form_Guide.md") == 1.0
    
    # Tier 3 documents (de-boosted)
    assert indexing.get_document_tier_weight("BC Labour Relations Code", "02_statutory/BC_Labour_Relations_Code.md") == 0.8
    assert indexing.get_document_tier_weight("BC Employment Standards Act", "02_statutory/BC_Employment_Standards_Act.md") == 0.8
    assert indexing.get_document_tier_weight("BC OHS Regulation - Part 07", "02_statutory/BC_OHS_Regulation_-_Part_07.md") == 0.8
    assert indexing.get_document_tier_weight("BCGEU Steward Resources", "03_resources/BCGEU_Steward_Resources.md") == 0.8


def test_search_index_boosts_tier_1_documents(monkeypatch):
    """search_index should boost a Tier 1 document (BCGEU 20th Main Agreement) to rank above a slightly closer Tier 3 document."""
    # 2 chunks: 
    # chunk 0: Tier 3 document (OHS Regulation)
    # chunk 1: Tier 1 document (20th Agreement)
    chunks = [
        {
            "text": "OHS regulation chunk",
            "page": 1,
            "source": "BC OHS Regulation - Part 07",
            "path": "02_statutory/BC_OHS_Regulation_-_Part_07.md",
            "chunk_index": 0
        },
        {
            "text": "20th main agreement chunk",
            "page": 5,
            "source": "BCGEU 20th Main Agreement",
            "path": "01_primary/BCGEU_20th_Main_Agreement.md",
            "chunk_index": 1
        }
    ]
    
    # We will construct a query vec q, and database vecs v0, v1 such that:
    # q . v0 = 0.9 (chunk 0, Tier 3)
    # q . v1 = 0.8 (chunk 1, Tier 1)
    
    EMBED_DIM = indexing.EMBED_DIM
    q = np.zeros((1, EMBED_DIM), dtype=np.float32)
    q[0, 0] = 1.0 # query is along axis 0
    
    v0 = np.zeros((1, EMBED_DIM), dtype=np.float32)
    v0[0, 0] = 0.9
    v0[0, 1] = np.sqrt(1 - 0.9**2) # norm is 1.0, dot product with q is 0.9
    
    v1 = np.zeros((1, EMBED_DIM), dtype=np.float32)
    v1[0, 0] = 0.8
    v1[0, 1] = np.sqrt(1 - 0.8**2) # norm is 1.0, dot product with q is 0.8
    
    vecs = np.vstack([v0, v1])
    
    monkeypatch.setattr(indexing, "embed_texts", _make_embed_fn(vecs))
    index = indexing.build_index(chunks)
    
    # For query search, mock embed_texts to return q
    monkeypatch.setattr(indexing, "embed_texts", lambda _texts: q.copy())
    
    # Unweighted search would find chunk 0 first since dot product 0.9 > 0.8.
    # But with tier boosting:
    # chunk 0 (Tier 3): 0.9 * 0.8 (Tier 3 weight) = 0.72
    # chunk 1 (Tier 1): 0.8 * 1.2 (Tier 1 weight) = 0.96
    # So chunk 1 should be boosted to the top!
    results = indexing.search_index(index, chunks, "query text", top_k=1)
    
    assert results[0] == chunks[1]



