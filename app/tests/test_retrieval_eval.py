"""Unit tests for the golden-question retrieval evaluation harness (#801)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieval_eval import (
    VALID_CATEGORIES,
    chunk_matches,
    evaluate_retrieval,
    load_golden_questions,
)

GOLDEN_PATH = Path(__file__).resolve().parent / "integration" / "golden_questions.json"


def test_golden_questions_schema_and_minimum_count():
    payload = load_golden_questions(GOLDEN_PATH)
    baseline = payload["baseline"]
    questions = payload["questions"]

    assert len(questions) >= 30
    assert 0.0 <= baseline["top5_hit_rate"] <= baseline["top10_hit_rate"] <= 1.0
    assert baseline["total"] == len(questions)

    seen_ids: set[str] = set()
    for q in questions:
        assert q["id"] not in seen_ids
        seen_ids.add(q["id"])
        assert q["category"] in VALID_CATEGORIES
        assert isinstance(q["query"], str) and q["query"].strip()
        assert isinstance(q["expected_source"], str) and q["expected_source"].strip()
        assert isinstance(q["expected_terms"], list) and q["expected_terms"]
        assert all(isinstance(t, str) and t.strip() for t in q["expected_terms"])


def test_chunk_matches_requires_exact_source_and_all_terms():
    chunk = {
        "source": "BCGEU 20th Main Agreement",
        "header": "10.8 RIGHT TO HAVE STEWARD PRESENT",
        "text": "An employee shall have the right to have their steward present.",
    }
    assert chunk_matches(chunk, "BCGEU 20th Main Agreement", ["10.8", "steward"])
    assert not chunk_matches(chunk, "BC Labour Relations Code", ["10.8", "steward"])
    assert not chunk_matches(chunk, "BCGEU 20th Main Agreement", ["10.8", "missing"])


def test_evaluate_retrieval_computes_top5_top10_and_categories():
    questions = [
        {
            "id": "q1",
            "category": "section_ref",
            "query": "Article 10.8",
            "expected_source": "Doc A",
            "expected_terms": ["10.8"],
        },
        {
            "id": "q2",
            "category": "section_ref",
            "query": "s. 21",
            "expected_source": "Doc B",
            "expected_terms": ["21"],
        },
        {
            "id": "q3",
            "category": "defined_term",
            "query": "Nexus test",
            "expected_source": "Doc C",
            "expected_terms": ["Millhaven"],
        },
    ]

    def fake_search(query: str, _top_k: int) -> list[dict]:
        filler = [{"source": "Other", "header": "", "text": "filler"} for _ in range(6)]
        if "10.8" in query:
            # Hit in top 5
            return [{"source": "Doc A", "header": "10.8", "text": "steward"}] + filler
        if "21" in query:
            # Hit only at rank 7 (top 10, not top 5)
            return filler + [{"source": "Doc B", "header": "21", "text": "duties"}]
        # Complete miss
        return filler

    report = evaluate_retrieval(questions, fake_search)
    assert report["total"] == 3
    assert report["top5_hits"] == 1
    assert report["top10_hits"] == 2
    assert report["top5_hit_rate"] == 1 / 3
    assert report["top10_hit_rate"] == 2 / 3
    assert report["by_category"]["section_ref"]["top5_hits"] == 1
    assert report["by_category"]["section_ref"]["top10_hits"] == 2
    assert report["by_category"]["defined_term"]["top10_hits"] == 0
    assert len(report["misses"]) == 1
    assert report["misses"][0]["id"] == "q3"
