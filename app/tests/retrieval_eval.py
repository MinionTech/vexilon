"""Evaluation harness for golden-question retrieval hit-rate benchmarks (#801)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

VALID_CATEGORIES = (
    "agreement",
    "section_ref",
    "defined_term",
    "statutory",
    "policy",
)


def load_golden_questions(path: Path) -> dict[str, Any]:
    """Load golden questions JSON payload from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def chunk_matches(
    chunk: dict[str, Any],
    expected_source: str,
    expected_terms: list[str],
) -> bool:
    """Return True when a retrieved chunk belongs to expected_source and contains all expected_terms."""
    if chunk.get("source") != expected_source:
        return False
    haystack = f"{chunk.get('header', '')}\n{chunk.get('text', '')}".lower()
    return all(term.lower() in haystack for term in expected_terms)


def find_oracle_matches(
    chunks: list[dict[str, Any]],
    expected_source: str,
    expected_terms: list[str],
) -> list[dict[str, Any]]:
    """Return all corpus chunks that satisfy expected_source and expected_terms."""
    return [
        chunk
        for chunk in chunks
        if chunk_matches(chunk, expected_source, expected_terms)
    ]


def evaluate_retrieval(
    questions: list[dict[str, Any]],
    search_fn: Callable[[str, int], list[dict[str, Any]]],
) -> dict[str, Any]:
    """Run search_fn(query, 10) for each golden question and compute top-5 / top-10 hit rates."""
    total = len(questions)
    top5_hits = 0
    top10_hits = 0
    by_category: dict[str, dict[str, Any]] = {}
    misses: list[dict[str, Any]] = []

    for question in questions:
        cat = question["category"]
        cat_stats = by_category.setdefault(
            cat,
            {"total": 0, "top5_hits": 0, "top10_hits": 0},
        )
        cat_stats["total"] += 1

        results = search_fn(question["query"], 10)
        expected_source = question["expected_source"]
        expected_terms = question["expected_terms"]

        hit_top5 = any(
            chunk_matches(chunk, expected_source, expected_terms)
            for chunk in results[:5]
        )
        hit_top10 = hit_top5 or any(
            chunk_matches(chunk, expected_source, expected_terms)
            for chunk in results[5:10]
        )

        if hit_top5:
            top5_hits += 1
            cat_stats["top5_hits"] += 1
        if hit_top10:
            top10_hits += 1
            cat_stats["top10_hits"] += 1
        if not hit_top10:
            misses.append(
                {
                    "id": question["id"],
                    "category": cat,
                    "query": question["query"],
                    "expected_source": expected_source,
                    "top_sources": [c.get("source", "") for c in results[:5]],
                }
            )

    for cat_stats in by_category.values():
        cat_total = cat_stats["total"]
        cat_stats["top5_hit_rate"] = cat_stats["top5_hits"] / cat_total if cat_total else 0.0
        cat_stats["top10_hit_rate"] = cat_stats["top10_hits"] / cat_total if cat_total else 0.0

    return {
        "total": total,
        "top5_hits": top5_hits,
        "top10_hits": top10_hits,
        "top5_hit_rate": top5_hits / total if total else 0.0,
        "top10_hit_rate": top10_hits / total if total else 0.0,
        "by_category": by_category,
        "misses": misses,
    }
