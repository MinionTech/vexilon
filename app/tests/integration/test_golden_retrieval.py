"""Integration: golden-question retrieval hit-rate evaluation over real FAISS index (#801)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import indexing
from retrieval_eval import (
    evaluate_retrieval,
    find_oracle_matches,
    load_golden_questions,
)

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_questions.json"


@pytest.fixture(scope="module")
def loaded_corpus():
    manifest = indexing._read_stored_manifest()
    if not manifest:
        pytest.fail("FAISS cache manifest.json is missing or unreadable")
    assert manifest.get("config") == indexing._index_config(), (
        "FAISS cache config is stale relative to current indexing configuration"
    )
    current_hashes = {
        str(f.relative_to(indexing.DATA_DIR)): indexing._hash_file(f)
        for f in indexing._get_rag_source_files()
    }
    stored_hashes = {
        k: v.get("content_hash") for k, v in manifest.get("files", {}).items()
    }
    assert stored_hashes == current_hashes, (
        "FAISS cache is stale relative to current source files in DATA_DIR"
    )

    index, chunks = indexing.load_precomputed_index()
    if index is None or not chunks:
        pytest.fail("Precomputed FAISS index and chunks must be available for retrieval eval")
    return index, chunks


def test_all_golden_questions_exist_in_corpus(loaded_corpus):
    """Every golden question's expected_source + expected_terms must match >= 1 chunk."""
    _, chunks = loaded_corpus
    payload = load_golden_questions(GOLDEN_PATH)
    unmatched = [
        q["id"]
        for q in payload["questions"]
        if not find_oracle_matches(chunks, q["expected_source"], q["expected_terms"])
    ]
    assert not unmatched, f"Golden questions with 0 oracle matches in corpus: {unmatched}"


def test_retrieval_hit_rate_meets_committed_baseline(loaded_corpus, capsys):
    """Run all golden questions through indexing.search_index and enforce baseline hit rates."""
    index, chunks = loaded_corpus
    payload = load_golden_questions(GOLDEN_PATH)
    baseline = payload["baseline"]

    report = evaluate_retrieval(
        payload["questions"],
        lambda query, top_k: indexing.search_index(index, chunks, query, top_k=top_k),
    )

    with capsys.disabled():
        print(
            f"\n[retrieval-eval] total={report['total']} "
            f"top5={report['top5_hits']}/{report['total']} ({report['top5_hit_rate']:.1%}) "
            f"[baseline={baseline['top5_hit_rate']:.1%}] | "
            f"top10={report['top10_hits']}/{report['total']} ({report['top10_hit_rate']:.1%}) "
            f"[baseline={baseline['top10_hit_rate']:.1%}]"
        )
        for cat, stats in sorted(report["by_category"].items()):
            print(
                f"  - {cat:12s}: top5={stats['top5_hits']}/{stats['total']} "
                f"({stats['top5_hit_rate']:.1%}), "
                f"top10={stats['top10_hits']}/{stats['total']} ({stats['top10_hit_rate']:.1%})"
            )
        if report["misses"]:
            print("  Misses (outside top-10):")
            for miss in report["misses"]:
                print(
                    f"    * {miss['id']} ({miss['category']}): {miss['query']} "
                    f"[expected={miss['expected_source']}, got={miss['top_sources']}]"
                )

    assert report["top5_hit_rate"] >= baseline["top5_hit_rate"], (
        f"Top-5 hit rate regressed: {report['top5_hit_rate']:.4f} < {baseline['top5_hit_rate']:.4f}"
    )
    assert report["top10_hit_rate"] >= baseline["top10_hit_rate"], (
        f"Top-10 hit rate regressed: {report['top10_hit_rate']:.4f} < {baseline['top10_hit_rate']:.4f}"
    )
