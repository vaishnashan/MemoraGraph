"""
Ingests your test documents, then runs every query in golden_dataset.json
through the real retrieval pipeline and computes:
  - Recall@5: % of queries where an expected keyword shows up somewhere
              in the top 5 returned chunks
  - MRR (Mean Reciprocal Rank): average of 1/rank of the first chunk
              containing an expected keyword

IMPORTANT: vector_index/bm25_index/fuzzy_index are in-memory, so they're
empty at the start of every new process. This script ingests
TEST_DOCUMENT_PATHS itself before scoring, so it's self-contained —
you don't need a uvicorn server running with prior uploads.

Run from your project ROOT:
    python eval/run_eval.py
"""
import json
import uuid
from pathlib import Path

from application.parsing3.docling_parser import parse_and_chunk
from application.embeddings4.embedd import embed_text
from application.retrieval5.vector_search import vector_index
from application.retrieval5.bm25_search import bm25_index
from application.retrieval5.fuzzy_search import fuzzy_index
from application.retrieval5.rrf_fusion import hybrid_search

GOLDEN_SET_PATH = Path(__file__).parent / "golden_dataset.json"
TOP_K = 5

# <-- EDIT THIS: point at the real files your golden_dataset.json questions
#     are written about (project_2.pdf, project_3.pdf, etc.) -->
TEST_DOCUMENT_PATHS = [
    r"E:\5.Project 2\codebase\project_2.pdf",
    # r"E:\5.Project 2\codebase\project_3.pdf",
]


def ingest_test_documents() -> dict[str, str]:
    """Parses + chunks + embeds + indexes each test doc. Returns {filename: doc_id}."""
    doc_ids = {}
    for path in TEST_DOCUMENT_PATHS:
        doc_id = str(uuid.uuid4())
        parsed, chunks = parse_and_chunk(path, doc_id)
        for chunk in chunks:
            embedding = embed_text(chunk.text)
            vector_index.add(chunk.chunk_id, chunk.text, embedding, doc_id)
            bm25_index.add(chunk.chunk_id, chunk.text)
            fuzzy_index.add(chunk.chunk_id, chunk.text)
        doc_ids[parsed.filename] = doc_id
        print(f"✅ Ingested {parsed.filename} -> doc_id={doc_id} ({len(chunks)} chunks)")
    return doc_ids


def _chunk_contains_any_keyword(text: str, keywords: list[str]) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def evaluate() -> None:
    print("=" * 60)
    print("INGESTING TEST DOCUMENTS")
    print("=" * 60)
    ingest_test_documents()

    with open(GOLDEN_SET_PATH) as f:
        data = json.load(f)

    entries = data["entries"]
    hits_at_5 = 0
    reciprocal_ranks = []
    per_query_results = []

    print("\n" + "=" * 60)
    print("RUNNING EVAL QUERIES")
    print("=" * 60)

    for entry in entries:
        query = entry["query"]
        expected_keywords = entry.get("expected_keywords", [])

        fused = hybrid_search(query, vector_index, bm25_index, fuzzy_index, top_k=TOP_K)

        rank_of_first_hit = None
        for rank, (chunk_id, _score) in enumerate(fused, start=1):
            text = vector_index.get_text(chunk_id) or ""
            if _chunk_contains_any_keyword(text, expected_keywords):
                rank_of_first_hit = rank
                break

        hit = rank_of_first_hit is not None
        if hit:
            hits_at_5 += 1
            reciprocal_ranks.append(1.0 / rank_of_first_hit)
        else:
            reciprocal_ranks.append(0.0)

        per_query_results.append({
            "id": entry["id"],
            "query": query,
            "difficulty": entry.get("difficulty", "unknown"),
            "hit": hit,
            "rank": rank_of_first_hit,
        })

    n = len(entries)
    recall_at_5 = hits_at_5 / n if n else 0.0
    mrr = sum(reciprocal_ranks) / n if n else 0.0

    print("\n" + "=" * 60)
    print("PER-QUERY RESULTS")
    print("=" * 60)
    for r in per_query_results:
        status = f"✅ rank {r['rank']}" if r["hit"] else "❌ not found in top 5"
        print(f"[{r['difficulty']:6s}] {r['id']}: {status}  — {r['query']}")

    print("\n" + "=" * 60)
    print("SUMMARY METRICS")
    print("=" * 60)
    print(f"Total queries:  {n}")
    print(f"Hits @ top-5:   {hits_at_5}")
    print(f"Recall@5:       {recall_at_5:.3f}")
    print(f"MRR:            {mrr:.3f}")

    print("\nBy difficulty:")
    for level in ["easy", "medium", "hard"]:
        subset = [r for r in per_query_results if r["difficulty"] == level]
        if not subset:
            continue
        subset_hits = sum(1 for r in subset if r["hit"])
        print(f"  {level:6s}: {subset_hits}/{len(subset)} ({subset_hits/len(subset):.0%})")


if __name__ == "__main__":
    evaluate()