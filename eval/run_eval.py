"""
Basic evaluation: Recall@5 and Mean Reciprocal Rank (MRR) against the
golden dataset. Run this AFTER you've uploaded the same test documents
the golden set was written against.

Usage:
    python eval/run_eval.py
"""
import json
from pathlib import Path
from app.retrieval.rrf_fusion import hybrid_search
from app.retrieval.vector_search import vector_index
from app.retrieval.bm25_search import bm25_index
from app.retrieval.fuzzy_search import fuzzy_index

GOLDEN_PATH = Path(__file__).parent / "golden_dataset.json"


def recall_at_k(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    if not expected_ids:
        return 0.0
    hits = sum(1 for eid in expected_ids if eid in retrieved_ids)
    return hits / len(expected_ids)


def reciprocal_rank(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in expected_ids:
            return 1.0 / rank
    return 0.0


def run_eval(top_k: int = 5):
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    recalls, rr_scores = [], []

    for item in golden["queries"]:
        query = item["query"]
        expected = item["expected_memory_ids"]

        fused = hybrid_search(query, vector_index, bm25_index, fuzzy_index, top_k=top_k)
        retrieved_ids = [cid for cid, _ in fused]

        recalls.append(recall_at_k(retrieved_ids, expected))
        rr_scores.append(reciprocal_rank(retrieved_ids, expected))

        print(f"Query: {query!r}")
        print(f"  Retrieved: {retrieved_ids}")
        print(f"  Expected:  {expected}")
        print(f"  Recall@{top_k}: {recalls[-1]:.2f}  RR: {rr_scores[-1]:.2f}\n")

    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    mrr = sum(rr_scores) / len(rr_scores) if rr_scores else 0.0

    print("=" * 40)
    print(f"Average Recall@{top_k}: {avg_recall:.3f}")
    print(f"MRR: {mrr:.3f}")


if __name__ == "__main__":
    run_eval()
