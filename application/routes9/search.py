"""
GET /search — hybrid retrieval endpoint.

Runs vector + BM25 + fuzzy search and fuses results with Reciprocal
Rank Fusion (RRF): score = sum over each ranked list of 1 / (k + rank).
k=60 is the standard default from the original RRF paper.

Langfuse tracing (Day 4): hybrid_search is wrapped with @observe() so
you get retrieval latency + inputs/outputs in the trace.
"""
from fastapi import APIRouter
from langfuse import observe

from application.retrieval5.vector_search import vector_index
from application.retrieval5.bm25_search import bm25_index
from application.retrieval5.fuzzy_search import fuzzy_index

router = APIRouter()


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    fused_scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, (chunk_id, _original_score) in enumerate(ranked_list, start=1):
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    result = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return result[:top_k]


@observe()
def hybrid_search(query: str, top_k: int = 5):
    vector_results = vector_index.search(query, top_k=10)
    bm25_results = bm25_index.search(query, top_k=10)
    fuzzy_results = fuzzy_index.search(query, top_k=10)

    return reciprocal_rank_fusion(
        [vector_results, bm25_results, fuzzy_results],
        top_k=top_k,
    )


@router.get("/search")
@observe()
async def search_endpoint(query: str, top_k: int = 5):
    fused = hybrid_search(query, top_k=top_k)
    return {
        "results": [
            {"chunk_id": cid, "score": round(score, 4), "text": vector_index.get_text(cid)}
            for cid, score in fused
        ]
    }