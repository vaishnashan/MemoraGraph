"""
Reciprocal Rank Fusion (RRF): combines rankings from vector, BM25, and
fuzzy search into one fair ranked list.

RRF score for a document = sum over each ranking list of 1 / (k + rank)
where rank is 1-indexed position in that list. k=60 is the standard
default from the original RRF paper — no need to tune unless you have
a reason to.

This is the ONLY place hybrid_search lives now (previously duplicated in
both routes9/search.py and here) — the search_memory_hybrid MCP tool calls it
directly. There is no FastAPI /search route anymore; search is an
agent-facing capability exposed only through MCP.
"""
from langfuse import observe

from application.mcp.vector_search import VectorIndex
from application.mcp.bm25_search import BM25Index
from application.mcp.fuzzy_search import FuzzyIndex


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
def hybrid_search(
    query: str,
    vector_index: VectorIndex,
    bm25_index: BM25Index,
    fuzzy_index: FuzzyIndex,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Runs all three retrievers and fuses results."""
    vector_results = vector_index.search(query, top_k=10)
    bm25_results = bm25_index.search(query, top_k=10)
    fuzzy_results = fuzzy_index.search(query, top_k=10)

    return reciprocal_rank_fusion(
        [vector_results, bm25_results, fuzzy_results],
        top_k=top_k,
    )
