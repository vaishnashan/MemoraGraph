"""
Reciprocal Rank Fusion (RRF): combines rankings from vector, BM25, and
fuzzy search into one fair ranked list.

RRF score for a document = sum over each ranking list of 1 / (k + rank)
where rank is 1-indexed position in that list. k=60 is the standard
default from the original RRF paper — no need to tune unless you have
a reason to.
"""

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """
    ranked_lists: e.g. [vector_results, bm25_results, fuzzy_results]
                  each is a list of (chunk_id, score) already sorted best-first
    Returns: [(chunk_id, fused_score)] sorted best-first
    """
    fused_scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, (chunk_id, _original_score) in enumerate(ranked_list, start=1):
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    result = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return result[:top_k]


def hybrid_search(query: str, vector_index, bm25_index, fuzzy_index, top_k: int = 5):
    """Runs all three retrievers and fuses results. Import indexes at call site
    to avoid circular imports."""
    vector_results = vector_index.search(query, top_k=10)
    bm25_results = bm25_index.search(query, top_k=10)
    fuzzy_results = fuzzy_index.search(query, top_k=10)

    return reciprocal_rank_fusion(
        [vector_results, bm25_results, fuzzy_results],
        top_k=top_k,
    )
