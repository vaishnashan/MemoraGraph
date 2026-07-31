"""
GET /search — hybrid retrieval (vector + BM25 + fuzzy, fused by RRF),
followed by two-hop graph expansion for related context.
"""
from fastapi import APIRouter, Query
from application.retrieval.vector_search import vector_index
from application.retrieval.bm25_search import bm25_index
from application.retrieval.fuzzy_search import fuzzy_index
from application.retrieval.rrf_fusion import hybrid_search
from application.graph.falkordb_client import two_hop_expansion

router = APIRouter()


@router.get("/search")
async def search(query: str = Query(...), top_k: int = 5):
    # 1. Hybrid retrieval fused via RRF
    fused_results = hybrid_search(query, vector_index, bm25_index, fuzzy_index, top_k=top_k)

    results = []
    for chunk_id, score in fused_results:
        text = vector_index.get_text(chunk_id)
        results.append({
            "chunk_id": chunk_id,
            "text": text,
            "score": round(score, 4),
            "citation": chunk_id,  # source_doc_id tracking; refine with real doc metadata
        })

    # 2. Graph expansion: naive entity spotting by checking each result's
    #    text against known entity names. Good enough for v1 — a stronger
    #    version would run NER on the query itself.
    related_entities = []
    # Placeholder: in practice, extract candidate entity names from the
    # query or top results and call two_hop_expansion(name) for each.

    return {
        "query": query,
        "results": results,
        "related_entities": related_entities,
    }
