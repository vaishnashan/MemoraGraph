"""
Basic smoke tests — not exhaustive, just enough to confirm each layer
imports and behaves sanely before you wire everything together.

Run with: pytest tests/
"""
from application.parsing3.docling_parser import chunk_text
from application.retrieval5.rrf_fusion import reciprocal_rank_fusion


def test_chunk_text_basic():
    text = "a" * 2000
    chunks = chunk_text(text, chunk_size=800, overlap=150)
    assert len(chunks) > 1
    assert all(len(c) <= 800 for c in chunks)


def test_chunk_text_empty():
    assert chunk_text("") == []


def test_rrf_fusion_combines_rankings():
    vector_results = [("doc1", 0.9), ("doc2", 0.8)]
    bm25_results = [("doc2", 5.0), ("doc1", 3.0)]
    fuzzy_results = [("doc1", 0.7)]

    fused = reciprocal_rank_fusion([vector_results, bm25_results, fuzzy_results], top_k=2)

    assert len(fused) == 2
    # doc1 appears in all three lists, should rank highest
    assert fused[0][0] == "doc1"


def test_rrf_fusion_empty_lists():
    assert reciprocal_rank_fusion([[], [], []], top_k=5) == []
