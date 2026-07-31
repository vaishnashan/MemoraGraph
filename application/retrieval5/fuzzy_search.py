"""
Fuzzy text matching — catches typos, partial names, and near-matches
that neither vector nor BM25 handle well (e.g. "Falkor DB" vs "FalkorDB").
"""
from rapidfuzz import fuzz


class FuzzyIndex:
    def __init__(self):
        self._store: dict[str, str] = {}  # chunk_id -> text

    def add(self, chunk_id: str, text: str) -> None:
        self._store[chunk_id] = text

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        scored = [
            (chunk_id, fuzz.partial_ratio(query.lower(), text.lower()))
            for chunk_id, text in self._store.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        # normalize 0-100 -> 0-1 to match the other retrievers' scale
        return [(cid, score / 100.0) for cid, score in scored[:top_k]]


fuzzy_index = FuzzyIndex()
