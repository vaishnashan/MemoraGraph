"""
BM25 keyword retrieval — catches exact-term matches that vector search
can miss (names, IDs, specific jargon).
"""
from rank_bm25 import BM25Okapi


class BM25Index:
    def __init__(self):
        self._chunk_ids: list[str] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def add(self, chunk_id: str, text: str) -> None:
        self._chunk_ids.append(chunk_id)
        self._tokenized_corpus.append(self._tokenize(text))
        # Rebuild index after each add — fine for small/medium corpora.
        # For large corpora, batch adds and rebuild once.
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(self._tokenize(query))
        scored = list(zip(self._chunk_ids, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


bm25_index = BM25Index()
