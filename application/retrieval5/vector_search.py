"""
Dense vector retrieval. For a portfolio project, an in-memory index is
fine to start — swap for FalkorDB's vector index or a proper vector DB
once you outgrow this.
"""
from application.embeddings4.embedd import embed_text, cosine_similarity


class VectorIndex:
    def __init__(self):
        # In-memory store: {chunk_id: (text, embedding, doc_id)}
        self._store: dict[str, tuple[str, list[float], str]] = {}

    def add(self, chunk_id: str, text: str, embedding: list[float], doc_id: str) -> None:
        self._store[chunk_id] = (text, embedding, doc_id)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Returns list of (chunk_id, similarity_score), best first."""
        query_emb = embed_text(query)
        scored = [
            (chunk_id, cosine_similarity(query_emb, emb))
            for chunk_id, (_, emb, _) in self._store.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_text(self, chunk_id: str) -> str | None:
        entry = self._store.get(chunk_id)
        return entry[0] if entry else None

    def get_chunks_by_doc(self, doc_id: str) -> list[str]:
        """Used by get_document_context — all chunk texts belonging to a document."""
        return [text for (text, _, d) in self._store.values() if d == doc_id]


# Single shared instance for the app (swap for a persisted store later)
vector_index = VectorIndex()