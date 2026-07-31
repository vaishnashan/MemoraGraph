"""
Dense vector retrieval — now backed by Supabase Postgres + pgvector
(see sql/schema.sql), instead of an in-memory dict. This means:
  - Data survives restarts.
  - The FastAPI process and the MCP server process see the SAME data,
    since both read/write the same Postgres table.

Interface (add / search / get_text / get_chunks_by_doc) is kept
identical to the old in-memory version, so upload.py, search.py, and
mcp_server/server.py don't need to change how they call this.
"""
from application.embeddings4.embedd import embed_text
from application.storage1 import supabase_client as db


class VectorIndex:
    def add(self, chunk_id: str, text: str, embedding: list[float], doc_id: str) -> None:
        db.upsert_indexed_text(chunk_id, text, embedding=embedding, doc_id=doc_id)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Returns list of (chunk_id, similarity_score), best first."""
        query_emb = embed_text(query)
        rows = db.vector_search(query_emb, top_k=top_k)
        return [(row["item_id"], row["score"]) for row in rows]

    def get_text(self, chunk_id: str) -> str | None:
        return db.get_indexed_text(chunk_id)

    def get_chunks_by_doc(self, doc_id: str) -> list[str]:
        """Used by get_document_context — all chunk texts belonging to a document."""
        return db.get_texts_by_doc(doc_id)


# Single shared instance for the app — now a thin wrapper over Postgres,
# not actual storage, so it's safe/cheap to share across modules.
vector_index = VectorIndex()