"""
BM25-equivalent keyword retrieval — Postgres full-text search (tsvector +
ts_rank, via the match_bm25 RPC in schema.sql). Persists across restarts
and is shared between the ingestion and MCP server processes.

`add()` is a no-op: the row (and its text) is already written by
VectorIndex.add() during ingestion, and Postgres computes the `tsv` column
from that same `text` automatically via a generated column. Kept only so
existing call sites that do `bm25_index.add(...)` don't break.
"""
from application.mcp import supabase_client as db


class BM25Index:
    def add(self, chunk_id: str, text: str) -> None:
        # No-op — see module docstring.
        pass

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        rows = db.bm25_search(query, top_k=top_k)
        return [(row["item_id"], row["score"]) for row in rows]


bm25_index = BM25Index()
