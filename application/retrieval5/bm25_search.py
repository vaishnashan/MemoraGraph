"""
BM25-equivalent keyword retrieval — now Postgres full-text search
(tsvector + ts_rank, via the match_bm25 RPC in sql/schema.sql) instead
of an in-memory rank_bm25 index. Persists across restarts and is shared
between the FastAPI and MCP server processes.

`add()` is now a no-op: the row (and its text) is already written by
VectorIndex.add() in upload.py, and Postgres computes the `tsv` column
from that same `text` automatically via a generated column — there's
nothing extra to store here. Kept only so existing call sites that do
`bm25_index.add(...)` don't break; feel free to remove those calls.
"""
from application.storage1 import supabase_client as db


class BM25Index:
    def add(self, chunk_id: str, text: str) -> None:
        # No-op — see module docstring. Left as a harmless call for
        # backward compatibility with existing upload.py / lifecycle.py
        # call sites during migration.
        pass

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        rows = db.bm25_search(query, top_k=top_k)
        return [(row["item_id"], row["score"]) for row in rows]


bm25_index = BM25Index()