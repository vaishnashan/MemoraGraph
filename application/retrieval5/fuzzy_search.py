"""
Fuzzy text matching — now Postgres trigram similarity (pg_trgm, via the
match_fuzzy RPC in sql/schema.sql) instead of an in-memory rapidfuzz
index. Persists across restarts and is shared between the FastAPI and
MCP server processes.

`add()` is now a no-op — same reason as bm25_search.py: the text is
already written once by VectorIndex.add(), and Postgres' trigram index
is built from that same `text` column automatically.
"""
from application.storage1 import supabase_client as db


class FuzzyIndex:
    def add(self, chunk_id: str, text: str) -> None:
        # No-op — see module docstring.
        pass

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        rows = db.fuzzy_search(query, top_k=top_k)
        return [(row["item_id"], row["score"]) for row in rows]


fuzzy_index = FuzzyIndex()