"""
Handles raw file storage + document/memory metadata in Supabase, plus the
persisted retrieval store: vector / BM25 / fuzzy search all live in one
Postgres table (`indexed_texts`) instead of in-memory dicts.

Run schema.sql once in the Supabase SQL Editor before using the functions
below — it creates the pgvector/pg_trgm extensions, the `documents`,
`memories`, and `indexed_texts` tables, and the RPC functions this file calls.
"""
from supabase import create_client, Client

from application.ingestion.config import settings
from application.ingestion.schemas import DocumentMetadata, MemoryRecord, MemoryStatus


def get_client() -> Client:
    return create_client(settings.supabase_url, settings.supabase_secret_key)


# ---------- Raw file + document metadata ----------

def upload_raw_file(local_path: str, storage_path: str) -> str:
    """Upload the original file to Supabase Storage. Returns storage path."""
    client = get_client()
    with open(local_path, "rb") as f:
        client.storage.from_(settings.supabase_storage_bucket).upload(storage_path, f)
    return storage_path


def save_document_metadata(meta: DocumentMetadata) -> None:
    client = get_client()
    client.table("documents").insert(meta.model_dump(mode="json")).execute()


def get_document_metadata(doc_id: str) -> DocumentMetadata | None:
    """Used by get_document_context — pulls filename/type/upload time for a doc."""
    client = get_client()
    res = client.table("documents").select("*").eq("doc_id", doc_id).execute()
    if not res.data:
        return None
    return DocumentMetadata(**res.data[0])


def list_documents_by_user(user_id: str) -> list[DocumentMetadata]:
    """Used by the list_documents MCP tool — every doc a user has uploaded."""
    client = get_client()
    res = client.table("documents").select("*").eq("user_id", user_id).execute()
    return [DocumentMetadata(**row) for row in res.data]


# ---------- Memory records ----------

def save_memory(memory: MemoryRecord) -> None:
    client = get_client()
    client.table("memories").insert(memory.model_dump(mode="json")).execute()


def get_memory(memory_id: str) -> MemoryRecord | None:
    client = get_client()
    res = client.table("memories").select("*").eq("memory_id", memory_id).execute()
    if not res.data:
        return None
    return MemoryRecord(**res.data[0])


def update_memory_status(memory_id: str, status: MemoryStatus, superseded_by: str | None = None) -> None:
    client = get_client()
    client.table("memories").update({
        "status": status.value,
        "superseded_by": superseded_by,
    }).eq("memory_id", memory_id).execute()


def delete_memory(memory_id: str) -> None:
    client = get_client()
    client.table("memories").delete().eq("memory_id", memory_id).execute()
    # Also remove it from the retrieval store, or it'll keep showing up in
    # vector/BM25/fuzzy search results even after "deletion".
    delete_indexed_text(memory_id)


def list_active_memories(user_id: str | None = None) -> list[MemoryRecord]:
    client = get_client()
    query = client.table("memories").select("*").eq("status", MemoryStatus.ACTIVE.value)
    res = query.execute()
    return [MemoryRecord(**row) for row in res.data]


# ---------- Persisted retrieval store ----------
# One row per chunk_id/memory_id. A single write here makes the item
# searchable by all three retrieval methods, since BM25 (tsvector) and
# fuzzy (trigram) are computed by Postgres from the `text` column
# automatically; only the embedding needs to be supplied explicitly.

def upsert_indexed_text(
    item_id: str,
    text: str,
    embedding: list[float] | None = None,
    doc_id: str | None = None,
    user_id: str | None = None,
) -> None:
    """
    Insert or update one row in indexed_texts. Safe to call again later
    with an embedding for a row that was first written without one.
    """
    client = get_client()
    payload = {"item_id": item_id, "text": text}
    if embedding is not None:
        payload["embedding"] = embedding
    if doc_id is not None:
        payload["doc_id"] = doc_id
    if user_id is not None:
        payload["user_id"] = user_id
    client.table("indexed_texts").upsert(payload, on_conflict="item_id").execute()


def delete_indexed_text(item_id: str) -> None:
    client = get_client()
    client.table("indexed_texts").delete().eq("item_id", item_id).execute()


def get_indexed_text(item_id: str) -> str | None:
    client = get_client()
    res = client.table("indexed_texts").select("text").eq("item_id", item_id).execute()
    return res.data[0]["text"] if res.data else None


def get_texts_by_doc(doc_id: str) -> list[str]:
    """Used by get_document_context — all chunk texts belonging to a document."""
    client = get_client()
    res = client.table("indexed_texts").select("text").eq("doc_id", doc_id).execute()
    return [row["text"] for row in res.data]


def vector_search(query_embedding: list[float], top_k: int = 10) -> list[dict]:
    """Returns [{item_id, text, doc_id, score}, ...] ranked by cosine similarity."""
    client = get_client()
    res = client.rpc(
        "match_vector",
        {"query_embedding": query_embedding, "match_count": top_k},
    ).execute()
    return res.data or []


def bm25_search(query_text: str, top_k: int = 10) -> list[dict]:
    """Returns [{item_id, text, doc_id, score}, ...] ranked by ts_rank."""
    client = get_client()
    res = client.rpc(
        "match_bm25",
        {"query_text": query_text, "match_count": top_k},
    ).execute()
    return res.data or []


def fuzzy_search(query_text: str, top_k: int = 10) -> list[dict]:
    """Returns [{item_id, text, doc_id, score}, ...] ranked by trigram similarity."""
    client = get_client()
    res = client.rpc(
        "match_fuzzy",
        {"query_text": query_text, "match_count": top_k},
    ).execute()
    return res.data or []


def find_duplicate_memory(query_embedding: list[float], similarity_threshold: float = 0.95) -> str | None:
    """
    Used by memory/lifecycle.py's duplicate detection. Does the nearest-
    neighbor check in one indexed Postgres query instead of looping over
    every active memory in Python and re-embedding each one.
    """
    client = get_client()
    res = client.rpc(
        "match_active_memory",
        {"query_embedding": query_embedding, "similarity_threshold": similarity_threshold},
    ).execute()
    if not res.data:
        return None
    row = res.data[0]
    return row["item_id"] if row["score"] >= similarity_threshold else None
