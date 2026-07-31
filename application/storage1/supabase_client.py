"""
Handles raw file storage + document/memory metadata in Supabase.
"""
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from application.models.schemas import DocumentMetadata, MemoryRecord, MemoryStatus

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SECRET_KEY = os.environ["SUPABASE_SECRET_KEY"]
SUPABASE_BUCKET = os.environ["SUPABASE_STORAGE_BUCKET"]


def get_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)


def upload_raw_file(local_path: str, storage_path: str) -> str:
    """Upload the original file to Supabase Storage. Returns storage path."""
    client = get_client()
    with open(local_path, "rb") as f:
        client.storage.from_(SUPABASE_BUCKET).upload(storage_path, f)
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


def list_active_memories(user_id: str | None = None) -> list[MemoryRecord]:
    client = get_client()
    query = client.table("memories").select("*").eq("status", MemoryStatus.ACTIVE.value)
    res = query.execute()
    return [MemoryRecord(**row) for row in res.data]