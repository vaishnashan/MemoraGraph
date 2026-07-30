"""
Handles raw file storage + document/memory metadata in Supabase.

Expected tables (create these in the Supabase SQL editor on Day 1):

    documents (
        doc_id text primary key,
        filename text,
        file_type text,
        user_id text,
        raw_storage_path text,
        uploaded_at timestamptz default now()
    )

    memories (
        memory_id text primary key,
        doc_id text references documents(doc_id),
        text text,
        status text default 'active',
        superseded_by text,
        created_at timestamptz default now(),
        updated_at timestamptz default now()
    )
"""
from supabase import create_client, Client
from app.config import settings
from app.models.schemas import DocumentMetadata, MemoryRecord, MemoryStatus


def get_client() -> Client:
    return create_client(settings.supabase_url, settings.supabase_key)


def upload_raw_file(local_path: str, storage_path: str) -> str:
    """Upload the original file to Supabase Storage. Returns storage path."""
    client = get_client()
    with open(local_path, "rb") as f:
        client.storage.from_(settings.supabase_bucket).upload(storage_path, f)
    return storage_path


def save_document_metadata(meta: DocumentMetadata) -> None:
    client = get_client()
    client.table("documents").insert(meta.model_dump(mode="json")).execute()


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
