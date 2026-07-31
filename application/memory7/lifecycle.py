"""
This is the piece that makes MemoraGraph a real memory system rather than
plain GraphRAG: duplicate detection, versioning, and marking memories as
active / outdated / superseded instead of just deleting or piling on.
Also home to get_document_context, which pulls together everything known
about a source document (metadata + extracted entities + stored chunks).

Persistence note: duplicate detection now uses one indexed Postgres
query (find_duplicate_memory, via the match_active_memory RPC) instead
of looping over every active memory in Python and re-embedding each one
on every single store_memory() call — same result, far less work as
your memory count grows. Storing a memory is now also a single
vector_index.add() call instead of three separate index writes, since
BM25/fuzzy are derived automatically from the same Postgres row.
"""
import uuid
from application.models6.schemas import MemoryRecord, MemoryStatus
from application.embeddings4.embedd import embed_text
from application.storage1 import supabase_client as db
from application.retrieval5.vector_search import vector_index
from application.graph2.falkordb_client import get_entities_by_doc_id

DUPLICATE_SIMILARITY_THRESHOLD = 0.95


def _is_duplicate(text: str) -> str | None:
    """Checks existing active memories for a near-identical match via a
    single indexed pgvector query. Returns the existing memory_id if a
    duplicate is found, else None."""
    new_emb = embed_text(text)
    return db.find_duplicate_memory(new_emb, similarity_threshold=DUPLICATE_SIMILARITY_THRESHOLD)


def store_memory(text: str, doc_id: str | None = None) -> tuple[MemoryRecord, str | None]:
    """
    Adds a new memory. If a near-duplicate already exists, returns the
    existing record instead of creating a new one (caller can decide
    what to tell the user/agent).
    """
    duplicate_id = _is_duplicate(text)
    if duplicate_id:
        existing = db.get_memory(duplicate_id)
        return existing, duplicate_id

    memory_id = str(uuid.uuid4())
    record = MemoryRecord(memory_id=memory_id, doc_id=doc_id or "", text=text, source_doc_id=doc_id)
    db.save_memory(record)

    # Index it for retrieval immediately (write-time processing) — one
    # write covers vector, BM25, and fuzzy search all at once now.
    embedding = embed_text(text)
    vector_index.add(memory_id, text, embedding, doc_id or "")

    return record, None


def update_memory(memory_id: str, new_text: str) -> MemoryRecord:
    """
    Doesn't overwrite — marks the old memory as 'superseded' and creates
    a new active memory pointing back at it. This preserves history,
    which is the whole point of temporal memory.
    """
    old_record = db.get_memory(memory_id)
    if not old_record:
        raise ValueError(f"Memory {memory_id} not found")

    new_record, duplicate_id = store_memory(new_text, doc_id=old_record.source_doc_id)
    if duplicate_id:
        db.update_memory_status(memory_id, MemoryStatus.SUPERSEDED, superseded_by=duplicate_id)
        return new_record

    db.update_memory_status(memory_id, MemoryStatus.SUPERSEDED, superseded_by=new_record.memory_id)
    return new_record


def forget_memory(memory_id: str, confirmed: bool) -> tuple[bool, str]:
    """
    Requires explicit confirmation before deleting — security requirement
    from the spec. Caller (MCP tool layer) is responsible for getting
    that confirmation from the user/agent before calling with confirmed=True.
    """
    if not confirmed:
        return False, "Deletion requires explicit confirmation. Pass confirm=true to proceed."

    record = db.get_memory(memory_id)
    if not record:
        return False, f"Memory {memory_id} not found."

    db.delete_memory(memory_id)  # also removes it from indexed_texts now
    return True, f"Memory {memory_id} deleted."


def get_document_context(doc_id: str) -> dict:
    """
    Pulls together everything MemoraGraph knows about a single source
    document: its metadata, the entities extracted from it, and its
    stored chunk text. Used by the get_document_context MCP tool so an
    agent can see full context on a source, not just an isolated
    retrieved chunk.
    """
    metadata = db.get_document_metadata(doc_id)
    entities = get_entities_by_doc_id(doc_id)
    chunks = vector_index.get_chunks_by_doc(doc_id)

    return {
        "doc_id": doc_id,
        "filename": metadata.filename if metadata else None,
        "file_type": metadata.file_type if metadata else None,
        "uploaded_at": metadata.uploaded_at.isoformat() if metadata else None,
        "found": metadata is not None,
        "entities": entities,
        "num_chunks": len(chunks),
        "chunks_preview": chunks[:3],
    }