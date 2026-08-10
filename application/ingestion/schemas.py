"""
Pydantic models shared across parsing, storage, retrieval, graph, ingestion,
and MCP. Keeping these in one place means the MCP tool schemas and the
internal function signatures never drift apart.
"""
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    OUTDATED = "outdated"
    SUPERSEDED = "superseded"


class DocumentMetadata(BaseModel):
    doc_id: str
    filename: str
    file_type: str  # "pdf" | "docx" | "xlsx" | "png" | "jpg" | ...
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    user_id: str
    raw_storage_path: str


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    embedding: Optional[list[float]] = None


class Entity(BaseModel):
    entity_id: str
    name: str
    type: str  # e.g. "person", "project", "concept", "deadline"
    source_doc_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Relationship(BaseModel):
    source_entity_id: str
    target_entity_id: str
    relation: str  # e.g. "works_on", "deployed_on"
    source_doc_id: str


class MemoryRecord(BaseModel):
    """A single stored memory (a chunk + its lifecycle state)."""
    memory_id: str
    doc_id: str
    text: str
    status: MemoryStatus = MemoryStatus.ACTIVE
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    superseded_by: Optional[str] = None
    source_doc_id: Optional[str] = None


# ---------- MCP tool I/O schemas ----------

class StoreMemoryInput(BaseModel):
    user_id: str
    text: str
    source_doc_id: Optional[str] = None
    metadata: Optional[dict] = None


class StoreMemoryOutput(BaseModel):
    memory_id: str
    status: MemoryStatus
    duplicate_of: Optional[str] = None


class SearchMemoryInput(BaseModel):
    user_id: str
    query: str
    top_k: int = 5


class SearchResultItem(BaseModel):
    memory_id: str
    text: str
    score: float
    source_doc_id: Optional[str] = None
    citation: Optional[str] = None


class SearchMemoryOutput(BaseModel):
    results: list[SearchResultItem]


class FindRelatedEntitiesInput(BaseModel):
    entity_name: str
    max_hops: int = 2


class RelatedEntity(BaseModel):
    name: str
    type: str
    relation_path: list[str]


class FindRelatedEntitiesOutput(BaseModel):
    related: list[RelatedEntity]


class UpdateMemoryInput(BaseModel):
    memory_id: str
    new_text: str
    user_id: str


class UpdateMemoryOutput(BaseModel):
    old_memory_id: str
    new_memory_id: str
    status: MemoryStatus


class ForgetMemoryInput(BaseModel):
    memory_id: str
    user_id: str
    confirm: bool = False


class ForgetMemoryOutput(BaseModel):
    memory_id: str
    deleted: bool
    message: str
