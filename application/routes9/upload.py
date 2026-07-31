"""
POST /upload — the full write-time pipeline in one endpoint:

  file -> Docling parse+chunk -> [Supabase raw storage + metadata]
       -> [embed + index each chunk for hybrid retrieval]
       -> [extract entities/relations -> FalkorDB, with entity resolution]

Langfuse tracing (Day 4): the route itself is wrapped with @observe()
so every /upload call becomes one trace, with parsing, embedding, and
extraction showing up as nested spans underneath it automatically.
"""
import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form
from langfuse import observe

from application.parsing3.docling_parser import parse_and_chunk
from application.storage1 import supabase_client as db
from application.models6.schemas import DocumentMetadata
from application.embeddings4.embedd import embed_text
from application.retrieval5.vector_search import vector_index
from application.retrieval5.bm25_search import bm25_index
from application.retrieval5.fuzzy_search import fuzzy_index
from application.graph2.entity_extraction_groq import extract_entities_and_relationships
from application.graph2.falkordb_client import add_entity, add_relationship

router = APIRouter()

TEMP_UPLOAD_DIR = Path("/tmp/memoragraph_uploads")
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
@observe()
async def upload_document(user_id: str = Form(...), file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())

    local_path = TEMP_UPLOAD_DIR / f"{doc_id}_{file.filename}"
    with open(local_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    parsed, chunks = parse_and_chunk(str(local_path), doc_id)

    storage_path = f"{user_id}/{doc_id}_{file.filename}"
    db.upload_raw_file(str(local_path), storage_path)
    db.save_document_metadata(DocumentMetadata(
        doc_id=doc_id,
        filename=parsed.filename,
        file_type=parsed.file_type,
        user_id=user_id,
        raw_storage_path=storage_path,
    ))

    for chunk in chunks:
        embedding = embed_text(chunk.text)
        vector_index.add(chunk.chunk_id, chunk.text, embedding, doc_id)
        bm25_index.add(chunk.chunk_id, chunk.text)
        fuzzy_index.add(chunk.chunk_id, chunk.text)

        entities, relationships = extract_entities_and_relationships(chunk.text, doc_id)

        id_map: dict[str, str] = {}
        for entity in entities:
            canonical_id = add_entity(entity)
            id_map[entity.entity_id] = canonical_id

        for rel in relationships:
            rel.source_entity_id = id_map.get(rel.source_entity_id, rel.source_entity_id)
            rel.target_entity_id = id_map.get(rel.target_entity_id, rel.target_entity_id)
            add_relationship(rel)

    local_path.unlink(missing_ok=True)

    return {
        "doc_id": doc_id,
        "filename": parsed.filename,
        "num_chunks": len(chunks),
        "status": "ingested",
    }