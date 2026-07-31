"""
POST /upload — the full write-time pipeline in one endpoint:

  file -> Docling parse+chunk -> [Supabase raw storage + metadata]
       -> [embed + index each chunk for hybrid retrieval]
       -> [extract entities/relations -> FalkorDB, with entity resolution]
"""
import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form

from application.parsing3.docling_parser import parse_and_chunk
from application.storage1 import supabase_client as db
from application.models.schemas import DocumentMetadata
from application.embeddings.embedd import embed_text
from application.retrieval.vector_search import vector_index
from application.retrieval.bm25_search import bm25_index
from application.retrieval.fuzzy_search import fuzzy_index
from application.graph2.entity_extraction_groq import extract_entities_and_relationships
from application.graph2.falkordb_client import add_entity, add_relationship

router = APIRouter()

TEMP_UPLOAD_DIR = Path("/tmp/memoragraph_uploads")
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_document(user_id: str = Form(...), file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())

    # 1. Save the incoming file locally, then hand off to Docling
    local_path = TEMP_UPLOAD_DIR / f"{doc_id}_{file.filename}"
    with open(local_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 2. Parse + chunk with Docling (structure-aware, tokenizer-aware chunks)
    parsed, chunks = parse_and_chunk(str(local_path), doc_id)

    # 3. Store raw file + metadata in Supabase
    storage_path = f"{user_id}/{doc_id}_{file.filename}"
    db.upload_raw_file(str(local_path), storage_path)
    db.save_document_metadata(DocumentMetadata(
        doc_id=doc_id,
        filename=parsed.filename,
        file_type=parsed.file_type,
        user_id=user_id,
        raw_storage_path=storage_path,
    ))

    # 4. Embed + index each chunk, and extract + resolve entities/relationships
    for chunk in chunks:
        embedding = embed_text(chunk.text)
        vector_index.add(chunk.chunk_id, chunk.text, embedding, doc_id)
        bm25_index.add(chunk.chunk_id, chunk.text)
        fuzzy_index.add(chunk.chunk_id, chunk.text)

        entities, relationships = extract_entities_and_relationships(chunk.text, doc_id)

        # Entity resolution: add_entity() may return a DIFFERENT id than
        # entity.entity_id if this real-world entity already exists in the
        # graph (e.g. "FalkorDB" seen in an earlier chunk/document). Build
        # a map from this chunk's local ids -> canonical graph ids, and use
        # it to remap relationships before writing them.
        id_map: dict[str, str] = {}
        for entity in entities:
            canonical_id = add_entity(entity)
            id_map[entity.entity_id] = canonical_id

        for rel in relationships:
            rel.source_entity_id = id_map.get(rel.source_entity_id, rel.source_entity_id)
            rel.target_entity_id = id_map.get(rel.target_entity_id, rel.target_entity_id)
            add_relationship(rel)

    # Clean up the temp local copy now that it's safely in Supabase
    local_path.unlink(missing_ok=True)

    return {
        "doc_id": doc_id,
        "filename": parsed.filename,
        "num_chunks": len(chunks),
        "status": "ingested",
    }