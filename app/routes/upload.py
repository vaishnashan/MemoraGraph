"""
POST /upload — the full write-time pipeline in one endpoint:

  file -> Docling parse -> chunk -> [Supabase raw storage] and
  [embed + index] and [extract entities/relations -> FalkorDB]
"""
import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form

from app.parsing.docling_parser import parse_document, chunk_text
from app.storage import supabase_client as db
from app.models.schemas import DocumentMetadata
from app.embeddings.ollama_embedder import embed_text
from app.retrieval.vector_search import vector_index
from app.retrieval.bm25_search import bm25_index
from app.retrieval.fuzzy_search import fuzzy_index
from app.graph.entity_extraction import extract_entities_and_relationships
from app.graph.falkordb_client import add_entity, add_relationship

router = APIRouter()

TEMP_UPLOAD_DIR = Path("/tmp/memoragraph_uploads")
TEMP_UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/upload")
async def upload_document(user_id: str = Form(...), file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())

    # 1. Save the incoming file locally, then hand off to Docling
    local_path = TEMP_UPLOAD_DIR / f"{doc_id}_{file.filename}"
    with open(local_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 2. Parse with Docling into unified structured text
    parsed = parse_document(str(local_path), doc_id)

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

    # 4. Chunk, embed, and index for hybrid retrieval
    chunks = chunk_text(parsed.full_text)
    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{i}"
        embedding = embed_text(chunk)
        vector_index.add(chunk_id, chunk, embedding, doc_id)
        bm25_index.add(chunk_id, chunk)
        fuzzy_index.add(chunk_id, chunk)

        # 5. Extract entities/relationships per chunk and write to FalkorDB
        entities, relationships = extract_entities_and_relationships(chunk, doc_id)
        for entity in entities:
            add_entity(entity)
        for rel in relationships:
            add_relationship(rel)

    return {
        "doc_id": doc_id,
        "filename": parsed.filename,
        "num_chunks": len(chunks),
        "status": "ingested",
    }
