"""
The full write-time pipeline, as a plain function with no FastAPI/HTTP
concerns in it:

  file -> Docling parse+chunk -> [Supabase raw storage + metadata]
       -> [embed + index each chunk for hybrid retrieval, persisted]
       -> [extract entities/relations -> FalkorDB, with entity resolution]

Kept separate from ingestion/main.py so the pipeline can be called from
anywhere (a route, a CLI script, a test) without pulling in FastAPI.
"""
import uuid
from pathlib import Path

from langfuse import observe

from application.ingestion.docling_parser import parse_and_chunk
from application.ingestion import supabase_client as db
from application.ingestion.schemas import DocumentMetadata
from application.ingestion.embedder import embed_text
from application.ingestion.vector_search import vector_index
from application.ingestion.entity_extraction import extract_entities_and_relationships
from application.ingestion.falkordb_client import add_entity, add_relationship

TEMP_UPLOAD_DIR = Path("/tmp/memoragraph_uploads")
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@observe()
def ingest_file(local_path: str, filename: str, user_id: str) -> dict:
    """
    Runs the full write-time pipeline on a file already saved to disk at
    `local_path`. Returns a summary dict (doc_id, filename, num_chunks, status).
    """
    doc_id = str(uuid.uuid4())
    print(f"[ingest] START doc_id={doc_id} filename={filename}")

    print("[ingest] step 1/4: parsing + chunking (Docling)...")
    parsed, chunks = parse_and_chunk(local_path, doc_id)
    print(f"[ingest] step 1/4: OK — {len(chunks)} chunks")

    print("[ingest] step 2/4: uploading raw file + metadata to Supabase...")
    storage_path = f"{user_id}/{doc_id}_{filename}"
    db.upload_raw_file(local_path, storage_path)
    db.save_document_metadata(DocumentMetadata(
        doc_id=doc_id,
        filename=filename,
        file_type=parsed.file_type,
        user_id=user_id,
        raw_storage_path=storage_path,
    ))
    print("[ingest] step 2/4: OK")

    for i, chunk in enumerate(chunks, start=1):
        print(f"[ingest] step 3/4: chunk {i}/{len(chunks)} — embedding + indexing...")
        embedding = embed_text(chunk.text)
        vector_index.add(chunk.chunk_id, chunk.text, embedding, doc_id)

        print(f"[ingest] step 3/4: chunk {i}/{len(chunks)} — extracting entities (Groq)...")
        entities, relationships = extract_entities_and_relationships(chunk.text, doc_id)
        print(f"[ingest] step 3/4: chunk {i}/{len(chunks)} — {len(entities)} entities, {len(relationships)} relationships")

        print(f"[ingest] step 4/4: chunk {i}/{len(chunks)} — writing to FalkorDB...")
        id_map: dict[str, str] = {}
        for entity in entities:
            canonical_id = add_entity(entity)
            id_map[entity.entity_id] = canonical_id

        for rel in relationships:
            rel.source_entity_id = id_map.get(rel.source_entity_id, rel.source_entity_id)
            rel.target_entity_id = id_map.get(rel.target_entity_id, rel.target_entity_id)
            add_relationship(rel)
        print(f"[ingest] step 4/4: chunk {i}/{len(chunks)} — OK")

    print(f"[ingest] DONE doc_id={doc_id}")
    return {
        "doc_id": doc_id,
        "filename": filename,
        "num_chunks": len(chunks),
        "status": "ingested",
    }
