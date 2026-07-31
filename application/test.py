"""
Full end-to-end test: runs the exact same steps as POST /upload followed
by GET /search, calling the pipeline functions directly (no server needed).

This is the real "Day 7" test — proves parsing, storage, embeddings,
BM25, fuzzy, RRF, and the graph all work together on one real file,
not just individually.

Run from your project ROOT:
    python application/test_end_to_end_pipeline.py

Edit TEST_FILE_PATH below to point at a real PDF/DOCX/XLSX/image.
"""
import uuid

from application.parsing3.docling_parser import parse_and_chunk
from application.storage1 import supabase_client as db
from application.models6.schemas import DocumentMetadata
from application.embeddings4.embedd import embed_text
from application.retrieval5.vector_search import vector_index
from application.retrieval5.bm25_search import bm25_index
from application.retrieval5.fuzzy_search import fuzzy_index
from application.retrieval5.rrf_fusion import hybrid_search
from application.graph2.entity_extraction_groq import extract_entities_and_relationships
from application.graph2.falkordb_client import add_entity, add_relationship, two_hop_expansion, find_entity_by_name

# <-- EDIT THIS to point at a real file on your machine -->
TEST_FILE_PATH = r"E:\5.Project 2\codebase\localtestdoc\project 3.pdf"
TEST_USER_ID = "test-user-e2e"

# <-- EDIT THIS to a query you expect to find in your test file's content -->
TEST_SEARCH_QUERY = "graph store"

doc_id = str(uuid.uuid4())

print("=" * 60)
print("STEP 1: Parse + chunk")
print("=" * 60)
parsed, chunks = parse_and_chunk(TEST_FILE_PATH, doc_id)
print(f"✅ Parsed '{parsed.filename}' -> {len(chunks)} chunks")

print("\n" + "=" * 60)
print("STEP 2: Store raw file + metadata in Supabase")
print("=" * 60)
storage_path = f"{TEST_USER_ID}/{doc_id}_{parsed.filename}"
db.upload_raw_file(TEST_FILE_PATH, storage_path)
db.save_document_metadata(DocumentMetadata(
    doc_id=doc_id,
    filename=parsed.filename,
    file_type=parsed.file_type,
    user_id=TEST_USER_ID,
    raw_storage_path=storage_path,
))
print(f"✅ Stored raw file at '{storage_path}' and saved metadata")

print("\n" + "=" * 60)
print("STEP 3: Embed + index chunks, extract + resolve entities")
print("=" * 60)
total_entities = 0
total_relationships = 0

for chunk in chunks:
    embedding = embed_text(chunk.text)
    vector_index.add(chunk.chunk_id, chunk.text, embedding, doc_id)
    bm25_index.add(chunk.chunk_id, chunk.text)
    fuzzy_index.add(chunk.chunk_id, chunk.text)

    entities, relationships = extract_entities_and_relationships(chunk.text, doc_id)

    id_map = {}
    for entity in entities:
        canonical_id = add_entity(entity)
        id_map[entity.entity_id] = canonical_id
    for rel in relationships:
        rel.source_entity_id = id_map.get(rel.source_entity_id, rel.source_entity_id)
        rel.target_entity_id = id_map.get(rel.target_entity_id, rel.target_entity_id)
        add_relationship(rel)

    total_entities += len(entities)
    total_relationships += len(relationships)

print(f"✅ Indexed {len(chunks)} chunks into vector/BM25/fuzzy indexes")
print(f"✅ Extracted {total_entities} entities and {total_relationships} relationships across all chunks")

print("\n" + "=" * 60)
print(f"STEP 4: Search — query = '{TEST_SEARCH_QUERY}'")
print("=" * 60)
fused_results = hybrid_search(TEST_SEARCH_QUERY, vector_index, bm25_index, fuzzy_index, top_k=5)

if not fused_results:
    print("⚠️  No search results returned — check your query matches the document's real content.")
else:
    print(f"✅ Got {len(fused_results)} fused results:")
    for chunk_id, score in fused_results:
        text_preview = vector_index.get_text(chunk_id)[:120].replace("\n", " ")
        print(f"   [{score:.4f}] {chunk_id}: {text_preview}...")

print("\n" + "=" * 60)
print("STEP 5: Graph expansion check")
print("=" * 60)
if fused_results:
    # Try two-hop expansion on any entity name that shows up in this doc
    sample_query_word = TEST_SEARCH_QUERY.split()[0]
    matched = find_entity_by_name(sample_query_word)
    if matched:
        related = two_hop_expansion(matched["name"])
        print(f"✅ Found entity '{matched['name']}', two-hop expansion returned {len(related)} related entities")
        for r in related[:5]:
            print(f"   - {r['name']} ({r['type']}) via {r['relation_path']}")
    else:
        print(f"ℹ️  No entity named '{sample_query_word}' found directly — try a different TEST_SEARCH_QUERY word "
              f"that matches an actual entity extracted from your document.")

print("\n" + "=" * 60)
print("🎉 END-TO-END PIPELINE TEST COMPLETE")
print("=" * 60)
print(f"doc_id for this run: {doc_id}")
print("You can now check Supabase (documents table + storage bucket) and")
print("FalkorDB (browser UI) to visually confirm this document's data landed correctly.")