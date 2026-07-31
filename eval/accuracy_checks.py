"""
Three quality checks that Recall@5/MRR alone don't cover:

1. Entity/relation extraction accuracy — did the LLM extraction actually
   find the entities you'd expect a human to find in each chunk?
2. Citation correctness — does a returned chunk_id actually trace back
   to a real, correct source document?
3. Groundedness — does the returned chunk's text actually SUPPORT the
   query, or is it just loosely/coincidentally related?

This is semi-automated: it runs the checks and produces a report, but
entity accuracy and groundedness ultimately need a human to look at the
printed comparisons and judge pass/fail — that's normal for this kind
of eval, not a bug in the script.

Run from your project ROOT:
    python eval/accuracy_checks.py
"""
import json
import uuid
from pathlib import Path

from application.parsing3.docling_parser import parse_and_chunk
from application.graph2.entity_extraction_groq import extract_entities_and_relationships
from application.graph2.falkordb_client import add_entity, add_relationship

GOLDEN_SET_PATH = Path(__file__).parent / "golden_dataset.json"

# <-- EDIT: same test files as run_eval.py -->
TEST_DOCUMENT_PATHS = [
    r"E:\5.Project 2\codebase\project_2.pdf",
]


def check_entity_extraction_accuracy():
    """
    For each test document, extracts entities per chunk and prints them
    next to any expected_entities from golden_dataset.json entries tied
    to that document — so you can eyeball whether extraction is finding
    the entities a human would expect.
    """
    print("=" * 60)
    print("1. ENTITY/RELATION EXTRACTION ACCURACY")
    print("=" * 60)

    with open(GOLDEN_SET_PATH) as f:
        golden = json.load(f)

    for path in TEST_DOCUMENT_PATHS:
        doc_id = str(uuid.uuid4())
        parsed, chunks = parse_and_chunk(path, doc_id)
        print(f"\n--- {parsed.filename} ---")

        all_extracted_names = set()
        all_relationships = []

        for chunk in chunks:
            entities, relationships = extract_entities_and_relationships(chunk.text, doc_id)
            for e in entities:
                all_extracted_names.add(e.name)
            all_relationships.extend(relationships)

        print(f"Extracted {len(all_extracted_names)} unique entity names:")
        for name in sorted(all_extracted_names):
            print(f"   - {name}")

        print(f"\nExtracted {len(all_relationships)} relationships (sample of 10):")
        for rel in all_relationships[:10]:
            print(f"   - {rel.source_entity_id} -[{rel.relation}]-> {rel.target_entity_id}")

        # Compare against expected_entities from matching golden set entries
        expected_for_this_doc = set()
        for entry in golden["entries"]:
            if entry.get("source_document") == parsed.filename:
                expected_for_this_doc.update(entry.get("expected_entities", []))

        if expected_for_this_doc:
            missing = {
                exp for exp in expected_for_this_doc
                if not any(exp.lower() in name.lower() or name.lower() in exp.lower()
                           for name in all_extracted_names)
            }
            print(f"\nExpected entities from golden set: {sorted(expected_for_this_doc)}")
            if missing:
                print(f"⚠️  Possibly MISSED expected entities: {sorted(missing)}")
            else:
                print("✅ All expected entities appear to have been extracted (fuzzy name match)")


def check_citation_correctness():
    """
    Confirms every chunk_id format used in the retrieval layer actually
    decodes back to a real doc_id (chunk_id = f"{doc_id}_chunk_{i}") —
    catches silent corruption where a citation points nowhere real.
    """
    print("\n" + "=" * 60)
    print("2. CITATION CORRECTNESS")
    print("=" * 60)

    from application.retrieval.vector_search import vector_index

    if not vector_index._store:
        print("⚠️  vector_index is empty — run this after ingesting documents "
              "(e.g. run eval/run_eval.py first in the same session, or add "
              "ingestion here too).")
        return

    broken = 0
    for chunk_id, (_text, _emb, doc_id) in vector_index._store.items():
        if not chunk_id.startswith(doc_id):
            broken += 1
            print(f"⚠️  Citation mismatch: chunk_id '{chunk_id}' doesn't start with its doc_id '{doc_id}'")

    total = len(vector_index._store)
    print(f"Checked {total} chunk citations — {total - broken} correct, {broken} broken")


def check_groundedness_prompts(sample_size: int = 5):
    """
    Prints (query, retrieved chunk text) pairs for manual groundedness
    review — this genuinely needs a human judgment call: does the chunk
    text actually support answering the query, or just superficially
    overlap on keywords?
    """
    print("\n" + "=" * 60)
    print("3. GROUNDEDNESS — MANUAL REVIEW NEEDED")
    print("=" * 60)
    print("For each pair below, ask: does this chunk ACTUALLY support")
    print("answering the query, or just share some keywords?\n")

    from application.retrieval.vector_search import vector_index
    from application.retrieval.bm25_search import bm25_index
    from application.retrieval.fuzzy_search import fuzzy_index
    from application.retrieval.rrf_fusion import hybrid_search

    with open(GOLDEN_SET_PATH) as f:
        golden = json.load(f)

    for entry in golden["entries"][:sample_size]:
        query = entry["query"]
        fused = hybrid_search(query, vector_index, bm25_index, fuzzy_index, top_k=1)
        if not fused:
            print(f"Q: {query}\n   (no results)\n")
            continue
        chunk_id, score = fused[0]
        text = vector_index.get_text(chunk_id)
        print(f"Q: {query}")
        print(f"   [score={score:.4f}] {text[:250]}...")
        print()


if __name__ == "__main__":
    check_entity_extraction_accuracy()
    check_citation_correctness()
    check_groundedness_prompts()
