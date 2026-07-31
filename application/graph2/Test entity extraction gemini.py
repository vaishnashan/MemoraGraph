"""
Standalone test for entity_extraction_gemini.py — confirms your GEMINI_API_KEY
works and the model returns usable, parseable entity/relationship JSON.

Run from your project root:
    python test_entity_extraction_gemini.py
"""
import os

from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("GEMINI_API_KEY"):
    raise SystemExit("❌ Missing GEMINI_API_KEY in .env")

print("✅ Env vars loaded")
print(f"   Model: {os.environ.get('EXTRACTION_MODEL', 'gemini-2.5-flash')}")

from graph2.entity_extraction_gemini import extract_entities_and_relationships
SAMPLE_TEXT = """
Vaishnavi is building MemoraGraph, an AI memory infrastructure project.
MemoraGraph uses FalkorDB as its graph store and Supabase for raw file storage.
The project deadline for Week 1 is this Friday.
"""

try:
    entities, relationships = extract_entities_and_relationships(
        text=SAMPLE_TEXT, doc_id="test-doc-001"
    )
except Exception as e:
    print(f"❌ Extraction call failed: {e}")
    raise SystemExit(1)

if not entities:
    print("⚠️  No entities extracted — check the model output / prompt formatting.")
else:
    print(f"✅ Extracted {len(entities)} entities:")
    for e in entities:
        print(f"   - {e.name} ({e.type})")

if not relationships:
    print("⚠️  No relationships extracted.")
else:
    print(f"✅ Extracted {len(relationships)} relationships:")
    for r in relationships:
        print(f"   - {r.source_entity_id} -[{r.relation}]-> {r.target_entity_id}")

if entities:
    print("\n🎉 Gemini entity extraction is working.")
else:
    print("\n⚠️  Call succeeded but nothing was extracted — inspect raw output/prompt.")