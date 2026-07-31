"""
Standalone test for docling_parser.py — confirms Docling can parse a real
PDF/DOCX and that HybridChunker produces sensible, embeddable chunks.

Run from your project ROOT (the folder that contains the `application` folder):
    python application/parsing/test_docling_parser.py

Before running:
    pip install docling transformers

Put a real PDF or DOCX file path in TEST_FILE_PATH below — for example,
your own project_2.pdf, or any lecture PDF / Word doc you have on hand.
"""
import uuid

from application.parsing.docling_parser import parse_and_chunk

# <-- EDIT THIS to point at a real file on your machine -->
TEST_FILE_PATH = r"E:\5.Project 2\codebase\localtestdoc\project 3.pdf"

doc_id = str(uuid.uuid4())

print(f"Parsing: {TEST_FILE_PATH}")
try:
    parsed, chunks = parse_and_chunk(TEST_FILE_PATH, doc_id)
except FileNotFoundError:
    print(f"❌ File not found at {TEST_FILE_PATH} — update TEST_FILE_PATH in this script.")
    raise SystemExit(1)
except Exception as e:
    print(f"❌ Parsing failed: {e}")
    raise SystemExit(1)

print(f"✅ Parsed successfully")
print(f"   Filename:  {parsed.filename}")
print(f"   File type: {parsed.file_type}")
print(f"   Full text length: {len(parsed.full_text)} chars")

if not parsed.full_text.strip():
    print("⚠️  Parsed text is empty — check the file isn't a scanned image-only PDF without OCR.")

print(f"\n✅ Chunked into {len(chunks)} chunks")

if not chunks:
    print("⚠️  No chunks produced — something's wrong with chunking, inspect docling_document directly.")
else:
    print("\n--- First 3 chunks (preview) ---")
    for c in chunks[:3]:
        preview = c.text[:200].replace("\n", " ")
        print(f"\n[{c.chunk_index}] chunk_id={c.chunk_id}")
        print(f"    {preview}...")

    print(f"\n--- Last chunk (preview) ---")
    last = chunks[-1]
    preview = last.text[:200].replace("\n", " ")
    print(f"[{last.chunk_index}] chunk_id={last.chunk_id}")
    print(f"    {preview}...")

    avg_len = sum(len(c.text) for c in chunks) / len(chunks)
    print(f"\n✅ Average chunk length: {avg_len:.0f} chars")

print("\n🎉 Docling parsing + chunking pipeline is working.")