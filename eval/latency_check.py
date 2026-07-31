"""
Measures retrieval latency across the golden set — how long does a
/search call actually take, end to end (vector + BM25 + fuzzy + RRF)?

Also includes a basic MCP tool-selection accuracy check: given a query,
does a simple keyword-based router pick the tool a human would expect?
This is a lightweight stand-in — real tool-selection accuracy is best
measured by actually running queries through an MCP-connected agent
(Claude, etc.) and logging which tool it picked, which needs a live
agent session rather than a standalone script.

Run from your project ROOT (after ingesting test docs — see run_eval.py):
    python eval/latency_check.py
"""
import json
import time
import uuid
from pathlib import Path
from statistics import mean, median

from application.parsing3.docling_parser import parse_and_chunk
from application.embeddings4.embedd import embed_text
from application.retrieval5.vector_search import vector_index
from application.retrieval5.bm25_search import bm25_index
from application.retrieval5.fuzzy_search import fuzzy_index
from application.retrieval5.rrf_fusion import hybrid_search

GOLDEN_SET_PATH = Path(__file__).parent / "golden_dataset.json"

TEST_DOCUMENT_PATHS = [
    r"E:\5.Project 2\codebase\project_2.pdf",
]


def ingest_test_documents():
    for path in TEST_DOCUMENT_PATHS:
        doc_id = str(uuid.uuid4())
        parsed, chunks = parse_and_chunk(path, doc_id)
        for chunk in chunks:
            embedding = embed_text(chunk.text)
            vector_index.add(chunk.chunk_id, chunk.text, embedding, doc_id)
            bm25_index.add(chunk.chunk_id, chunk.text)
            fuzzy_index.add(chunk.chunk_id, chunk.text)
        print(f"✅ Ingested {parsed.filename} ({len(chunks)} chunks)")


def measure_search_latency():
    print("=" * 60)
    print("RETRIEVAL LATENCY")
    print("=" * 60)

    with open(GOLDEN_SET_PATH) as f:
        golden = json.load(f)

    latencies_ms = []
    for entry in golden["entries"]:
        start = time.perf_counter()
        hybrid_search(entry["query"], vector_index, bm25_index, fuzzy_index, top_k=5)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

    print(f"Queries measured: {len(latencies_ms)}")
    print(f"Mean latency:     {mean(latencies_ms):.2f} ms")
    print(f"Median latency:   {median(latencies_ms):.2f} ms")
    print(f"Max latency:      {max(latencies_ms):.2f} ms")
    print(f"Min latency:      {min(latencies_ms):.2f} ms")


# --- Lightweight MCP tool-selection accuracy stand-in ---

TOOL_KEYWORDS = {
    "search_memory": ["what", "find", "search", "how", "who", "where", "when"],
    "store_memory": ["remember", "note that", "save", "store"],
    "update_memory": ["update", "change", "correct", "actually it's"],
    "forget_memory": ["forget", "delete", "remove"],
    "find_related_entities": ["related to", "connected to", "linked with"],
    "get_document_context": ["document", "file", "source", "uploaded"],
}


def guess_expected_tool(query: str) -> str:
    """Very rough keyword-based expected-tool guesser for the golden set —
    this is just a scoring baseline, not the real routing logic."""
    q = query.lower()
    for tool, keywords in TOOL_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return tool
    return "search_memory"  # default assumption for factual questions


def check_tool_selection_baseline():
    print("\n" + "=" * 60)
    print("MCP TOOL-SELECTION BASELINE (heuristic, not a real agent test)")
    print("=" * 60)
    print("NOTE: this only checks a simple keyword heuristic against your")
    print("golden queries. Real tool-selection accuracy should also be")
    print("measured by connecting an actual MCP client (e.g. Claude) and")
    print("logging which tool IT picks for each query — this script is a")
    print("quick proxy, not a substitute for that.\n")

    with open(GOLDEN_SET_PATH) as f:
        golden = json.load(f)

    for entry in golden["entries"][:10]:
        guessed = guess_expected_tool(entry["query"])
        print(f"'{entry['query']}' -> expected tool guess: {guessed}")


if __name__ == "__main__":
    ingest_test_documents()
    measure_search_latency()
    check_tool_selection_baseline()
