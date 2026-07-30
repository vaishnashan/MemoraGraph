# MemoraGraph

Private multimodal AI memory infrastructure exposed through the Model Context Protocol (MCP).

A reusable memory service that external AI agents (Claude-compatible clients, LangGraph apps, etc.)
can call as a tool — to store, search, update, and forget memories, backed by hybrid retrieval
(vector + BM25 + fuzzy) and a knowledge graph (FalkorDB) for relationship-aware context.

## Architecture

```
PDF / DOCX
    │
    ▼
Docling (parse into unified structured text)
    │
    ├──► Supabase (raw file + document metadata)
    │
    ├──► Chunk → Ollama embeddings ──► Vector index ─┐
    │                                  BM25 index    ├─► RRF fusion ──► ranked results
    │                                  Fuzzy index   ─┘
    │
    └──► LLM entity/relation extraction ──► FalkorDB (property graph)
                                                  │
                                                  ▼
                                    Two-hop graph expansion (related entities)

                    All of the above wrapped as 5 MCP tools:
        store_memory · search_memory · find_related_entities
                    update_memory · forget_memory
                              │
                              ▼
                Permission checks + audit log (security layer)
```

## Project structure

```
app/
  main.py                 FastAPI entrypoint (upload + search HTTP endpoints)
  config.py               Settings loaded from .env
  models/schemas.py        Pydantic models shared across all layers
  parsing/docling_parser.py       PDF/DOCX parsing + chunking
  storage/supabase_client.py      Raw file + metadata storage
  embeddings/ollama_embedder.py   Local embeddings
  retrieval/
    vector_search.py       Dense retrieval
    bm25_search.py          Keyword retrieval
    fuzzy_search.py         Fuzzy text matching
    rrf_fusion.py           Combines all three via RRF
  graph/
    falkordb_client.py     Graph writes + two-hop expansion
    entity_extraction.py   LLM-based entity/relation extraction
  memory/lifecycle.py      Store/update/forget + dedup + versioning
  security/permissions.py  Access control + audit log
  routes/                  FastAPI HTTP routes
  mcp_server/server.py     MCP server exposing the 5 tools

eval/
  golden_dataset.json      Test queries + expected results
  run_eval.py              Recall@5 / MRR scoring

tests/test_basic.py        Smoke tests
docker-compose.yml          FalkorDB + Ollama + backend
```

## Setup

1. **Install dependencies**
   ```bash
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Environment**
   ```bash
   cp .env.example .env
   # fill in SUPABASE_URL, SUPABASE_KEY, etc.
   ```

3. **Start FalkorDB + Ollama**
   ```bash
   docker-compose up falkordb ollama -d
   ollama pull nomic-embed-text
   ollama pull llama3.1   # or whatever you set as EXTRACTION_MODEL
   ```

4. **Create Supabase tables** — run the SQL in `app/storage/supabase_client.py`'s
   docstring in the Supabase SQL editor, and create a storage bucket named
   `memoragraph-files` (or whatever you set in `.env`).

5. **Run the API**
   ```bash
   uvicorn app.main:app --reload
   ```
   Visit `http://localhost:8000/docs` for interactive API docs.

6. **Run the MCP server** (separate process, for AI agents to connect to)
   ```bash
   python -m app.mcp_server.server
   ```

7. **Run tests**
   ```bash
   pytest tests/
   ```

8. **Run eval** (after uploading test documents and filling in `eval/golden_dataset.json`)
   ```bash
   python eval/run_eval.py
   ```

## Day-by-day build order

Follow the 3-week plan you already have:
- **Week 1**: `parsing/` → `storage/` → `embeddings/` → `retrieval/` → `/search` endpoint
- **Week 2**: `graph/` → `memory/lifecycle.py` → `mcp_server/server.py` → `security/permissions.py`
- **Week 3**: `eval/` → Docker deployment → demo + README polish

## Notes

- `vector_index`, `bm25_index`, `fuzzy_index` are in-memory for v1 — fine for building and
  demoing. If you want persistence across restarts, back these with FalkorDB's own vector/full-text
  indexes or a dedicated store.
- Entity extraction runs per-chunk at upload time — tune the prompt in
  `entity_extraction.py` once you see real outputs on your sample documents.
- `forget_memory` will refuse to delete anything unless called with `confirm=true` —
  this is intentional (Security requirement: confirmation before deletion).
