# MemoraGraph

Private multimodal AI memory infrastructure, exposed to AI agents via MCP.

`application/` has exactly two folders in it, on purpose:

- **`application/ingestion`** — a tiny FastAPI app with exactly two HTTP endpoints:
  `GET /health` and `POST /upload`. Its only job is to take a file in and run it
  through the write-time pipeline (parse → chunk → embed → store → extract
  entities/relations → graph write). Nothing else is exposed over HTTP.
- **`application/mcp`** — the MCP server. This is the *only* place search and
  memory-management operations live (`search_memory`, `store_memory`,
  `find_related_entities`, `update_memory`, `forget_memory`, etc. — 11 tools
  total). AI agents talk to it over stdio via MCP, not HTTP.

Everything else both apps depend on — config, schemas, parsing, embeddings,
storage, the graph client, retrieval, memory lifecycle — is a flat module
directly under `application/`, not its own folder. Both `ingestion` and `mcp`
import from these the same way; neither owns them.

## Project layout

```
application/
  __init__.py
  config.py              # every env var, read once, in one place
  schemas.py              # shared Pydantic models
  security.py              # permission checks + audit log
  docling_parser.py         # PDF/DOCX/XLSX/image -> structured chunks
  embedder.py                # local sentence-transformers embeddings
  supabase_client.py           # raw files, document/memory metadata, indexed_texts
  falkordb_client.py            # entity/relationship graph, 2-hop expansion
  entity_extraction.py           # LLM (Groq) entity/relation extraction
  vector_search.py                # dense retrieval
  bm25_search.py                    # keyword retrieval (Postgres full-text)
  fuzzy_search.py                     # trigram retrieval
  rrf_fusion.py                        # combines the three into one ranked list
  lifecycle.py                          # store/update/forget, dedup, active/superseded

  ingestion/
    __init__.py
    main.py                # FastAPI app: /health, /upload ONLY
    pipeline.py             # the actual write-time pipeline logic

  mcp/
    __init__.py
    server.py                # MCP server, 11 tools, stdio transport

schema.sql                 # run once in Supabase SQL editor
requirements.txt
.env.example
Dockerfile
docker-compose.yml
scripts/smoke_test.py       # local end-to-end test, no server needed
```

## Design notes

- **No FastAPI `/search` route.** Search is agent-facing only — it lives
  solely in the MCP server (`search_memory`, `search_chunks` tools).
  `application/ingestion` truly only ingests.
- **Pipeline logic is separate from the HTTP route.** `ingestion/pipeline.py`
  does the actual work; `ingestion/main.py` is a thin HTTP layer that saves
  the upload to disk and calls `pipeline.ingest_file(...)`. This means the
  same pipeline is callable from a script or test without spinning up FastAPI.
- **One config module.** Every environment variable is read in `config.py`
  and exposed as `settings`. No other file calls `os.environ[...]` or
  `load_dotenv()` directly — this catches missing/misspelled env vars at
  import time instead of deep inside a request.
- **`documents` and `memories` tables were missing from the original
  `schema.sql`** even though the code queries them — fixed, both are defined
  now.
- **FalkorDB SSL wasn't being read** even though `.env` had `FALKORDB_SSL` —
  fixed, `falkordb_client.py` now passes it through.
- `rank_bm25` / `rapidfuzz` removed from `requirements.txt` — unused, since
  BM25 and fuzzy matching both run in Postgres now, not in-process.

## Running locally

1. **Install dependencies**
   ```bash
   python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```

2. **Set up Supabase**
   - Create a project at supabase.com (or use your existing one).
   - Open the SQL Editor and run all of `schema.sql`.
   - Create a Storage bucket and note its name.

3. **Set up FalkorDB** — easiest is Docker:
   ```bash
   docker run -p 6379:6379 -it --rm falkordb/falkordb:latest
   ```
   (Or use FalkorDB Cloud — set `FALKORDB_SSL=true` in that case.)

4. **Copy `.env.example` to `.env`** and fill in every value: Supabase URL/key/bucket,
   FalkorDB host/port/password, your Groq API key. Langfuse vars are optional.

5. **Run the ingestion API**
   ```bash
   uvicorn application.ingestion.main:app --reload
   ```
   Check `http://localhost:8000/health`, then upload a file:
   ```bash
   curl -F "user_id=me" -F "file=@/path/to/some.pdf" http://localhost:8000/upload
   ```

6. **Run the MCP server** (separate terminal)
   ```bash
   python -m application.mcp.server
   ```
   Inspect it with the official MCP inspector if you want a UI to poke at the tools:
   ```bash
   npx @modelcontextprotocol/inspector python -m application.mcp.server
   ```

7. **Or run the full pipeline without any server**, for a quick sanity check:
   ```bash
   python scripts/smoke_test.py /path/to/file.pdf "some query in that file"
   ```

## Docker

```bash
docker compose up memoragraph-ingestion          # ingestion API on :8000
docker compose run --rm memoragraph-mcp          # MCP server, interactive stdio
```
