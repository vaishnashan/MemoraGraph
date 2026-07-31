-- MemoraGraph — persisted retrieval schema
-- Run this ONCE in the Supabase SQL Editor (Project -> SQL Editor -> New query).
--
-- Replaces the old in-memory VectorIndex / BM25Index / FuzzyIndex with one
-- Postgres table + pgvector + full-text search + trigram fuzzy matching.
-- This is what both the FastAPI process and the MCP server process read
-- from and write to, so data no longer disappears on restart and both
-- processes see the same data.

-- 1. Extensions -----------------------------------------------------------
create extension if not exists vector;
create extension if not exists pg_trgm;

-- 2. Unified table ----------------------------------------------------------
-- Holds BOTH document chunks (from /upload) and standalone memories
-- (from store_memory) — same as the old in-memory indexes did, just persisted.
--
-- embedding dimension = 384, matching sentence-transformers/all-MiniLM-L6-v2
create table if not exists indexed_texts (
    item_id     text primary key,      -- chunk_id OR memory_id
    doc_id      text,
    user_id     text,
    text        text not null,
    embedding   vector(384),
    tsv         tsvector generated always as (to_tsvector('english', text)) stored,
    created_at  timestamptz default now()
);

-- 3. Indexes ----------------------------------------------------------------
-- Vector similarity (cosine distance) — powers vector_search()
create index if not exists indexed_texts_embedding_idx
    on indexed_texts using hnsw (embedding vector_cosine_ops);

-- Full-text search (BM25-equivalent) — powers bm25_search()
create index if not exists indexed_texts_tsv_idx
    on indexed_texts using gin (tsv);

-- Trigram fuzzy matching — powers fuzzy_search()
create index if not exists indexed_texts_trgm_idx
    on indexed_texts using gin (text gin_trgm_ops);

create index if not exists indexed_texts_doc_id_idx
    on indexed_texts (doc_id);

-- 4. RPC functions (called from Python via client.rpc(...)) -----------------

-- Dense vector similarity search
create or replace function match_vector(
    query_embedding vector(384),
    match_count int default 10
)
returns table (item_id text, text text, doc_id text, score float)
language sql stable
as $$
    select item_id, text, doc_id,
           1 - (embedding <=> query_embedding) as score
    from indexed_texts
    where embedding is not null
    order by embedding <=> query_embedding
    limit match_count;
$$;

-- BM25-equivalent keyword search
create or replace function match_bm25(
    query_text text,
    match_count int default 10
)
returns table (item_id text, text text, doc_id text, score float)
language sql stable
as $$
    select item_id, text, doc_id,
           ts_rank(tsv, plainto_tsquery('english', query_text)) as score
    from indexed_texts
    where tsv @@ plainto_tsquery('english', query_text)
    order by score desc
    limit match_count;
$$;

-- Fuzzy / trigram search
create or replace function match_fuzzy(
    query_text text,
    match_count int default 10
)
returns table (item_id text, text text, doc_id text, score float)
language sql stable
as $$
    select item_id, text, doc_id,
           similarity(text, query_text) as score
    from indexed_texts
    order by score desc
    limit match_count;
$$;

-- Nearest-active-memory lookup, used by memory/lifecycle.py's duplicate
-- detection instead of looping over every memory in Python.
create or replace function match_active_memory(
    query_embedding vector(384),
    similarity_threshold float default 0.95
)
returns table (item_id text, score float)
language sql stable
as $$
    select it.item_id,
           1 - (it.embedding <=> query_embedding) as score
    from indexed_texts it
    join memories m on m.memory_id = it.item_id
    where m.status = 'active'
      and it.embedding is not null
    order by it.embedding <=> query_embedding
    limit 1;
$$;
