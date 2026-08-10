-- MemoraGraph — Supabase schema
-- Run this ONCE in the Supabase SQL Editor (Project -> SQL Editor -> New query).

-- 1. Extensions -----------------------------------------------------------
create extension if not exists vector;
create extension if not exists pg_trgm;

-- 2. Documents (raw file metadata; the actual file bytes live in Storage) --
create table if not exists documents (
    doc_id            text primary key,
    filename          text not null,
    file_type         text not null,
    uploaded_at       timestamptz default now(),
    user_id           text not null,
    raw_storage_path  text not null
);

create index if not exists documents_user_id_idx on documents (user_id);

-- 3. Memories (lifecycle: active / outdated / superseded) -----------------
create table if not exists memories (
    memory_id      text primary key,
    doc_id         text,
    text           text not null,
    status         text not null default 'active',
    created_at     timestamptz default now(),
    updated_at     timestamptz default now(),
    superseded_by  text,
    source_doc_id  text
);

create index if not exists memories_status_idx on memories (status);

-- 4. Unified retrieval table ------------------------------------------------
-- Holds BOTH document chunks (from /upload) and standalone memories
-- (from store_memory) — one row per chunk_id/memory_id.
--
-- embedding dimension = 384, matching sentence-transformers/all-MiniLM-L6-v2.
-- If you change EMBEDDING_MODEL in .env to a model with a different output
-- size, update the `vector(384)` below (and the match_* functions) to match.
create table if not exists indexed_texts (
    item_id     text primary key,      -- chunk_id OR memory_id
    doc_id      text,
    user_id     text,
    text        text not null,
    embedding   vector(384),
    tsv         tsvector generated always as (to_tsvector('english', text)) stored,
    created_at  timestamptz default now()
);

-- 5. Indexes ----------------------------------------------------------------
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

-- 6. RPC functions (called from Python via client.rpc(...)) -----------------

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

-- 7. (Optional, not wired up yet) Audit log table ----------------------------
-- application/core/security.py currently keeps the audit log in memory,
-- which resets on every restart. Swap it to write here when you want the
-- audit trail to persist:
-- create table if not exists audit_logs (
--     id          bigserial primary key,
--     user_id     text,
--     tool_name   text,
--     memory_id   text,
--     action      text,
--     created_at  timestamptz default now()
-- );
