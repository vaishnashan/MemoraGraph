"""
Langfuse tracing setup notes for MemoraGraph (Day 4).

Langfuse SDK v4 (current as of mid-2026) uses:
    from langfuse import observe, get_client

Add these to .env:
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_HOST=https://cloud.langfuse.com   # or your region's endpoint

Install:
    pip install langfuse

HOW TO WIRE IT IN — add the @observe() decorator directly onto the
functions you want traced. No wrapper module needed; just decorate.

In application/graph2/entity_extraction_groq.py, add at the top:

    from langfuse import observe

Then decorate the traced functions:

    @observe()
    def _call_groq(prompt: str) -> str:
        ...

    @observe()
    def extract_entities_and_relationships(text: str, doc_id: str):
        ...

In application/retrieval/rrf_fusion.py:

    from langfuse import observe

    @observe()
    def hybrid_search(query, vector_index, bm25_index, fuzzy_index, top_k=5):
        ...

In application/routes/upload.py and application/routes/search.py, decorate
the route handlers themselves so each HTTP call becomes one trace with
the above functions nested inside it as child spans automatically:

    from langfuse import observe

    @router.post("/upload")
    @observe()
    async def upload_document(...):
        ...

    @router.get("/search")
    @observe()
    async def search(...):
        ...

That's it — @observe() nests automatically via OpenTelemetry context, so
decorating the outer route handler PLUS the inner functions gives you a
full trace tree per request: upload -> parse -> chunk -> embed -> extract
-> add_entity -> add_relationship, all under one trace, in the Langfuse
dashboard.

Since your entity extraction supports multiple providers (Groq/Gemini),
decorate whichever _call_groq / _call_gemini function you're actually
using — that's the piece that costs tokens and is worth tracking latency
+ cost on.