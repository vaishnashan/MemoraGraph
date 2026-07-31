"""
FastAPI entrypoint. Run with:
    uvicorn app.main:app --reload
"""
from fastapi import FastAPI
from application.routes import upload, search

app = FastAPI(
    title="MemoraGraph",
    description="Private multimodal AI memory infrastructure with MCP",
    version="0.1.0",
)

app.include_router(upload.router, tags=["ingestion"])
app.include_router(search.router, tags=["retrieval"])


@app.get("/health")
async def health():
    return {"status": "ok"}
