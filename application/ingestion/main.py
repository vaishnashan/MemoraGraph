"""
Ingestion API — the ONLY FastAPI app in this project, and it exposes
exactly two endpoints:

    GET  /health   liveness check
    POST /upload   take a file in, run the full write-time pipeline, done

There is deliberately no /search here. Reading/searching memory is an
agent capability, exposed only through the MCP server (application/mcp) —
not through a public HTTP API. This endpoint's only job is: accept a file,
save it, and hand off to the pipeline.

Run with:
    uvicorn application.ingestion.main:app --reload
"""
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, Form
from langfuse import observe

from application.ingestion.pipeline import ingest_file, TEMP_UPLOAD_DIR

app = FastAPI(
    title="MemoraGraph — Ingestion API",
    description="Write-time ingestion endpoint for MemoraGraph. Reads/search live in the MCP server, not here.",
    version="0.1.0",
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload")
@observe()
async def upload_document(user_id: str = Form(...), file: UploadFile = File(...)):
    local_path = TEMP_UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
    try:
        with open(local_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        return ingest_file(str(local_path), file.filename, user_id)
    finally:
        local_path.unlink(missing_ok=True)
