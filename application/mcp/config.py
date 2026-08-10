"""
Single place where every environment variable is read. Every other module
imports `settings` from here instead of calling load_dotenv()/os.environ[]
itself — one .env load, one place to see every required variable, and
typos in an env var name fail immediately at import time instead of deep
inside a request.
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class Settings:
    # Supabase
    supabase_url: str
    supabase_secret_key: str
    supabase_storage_bucket: str

    # FalkorDB
    falkordb_host: str
    falkordb_port: int
    falkordb_username: str
    falkordb_password: str
    falkordb_ssl: bool
    falkordb_graph_name: str

    # Entity extraction (Groq)
    groq_api_key: str
    groq_extraction_model: str

    # Embeddings
    embedding_model: str

    # Langfuse (optional — tracing no-ops cleanly if unset)
    langfuse_secret_key: str | None
    langfuse_public_key: str | None
    langfuse_base_url: str | None


def _load() -> Settings:
    return Settings(
        supabase_url=_require("SUPABASE_URL"),
        supabase_secret_key=_require("SUPABASE_SECRET_KEY"),
        supabase_storage_bucket=_require("SUPABASE_STORAGE_BUCKET"),
        falkordb_host=_require("FALKORDB_HOST"),
        falkordb_port=int(_require("FALKORDB_PORT")),
        falkordb_username=os.environ.get("FALKORDB_USERNAME", "falkordb"),
        falkordb_password=_require("FALKORDB_PASSWORD"),
        falkordb_ssl=os.environ.get("FALKORDB_SSL", "false").lower() == "true",
        falkordb_graph_name=os.environ.get("FALKORDB_GRAPH_NAME", "memoragraph"),
        groq_api_key=_require("GROQ_API_KEY"),
        groq_extraction_model=os.environ.get("GROQ_EXTRACTION_MODEL", "llama-3.3-70b-versatile"),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        langfuse_secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
        langfuse_public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        langfuse_base_url=os.environ.get("LANGFUSE_BASE_URL"),
    )


settings = _load()
