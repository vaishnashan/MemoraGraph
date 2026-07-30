"""
Central configuration for MemoraGraph.
Loads from .env — copy .env.example to .env and fill in your values.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Supabase
    supabase_url: str = ""
    supabase_key: str = ""
    supabase_bucket: str = "memoragraph-files"

    # FalkorDB
    falkordb_host: str = "localhost"
    falkordb_port: int = 6379
    falkordb_graph_name: str = "memoragraph"

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"

    # Extraction LLM
    extraction_model: str = "llama3.1"
    extraction_provider: str = "ollama"
    anthropic_api_key: str = ""

    # Observability
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # App
    app_env: str = "development"
    local_embedding_only: bool = True


settings = Settings()
