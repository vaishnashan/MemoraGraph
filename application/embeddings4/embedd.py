"""
Generates local embeddings using sentence-transformers — replaces Ollama.
Runs fully on-device (no API calls), so private content never leaves the
machine (Security requirement: local embedding mode).

Model: sentence-transformers/all-MiniLM-L6-v2 by default — this matches
the tokenizer docling_parser.py's HybridChunker is built against, so
chunk sizing and embedding stay aligned to the same model's token limits.
Override via EMBED_MODEL_ID in .env if you want a different model later.
"""
import os

import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL_ID = os.environ.get("EMBED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")

# Loaded once at import time — reused across every embed call.
_model = SentenceTransformer(EMBED_MODEL_ID)


def embed_text(text: str) -> list[float]:
    """Embed a single piece of text."""
    embedding = _model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return embedding.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed multiple chunks in one batched call — much faster than looping
    embed_text() one at a time. Use this when embedding all chunks from
    a freshly-parsed document.
    """
    if not texts:
        return []
    embeddings = _model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32
    )
    return embeddings.tolist()


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a, b = np.array(vec_a), np.array(vec_b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)