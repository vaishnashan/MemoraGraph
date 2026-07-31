"""
Generates local embeddings using Ollama — keeps private content from
ever leaving the machine (Security requirement: local embedding mode).

Day 1: run `ollama pull nomic-embed-text` before using this.
"""
import ollama
from application.config import settings


def embed_text(text: str) -> list[float]:
    """Embed a single piece of text."""
    response = ollama.embeddings(model=settings.ollama_embed_model, prompt=text)
    return response["embedding"]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple chunks. Ollama's python client doesn't batch natively,
    so we loop — fine for local dev; parallelize later if it's too slow."""
    return [embed_text(t) for t in texts]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    import numpy as np
    a, b = np.array(vec_a), np.array(vec_b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
