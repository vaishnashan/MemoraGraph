"""
Standalone test for sentence_transformers_embedder.py — confirms the model
loads, embeds text, and cosine similarity behaves sanely (similar sentences
score higher than unrelated ones).

Run from your project ROOT:
    python application/embeddings/test_sentence_transformers_embedder.py
"""
from application.embeddings.embedd import (
    embed_text,
    embed_batch,
    cosine_similarity,
)

print("Loading model and embedding test sentences...")

sentence_a = "MemoraGraph uses FalkorDB as its graph store."
sentence_b = "FalkorDB is used by MemoraGraph for graph storage."  # similar meaning
sentence_c = "The weather in Colombo is sunny today."              # unrelated

emb_a = embed_text(sentence_a)
emb_b = embed_text(sentence_b)
emb_c = embed_text(sentence_c)

print(f"✅ Embedded 3 sentences, embedding dim: {len(emb_a)}")

sim_ab = cosine_similarity(emb_a, emb_b)
sim_ac = cosine_similarity(emb_a, emb_c)

print(f"\nSimilarity (related sentences A vs B):   {sim_ab:.4f}")
print(f"Similarity (unrelated sentences A vs C):  {sim_ac:.4f}")

if sim_ab > sim_ac:
    print("\n✅ Sanity check passed: related sentences scored higher than unrelated ones.")
else:
    print("\n⚠️  Unexpected: unrelated sentence scored higher — investigate the model/setup.")

# Test batch embedding
batch = embed_batch([sentence_a, sentence_b, sentence_c])
assert len(batch) == 3
print(f"\n✅ Batch embedding works: got {len(batch)} embeddings from embed_batch()")

print("\n🎉 sentence-transformers embedder is working.")