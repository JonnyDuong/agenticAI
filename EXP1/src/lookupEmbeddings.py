import os
import ssl
import certifi

from pathlib import Path
import faiss
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings

os.environ["SSL_CERT_FILE"] = certifi.where()
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ.pop("CURL_CA_BUNDLE", None)
os.environ["CURL_CA_BUNDLE"] = ""

embedding_path = Path(__file__).parent / "RAG" / "embeddings" / "mistral_index" / "index.faiss"

if not embedding_path.exists():
    raise FileNotFoundError(f"Index file not found at: {embedding_path.resolve()}")

index = faiss.read_index(str(embedding_path))
print("✅ FAISS index loaded successfully.")



chunk_lookup_path = Path(__file__).parent.parent / "src" / "RAG" / "mistral_chunk_lookup.txt"

with open(chunk_lookup_path, encoding="utf-8") as f:
    chunks = f.read().split("--- Chunk ")[1:]  # Skip the first empty split
    chunks = [chunk.split("\n", 1)[1].strip() for chunk in chunks]  # Remove labels


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



query = "What is Magistral?"
query_vec = embedding_model.embed_query(query)  # <- Correct method
query_vec = np.array([query_vec]).astype("float32")  # FAISS expects 2D array

faiss.normalize_L2(query_vec)  # normalize if index was normalized

k = 5  # number of top results
D, I = index.search(query_vec, k)

print(f"\n🔍 Top {k} Matches for Query:")
for rank, idx in enumerate(I[0], start=1):
    print(f"\n--- Match {rank} (Chunk {idx + 1}) ---\n{chunks[idx]}")