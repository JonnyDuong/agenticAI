import os
import ssl
import certifi
from dotenv import load_dotenv
from pathlib import Path
import re
from typing import List

import fitz  # PyMuPDF
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
# Updated import to address deprecation warning
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------
# SSL + Env Setup
# -------------------------
os.environ["SSL_CERT_FILE"] = certifi.where()
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ.pop("CURL_CA_BUNDLE", None)
os.environ["CURL_CA_BUNDLE"] = ""

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# -------------------------
# Paths
# -------------------------
base_dir = Path(__file__).parent
# Use the actual path where the PDF exists
mistral_pdf_path = base_dir / "RAG" / "Mistral_paper.pdf"
embedding_output_dir = base_dir / "RAG" / "embeddings"
embedding_output_dir.mkdir(parents=True, exist_ok=True)

# -------------------------
# Extract and Chunk Text (Sentence-Level)
# -------------------------
def extract_sentences_from_pdf(pdf_path: Path) -> List[str]:
    sentences = []
    sentence_splitter = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")  # basic sentence boundary

    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            text = page.get_text("text")
            split_sentences = sentence_splitter.split(text)
            clean_sentences = [s.strip() for s in split_sentences if len(s.strip()) > 30]  # remove very short junk
            sentences.extend(clean_sentences)
    
    return sentences

chunks = extract_sentences_from_pdf(mistral_pdf_path)

# -------------------------
# Convert to LangChain Documents
# -------------------------
documents = [Document(page_content=chunk, metadata={"source": "Mistral"}) for chunk in chunks]

# -------------------------
# Embedding + FAISS
# -------------------------

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
faiss_index = FAISS.from_documents(documents, embedding_model)
# -------------------------
# Save the index
# -------------------------
index_dir = embedding_output_dir / "mistral_index_sentences"
faiss_index.save_local(str(index_dir))

print(f"✅ Sentence-level FAISS index saved to: {index_dir}")

# -------------------------
# Save chunks for lookup
# -------------------------
rag_dir = base_dir / "RAG"
rag_dir.mkdir(exist_ok=True)
chunk_lookup_path = rag_dir / "mistral_chunk_lookup.txt"

with open(chunk_lookup_path, "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        f.write(f"--- Chunk {i} ---\n{chunk}\n")

print(f"✅ Chunk lookup file saved to: {chunk_lookup_path}")
