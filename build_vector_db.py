# build_vector_db.py

import chromadb
from sentence_transformers import SentenceTransformer
from file_loader import load_document
from chunker import chunk_text

# --- Load embedding model locally ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Setup ChromaDB using NEW API ---
client = chromadb.PersistentClient(path="./rag_db")
collection = client.get_or_create_collection("documents")

# --- Load your document ---
text = load_document("AgenticAI with Gamification.pdf")  # change file name here
chunks = chunk_text(text)

embeddings = embedder.encode(chunks).tolist()

# --- Store into vector DB ---
collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"id_{i}" for i in range(len(chunks))]
)

print("Vector DB created successfully.")