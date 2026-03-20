# rag_query.py

from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from azure_config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_MODEL,
    AZURE_API_VERSION
)

# --- Azure Client ---
client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_API_VERSION
)

# --- Load embedding model ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Load vector DB ---
db = chromadb.PersistentClient(path="./rag_db")
collection = db.get_collection("documents")

# --- Query ---
while(True):
    question = input("Ask a question: ")

    query_emb = embedder.encode([question]).tolist()

    result = collection.query(
        query_embeddings=query_emb,
        n_results=3
    )   

    context = "\n".join(result["documents"][0])

    prompt = f"""
    You are an AI assistant. Use ONLY the context below to answer the question. 
    If the answer is not in the context, say "Sorry, I don't know.."

    CONTEXT:
        {context}

    QUESTION:
    {question}

    ANSWER:
    """

    response = client.chat.completions.create(
    model=AZURE_OPENAI_MODEL,
    messages=[{"role": "user", "content": prompt}]
    )

    print("\n--- Answer ---")
    print(response.choices[0].message.content)