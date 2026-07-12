from app.core.config import TOP_K
from app.services.embeddings import embed_text

def retrieve(query, vectorstore):
    query_embedding = embed_text([query])
    results = vectorstore.search(query_embedding, TOP_K)
    return results