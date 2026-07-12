from langchain_postgres import PGVector
from app.core.config import POSTGRES_URL
from app.services.embeddings import embeddings

class VectorStore:
    def __init__(self, collection_name="default_collection"):
        self.vectorstore = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=POSTGRES_URL,
            use_jsonb=True,
        )

    def add(self, texts, metadata=None):
        self.vectorstore.add_texts(texts, metadatas=metadata)

    def search(self, query, k=3, user_id=None):
        filter = {"user_id": user_id} if user_id else None
        results = self.vectorstore.similarity_search(query, k=k, filter=filter)
        return [doc.page_content for doc in results]
    
    def as_retriever(self, **kwargs):
        """Returns the native Langchain retriever."""
        return self.vectorstore.as_retriever(**kwargs)