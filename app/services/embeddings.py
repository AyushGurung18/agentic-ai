from langchain_huggingface import HuggingFaceEmbeddings

# Using Langchain's wrapper for sentence-transformers to be compatible with PGVector
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

def embed_text(texts):
    """Encode a list of strings into float32 embeddings."""
    return embeddings.embed_documents(texts)