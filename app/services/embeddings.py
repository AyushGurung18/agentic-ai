from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_real_embeddings: HuggingFaceEmbeddings | None = None


def _get_real_embeddings() -> HuggingFaceEmbeddings:
    # Deferred so the model loads on first actual use instead of at import
    # time — the FastAPI and Celery processes both import this module, and
    # loading a full sentence-transformers model in each at the exact same
    # moment during container boot is a real memory spike on small hosts.
    global _real_embeddings
    if _real_embeddings is None:
        _real_embeddings = HuggingFaceEmbeddings(
            model_name=_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
    return _real_embeddings


class _LazyEmbeddings(Embeddings):
    """Same interface as HuggingFaceEmbeddings, but doesn't load the model
    until the first embed_documents/embed_query call."""

    def embed_documents(self, texts):
        return _get_real_embeddings().embed_documents(texts)

    def embed_query(self, text):
        return _get_real_embeddings().embed_query(text)


# Using Langchain's wrapper for sentence-transformers to be compatible with PGVector
embeddings = _LazyEmbeddings()

def embed_text(texts):
    """Encode a list of strings into float32 embeddings."""
    return embeddings.embed_documents(texts)