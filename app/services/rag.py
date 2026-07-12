"""
app/services/rag.py
────────────────────
Conversational RAG entry points wired to:
  • LangGraph Self-RAG + CRAG graph  (langgraph_rag.py)
  • Groq / Ollama LLMs
  • PGVector + HNSW for similarity retrieval
  • Normalized messages table for chat history  (via history.py)
  • Normalized documents / chunks / embeddings tables (via document_service.py)

The ask_question() generator is kept streaming-compatible:
  — LangGraph runs synchronously to produce the full answer
  — We yield it in one shot (or chunk it for perceived streaming)
  — Semantic cache + HF cache writes happen in background threads (unchanged)
"""

import os
import logging
import httpx
import threading

from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import langchain
langchain.debug = False  # set True only when actively debugging chains

from app.services.chunking import hierarchical_chunk_text
from app.services.vectorstore import VectorStore
from app.services.document_service import ingest_document, ensure_session, ensure_user
from app.services.hf_cache import append_messages_to_cache
from app.services.langgraph_rag import run_rag_graph
from app.services.embeddings import embed_text
from app.core.config import (
    CHUNK_SIZE, CHUNK_OVERLAP,
    OLLAMA_MODEL, OLLAMA_URL,
    GROQ_API_KEY, GEMINI_API_KEY, NVIDIA_API_KEY,
    DEV_USER_ID, DEV_USER_EMAIL,
)

logger = logging.getLogger("rag")

# ── Singletons ─────────────────────────────────────────────────────────────────
vectorstore = VectorStore()


# ── Internal semantic cache write helper ───────────────────────────────────────

def _get_local_embedding(text: str) -> list[float] | None:
    """Embed *text* with the local sentence-transformers MiniLM model."""
    try:
        vecs = embed_text([text])
        return vecs[0]
    except Exception as exc:
        logger.warning("[SemanticCache] Local embedding failed: %s", exc)
        return None


def _write_to_semantic_cache(question: str, answer: str) -> None:
    """Persist a Q&A pair to the semantic cache via the internal endpoint."""
    embedding = _get_local_embedding(question)
    if embedding is None:
        return
    internal_secret = os.getenv("INTERNAL_API_SECRET", "")
    base_url = os.getenv("INTERNAL_BASE_URL", "http://127.0.0.1:7860")
    try:
        resp = httpx.post(
            f"{base_url}/api/internal/cache-write",
            headers={"X-Internal-Secret": internal_secret, "Content-Type": "application/json"},
            json={"raw_question": question, "cached_answer": answer, "embedding": embedding},
            timeout=10,
        )
        if resp.status_code == 200:
            logger.info("[SemanticCache] ✅ Cache write OK for question: %s", question[:60])
        else:
            logger.warning("[SemanticCache] Cache write failed: HTTP %s %s", resp.status_code, resp.text[:120])
    except Exception as exc:
        logger.warning("[SemanticCache] Cache write request failed: %s", exc)


# ── Intent Router & LLM factory ────────────────────────────────────────────────

def route_intent(question: str) -> str:
    """Classifies user intent as 'cheap' or 'complex' using a fast, cheap model."""
    try:
        # We use a fast local or cheap Groq model for routing
        router_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0, groq_api_key=GROQ_API_KEY)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an intent router. Classify the user query into one of two categories: 'cheap' or 'complex'.\n"
                       "- 'cheap': Simple factual questions, greetings, short definitions, basic info retrieval.\n"
                       "- 'complex': Coding, deep synthesis, reasoning, comparisons, highly technical questions.\n"
                       "Output ONLY 'cheap' or 'complex'."),
            ("human", "{question}")
        ])
        
        chain = prompt | router_llm | StrOutputParser()
        intent = chain.invoke({"question": question[:500]}).strip().lower()
        if intent not in ["cheap", "complex"]:
            intent = "complex"
        return intent
    except Exception as e:
        logger.warning(f"Intent router failed: {e}. Defaulting to 'complex'.")
        return "complex"

def get_llm_by_intent(intent: str):
    """Returns an instantiated LLM based on intent."""
    if intent == "cheap":
        # Prefer Gemini Flash, fallback to Groq Llama 3.1 8B
        if GEMINI_API_KEY:
            logger.info("🤖 Routing [CHEAP] -> Gemini 1.5 Flash")
            return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.1)
        else:
            logger.info("🤖 Routing [CHEAP] -> Groq Llama 3.1 8B")
            return ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1, groq_api_key=GROQ_API_KEY)
    else:
        # Prefer Nvidia (Llama 70B), fallback to Gemini Pro or Groq 70B
        if NVIDIA_API_KEY:
            logger.info("🤖 Routing [COMPLEX] -> Nvidia (meta/llama-3.1-70b-instruct)")
            return ChatNVIDIA(model="meta/llama-3.1-70b-instruct", nvidia_api_key=NVIDIA_API_KEY, temperature=0.2)
        elif GEMINI_API_KEY:
            logger.info("🤖 Routing [COMPLEX] -> Gemini 1.5 Pro")
            return ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GEMINI_API_KEY, temperature=0.2)
        else:
            logger.info("🤖 Routing [COMPLEX] -> Groq Llama 3.3 70B")
            return ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2, groq_api_key=GROQ_API_KEY)


# ── Document ingestion (unchanged — also called by Celery worker) ──────────────

def process_document(
    text: str,
    filename: str = "unknown.pdf",
    user_id: str | None = None,
    session_id: str | None = None,
    metadata_tags: dict[str, str] | None = None,
    progress_callback = None,
) -> str:
    """
    Chunk text → persist to DB (documents / chunks / embeddings) → add to PGVector.
    Returns the document_id UUID string.

    This function is called both:
      • Directly (legacy sync path, if needed)
      • By the Celery worker task (app/worker/tasks.py)
    """
    if user_id is None:
        user_id = DEV_USER_ID

    ensure_user(user_id, email=DEV_USER_EMAIL)

    print(f"📄 Processing '{filename}' ({len(text)} chars) for user {user_id}")
    hierarchical_chunks = hierarchical_chunk_text(text)
    
    flat_children = []
    for p in hierarchical_chunks:
        flat_children.extend(p["children"])
        
    print(f"  → {len(hierarchical_chunks)} parents, {len(flat_children)} children")

    tags = metadata_tags or {}
    if session_id:
        tags["session_id"] = session_id

    document_id, chunk_metadatas = ingest_document(
        text=text,
        hierarchical_chunks=hierarchical_chunks,
        filename=filename,
        user_id=user_id,
        metadata_tags=tags,
        progress_callback=progress_callback,
    )

    vectorstore.add(flat_children, metadata=chunk_metadatas)
    print(f"  → Added to PGVector/HNSW (document_id={document_id})")

    return document_id


# ── Question answering — now powered by LangGraph Self-RAG + CRAG ─────────────

def ask_question(
    question: str,
    session_id: str = "00000000-0000-0000-0000-000000000000",
    user_id: str | None = None,
):
    """
    Stream answer tokens via the LangGraph Self-RAG + CRAG graph.

    Graph flow:
      retrieve → grade_documents → [rewrite / web_search] → generate → grade_generation

    After the graph completes, writes the Q&A pair to:
      • Semantic cache (Cloudflare KV via internal endpoint)
      • HF disk cache (local session JSON)
    Both writes are non-blocking background threads.
    """
    if user_id is None:
        user_id = DEV_USER_ID

    ensure_user(user_id, email=DEV_USER_EMAIL)
    ensure_session(session_id, user_id)

    # Route intent to determine which model to use
    intent = route_intent(question)
    llm = get_llm_by_intent(intent)

    logger.info("[RAG] intent=%s user=%s session=%s q=%s", intent, user_id, session_id, question[:80])

    try:
        # Run the LangGraph graph (synchronous — returns full answer)
        full_answer = run_rag_graph(
            question=question,
            session_id=session_id,
            user_id=user_id,
            llm=llm,
        )
    except Exception as e:
        yield f"Error generating answer: {str(e)}"
        return

    if not full_answer.strip():
        full_answer = "I was unable to find relevant information to answer your question."

    # ── Yield answer in chunks for perceived streaming ─────────────────────────
    # Chunk size of ~20 chars gives a natural typewriter feel
    STREAM_CHUNK = 20
    for i in range(0, len(full_answer), STREAM_CHUNK):
        yield full_answer[i:i + STREAM_CHUNK]

    # ── Write to semantic cache in background thread ───────────────────────────
    if full_answer.strip():
        logger.info("[SemanticCache] 📝 Queuing cache write (len=%d chars)", len(full_answer))
        t = threading.Thread(
            target=_write_to_semantic_cache,
            args=(question, full_answer),
            daemon=True,
        )
        t.start()

        # ── Write to HF disk cache ─────────────────────────────────────────────
        def _hf_cache_write():
            try:
                append_messages_to_cache(
                    user_id=user_id,
                    session_id=session_id,
                    question=question,
                    answer=full_answer,
                )
            except Exception as exc:
                logger.warning("[HFCache] Post-response write failed: %s", exc)

        t2 = threading.Thread(target=_hf_cache_write, daemon=True)
        t2.start()
    else:
        logger.warning("[SemanticCache] Empty answer – skipping cache write")