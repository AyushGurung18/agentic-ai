"""
app/services/rag.py
────────────────────
Conversational RAG entry points wired to:
  • LangGraph Self-RAG + CRAG graph  (langgraph_rag.py)
  • Groq / Gemini / NVIDIA / self-hosted vLLM LLMs
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
import concurrent.futures

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langsmith import traceable
import langchain
langchain.debug = False  # set True only when actively debugging chains

# Some provider SDKs (ChatNVIDIA in particular — it exposes no timeout
# parameter at all) can hang far longer than expected on a rate limit or
# slow response, which defeats the whole point of chaining fallbacks: a
# single stuck provider blocks the fallback from ever being tried. This
# wraps any LLM in a hard, externally-enforced wall-clock cutoff so a
# stuck call is abandoned (not waited on) and the next provider in the
# .with_fallbacks() chain runs instead — independent of whatever timeout
# behavior (or lack of one) that provider's own client library has.
_TIMEOUT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=8, thread_name_prefix="llm-timeout")

# Per-provider hard cutoff. Was 12s — measured OpenRouter's free auto-router
# directly (5 back-to-back calls) and got 0.8s, 1.1s, 3.0s, 5.9s, 14.6s: the
# shared free pool routes to whatever model has room, and that 14.6s sample
# alone exceeded the old 12s cutoff.
#
# 18s, not 20s: get_llm_by_intent() below builds up to a 6-candidate chain
# (OpenRouter appears twice — its auto-router draws a different model per
# call, so a second draw is a real extra shot, not a wasted repeat), and a
# single graph node's worst case is every candidate hitting this cutoff:
# 6 x 20s = 120s would land EXACTLY on the edge Worker's 120s proxy
# timeout with zero margin, relying on Groq/vLLM happening to fail fast
# rather than actually being bounded by anything. 6 x 18s = 108s leaves
# real headroom regardless of which providers are currently slow-to-fail
# vs instant-to-fail.
PROVIDER_TIMEOUT_S = 18


def _with_hard_timeout(llm, seconds: float, name: str):
    def _invoke(input, config=None, **kwargs):
        future = _TIMEOUT_EXECUTOR.submit(llm.invoke, input, config=config, **kwargs)
        try:
            return future.result(timeout=seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"{name} did not respond within {seconds}s")
    return RunnableLambda(_invoke, name=f"{name}_bounded")

from app.services.chunking import hierarchical_chunk_text
from app.services.document_service import (
    ingest_document, ensure_session, ensure_user, get_session_document_ids, get_documents_by_ids,
)
from app.services.hf_cache import append_messages_to_cache
from app.services.history import save_message
from app.services.langgraph_rag import run_rag_graph, invoke_with_retry, group_documents_by_source
from app.services.quote_annotator import annotate_quotes, DocRef
from app.services.agent_research import run_agent_research
from app.services.embeddings import embed_text
import json as _json
from app.core.config import (
    CHUNK_SIZE, CHUNK_OVERLAP,
    VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY,
    GROQ_API_KEY, GEMINI_API_KEY, NVIDIA_API_KEY, OPENROUTER_API_KEY,
    GROQ_API_KEY_2, GEMINI_API_KEY_2,
    DEV_USER_ID, DEV_USER_EMAIL,
)

logger = logging.getLogger("rag")


# ── Internal semantic cache write helper ───────────────────────────────────────

def _get_local_embedding(text: str) -> list[float] | None:
    """Embed *text* with the local sentence-transformers MiniLM model."""
    try:
        vecs = embed_text([text])
        return vecs[0]
    except Exception as exc:
        logger.warning("[SemanticCache] Local embedding failed: %s", exc)
        return None


def _write_to_semantic_cache(question: str, answer: str, session_id: str) -> None:
    """Persist a Q&A pair to the semantic cache via the internal endpoint.

    Scoped by session_id — see cache_routes.py / the 2026-07-19 migration
    for why an unscoped cache incorrectly leaked answers across documents.
    """
    embedding = _get_local_embedding(question)
    if embedding is None:
        return
    internal_secret = os.getenv("INTERNAL_API_SECRET", "")
    base_url = os.getenv("INTERNAL_BASE_URL", "http://127.0.0.1:7860")
    try:
        resp = httpx.post(
            f"{base_url}/api/internal/cache-write",
            headers={"X-Internal-Secret": internal_secret, "Content-Type": "application/json"},
            json={"raw_question": question, "cached_answer": answer, "embedding": embedding, "session_id": session_id},
            timeout=10,
        )
        if resp.status_code == 200:
            logger.info("[SemanticCache] ✅ Cache write OK for question: %s", question[:60])
        else:
            logger.warning("[SemanticCache] Cache write failed: HTTP %s %s", resp.status_code, resp.text[:120])
    except Exception as exc:
        logger.warning("[SemanticCache] Cache write request failed: %s", exc)


# ── Intent Router & LLM factory ────────────────────────────────────────────────

@traceable(name="route_intent", run_type="chain")
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
        intent = invoke_with_retry(chain, {"question": question[:500]}).strip().lower()
        if intent not in ["cheap", "complex"]:
            intent = "complex"
        return intent
    except Exception as e:
        logger.warning(f"Intent router failed: {e}. Defaulting to 'complex'.")
        return "complex"

def _get_vllm_llm(temperature: float) -> ChatOpenAI:
    """Self-hosted fallback — talks to a vLLM OpenAI-compatible server.

    vLLM's PagedAttention KV-cache manager is what makes this a viable
    no-cloud-API-key fallback under concurrent requests, unlike a naive
    single-request local model server.
    """
    return ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=VLLM_API_KEY,
        temperature=temperature,
        request_timeout=PROVIDER_TIMEOUT_S,
    )


def _get_openrouter_llm(temperature: float) -> ChatOpenAI:
    """OpenRouter's free auto-router — OpenAI-compatible, so this reuses
    ChatOpenAI exactly like the vLLM fallback, just pointed at OpenRouter's
    endpoint. "openrouter/free" picks whichever free model is actually
    available behind the scenes (the free lineup rotates over time), so
    there's no specific free model name to hardcode and keep up to date.
    """
    return ChatOpenAI(
        model="openrouter/free",
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        temperature=temperature,
        request_timeout=PROVIDER_TIMEOUT_S,
    )


def get_llm_by_intent(intent: str):
    """Returns an LLM with automatic cross-provider fallback baked in.

    Previously this picked exactly one provider (whichever key was
    configured first) and invoke_with_retry() would retry ONLY that same
    provider up to 4 times with growing backoff — so a rate-limited
    provider got hammered again and again instead of the request moving
    on, turning a single 429 into 30-45+ seconds of dead waiting per LLM
    call. A full graph run can hit several LLM calls (routing, grading,
    generation, hallucination/usefulness checks, possible rewrites), so
    that compounded into multi-minute hangs and "origin_timeout" errors
    at the edge — the timeout itself was never really the problem.

    LangChain's .with_fallbacks() tries every configured provider in
    priority order within a single invoke() call, moving to the next one
    immediately on any failure instead of retrying the one that just
    failed.
    """
    # The FIRST Groq/Gemini accounts are broken (org-restricted / quota-
    # exhausted, account-level — confirmed by testing every model
    # directly). Second accounts for both were added and confirmed
    # working via direct API test, so they're placed ahead of the broken
    # ones. The broken originals are kept, just demoted to the very back:
    # either could get fixed later (quotas reset, restrictions lifted),
    # but until then they were burning a full PROVIDER_TIMEOUT_S per
    # request for a guaranteed failure. OpenRouter appears twice — its
    # free auto-router hands back a different model per call (observed 5
    # different models in 5 calls), so a second draw is a real extra
    # chance, not a wasted repeat.
    if intent == "cheap":
        temperature = 0.1
        candidates = []
        if GROQ_API_KEY_2:
            candidates.append(("Groq #2 Llama 3.1 8B", ChatGroq(model_name="llama-3.1-8b-instant", temperature=temperature, groq_api_key=GROQ_API_KEY_2, request_timeout=PROVIDER_TIMEOUT_S)))
        if GEMINI_API_KEY_2:
            candidates.append(("Gemini #2 Flash", ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=GEMINI_API_KEY_2, temperature=temperature, timeout=PROVIDER_TIMEOUT_S)))
        if OPENROUTER_API_KEY:
            candidates.append(("OpenRouter (free) #1", _get_openrouter_llm(temperature=temperature)))
        if OPENROUTER_API_KEY:
            candidates.append(("OpenRouter (free) #2", _get_openrouter_llm(temperature=temperature)))
        if GROQ_API_KEY:
            candidates.append(("Groq #1 Llama 3.1 8B", ChatGroq(model_name="llama-3.1-8b-instant", temperature=temperature, groq_api_key=GROQ_API_KEY, request_timeout=PROVIDER_TIMEOUT_S)))
        if GEMINI_API_KEY:
            candidates.append(("Gemini #1 Flash", ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY, temperature=temperature, timeout=PROVIDER_TIMEOUT_S)))
    else:
        temperature = 0.2
        candidates = []
        if NVIDIA_API_KEY:
            candidates.append(("Nvidia Llama 3.1 70B", ChatNVIDIA(model="meta/llama-3.1-70b-instruct", nvidia_api_key=NVIDIA_API_KEY, temperature=temperature)))
        if GROQ_API_KEY_2:
            candidates.append(("Groq #2 Llama 3.3 70B", ChatGroq(model_name="llama-3.3-70b-versatile", temperature=temperature, groq_api_key=GROQ_API_KEY_2, request_timeout=PROVIDER_TIMEOUT_S)))
        if GEMINI_API_KEY_2:
            candidates.append(("Gemini #2 Pro", ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=GEMINI_API_KEY_2, temperature=temperature, timeout=PROVIDER_TIMEOUT_S)))
        if OPENROUTER_API_KEY:
            candidates.append(("OpenRouter (free) #1", _get_openrouter_llm(temperature=temperature)))
        if OPENROUTER_API_KEY:
            candidates.append(("OpenRouter (free) #2", _get_openrouter_llm(temperature=temperature)))
        if GROQ_API_KEY:
            candidates.append(("Groq #1 Llama 3.3 70B", ChatGroq(model_name="llama-3.3-70b-versatile", temperature=temperature, groq_api_key=GROQ_API_KEY, request_timeout=PROVIDER_TIMEOUT_S)))
        if GEMINI_API_KEY:
            candidates.append(("Gemini #1 Pro", ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=GEMINI_API_KEY, temperature=temperature, timeout=PROVIDER_TIMEOUT_S)))

    # Self-hosted vLLM always closes out the chain — no API key, no rate limit.
    candidates.append(("Self-hosted vLLM", _get_vllm_llm(temperature=temperature)))

    names = [n for n, _ in candidates]
    logger.info("🤖 Routing [%s] -> %s (fallbacks: %s)", intent.upper(), names[0], ", ".join(names[1:]) or "none")

    # Every candidate gets the hard external timeout regardless of whether
    # it also has a native one set above — ChatNVIDIA has no native timeout
    # parameter at all, so this is the only thing bounding it.
    bounded = [_with_hard_timeout(llm, PROVIDER_TIMEOUT_S, name) for name, llm in candidates]
    primary, fallbacks = bounded[0], bounded[1:]
    return primary.with_fallbacks(fallbacks) if fallbacks else primary


# ── Document ingestion (unchanged — also called by Celery worker) ──────────────

def process_document(
    text: str,
    filename: str = "unknown.pdf",
    user_id: str | None = None,
    session_id: str | None = None,
    metadata_tags: dict[str, str] | None = None,
    progress_callback = None,
    doc_type: str = "pdf",
    pages: list | None = None,
    r2_url: str | None = None,
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
        doc_type=doc_type,
        pages=pages,
        r2_url=r2_url,
        home_session_id=session_id,
    )

    # Note: child chunks are NOT also written to the separate LangChain
    # PGVector `default_collection` here. That write used to happen
    # unconditionally, but nothing in the live retrieval path ever reads from
    # it — langgraph_rag.retrieve() queries the hand-rolled hybrid_search()
    # SQL function directly against document_chunks/embeddings (already
    # populated by ingest_document() above), not through VectorStore/
    # retriever.py. The write was pure waste on every upload, and worse: a
    # transient failure in it (observed live: a Postgres SSL "decryption
    # failed" blip) propagated up and triggered Celery's retry, which re-ran
    # this whole function from scratch — duplicating the already-committed
    # document, chunks, and embeddings rather than just the unrelated write
    # that actually failed.

    # Attach to the session this upload was made in, if any — this is what
    # scopes retrieval (hybrid_search's filter_document_ids) to exactly the
    # document(s) the user is actually chatting with, instead of pooling
    # across every document they've ever uploaded.
    if session_id:
        from app.services.document_service import attach_document_to_session
        attach_document_to_session(session_id, document_id)

    return document_id


# ── Question answering — now powered by LangGraph Self-RAG + CRAG ─────────────

def ask_question(
    question: str,
    session_id: str = "00000000-0000-0000-0000-000000000000",
    user_id: str | None = None,
    mode: str = "standard",
):
    """
    Stream answer tokens via either:
      • the standard LangGraph CRAG graph (retrieve → grade → [rewrite/web] → generate), or
      • the Part C agentic research loop (mode="agentic") — a bounded
        multi-round tool-calling loop over a single document (search_document /
        get_section / list_clauses), shown live as it runs.

    Both paths converge on the same quote-verification step before anything
    is persisted or streamed, and write the Q&A pair to the semantic cache +
    HF disk cache in background threads afterward.
    """
    if user_id is None:
        user_id = DEV_USER_ID

    ensure_user(user_id, email=DEV_USER_EMAIL)
    ensure_session(session_id, user_id)

    # Persist the user's question immediately — before running the graph —
    # so it survives even if generation fails or times out. Without this,
    # a session reload after any failure shows nothing at all, because
    # nothing was ever saved: this pipeline previously never wrote to the
    # messages table at all, only read from it for context.
    try:
        save_message(session_id, "user", question)
    except Exception as exc:
        logger.warning("[history] Failed to save user message: %s", exc)

    # Route intent to determine which model to use
    intent = route_intent(question)
    llm = get_llm_by_intent(intent)

    document_ids = get_session_document_ids(session_id)

    logger.info("[RAG] mode=%s intent=%s user=%s session=%s docs=%d q=%s", mode, intent, user_id, session_id, len(document_ids), question[:80])

    agentic = mode == "agentic" and len(document_ids) == 1
    if mode == "agentic" and len(document_ids) != 1:
        logger.info("[RAG] Agentic mode requested but session has %d documents (needs exactly 1) — falling back to standard.", len(document_ids))

    full_answer = ""
    doc_refs: dict[int, DocRef] = {}

    try:
        if agentic:
            fetched = get_documents_by_ids(document_ids, user_id)
            doc_data = fetched.get(document_ids[0])
            if doc_data is None:
                raise RuntimeError("Document not found or no longer accessible.")
            doc_ref = DocRef(
                document_id=document_ids[0], filename=doc_data["filename"],
                text=doc_data["content"], pages=doc_data["pages"],
            )
            doc_refs = {1: doc_ref}
            for event in run_agent_research(question=question, doc=doc_ref, llm=llm):
                if event["type"] == "status":
                    yield f"\x1e{event['label']}\x1e"
                elif event["type"] == "done":
                    full_answer = event["answer"]
        else:
            # run_rag_graph is now a generator: it yields {"type": "status", ...}
            # after every graph node completes (real progress, not a blind
            # spinner) and finally {"type": "done", "answer": ..., "documents": [...]}.
            # Status events are wrapped in \x1e (ASCII Record Separator) markers so
            # the frontend can pull them out of the token stream and show them
            # as a live status line instead of appending them to the visible
            # answer — \x1e never occurs in normal generated text, so it's a
            # safe, unambiguous delimiter without needing a bigger protocol
            # change (SSE event types, etc.) on either side.
            retrieved_docs = []
            for event in run_rag_graph(
                question=question,
                session_id=session_id,
                user_id=user_id,
                llm=llm,
                document_ids=document_ids,
            ):
                if event["type"] == "status":
                    yield f"\x1e{event['label']}\x1e"
                elif event["type"] == "done":
                    full_answer = event["answer"]
                    retrieved_docs = event.get("documents", [])

            doc_order, doc_filenames = group_documents_by_source(retrieved_docs)
            if doc_order:
                fetched = get_documents_by_ids(doc_order, user_id)
                doc_refs = {
                    i + 1: DocRef(
                        document_id=doc_id, filename=doc_filenames.get(doc_id, "document"),
                        text=fetched[doc_id]["content"], pages=fetched[doc_id]["pages"],
                    )
                    for i, doc_id in enumerate(doc_order)
                    if doc_id in fetched
                }
    except Exception as e:
        error_answer = f"Error generating answer: {str(e)}"
        try:
            save_message(session_id, "assistant", error_answer)
        except Exception as exc:
            logger.warning("[history] Failed to save error message: %s", exc)
        yield error_answer
        return

    if not full_answer.strip():
        full_answer = "I was unable to find relevant information to answer your question."

    # ── Verify every quote before anything is shown to the user ────────────────
    # This is the core "verified quotes" requirement: re-locate each quote
    # the model produced in the actual document text ourselves (never trust
    # the model's own claim about where it came from), and rewrite the
    # answer so the frontend can render verified quotes as clickable/
    # highlighted and unverified ones as clearly flagged — never silently
    # as if they were genuine.
    quote_meta: list[dict] = []
    try:
        full_answer, quote_meta = annotate_quotes(full_answer, doc_refs)
    except Exception as exc:
        logger.warning("[quotes] Verification failed, stripping quote markers: %s", exc)
        full_answer, quote_meta = annotate_quotes(full_answer, {})

    # Persist the assistant's answer (with quote markers resolved to
    # verbatim/verified text) as soon as it's ready — before the perceived-
    # streaming yield loop below — so a reload reflects the real exchange
    # even if the client disconnects mid-stream.
    try:
        save_message(session_id, "assistant", full_answer, metadata={"quotes": quote_meta} if quote_meta else None)
    except Exception as exc:
        logger.warning("[history] Failed to save assistant message: %s", exc)

    # ── Yield answer in chunks for perceived streaming ─────────────────────────
    # Chunk size of ~20 chars gives a natural typewriter feel
    STREAM_CHUNK = 20
    for i in range(0, len(full_answer), STREAM_CHUNK):
        yield full_answer[i:i + STREAM_CHUNK]

    # Quote verification metadata, one entry per QUOTE_MARK pair in the
    # stream above, in order. Sent as a single marker (reusing the \x1e
    # status-line channel with a reserved prefix) after all the answer text,
    # since the frontend needs the whole answer in hand to walk quote marks
    # in order and attach metadata by index.
    if quote_meta:
        yield f"\x1e__QUOTES__:{_json.dumps(quote_meta)}\x1e"

    # ── Write to semantic cache in background thread ───────────────────────────
    if full_answer.strip():
        logger.info("[SemanticCache] 📝 Queuing cache write (len=%d chars)", len(full_answer))
        t = threading.Thread(
            target=_write_to_semantic_cache,
            args=(question, full_answer, session_id),
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
                    answer_metadata={"quotes": quote_meta} if quote_meta else None,
                )
            except Exception as exc:
                logger.warning("[HFCache] Post-response write failed: %s", exc)

        t2 = threading.Thread(target=_hf_cache_write, daemon=True)
        t2.start()
    else:
        logger.warning("[SemanticCache] Empty answer – skipping cache write")