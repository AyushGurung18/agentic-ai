# app/routes/cache.py
"""Internal cache management endpoints.
These endpoints are *not* exposed publicly – they require a shared secret
passed in the ``X-Internal-Secret`` header.  The secret is stored in the
environment variable ``INTERNAL_API_SECRET`` (add it to your ``.env``).
All actions are logged to the standard FastAPI logger so they appear in the
Cloudflare Worker console (when the worker proxies the request) and in the
Docker container logs.
"""

import os
import logging
from fastapi import APIRouter, Header, HTTPException, Depends
from pydantic import BaseModel, conlist
import httpx
from app.db.database import get_conn
from app.services.embeddings import embed_text
from psycopg.rows import dict_row

# ---------------------------------------------------------------------------
# Logging configuration – FastAPI uses the ``uvicorn.error`` logger by default.
# We create a dedicated logger for cache operations.
# ---------------------------------------------------------------------------
logger = logging.getLogger("cache")
if not logger.handlers:
    # Ensure at least one handler (stdout) when the module is imported directly.
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

router = APIRouter(prefix="/api/internal", tags=["internal"])

# ---------------------------------------------------------------------------
# Dependency that validates the shared secret header.
# ---------------------------------------------------------------------------
def validate_internal_secret(x_internal_secret: str = Header(..., alias="X-Internal-Secret")):
    expected = os.getenv("INTERNAL_API_SECRET")
    if expected is None:
        logger.error("INTERNAL_API_SECRET not set in environment")
        raise HTTPException(status_code=500, detail="Server mis-configuration")
    if x_internal_secret != expected:
        logger.warning("Invalid internal secret supplied")
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

# ---------------------------------------------------------------------------
# Pydantic models for request/response payloads
# ---------------------------------------------------------------------------
class VectorCacheCheckRequest(BaseModel):
    # Accept either a raw question string (preferred) or a pre-computed vector
    # (legacy — kept for backward compat but ignored when question is provided)
    question: str | None = None
    vector: conlist(float, min_length=384, max_length=384) | None = None
    # Required — a cache hit must never cross session/document boundaries.
    # See supabase/migrations/20260719_scope_semantic_cache_by_session.sql
    session_id: str

    def model_post_init(self, __context):
        if self.question is None and self.vector is None:
            raise ValueError("Either 'question' or 'vector' must be provided")

class VectorCacheCheckResponse(BaseModel):
    hit: bool
    cached_answer: str | None = None
    similarity: float | None = None

class CacheWriteRequest(BaseModel):
    raw_question: str
    cached_answer: str
    embedding: conlist(float, min_length=384, max_length=384)
    session_id: str

# ---------------------------------------------------------------------------
# 1️⃣ Vector cache check — scoped to session_id so a hit can only ever replay
#    an answer generated within the same conversation/document context.
#    (Previously matched on question-text similarity alone, globally, across
#    every user and every document — see the 2026-07-19 migration.)
# ---------------------------------------------------------------------------
@router.post("/vector-cache-check", response_model=VectorCacheCheckResponse, dependencies=[Depends(validate_internal_secret)])
def vector_cache_check(payload: VectorCacheCheckRequest):
    # Embed the raw question using the local MiniLM model (same model as cache writes)
    if payload.question is not None:
        try:
            vecs = embed_text([payload.question])
            query_vec = str(vecs[0])
        except Exception as exc:
            logger.error("[Cache] Embedding failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"Embedding error: {exc}")
    elif payload.vector is not None:
        # Legacy path: pre-computed vector passed by caller
        query_vec = str(payload.vector)
    else:
        raise HTTPException(status_code=400, detail="No question or vector provided")

    threshold = 0.85  # relaxed from 0.96 — 0.96 was too strict for paraphrased questions
    with get_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            sql = """
                SELECT cached_answer,
                       1 - ( %s::vector <=> embedding ) AS similarity
                FROM public.semantic_responses_cache
                WHERE session_id = %s::uuid
                  AND ( %s::vector <=> embedding ) <= (1 - %s)
                ORDER BY similarity DESC
                LIMIT 1;
            """
            cur.execute(sql, (query_vec, payload.session_id, query_vec, threshold))
            row = cur.fetchone()

            if row:
                logger.info("[Cache Hit] session=%s similarity=%.4f", payload.session_id, row["similarity"])
                return VectorCacheCheckResponse(hit=True, cached_answer=row["cached_answer"], similarity=row["similarity"])
            else:
                logger.info("[Cache Miss] session=%s No semantic match (threshold=%.2f)", payload.session_id, threshold)
                return VectorCacheCheckResponse(hit=False)

# ---------------------------------------------------------------------------
# 2️⃣ Write a fresh Q&A into the cache after a successful LLM response.
# ---------------------------------------------------------------------------
@router.post("/cache-write", dependencies=[Depends(validate_internal_secret)])
def cache_write(request: CacheWriteRequest):
    with get_conn() as conn:
        with conn.cursor() as cur:
            sql = """
                INSERT INTO public.semantic_responses_cache (raw_question, cached_answer, embedding, session_id)
                VALUES (%s, %s, %s::vector, %s::uuid);
            """
            cur.execute(sql, (request.raw_question, request.cached_answer, str(request.embedding), request.session_id))
        conn.commit()
    logger.info("[Cache Write] New semantic cache entry stored for question: %s", request.raw_question[:50])
    return {"status": "written"}

# ---------------------------------------------------------------------------
# 3️⃣ Invalidate the user-chat KV entry when the chat list changes.
# ---------------------------------------------------------------------------
@router.post("/invalidate-user-chat/{user_id}", dependencies=[Depends(validate_internal_secret)])
async def invalidate_user_chat(user_id: str):
    cf_account = os.getenv("CF_ACCOUNT_ID")
    kv_namespace = os.getenv("CF_KV_NAMESPACE_ID")
    api_token = os.getenv("CF_API_TOKEN")
    if not all([cf_account, kv_namespace, api_token]):
        logger.error("Cloudflare credentials missing for KV invalidation")
        raise HTTPException(status_code=500, detail="Cloudflare credentials not configured")
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{cf_account}"
        f"/storage/kv/namespaces/{kv_namespace}/values/user_chats:{user_id}"
    )
    async with httpx.AsyncClient() as client:
        resp = await client.delete(url, headers={"Authorization": f"Bearer {api_token}"})
        if resp.status_code != 200:
            logger.error("KV invalidation failed for user %s – status %s", user_id, resp.status_code)
            raise HTTPException(status_code=resp.status_code, detail="KV deletion failed")
    logger.info("[KV Invalidate] Removed chat cache for user %s", user_id)
    return {"status": "invalidated"}
