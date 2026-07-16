"""
app/routes.py
──────────────
API route definitions for thotqen.

Auth model
──────────
  Every endpoint accepts an optional Bearer token (Supabase JWT).
  • Token present  → user_id extracted from `sub` claim
  • No token       → falls back to DEV_USER_ID (local dev only)
  • Bad token      → HTTP 401

This guarantees per-user data isolation: every DB query is scoped to
the caller's UUID, so anonymous and real users never see each other's data.

Caching
───────
  Session lists and message histories are cached on HF Space persistent
  storage (/data/chat_history) to avoid a Supabase round-trip on every
  chat switch.  Reads hit disk first; DB is only queried on a cold miss.
  Writes go to both the HF cache and Supabase.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Depends, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
import uuid as _uuid
from typing import Optional
from pydantic import BaseModel

from app.services.pdf_loader import extract_text
from app.services.rag import ask_question
from app.services.document_service import ensure_user, ensure_session
from app.db.database import get_conn
from app.models.schemas import (
    UploadResponse,
    CreateSessionRequest,
    SessionResponse, SessionListResponse,
    DeleteResponse,
    MessageResponse, MessageListResponse,
)
from app.core.auth import get_current_user_id
from app.core.config import DEV_USER_EMAIL
import app.services.hf_cache as hf_cache

router = APIRouter()


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    return {"status": "ok"}


# ── Upload (async — dispatches to Celery worker) ─────────────────────────────

from app.services.storage import upload_file_stream_to_r2
from app.worker.tasks import ingest_document_task


class UploadJobResponse(BaseModel):
    """Returned immediately when a PDF is queued for background processing."""
    job_id: str
    filename: str
    status: str = "queued"
    message: str = "Document queued for processing. Poll /upload/status/{job_id} for updates."


class JobStatusResponse(BaseModel):
    job_id: str
    filename: str
    status: str                  # pending | processing | done | failed
    progress_percent: int = 0
    document_id: Optional[str] = None
    error: Optional[str] = None


@router.post("/upload", response_model=UploadJobResponse)
async def upload(
    file: UploadFile = File(...),
    session_id: Optional[str] = Query(None, description="Attach doc to this session."),
    user_id: str = Depends(get_current_user_id),
):
    """
    Upload a PDF and queue it for background processing via Celery + RabbitMQ.

    Returns immediately with a job_id.  Large PDFs (100 pages) are chunked
    and embedded asynchronously — poll GET /upload/status/{job_id} for progress.

    In-memory strategy
    ──────────────────
    The raw PDF bytes are read once into a BytesIO buffer in the API process.
    That same buffer (seeked back to 0 between uses) is passed to:
      1. R2 upload  — stores the original PDF for durable retrieval
      2. extract_text() — parses text with fitz entirely in-memory (no disk I/O)
    The extracted text string is then handed to Celery, which does chunking +
    embedding in the worker process — keeping the API process lightweight.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Ensure the user row exists (idempotent upsert)
    ensure_user(user_id, email=f"anon-{user_id}@thotqen.internal")

    try:
        import io as _io
        # ── Read the entire upload once into memory ──────────────────────────
        # This guarantees both R2 upload and text extraction see the full bytes
        # without any disk writes — critical for HF Space containers.
        raw_bytes = await file.read()
        pdf_buffer = _io.BytesIO(raw_bytes)

        # 1. Upload the raw PDF to R2 for storage
        try:
            pdf_buffer.seek(0)
            r2_url = upload_file_stream_to_r2(pdf_buffer, file.filename)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload to R2: {str(e)}")

        # 2. Extract text in-process (fast; just PDF parsing, not embedding)
        pdf_buffer.seek(0)
        text = extract_text(pdf_buffer)
        if not text.strip():
            raise HTTPException(status_code=422, detail="PDF appears to be empty or unreadable.")

        # 3. Create a document_jobs row (status=pending)
        job_id = str(_uuid.uuid4())
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO document_jobs (id, user_id, filename, r2_url, status)
                    VALUES (%s::uuid, %s::uuid, %s, %s, 'pending')
                    """,
                    (job_id, user_id, file.filename, r2_url),
                )
            conn.commit()

        # 4. Dispatch Celery task — returns immediately
        ingest_document_task.delay(
            job_id=job_id,
            text=text,
            filename=file.filename,
            user_id=user_id,
            session_id=session_id,
            metadata_tags={"r2_url": r2_url},
        )

        return UploadJobResponse(
            job_id=job_id,
            filename=file.filename,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/upload/status/{job_id}", response_model=JobStatusResponse)
def upload_status(
    job_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """
    Poll the processing status of a previously queued PDF upload.

    Status values:
      pending    — job is in the queue, not yet picked up by a worker
      processing — worker is actively chunking + embedding
      done       — ingestion complete; document_id is available
      failed     — ingestion failed; error field contains the reason
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, user_id, filename, status, progress_percent, document_id, error
                FROM   document_jobs
                WHERE  id = %s::uuid
                """,
                (job_id,),
            )
            row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Job not found.")
    if str(row[1]) != user_id:
        raise HTTPException(status_code=403, detail="Not your job.")

    return JobStatusResponse(
        job_id=str(row[0]),
        filename=row[2],
        status=row[3],
        progress_percent=row[4] or 0,
        document_id=str(row[5]) if row[5] else None,
        error=row[6],
    )

import asyncio
import json

@router.get("/upload/stream/{job_id}")
async def upload_stream(
    job_id: str,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    """
    SSE Endpoint for real-time progress updates on an upload job.
    """
    # Verify ownership first
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id FROM document_jobs WHERE id = %s::uuid", (job_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Job not found.")
            if str(row[0]) != user_id:
                raise HTTPException(status_code=403, detail="Not your job.")

    async def event_generator():
        last_progress = -1
        while True:
            if await request.is_disconnected():
                break

            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT status, progress_percent, error, document_id 
                        FROM document_jobs 
                        WHERE id = %s::uuid
                        """, (job_id,)
                    )
                    job_row = cur.fetchone()
            
            if not job_row:
                yield {"event": "error", "data": json.dumps({"error": "Job disappeared"})}
                break
                
            status, progress_percent, error, document_id = job_row
            progress_percent = progress_percent or 0
            
            if progress_percent != last_progress:
                last_progress = progress_percent
                yield {
                    "event": "message", 
                    "data": json.dumps({
                        "job_id": job_id,
                        "status": status,
                        "progress_percent": progress_percent,
                        "error": error,
                        "document_id": str(document_id) if document_id else None
                    })
                }
            
            if status in ("done", "failed"):
                break
                
            await asyncio.sleep(1.0)

    return EventSourceResponse(event_generator())


# ── Ask / Chat ────────────────────────────────────────────────────────────────

@router.get("/ask")
def ask(
    q: str,
    session_id: Optional[str] = Query(
        "00000000-0000-0000-0000-000000000000",
        description="Session UUID for chat history.",
    ),
    user_id: str = Depends(get_current_user_id),
):
    """Legacy GET endpoint for backward compat."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    ensure_user(user_id, email=f"anon-{user_id}@thotqen.internal")
    response_stream = ask_question(q, session_id, user_id=user_id)
    return StreamingResponse(response_stream, media_type="text/plain")


class ChatRequest(BaseModel):
    q: str
    session_id: str = "00000000-0000-0000-0000-000000000000"


@router.post("/chat")
def chat(
    body: ChatRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Streaming chat endpoint — user_id comes from the verified JWT."""
    if not body.q.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    ensure_user(user_id, email=f"anon-{user_id}@thotqen.internal")
    response_stream = ask_question(
        body.q, body.session_id, user_id=user_id
    )
    return StreamingResponse(response_stream, media_type="text/plain")


# ── Sessions ──────────────────────────────────────────────────────────────────

@router.post("/sessions", response_model=SessionResponse)
def create_session(
    body: CreateSessionRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Create a brand-new chat session and return it."""
    ensure_user(user_id, email=f"anon-{user_id}@thotqen.internal")

    new_id = str(_uuid.uuid4())
    title = body.title or "New Chat"
    ensure_session(new_id, user_id, title=title)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, user_id, title, created_at FROM sessions WHERE id = %s::uuid",
                (new_id,),
            )
            row = cur.fetchone()

    session_resp = SessionResponse(
        id=str(row[0]),
        user_id=str(row[1]) if row[1] else None,
        title=row[2],
        created_at=row[3],
    )

    # ── Warm the HF cache immediately so first load is instant ────────────────
    is_anon = user_id.endswith(".internal") or not row[1]
    hf_cache.write_session_cache(
        user_id=user_id,
        session_id=new_id,
        title=title,
        messages=[],
        is_anon=is_anon,
    )
    hf_cache.update_index(
        user_id=user_id,
        session_id=new_id,
        title=title,
        is_anon=is_anon,
        created_at=row[3].isoformat() if row[3] else None,
    )

    return session_resp


@router.delete("/sessions/{session_id}", response_model=DeleteResponse)
def delete_session(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Delete a session — only if it belongs to the caller."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Ownership check
            cur.execute(
                "SELECT user_id FROM sessions WHERE id = %s::uuid",
                (session_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Session not found.")
            if str(row[0]) != user_id:
                raise HTTPException(status_code=403, detail="Not your session.")

            cur.execute("DELETE FROM messages WHERE session_id = %s::uuid", (session_id,))
            cur.execute("DELETE FROM sessions WHERE id = %s::uuid", (session_id,))
        conn.commit()

    # ── Evict from HF cache (both auth + anon dirs searched internally) ───────
    hf_cache.delete_from_cache(user_id=user_id, session_id=session_id)

    return DeleteResponse(deleted=True, id=session_id)


@router.get("/sessions", response_model=SessionListResponse)
def list_sessions(user_id: str = Depends(get_current_user_id)):
    """
    List sessions belonging to the authenticated user.

    Cache strategy
    ──────────────
    1. Try reading index.json from HF disk cache (zero DB cost).
    2. On cold miss, query Supabase and populate both the index and
       any missing session files.
    """
    # ── 1. Try HF cache first ─────────────────────────────────────────────────
    for is_anon in (False, True):
        cached_index = hf_cache.read_index(user_id, is_anon)
        if cached_index:
            sessions = [
                SessionResponse(
                    id=item["id"],
                    user_id=user_id,
                    title=item.get("title"),
                    created_at=item["created_at"],
                )
                for item in cached_index
            ]
            return SessionListResponse(sessions=sessions)

    # ── 2. Cold miss → query Supabase ─────────────────────────────────────────
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, user_id, title, created_at
                FROM   sessions
                WHERE  user_id = %s::uuid
                ORDER  BY created_at DESC
                """,
                (user_id,),
            )
            rows = cur.fetchall()

    sessions = [
        SessionResponse(
            id=str(row[0]),
            user_id=str(row[1]) if row[1] else None,
            title=row[2],
            created_at=row[3],
        )
        for row in rows
    ]

    # ── Populate HF index cache so next request is instant ───────────────────
    if rows:
        is_anon = False  # authenticated users hit this path
        for row in rows:
            hf_cache.update_index(
                user_id=user_id,
                session_id=str(row[0]),
                title=row[2],
                is_anon=is_anon,
                created_at=row[3].isoformat() if row[3] else None,
            )

    return SessionListResponse(sessions=sessions)


@router.patch("/sessions/{session_id}/title")
def rename_session(
    session_id: str,
    title: str = Query(...),
    user_id: str = Depends(get_current_user_id),
):
    """Rename a session title — ownership enforced."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET title = %s WHERE id = %s::uuid AND user_id = %s::uuid",
                (title, session_id, user_id),
            )
        conn.commit()

    # ── Keep HF cache in sync — update title in session file + index ─────────
    cached = hf_cache.read_session_cache(user_id, session_id)
    if cached:
        cached["title"] = title
        # Determine is_anon from what we found
        is_anon = "anon_" in str(
            hf_cache.get_session_path(user_id, session_id, is_anon=False)
        )
        hf_cache.write_session_cache(
            user_id=user_id,
            session_id=session_id,
            title=title,
            messages=cached.get("messages", []),
            is_anon=False,
        )
    # Update index in both possible dirs
    for _is_anon in (False, True):
        idx = hf_cache.read_index(user_id, _is_anon)
        if any(item.get("id") == session_id for item in idx):
            hf_cache.update_index(
                user_id=user_id,
                session_id=session_id,
                title=title,
                is_anon=_is_anon,
            )
            break

    return {"id": session_id, "title": title}


@router.get("/sessions/{session_id}/messages", response_model=MessageListResponse)
def get_session_messages(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """
    Fetch messages for a session — verifies the session belongs to the caller.

    Cache strategy
    ──────────────
    1. Try reading the session JSON from HF disk cache (zero DB cost).
    2. On cold miss, query Supabase, populate HF cache, then return.

    Note: ownership check is still enforced — on a cache hit we verify
    user_id matches what's stored in the file; on a miss the DB check runs.
    """
    # ── 1. Try HF cache first ─────────────────────────────────────────────────
    cached = hf_cache.read_session_cache(user_id, session_id)
    if cached is not None:
        # Ownership guard: file must belong to the requesting user
        if cached.get("user_id") == user_id:
            msgs = [
                MessageResponse(
                    id=m["id"],
                    session_id=session_id,
                    role=m["role"],
                    content=m["content"],
                    created_at=m["created_at"],
                )
                for m in cached.get("messages", [])
            ]
            return MessageListResponse(messages=msgs)

    # ── 2. Cold miss → query Supabase ─────────────────────────────────────────
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Ownership check
            cur.execute(
                "SELECT user_id FROM sessions WHERE id = %s::uuid",
                (session_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Session not found.")
            if str(row[0]) != user_id:
                raise HTTPException(status_code=403, detail="Not your session.")

            cur.execute(
                """
                SELECT id, session_id, role, content, created_at
                FROM   messages
                WHERE  session_id = %s::uuid
                ORDER  BY created_at ASC
                """,
                (session_id,),
            )
            rows = cur.fetchall()

            # Also grab session title for caching
            cur.execute(
                "SELECT title FROM sessions WHERE id = %s::uuid",
                (session_id,),
            )
            title_row = cur.fetchone()
            session_title = title_row[0] if title_row else None

    msgs_db = [
        {
            "id": row[0],
            "role": row[2],
            "content": row[3],
            "created_at": row[4].isoformat() if hasattr(row[4], "isoformat") else str(row[4]),
        }
        for row in rows
    ]

    # ── Populate HF cache so next request is instant ──────────────────────────
    hf_cache.write_session_cache(
        user_id=user_id,
        session_id=session_id,
        title=session_title,
        messages=msgs_db,
        is_anon=False,
    )

    msgs = [
        MessageResponse(
            id=m["id"],
            session_id=session_id,
            role=m["role"],
            content=m["content"],
            created_at=m["created_at"],
        )
        for m in msgs_db
    ]
    return MessageListResponse(messages=msgs)