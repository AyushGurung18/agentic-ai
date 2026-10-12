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

from app.services.pdf_loader import extract_pdf
from app.services.docx_loader import extract_docx
from app.services.rag import ask_question
from app.services.document_service import (
    ensure_user, ensure_session,
    list_documents, get_document, get_documents_by_ids, delete_document,
    get_session_document_ids,
)
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

from app.services.storage import upload_file_stream_to_r2, delete_file_from_r2, download_file_from_r2
from app.worker.tasks import ingest_document_task

MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024  # 20MB — keep in sync with the frontend's client-side check

ACCEPTED_EXTENSIONS = {".pdf": "pdf", ".docx": "docx"}


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
    file: Optional[UploadFile] = File(None),
    session_id: Optional[str] = Query(None, description="Attach doc to this session."),
    user_id: str = Depends(get_current_user_id),
):
    """
    Upload a PDF or DOCX and queue it for background processing via Celery + RabbitMQ.

    Returns immediately with a job_id.  Large documents (150+ pages) are
    chunked and embedded asynchronously — poll GET /upload/status/{job_id}
    for progress.

    In-memory strategy
    ──────────────────
    The raw file bytes are read once into memory in the API process and used
    for both the R2 upload and text extraction — no disk I/O.  The extracted
    text is then handed to Celery, which does chunking + embedding in the
    worker process, keeping the API process lightweight.
    """
    if file is None:
        raise HTTPException(
            status_code=422,
            detail="No file was received. Attach a PDF or DOCX and try again.",
        )
    filename = file.filename or ""
    ext = next((e for e in ACCEPTED_EXTENSIONS if filename.lower().endswith(e)), None)
    if not ext:
        accepted = ", ".join(sorted(ACCEPTED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Only {accepted} files are supported.",
        )
    doc_type = ACCEPTED_EXTENSIONS[ext]
    content_type = "application/pdf" if doc_type == "pdf" else (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

    # Ensure the user row exists (idempotent upsert)
    ensure_user(user_id, email=f"anon-{user_id}@thotqen.internal")

    try:
        import io as _io
        # ── Read the entire upload once into memory ──────────────────────────
        # Read one byte past the cap so an oversized file is rejected cleanly
        # instead of being fully buffered into memory first.
        raw_bytes = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)
        if len(raw_bytes) > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB upload limit.",
            )
        if len(raw_bytes) == 0:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        # 1. Upload the raw file to R2 for storage (viewer + re-download)
        # boto3's upload_fileobj() closes the buffer it's given once the
        # transfer completes — give it its own BytesIO instead of sharing
        # one with text extraction, or the second read fails with
        # "I/O operation on closed file."
        try:
            r2_url = upload_file_stream_to_r2(_io.BytesIO(raw_bytes), filename, content_type=content_type)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload to R2: {str(e)}")

        # 2. Extract text in-process (fast; just parsing, not embedding)
        try:
            if doc_type == "pdf":
                extracted = extract_pdf(_io.BytesIO(raw_bytes))
            else:
                extracted = extract_docx(_io.BytesIO(raw_bytes))
        except Exception as e:
            delete_file_from_r2(r2_url)
            raise HTTPException(
                status_code=422,
                detail=f"Could not parse this {doc_type.upper()} file — it may be corrupted. ({e})",
            )

        if not extracted.text.strip():
            delete_file_from_r2(r2_url)
            # A PDF with zero extractable text is almost always a scanned
            # image with no OCR layer — tell the user plainly rather than
            # silently saving an empty document and calling it a success.
            hint = (
                "This looks like a scanned PDF with no extractable text (no OCR layer). "
                "Upload a text-based PDF, or an OCR'd version of this one."
                if doc_type == "pdf" else
                "This DOCX file has no readable text content."
            )
            raise HTTPException(status_code=422, detail=hint)

        # Pages cross a Celery/JSON boundary — plain dicts, not dataclasses.
        pages_payload = [
            {"page_number": p.page_number, "char_start": p.char_start, "char_end": p.char_end}
            for p in extracted.pages
        ]

        # 3. Create a document_jobs row (status=pending)
        job_id = str(_uuid.uuid4())
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO document_jobs (id, user_id, filename, r2_url, status)
                        VALUES (%s::uuid, %s::uuid, %s, %s, 'pending')
                        """,
                        (job_id, user_id, filename, r2_url),
                    )
                conn.commit()
        except Exception:
            # The R2 object is already durably stored — if the DB row never
            # gets created, delete it rather than leaving it orphaned.
            delete_file_from_r2(r2_url)
            raise

        # 4. Dispatch Celery task — returns immediately
        ingest_document_task.delay(
            job_id=job_id,
            text=extracted.text,
            filename=filename,
            user_id=user_id,
            session_id=session_id,
            metadata_tags={"r2_url": r2_url},
            doc_type=doc_type,
            pages=pages_payload,
            r2_url=r2_url,
        )

        return UploadJobResponse(
            job_id=job_id,
            filename=filename,
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


class AnonymizeRequest(BaseModel):
    text: str


class DeanonymizeRequest(BaseModel):
    text: str
    mapping: dict[str, str]


@router.post("/anonymize")
def anonymize_text(body: AnonymizeRequest):
    """Reversibly anonymizes PII (names, emails, phones, tax IDs, orgs) in contract text."""
    from app.services.anonymizer import anonymizer
    anon_text, mapping, entities = anonymizer.anonymize(body.text)
    return {
        "anonymized_text": anon_text,
        "mapping": mapping,
        "entities": entities
    }


@router.post("/deanonymize")
def deanonymize_text(body: DeanonymizeRequest):
    """Restores placeholders to original entity values using reversible mapping."""
    from app.services.anonymizer import anonymizer
    restored = anonymizer.deanonymize(body.text, body.mapping)
    return {
        "original_text": restored
    }


class ChatRequest(BaseModel):
    q: str
    session_id: str = "00000000-0000-0000-0000-000000000000"
    # "agentic" = Part C tool-calling research mode (search_document /
    # get_section / list_clauses). Only applies when the session has exactly
    # one attached document — ask_question() falls back to "standard"
    # otherwise rather than erroring.
    mode: str = "standard"
    anonymize: bool = False


@router.post("/chat")
def chat(
    body: ChatRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Streaming chat endpoint — user_id comes from the verified JWT."""
    if not body.q.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    query_text = body.q
    if body.anonymize:
        from app.services.anonymizer import anonymizer
        query_text, _, _ = anonymizer.anonymize(query_text)

    ensure_user(user_id, email=f"anon-{user_id}@thotqen.internal")
    response_stream = ask_question(
        query_text, body.session_id, user_id=user_id, mode=body.mode,
    )
    return StreamingResponse(response_stream, media_type="text/plain")


class TruncateLastAnswerRequest(BaseModel):
    content: str  # exactly what the client had rendered at the moment "stop" was clicked


@router.post("/sessions/{session_id}/truncate-last-answer")
def truncate_last_answer(
    session_id: str,
    body: TruncateLastAnswerRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    Called when the user clicks "stop" mid-answer.

    ask_question() generates, quote-verifies, and saves the FULL answer to
    the DB before it ever starts streaming to the client (verification
    needs the complete text) — so without this endpoint, a stopped answer
    looks truncated live but shows the full, un-stopped version again on
    reopen. The spec's "whatever was generated is kept" only held in the
    moment, not across a reopen, without this.

    Trims to the last *complete* QUOTE_MARK pair (an odd count means the
    stop landed mid-quote — the trailing partial quote is dropped rather
    than left dangling/unclosed) and keeps only the quote-metadata entries
    that still fall within the truncated text, since quote_meta[i]
    describes the i-th QUOTE_MARK pair in order (see annotate_quotes()).
    """
    from app.services.quote_annotator import QUOTE_MARK

    content = body.content
    mark_positions = [i for i, ch in enumerate(content) if ch == QUOTE_MARK]
    if len(mark_positions) % 2 == 1:
        content = content[:mark_positions[-1]]
        mark_positions = mark_positions[:-1]
    kept_quote_count = len(mark_positions) // 2

    with get_conn() as conn:
        with conn.cursor() as cur:
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
                SELECT id, metadata FROM messages
                WHERE session_id = %s::uuid AND role = 'assistant'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (session_id,),
            )
            msg_row = cur.fetchone()
            if not msg_row:
                return {"status": "no_message_to_truncate"}

            msg_id, raw_meta = msg_row
            if isinstance(raw_meta, dict):
                meta = raw_meta
            elif raw_meta:
                try:
                    meta = json.loads(raw_meta)
                except (TypeError, ValueError):
                    meta = {}
            else:
                meta = {}
            quotes = meta.get("quotes") or []
            new_meta = {**meta, "quotes": quotes[:kept_quote_count]} if quotes else (meta or None)

            cur.execute(
                "UPDATE messages SET content = %s, metadata = %s::jsonb WHERE id = %s::uuid",
                (content, json.dumps(new_meta) if new_meta else None, msg_id),
            )
        conn.commit()

    hf_cache.invalidate_session_cache(user_id, session_id)
    return {"status": "truncated"}


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

    if body.document_ids:
        # Ownership-checked via get_documents_by_ids — an id the caller
        # doesn't own is silently dropped rather than attached.
        owned = get_documents_by_ids(body.document_ids, user_id)
        for doc_id in body.document_ids:
            if doc_id in owned:
                from app.services.document_service import attach_document_to_session
                attach_document_to_session(new_id, doc_id)

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


# Defined here (ahead of the rest of the document-library section further
# below) because get_session_documents, right after this, needs them at
# decoration time.
class DocumentResponse(BaseModel):
    id: str
    filename: str
    doc_type: str
    status: str
    page_count: Optional[int] = None
    char_count: Optional[int] = None
    created_at: str
    home_session_id: Optional[str] = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]


@router.get("/sessions/{session_id}/documents", response_model=DocumentListResponse)
def get_session_documents(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Which document(s) a session is grounded in — drives the chat page's
    viewer launches, the agentic-mode toggle (single doc only), and the
    "N documents" badge for a multi-document session."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id FROM sessions WHERE id = %s::uuid", (session_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Session not found.")
            if str(row[0]) != user_id:
                raise HTTPException(status_code=403, detail="Not your session.")

    doc_ids = get_session_document_ids(session_id)
    docs = get_documents_by_ids(doc_ids, user_id)
    return DocumentListResponse(documents=[
        DocumentResponse(
            id=doc_id, filename=d["filename"], doc_type=d["doc_type"] or "pdf",
            status=d.get("status") or "ready", page_count=len(d["pages"]) if d.get("doc_type") == "pdf" else None,
            char_count=len(d["content"]),
            created_at=d["created_at"].isoformat() if hasattr(d.get("created_at"), "isoformat") else str(d.get("created_at") or ""),
        )
        for doc_id, d in docs.items()
    ])


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
                    quotes=(m.get("metadata") or {}).get("quotes"),
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
                SELECT id, session_id, role, content, created_at, metadata
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

    def _parse_metadata(raw):
        # psycopg normally adapts jsonb -> dict automatically, but handle a
        # raw string defensively too rather than assume the driver config.
        if raw is None:
            return None
        if isinstance(raw, dict):
            return raw
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return None

    msgs_db = [
        {
            "id": row[0],
            "role": row[2],
            "content": row[3],
            "created_at": row[4].isoformat() if hasattr(row[4], "isoformat") else str(row[4]),
            "metadata": _parse_metadata(row[5]),
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
            quotes=(m.get("metadata") or {}).get("quotes"),
        )
        for m in msgs_db
    ]
    return MessageListResponse(messages=msgs)


# ── Document library ──────────────────────────────────────────────────────────
# Documents are first-class, independent of any one chat session: a session
# just records *which* document(s) it's grounded in (session_documents).
# This is what lets the library list/open/delete uploaded files on its own,
# and what lets multiple documents be selected for one cross-document question.
# (DocumentResponse / DocumentListResponse are defined earlier, above
# get_session_documents, which needs them sooner.)

@router.get("/documents", response_model=DocumentListResponse)
def get_documents(user_id: str = Depends(get_current_user_id)):
    docs = list_documents(user_id)
    return DocumentListResponse(documents=[
        DocumentResponse(
            id=d["id"], filename=d["filename"], doc_type=d["doc_type"] or "pdf",
            status=d["status"] or "ready", page_count=d["page_count"], char_count=d["char_count"],
            created_at=d["created_at"].isoformat() if hasattr(d["created_at"], "isoformat") else str(d["created_at"]),
            home_session_id=d.get("home_session_id"),
        )
        for d in docs
    ])


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
def remove_document(document_id: str, user_id: str = Depends(get_current_user_id)):
    """Delete a document from the library. Cascades to its chunks,
    embeddings, page map, and session links — chat sessions that referenced
    it remain (their message history is independent), they just lose the
    ability to retrieve from a now-deleted document."""
    ok = delete_document(document_id, user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DeleteResponse(deleted=True, id=document_id)


@router.get("/documents/{document_id}/file")
def get_document_file(document_id: str, user_id: str = Depends(get_current_user_id)):
    """Streams the original PDF/DOCX bytes back — used by the in-browser
    viewer (react-pdf / mammoth) for citation highlighting. Proxied through
    this authenticated endpoint rather than a public bucket URL so ownership
    is enforced the same way as every other route."""
    doc = get_document(document_id, user_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    if not doc.get("r2_url"):
        raise HTTPException(status_code=404, detail="Original file is not available for this document.")

    try:
        raw_bytes = download_file_from_r2(doc["r2_url"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not fetch the original file: {e}")

    media_type = (
        "application/pdf" if doc["doc_type"] == "pdf"
        else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    from fastapi.responses import Response as _Response
    return _Response(content=raw_bytes, media_type=media_type)


@router.get("/documents/{document_id}/text")
def get_document_text(document_id: str, user_id: str = Depends(get_current_user_id)):
    """Extracted text + page offset map — used by the DOCX viewer (which
    renders from server-converted HTML, not the raw bytes) and by the
    frontend's own quote-locating fallback."""
    doc = get_document(document_id, user_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {
        "id": doc["id"],
        "filename": doc["filename"],
        "doc_type": doc["doc_type"],
        "content": doc["content"],
        "pages": doc["pages"],
    }


# ── Document comparison (Part B #7) ─────────────────────────────────────────

class CompareRequest(BaseModel):
    document_id_a: str
    document_id_b: str


@router.post("/documents/compare")
def compare_documents_route(
    body: CompareRequest,
    user_id: str = Depends(get_current_user_id),
):
    doc_a = get_document(body.document_id_a, user_id)
    doc_b = get_document(body.document_id_b, user_id)
    if not doc_a or not doc_b:
        raise HTTPException(status_code=404, detail="One or both documents were not found.")

    from app.services.document_compare import compare_documents
    from app.services.rag import get_llm_by_intent

    llm = get_llm_by_intent("complex")
    try:
        result = compare_documents(doc_a["content"], doc_a["filename"], doc_b["content"], doc_b["filename"], llm)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")
    return result